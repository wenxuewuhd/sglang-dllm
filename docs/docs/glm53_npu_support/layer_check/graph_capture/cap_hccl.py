"""Can an HCCL all-reduce be captured into an NPUGraph and replayed?

TP16 puts one all-reduce per layer INSIDE the captured region, so if this does
not work, nothing else about single-layer capture matters.  Two ranks on the two
free dies is enough to answer it: the question is whether the collective can be
recorded and replayed at all, not how fast it is.

Same three questions as the layer probes:
  A  replay on the capture inputs == eager
  B  overwrite the same device buffers, replay == eager on the NEW inputs
     (a collective whose payload was frozen at capture time fails here)
"""
import os, sys, torch, torch_npu  # noqa
import torch.distributed as dist

rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.npu.set_device(rank)
torch.set_grad_enabled(False)
dist.init_process_group(backend="hccl", rank=rank, world_size=world)
grp = dist.group.WORLD

N = 4096
x = torch.full((N, N), float(rank + 1), device="npu", dtype=torch.bfloat16)
w = torch.randn(N, N, device="npu", dtype=torch.bfloat16)

def step():
    y = x @ w                       # a real kernel before the collective
    dist.all_reduce(y, group=grp)   # the thing under test
    return y

def eager():
    torch.npu.synchronize(); dist.barrier()
    return step().float().cpu().clone()

refA = eager()

pool = torch.npu.graph_pool_handle()
stream = torch.npu.Stream()
for _ in range(3):
    step()
torch.npu.synchronize(); dist.barrier()
g = torch.npu.NPUGraph()
try:
    with torch.npu.graph(g, pool=pool, stream=stream, auto_dispatch_capture=True):
        out = step()
    torch.npu.synchronize()
except Exception as e:
    if rank == 0:
        print("CAPTURE FAILED:", type(e).__name__, str(e)[:900], flush=True)
    dist.destroy_process_group(); raise SystemExit(1)
if rank == 0:
    print("  capture OK (HCCL all_reduce inside the graph)", flush=True)

def rel(a, b):
    return (a - b).norm().item() / max(b.norm().item(), 1e-12)

dist.barrier(); g.replay(); torch.npu.synchronize(); dist.barrier()
gA = out.float().cpu().clone()
okA = torch.equal(gA, refA)

# B: new payload into the SAME buffers
x.fill_(float(rank + 1) * 3.5)
refB = eager()
dist.barrier(); g.replay(); torch.npu.synchronize(); dist.barrier()
gB = out.float().cpu().clone()
okB = torch.equal(gB, refB)

# C: does the result actually depend on the OTHER rank?  Only rank 1 changes.
if rank == 1:
    x.fill_(99.0)
refC = eager()
dist.barrier(); g.replay(); torch.npu.synchronize(); dist.barrier()
gC = out.float().cpu().clone()
okC = torch.equal(gC, refC)

if rank == 0:
    print(f"    replay(A same input)            {'bitwise' if okA else f'MISMATCH rel={rel(gA, refA):.3e}'}")
    print(f"    replay(B new local payload)     {'bitwise' if okB else f'MISMATCH rel={rel(gB, refB):.3e}'}")
    print(f"    replay(C only the PEER changed) {'bitwise' if okC else f'MISMATCH rel={rel(gC, refC):.3e}'}")
    print(f"    (A->B rel={rel(refB, refA):.3e}  B->C rel={rel(refC, refB):.3e} -- non-vacuous if both > 0)")
    print("VERDICT:", "PASS" if (okA and okB and okC) else "FAIL")

g.reset()
dist.barrier()
dist.destroy_process_group()
torch.npu.synchronize(); torch.npu.empty_cache()
