"""KDA decode under NPU-graph capture, driven through the RUNNER's graph contract
(init_cuda_graph_state / init_forward_metadata_out_graph) rather than the eager
metadata path, and replayed with a PADDED batch -- the two things cap_kda.py
did not cover.

KDA decode MUTATES the conv/ssm cache, so every timed or scored run is
sandwiched between a save/restore of the two state tensors -- otherwise
'eager then replay' would be comparing step N with step N+1.
"""
import os
import sys, argparse, time, torch
import os as _os
from pathlib import Path as _Path
LC = str(_Path(__file__).resolve().parent.parent)          # .../layer_check
G = str(_Path(__file__).resolve().parent)                  # .../layer_check/graph_capture
# The DSA fp32 references are multi-hundred-MB dumps that do not go in the repo.
# Point SCRATCH at wherever dump_reference.py / reference_dsa.py wrote them.
SP = _os.environ.get("SCRATCH", "/tmp/glm53_scratch")

sys.path.insert(0, LC); sys.path.insert(0, G)
import gcap
ap = argparse.ArgumentParser()
ap.add_argument("--case", default=f"{_GLM53_ROOT}/env/goldens/kda_layer00.pt")
ap.add_argument("--tp", type=int, default=16)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--ctx", type=int, default=32768)
ap.add_argument("--prefill", type=int, default=0)
a = ap.parse_args()
torch.set_grad_enabled(False)
import torch_npu, custom_ops  # noqa
torch.npu.set_device(0)
DEV = "npu"
from harness import Case
import check_kda as CK
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
get_context().set_server_args(ServerArgs(model_path=str(CK.MODEL), device="npu", tp_size=1))

case = Case.load(a.case); meta = case.meta
print(f"KDA case {case.name}: {meta['num_heads']}H hd={meta['head_dim']} conv_k={meta['conv_kernel']} layer={meta['layer']}")
full = CK.load_layer_weights(CK.MODEL, int(meta["layer"]))
w = CK.ShardedKDAWeights(full, rank=0, tp=a.tp)
page = 64
max_ctx = ((a.ctx + page) // page + 1) * page
runner, backend = CK.build_backend(tp=a.tp, batch=a.batch, max_context_len=max_ctx,
    num_heads=int(meta["num_heads"]), head_dim=int(meta["head_dim"]),
    conv_kernel=int(meta["conv_kernel"]), page_size=page)
layer = CK.make_layer(w, meta["gate_lower_bound"])
from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
o_norm = FusedRMSNormGated(int(meta["head_dim"]), eps=float(meta["rms_norm_eps"]), activation="sigmoid").to(DEV)
o_norm.weight.data.copy_(w.o_norm_weight)
print(f"  TP{a.tp} rank0: local heads = {w.num_heads}")

hidden = case.inputs["hidden_states"].to(DEV, torch.bfloat16).contiguous()
H = hidden.shape[1]
if a.prefill <= 0:
    a.prefill = int(meta["prefill"])
pass
from sglang.srt.model_executor.forward_batch_info import ForwardMode

# MambaSlotAllocator reserves mamba slot 0 as the dummy write target for padded
# tokens (mem_cache/allocator/mamba.py: free_slots = arange(1, size+1)), and
# ReqToTokenPool reserves req slot 0 for the same reason.  check_kda's fixture
# hands requests slots 0..bs-1, which puts a REAL request on the reserved slot
# and makes the padding question untestable.  Shift by one.
_mk = CK.make_forward_batch
def make_forward_batch(**kw):
    b = _mk(**kw)
    n = b.req_pool_indices.shape[0]
    runner_ = kw["runner"]
    runner_.req_to_token_pool.req_index_to_mamba_index_mapping[
        b.req_pool_indices.to(torch.int64)
    ] = (torch.arange(n, dtype=torch.int32, device=b.req_pool_indices.device) + 1)
    return b
CK.make_forward_batch = make_forward_batch

# --- prefill so the states are real, not zeros
rows_p = hidden[: a.prefill].repeat(1, 1)
seen = [0] * a.batch
fbP = CK.make_forward_batch(mode=ForwardMode.EXTEND, runner=runner,
    seq_lens=[a.prefill] * a.batch, input_lens=[a.prefill] * a.batch,
    max_context_len=max_ctx, page_size=page)
backend.init_forward_metadata(fbP)
xp = torch.cat([hidden[: a.prefill] for _ in range(a.batch)], 0)
mq, fg, be, og = w.project(xp)
backend.forward_extend(layer=layer, forward_batch=fbP, mixed_qkv=mq,
                       a=fg.unsqueeze(0), b=be.unsqueeze(0))
torch.npu.synchronize()
print(f"  prefill {a.prefill} x {a.batch} done", flush=True)

# --- decode step, static buffers
seq_lens = [a.prefill + 1] * a.batch
fb = CK.make_forward_batch(mode=ForwardMode.DECODE, runner=runner, seq_lens=seq_lens,
    input_lens=[1] * a.batch, max_context_len=max_ctx, page_size=page)
backend.init_cuda_graph_state(max_bs=a.batch, max_num_tokens=a.batch)
backend.init_forward_metadata_out_graph(fb, in_capture=True)   # capture-time prep
def prep(): backend.init_forward_metadata_out_graph(fb, in_capture=False)
x = torch.stack([hidden[(a.prefill + i) % hidden.shape[0]] for i in range(a.batch)]).contiguous()

cache = runner.req_to_token_pool.mamba2_layer_cache(0)
conv, ssm = cache.conv[0], cache.temporal
saved = (conv.clone(), ssm.clone())
def restore():
    conv.copy_(saved[0]); ssm.copy_(saved[1])

def step():
    mixed_qkv, forget_gate, beta, o_gate = w.project(x)
    core = backend.forward_decode(layer=layer, forward_batch=fb, mixed_qkv=mixed_qkv,
                                  a=forget_gate, b=beta.unsqueeze(0))
    g = o_gate.unflatten(-1, (-1, w.head_dim))
    core = o_norm(core, g).squeeze(0).flatten(-2)
    return {"out": core @ w.wo.T}

def run_eager():
    restore(); prep(); r = gcap.snap(step()); r["conv"] = conv.detach().float().cpu().clone(); r["ssm"] = ssm.detach().float().cpu().clone(); return r

refA = run_eager()
cap = gcap.Cap("kda")
restore()
try:
    gout = cap.capture(lambda: step(), warmup=0)   # warmups mutate state; do them by hand
except Exception as e:
    print("  CAPTURE FAILED:", type(e).__name__, str(e)[:900]); raise SystemExit(1)
print("  capture OK (0 warmup)")
restore(); prep(); cap.replay()
gA = gcap.snap(gout); gA["conv"] = conv.detach().float().cpu().clone(); gA["ssm"] = ssm.detach().float().cpu().clone()
badA = gcap.compare("replay(A)", gA, refA)

# --- B: new hidden rows into the same buffer
g_ = torch.Generator().manual_seed(23)
x.copy_((torch.randn(x.shape, generator=g_) * 0.5).to(torch.bfloat16).to(DEV))
refB = run_eager()
restore(); prep(); cap.replay()
gB = gcap.snap(gout); gB["conv"] = conv.detach().float().cpu().clone(); gB["ssm"] = ssm.detach().float().cpu().clone()
badB = gcap.compare("replay(B)", gB, refB)
d = gcap.rel(refB["out"], refA["out"]); print(f"    (A vs B differ rel={d:.3e}){'' if d>1e-3 else '  !! VACUOUS'}")

# --- timing (state drifts, but shapes/cost do not)
restore(); te = gcap.bench(step)
restore(); tg = gcap.bench(cap.replay)
print(f"  eager {te:.3f} ms  graph {tg:.3f} ms  speedup {te/tg:.2f}x")

# --- C: PADDED replay.  A captured graph has a fixed width; when fewer requests
# are running the runner zeroes the tail rows (seq_lens -> 0, req_pool_indices
# -> 0).  Padding rows therefore *name request 0*.  If the mamba state scatter
# does not exclude them, request 0's conv/ssm state is clobbered -- silently.
REAL = max(1, a.batch // 4)
# Dirty the reserved dummy slot 0 first.  Otherwise it is all-zero, a padded
# write of zeros is value-preserving, and "slot 0 did not move" would prove
# nothing about where the padded rows actually landed.
conv[0].normal_(0, 1); ssm[0].normal_(0, 1)
saved = (conv.clone(), ssm.clone())
fbp = CK.make_forward_batch(mode=ForwardMode.DECODE, runner=runner,
    seq_lens=[a.prefill + 1] * REAL, input_lens=[1] * REAL,
    max_context_len=max_ctx, page_size=page)
restore()
backend.init_forward_metadata(fbp)
x_small = x[:REAL].contiguous()
mq, fg2, b2, og = w.project(x_small)
core = backend.forward_decode(layer=layer, forward_batch=fbp, mixed_qkv=mq,
                              a=fg2, b=b2.unsqueeze(0))
gg = og.unflatten(-1, (-1, w.head_dim))
o_small = (o_norm(core, gg).squeeze(0).flatten(-2) @ w.wo.T).float().cpu().clone()
conv_small = conv.detach().float().cpu().clone(); ssm_small = ssm.detach().float().cpu().clone()

# now the padded replay: rows [REAL:] get seq_len 0 and req_pool_index 0
restore()
fb.seq_lens[REAL:] = 0
fb.seq_lens_cpu[REAL:] = 0
fb.req_pool_indices[REAL:] = 0
fb.num_padding = a.batch - REAL          # decode_cuda_graph_runner.py:188
x[REAL:] = 0
prep()
cap.replay(); torch.npu.synchronize()
o_pad = gout["out"].float().cpu().clone()
conv_pad = conv.detach().float().cpu().clone(); ssm_pad = ssm.detach().float().cpu().clone()
eo = gcap.rel(o_pad[:REAL], o_small)
# which mamba slots actually moved?
import numpy as _np

# Root of the workspace holding env/, the goldens and the sibling checkouts.
_GLM53_ROOT = os.environ.get("GLM53_ROOT") or os.environ.get("GLM53_WORKSPACE") or ""
base_conv = saved[0].float().cpu(); base_ssm = saved[1].float().cpu()
def moved(t, b):
    return [i for i in range(t.shape[0])
            if not torch.equal(t[i], b[i])]
print(f"    [diag] num_padding attr on fb = {getattr(fb, 'num_padding', 'ABSENT')!r}")
print(f"    [diag] state_indices_list[{a.batch-1}] after prep = "
      f"{backend.state_indices_list[a.batch-1].tolist()}")
print(f"    [diag] conv slots moved by unpadded eager : {moved(conv_small, base_conv)}")
print(f"    [diag] conv slots moved by padded replay  : {moved(conv_pad, base_conv)}")
print(f"    [diag] ssm  slots moved by unpadded eager : {moved(ssm_small, base_ssm)}")
print(f"    [diag] ssm  slots moved by padded replay  : {moved(ssm_pad, base_ssm)}")
print(f"    [diag] slot0 conv padded-vs-unpadded rel  = {gcap.rel(conv_pad[0], conv_small[0]):.3e}")
print(f"    [diag] slot0 ssm  padded-vs-unpadded rel  = {gcap.rel(ssm_pad[0], ssm_small[0]):.3e}")
# Slot 0 is the reserved dummy write target -- padded rows are SUPPOSED to land
# there and nothing reads it, so score slots 1.. only.  (Scoring slot 0 too
# would call a correct implementation a failure.)
ec = gcap.rel(conv_pad[1:], conv_small[1:]); es = gcap.rel(ssm_pad[1:], ssm_small[1:])
print(f"    [diag] reserved slot 0 moved by padded replay (expected): "
      f"conv {gcap.rel(conv_pad[0], conv_small[0]):.3e}  ssm {gcap.rel(ssm_pad[0], ssm_small[0]):.3e}")
print(f"\n  padded replay: bs={a.batch}, {REAL} real rows, {a.batch-REAL} padding rows")
print(f"    out[:{REAL}] vs unpadded eager   rel={eo:.3e}  {'ok' if eo < 1e-6 else '<-- MISMATCH'}")
print(f"    conv state slots 1..            rel={ec:.3e}  {'ok' if ec < 1e-6 else '<-- CLOBBERED'}")
print(f"    ssm  state slots 1..            rel={es:.3e}  {'ok' if es < 1e-6 else '<-- CLOBBERED'}")
badC = [] if (eo < 1e-6 and ec < 1e-6 and es < 1e-6) else ["padded replay"]

print("VERDICT:", "PASS" if not badA and not badB and not badC else f"FAIL A={badA} B={badB} C={badC}")
