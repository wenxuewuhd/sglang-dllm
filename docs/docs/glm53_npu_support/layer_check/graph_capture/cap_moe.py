"""MoE decode-shape NPU-graph capture + baked-in-value check (rank 0 of TP16)."""
import sys, os, argparse, time, torch
import os as _os
from pathlib import Path as _Path
LC = str(_Path(__file__).resolve().parent.parent)          # .../layer_check
G = str(_Path(__file__).resolve().parent)                  # .../layer_check/graph_capture
# The DSA fp32 references are multi-hundred-MB dumps that do not go in the repo.
# Point SCRATCH at wherever dump_reference.py / reference_dsa.py wrote them.
SP = _os.environ.get("SCRATCH", "/tmp/glm53_scratch")

sys.path.insert(0, LC); sys.path.insert(0, G)
sys.path.insert(0, LC + "/../operator_handoff")
import gcap
ap = argparse.ArgumentParser()
ap.add_argument("--case", default="/mnt/workspace/y00359136/work/glm53_dev/env/goldens/moe_layer03_s1024.pt")
ap.add_argument("-M", type=int, default=16)
ap.add_argument("--tp", type=int, default=16)
ap.add_argument("--port", type=int, default=29711)
ap.add_argument("--parts", default="router,topk,dispatch,routed,shared,full")
ap.add_argument("--golden", action="store_true", help="full-TP16 sum, every rank produced by REPLAYING one captured graph")
a = ap.parse_args()
torch.set_grad_enabled(False)
import torch_npu, custom_ops  # noqa
torch.npu.set_device(0)
dev = "npu"
from harness import Case
import check_moe as CM

case = Case.load(a.case); meta = case.meta
print(f"MoE case {case.name}: layer {meta['layer']} {meta['n_routed_experts']}E top-{meta['top_k']} M={a.M}")
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
get_context().set_server_args(ServerArgs(model_path=CM.MODEL, device="npu", tp_size=a.tp,
    dtype="bfloat16", moe_a2a_backend="none", trust_remote_code=True))
from sglang.srt.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                             distributed_init_method=f"tcp://127.0.0.1:{a.port}")
initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")

cfg = CM.build_config(meta)
inter = cfg.moe_intermediate_size // a.tp
t0 = time.time()
w13_all, w2_all, shared_w, gate_w, bias = CM.load_experts_host(meta["layer"], cfg)
print(f"  experts loaded host {time.time()-t0:.0f}s", flush=True)
w13_all_keep, w2_all_keep = w13_all.to(dev), w2_all.to(dev)
w13, w2 = CM.make_shard(w13_all_keep, w2_all_keep, 0, inter, cfg.moe_intermediate_size)
del w13_all, w2_all
torch.npu.empty_cache()
gate = CM.build_router(cfg, gate_w, bias, dev)
topk = CM.build_topk(cfg, gate)
layer, rc, runner, dispatcher = CM.build_moe_runner(cfg, inter, meta["layer"])
qi = CM.attach_weights(layer, w13, w2)
sh_mlp = CM.build_shared(cfg, inter, shared_w, 0, dev)
print(f"  runner {type(runner.activation).__name__} dispatcher {type(dispatcher).__name__}", flush=True)

x_all = case.inputs["hidden_states"]
x = x_all[x_all.shape[0] - a.M:].to(torch.bfloat16).to(dev).contiguous()

parts = set(a.parts.split(","))
results = {}

def report(name, step, mutate):
    print(f"\n--- {name}")
    try:
        refA = gcap.snap(step())
    except Exception as e:
        print("  EAGER FAILED:", type(e).__name__, str(e)[:400]); results[name]="eager-fail"; return
    cap = gcap.Cap(name)
    try:
        gout = cap.capture(step)
    except Exception as e:
        print("  CAPTURE FAILED:", type(e).__name__, str(e)[:700]); results[name]="capture-fail"; return
    print("  capture OK")
    cap.replay()
    badA = gcap.compare("replay(A)", gcap.snap(gout), refA)
    mutate()
    refB = gcap.snap(step())
    cap.replay()
    badB = gcap.compare("replay(B)", gcap.snap(gout), refB)
    k0 = list(refA)[0]
    d = gcap.rel(refB[k0], refA[k0])
    print(f"    (A vs B differ rel={d:.3e}){'' if d>1e-3 else '  !! VACUOUS'}")
    te = gcap.bench(step); tg = gcap.bench(cap.replay)
    print(f"  eager {te:.3f} ms  graph {tg:.3f} ms  speedup {te/tg:.2f}x")
    results[name] = "PASS" if not badA and not badB else f"FAIL {badA} {badB}"

g = torch.Generator().manual_seed(11)
def mut():
    x.copy_((torch.randn(x.shape, generator=g) * 0.6).to(torch.bfloat16).to(dev))

if "router" in parts:
    report("router(gate)", lambda: {"logits": gate(x)}, mut)
if "topk" in parts:
    def s_topk():
        o = topk(x, gate(x))
        return {"ids": o.topk_ids.float(), "w": o.topk_weights.float()}
    report("router+topk", s_topk, mut)
if "routed" in parts:
    def s_routed():
        return {"out": CM.run_routed(x, topk(x, gate(x)), qi, runner, dispatcher)}
    report("routed experts (dispatch+gmm+combine)", s_routed, mut)
if "shared" in parts:
    report("shared expert MLP", lambda: {"out": sh_mlp(x)}, mut)
if "full" in parts:
    def s_full():
        r = CM.run_routed(x, topk(x, gate(x)), qi, runner, dispatcher)
        return {"out": r * cfg.routed_scaling_factor + sh_mlp(x)}
    report("full MoE block", s_full, mut)

print("\n==== MoE summary ====")
for k, v in results.items():
    print(f"  {k:<42} {v}")


# ---------------------------------------------------------------- golden
# The blocking question a self-consistency check cannot answer: is the number
# that falls out of the *replayed* graph still the right number?  So build the
# whole TP16 sum out of graph replays -- capture once on rank 0's shard, then
# for every other rank copy that rank's weights into the same buffers and
# replay.  Nothing is recaptured, so any host value baked at capture time on
# rank 0 would corrupt all 16 terms.
if a.golden:
    import gold
    from harness import Case as HCase
    print("\n--- golden: TP16 sum assembled from graph replays")
    # the bake-check above deliberately overwrote x with noise; put the golden
    # rows back before scoring against the golden.
    x.copy_(x_all[x_all.shape[0] - a.M:].to(torch.bfloat16).to(dev))
    rl = gate(x)
    tko = topk(x, rl)
    routed_buf = torch.zeros(a.M, cfg.hidden_size, dtype=torch.float32, device=dev)
    shared_buf = torch.zeros(a.M, cfg.hidden_size, dtype=torch.float32, device=dev)

    sh_state = sh_mlp                      # rank-0 shared MLP, weights swapped below
    def step_g():
        r = CM.run_routed(x, tko, qi, runner, dispatcher)
        return {"routed": r, "shared": sh_state(x)}

    capg = gcap.Cap("moe-golden")
    gg = capg.capture(step_g)
    print("  capture OK (rank 0 shard)")
    w13_full = w13_all_keep
    w2_full = w2_all_keep
    for r in range(a.tp):
        nw13, nw2 = CM.make_shard(w13_full, w2_full, r, inter, cfg.moe_intermediate_size)
        w13.copy_(nw13); w2.copy_(nw2)
        del nw13, nw2
        rank_sh = CM.build_shared(cfg, inter, shared_w, r, dev)
        for dst, src in zip(sh_state.parameters(), rank_sh.parameters()):
            dst.data.copy_(src.data)
        del rank_sh
        capg.replay()
        routed_buf += gg["routed"].float()
        shared_buf += gg["shared"].float()
        print(f"    rank {r:>2} replayed", flush=True)
    routed = routed_buf * cfg.routed_scaling_factor
    cand = {"routed_out": routed, "shared_out": shared_buf,
            "moe_out": routed + shared_buf, "router_logits": rl.float()}
    rows = slice(x_all.shape[0] - a.M, x_all.shape[0])
    sub = HCase(case.name, {}, {k: v[rows] for k, v in case.ref_fp32.items() if k in cand},
                {k: v[rows] for k, v in case.ref_bf16.items() if k in cand}, meta)
    from harness import check as hcheck, report as hreport
    rc = hreport(f"MoE graph-replay TP16 sum vs golden (M={a.M})", hcheck(sub, cand),
                 extra="all 16 rank terms came out of ONE captured graph, replayed")
    print("GOLDEN VERDICT:", "PASS" if rc == 0 else "FAIL")
