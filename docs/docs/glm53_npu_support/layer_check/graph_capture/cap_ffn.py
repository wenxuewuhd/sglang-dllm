"""Dense FFN (layers 0-2) NPU-graph capture, baked-value check, and golden score.

Decode shape is M = max-running-requests = 16 at TP16 rank 0:
gate_up [M, 2*768], down [M, 768] partial sum.

Three questions, in order:
  1. does capture succeed?
  2. does the replay still track its device inputs (or was a host value baked)?
  3. is the number that comes out of the replayed graph still correct?

(3) is answered against the layer_check two-reference golden, and the full-TP
`out` is assembled by REPLAYING the one captured graph 16 times, copying each
rank's weights into the same buffers between replays -- so a value baked at
capture time would corrupt all 16 terms.
"""
import sys, argparse, torch
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
ap.add_argument("--case", default="/mnt/workspace/y00359136/work/glm53_dev/env/goldens/dense_ffn_layer02_t8192.pt")
ap.add_argument("-M", type=int, default=16)
ap.add_argument("--tp", type=int, default=16)
ap.add_argument("--port", type=int, default=29755)
a = ap.parse_args()
torch.set_grad_enabled(False)
import torch_npu, custom_ops  # noqa
from pathlib import Path
from harness import Case, check, report
import check_dense_ffn as CF

CF.init_single_process_group(a.port)
case = Case.load(Path(a.case))
m = case.meta
print(f"dense FFN case {case.name}: layer {m['layer']} hidden={m['hidden_size']} "
      f"inter={m['intermediate_size']} TP{a.tp} -> per-rank {m['intermediate_size']//a.tp}  M={a.M}")

x = case.inputs["hidden_states"][: a.M].to("npu", torch.bfloat16).contiguous()
w0 = CF.load_shard(Path(CF.DEFAULT_MODEL), m["layer"], 0, a.tp)
mlp = CF.build_mlp(m, 0, a.tp, w0)

box = {}
def gu_hook(_mod, _args, out):
    box["gate_up"] = out[0] if isinstance(out, tuple) else out
def down_pre(_mod, args, _kw):
    box["act"] = args[0]
mlp.gate_up_proj.register_forward_hook(gu_hook)
mlp.down_proj.register_forward_pre_hook(down_pre, with_kwargs=True)

def step():
    out = mlp(x)
    out = out[0] if isinstance(out, tuple) else out
    assert "gate_up" in box and "act" in box, "hooks did not fire -- fused fast path"
    return {"gate_up": box["gate_up"], "act": box["act"], "out": out}

refA = gcap.snap(step())
cap = gcap.Cap("ffn")
try:
    gout = cap.capture(step)
except Exception as e:
    print("  CAPTURE FAILED:", type(e).__name__, str(e)[:800]); raise SystemExit(1)
print("  capture OK")
cap.replay()
badA = gcap.compare("replay(A)", gcap.snap(gout), refA)

# ---- golden, rank 0 tensors, straight out of the replayed graph
r32, r16 = CF.rank_slices(case, 0, a.tp, a.M)
sub = Case(case.name, {}, r32, r16, m)
g_snap = gcap.snap(gout)
rc1 = report(f"dense FFN graph-replay vs golden, rank-0 tensors (M={a.M})",
             check(sub, {"gate_up": g_snap["gate_up"], "act": g_snap["act"]}),
             extra="candidate read back out of the replayed NPUGraph")

# ---- golden, full TP16 `out`, assembled from 16 replays of the SAME graph
acc = torch.zeros(a.M, m["hidden_size"], dtype=torch.float32, device="npu")
for r in range(a.tp):
    wr = CF.load_shard(Path(CF.DEFAULT_MODEL), m["layer"], r, a.tp)
    mlp.gate_up_proj.weight.data.copy_(wr["gate_up_proj.weight"].to(torch.bfloat16))
    mlp.down_proj.weight.data.copy_(wr["down_proj.weight"].to(torch.bfloat16))
    del wr
    cap.replay()
    acc += gout["out"].float()
sub_out = Case(case.name, {}, {"out": case.ref_fp32["out"][: a.M]},
               {"out": case.ref_bf16["out"][: a.M]}, m)
rc2 = report(f"dense FFN graph-replay TP16 sum vs golden (M={a.M})",
             check(sub_out, {"out": acc}),
             extra="all 16 rank partials came out of ONE captured graph, replayed")

# restore rank 0 for the bake test
mlp.gate_up_proj.weight.data.copy_(w0["gate_up_proj.weight"].to(torch.bfloat16))
mlp.down_proj.weight.data.copy_(w0["down_proj.weight"].to(torch.bfloat16))
cap.replay()

# ---- B: new activations into the SAME buffer
g = torch.Generator().manual_seed(17)
x.copy_((torch.randn(x.shape, generator=g) * 0.5).to(torch.bfloat16).to("npu"))
refB = gcap.snap(step())
cap.replay()
badB = gcap.compare("replay(B new x)", gcap.snap(gout), refB)
d = gcap.rel(refB["out"], refA["out"])
print(f"    (A vs B differ rel={d:.3e}){'' if d > 1e-3 else '  !! VACUOUS'}")

te = gcap.bench(step); tg = gcap.bench(cap.replay)
print(f"  eager {te:.3f} ms  graph {tg:.3f} ms  ({te/tg:.2f}x)  "
      f"[CONTENDED MACHINE -- reference only, not a conclusion]")
print("VERDICT:", "PASS" if not badA and not badB and rc1 == 0 and rc2 == 0
      else f"FAIL bake={badA}{badB} golden={rc1},{rc2}")
