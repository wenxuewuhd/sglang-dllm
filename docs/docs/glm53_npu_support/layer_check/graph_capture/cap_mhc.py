"""mHC decode-shape NPU-graph capture + baked-in-value check."""
import sys, argparse, torch, torch_npu  # noqa
import os as _os
from pathlib import Path as _Path
LC = str(_Path(__file__).resolve().parent.parent)          # .../layer_check
G = str(_Path(__file__).resolve().parent)                  # .../layer_check/graph_capture
# The DSA fp32 references are multi-hundred-MB dumps that do not go in the repo.
# Point SCRATCH at wherever dump_reference.py / reference_dsa.py wrote them.
SP = _os.environ.get("SCRATCH", "/tmp/glm53_scratch")

sys.path.insert(0, LC); sys.path.insert(0, G)
import gcap
import custom_ops  # registers torch.ops.custom
from harness import Case

ap = argparse.ArgumentParser()
ap.add_argument("--case", default="${GLM53_ROOT}/env/goldens/mhc_attn_layer20.pt")
ap.add_argument("-M", type=int, default=16)
a = ap.parse_args()
torch.npu.set_device(0); torch.set_grad_enabled(False)
import gold

case = Case.load(a.case)
m = case.meta
n, d = m["hc_mult"], m["hidden_size"]
print(f"mHC layer case {case.name}  hc_mult={n} hidden={d} sinkhorn={m['hc_sinkhorn_iters']} M={a.M}")

streams = case.inputs["hidden_streams"][: a.M].to("npu", torch.bfloat16).contiguous()
xin     = case.inputs["sublayer_out"][: a.M].to("npu", torch.bfloat16).contiguous()
fn_w    = case.inputs["weight.fn"].to("npu", torch.float32).contiguous()
base    = case.inputs["weight.base"].to("npu", torch.float32).contiguous()
scale   = case.inputs["weight.scale"].to("npu", torch.float32).contiguous()
flat    = streams.reshape(-1, n * d).contiguous()

from sglang.kernels.ops.layernorm.mhc import hc_pre, hc_post

# static output buffers so replay lands somewhere we can read
def step():
    layer_input, h_res, h_post, norm_fused = hc_pre(
        x=flat, hc_fn=fn_w, hc_scale=scale, hc_base=base, hc_mult=n,
        rms_eps=m["rms_norm_eps"], hc_eps=m["hc_eps"],
        sinkhorn_iters=m["hc_sinkhorn_iters"],
        post_mult_value=m["post_mult_value"],
        hc_norm_weight=None, out_norm_weight=None, out_norm_eps=None)
    out = hc_post(x=xin, residual=flat, h_post=h_post, h_res=h_res, hc_mult=n)
    return {"collapsed": layer_input, "post": h_post, "comb": h_res, "out": out}

# ---- A
refA = gcap.snap(step())
cap = gcap.Cap("mhc")
try:
    gout = cap.capture(step)
except Exception as e:
    print("CAPTURE FAILED:", type(e).__name__, str(e)[:600]); raise SystemExit(1)
print("  capture OK")
cap.replay()
badA = gcap.compare("replay(A)", gcap.snap(gout), refA)

# ---- golden: score the REPLAYED graph output against the two-reference budget
gA = {k: v for k, v in gout.items()}
n_, d_ = m["hc_mult"], m["hidden_size"]
S_ = gA["collapsed"].shape[0]
cand = {"pre.post":      gA["post"].reshape(S_, n_),
        "pre.comb":      gA["comb"].reshape(S_, n_, n_),
        "pre.collapsed": gA["collapsed"],
        "post.out":      gA["out"].reshape(S_, n_, d_)}
rc_gold = gold.score(f"mHC graph-replay vs golden (M={a.M})", case, a.M, cand,
                     extra="candidate = tensors read back out of the replayed NPUGraph")

# ---- B : new data into the SAME buffers
g = torch.Generator(device="cpu").manual_seed(7)
newS = (torch.randn(streams.shape, generator=g) * 0.7).to(torch.bfloat16)
newX = (torch.randn(xin.shape, generator=g) * 0.7).to(torch.bfloat16)
streams.copy_(newS.to("npu")); flat.copy_(newS.reshape(-1, n*d).to("npu")); xin.copy_(newX.to("npu"))
refB = gcap.snap(step())          # eager on the new data
cap.replay()
badB = gcap.compare("replay(B)", gcap.snap(gout), refB)
# and confirm B actually differs from A (otherwise the test is vacuous)
delta = gcap.rel(refB["out"], refA["out"])
print(f"    (A vs B differ by rel={delta:.3e} -- test is non-vacuous)" if delta > 1e-3
      else f"    !! A and B nearly identical (rel={delta:.3e}), test is VACUOUS")

# ---- timing
te = gcap.bench(step)
tg = gcap.bench(cap.replay)
print(f"  eager p50 = {te:.3f} ms   graph replay p50 = {tg:.3f} ms   speedup {te/tg:.2f}x")
print("VERDICT:", "PASS" if (not badA and not badB and rc_gold==0) else f"FAIL bake:{badA}{badB} golden_rc={rc_gold}")
