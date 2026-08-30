#!/usr/bin/env python
"""Stage A for the MoE layers: CPU/fp32 reference from the real HF implementation.

Runs GLM-5.3-Flash on CPU with real weights up to the MoE layer under test, captures
the tensor that layer's ``mlp`` actually receives, and evaluates the *reference* MoE
twice on it -- once fp32, once bf16 -- so ``harness.check`` has a measured noise floor
(``ACCEPTANCE.md`` §A).  Saves a ``harness.Case``.

The model does not fit in memory 45 layers at a time (one MoE layer alone is 29 GB in
fp32), so layers are **streamed**: build layer i on meta, fill it from the checkpoint,
run it, free it.  The decoder-layer plumbing that ``Glm5NextTextModel.forward`` does is
reproduced here (it is three lines: an all-ones bool mask, arange position ids, and
``position_embeddings=None`` because GLM is NoPE).

    $ROOT/.venv-ref/bin/python reference_moe.py --layer 3 --seq 1024 \
        --out $ROOT/goldens/moe_layer03.pt

``--defects`` additionally prints the measurements for the two PLAN §4 defects, which
are both properties of the *reference* and so belong on this side:

  1. the DeepEP routed path drops ``swiglu_limit=10.0``
     (``moe_runner/ascend.py`` forwards ``gemm1_clamp_limit``, which GLM leaves None),
  2. the NPU router GEMM runs in bf16 while the config asks for float32
     (``deepseek_v2.py`` MoEGate: the non-CUDA branch is a plain bf16 ``F.linear``).

Both are measured against this case's own noise floor, because "the error is small"
means nothing without the floor next to it.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Case  # noqa: E402
from tolerance import ABS_MIN, SLACK, noise_floor, rel_err  # noqa: E402

MODEL = os.environ.get("GLM53_MODEL", "/mnt/workspace/models/GLM-5.3-Flash-BF16")
CKPT_PREFIX = "model.language_model."


# --------------------------------------------------------------------------- weights
class Checkpoint:
    """Lazy per-tensor reader. Handles stay open; the OS page cache does the rest."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.map = json.loads(
            (self.path / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self._open = {}

    def get(self, name: str) -> torch.Tensor:
        full = CKPT_PREFIX + name
        shard = self.map[full]
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.path / shard), framework="pt")
        return self._open[shard].get_tensor(full)

    def has(self, name: str) -> bool:
        return CKPT_PREFIX + name in self.map


def load_layer_(layer: torch.nn.Module, li: int, ck: Checkpoint, dtype=torch.float32):
    """Fill one materialised decoder layer from the checkpoint.

    The checkpoint names differ from the module names in three places, all of them
    load-bearing: the hyper-connections (``hc_attn_*`` -> ``attn_hc.*``), the KDA
    depthwise conv (three separate q/k/v conv weights -> one stacked ``conv1d``), and
    the MoE experts (per-expert 2-D matrices -> one stacked 3-D parameter).
    """
    sd = layer.state_dict()
    p = f"layers.{li}."
    done = set()

    def put(key: str, tensor: torch.Tensor):
        assert sd[key].shape == tensor.shape, (key, sd[key].shape, tensor.shape)
        sd[key].copy_(tensor)
        done.add(key)

    # hyper-connections
    for site, pre in (("attn_hc", "hc_attn"), ("ffn_hc", "hc_ffn")):
        for sub in ("fn", "base", "scale"):
            put(f"{site}.{sub}", ck.get(f"{p}{pre}_{sub}"))
    for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        put(n, ck.get(p + n))

    # attention
    sa = "self_attn."
    if ck.has(p + sa + "q_conv1d.weight"):  # linear (KDA) layer
        for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "b_proj.weight",
                  "g_a_proj.weight", "g_b_proj.weight", "o_proj.weight",
                  "o_norm.weight"):
            put(sa + n, ck.get(p + sa + n))
        put(sa + "conv1d.weight", torch.cat(
            [ck.get(p + sa + f"{c}_conv1d.weight") for c in ("q", "k", "v")], 0
        ))
        for n in ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight"):
            put(sa + "forget_gate." + n, ck.get(p + sa + n))
    else:  # deepseek_sparse_attention layer
        for n in ("q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
                  "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight",
                  "kv_b_proj.weight", "o_proj.weight"):
            put(sa + n, ck.get(p + sa + n))
        for n in ("index_kpool_compress_ape", "index_kpool_compress_gate",
                  "wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias",
                  "weights_proj.weight"):
            put(sa + "indexer." + n, ck.get(p + sa + "indexer." + n))

    # mlp
    if "mlp.experts.gate_up_proj" in sd:
        n_experts, two_i, _ = sd["mlp.experts.gate_up_proj"].shape
        inter = two_i // 2
        gu = sd["mlp.experts.gate_up_proj"]
        dn = sd["mlp.experts.down_proj"]
        for e in range(n_experts):
            ep = f"{p}mlp.experts.{e}."
            gu[e, :inter].copy_(ck.get(ep + "gate_proj.weight"))
            gu[e, inter:].copy_(ck.get(ep + "up_proj.weight"))
            dn[e].copy_(ck.get(ep + "down_proj.weight"))
        done.update(("mlp.experts.gate_up_proj", "mlp.experts.down_proj"))
        put("mlp.gate.weight", ck.get(p + "mlp.gate.weight"))
        put("mlp.gate.e_score_correction_bias",
            ck.get(p + "mlp.gate.e_score_correction_bias"))
        for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            put("mlp.shared_experts." + n, ck.get(p + "mlp.shared_experts." + n))
    else:
        for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            put("mlp." + n, ck.get(p + "mlp." + n))

    missing = [k for k in sd if k not in done]
    assert not missing, f"layer {li}: unloaded params {missing}"


# --------------------------------------------------------------------------- corpus
def make_input_ids(seq: int):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    repo = Path(__file__).resolve().parents[4]
    srcs = ["README.md", "docs/CONTRIBUTING.md", "docs/AGENTS.md",
            "python/sglang/srt/layers/attention/dsa/dsa_indexer.py",
            "python/sglang/srt/managers/scheduler.py", "test/README.md",
            "CODE_OF_CONDUCT.md"]
    text = ""
    for s in srcs:
        f = repo / s
        if f.exists():
            text += f.read_text(errors="ignore") + "\n\n"
    ids = tok(text, return_tensors="pt").input_ids[0]
    assert ids.numel() >= seq, f"corpus has only {ids.numel()} tokens, need {seq}"
    return ids[:seq][None]


# --------------------------------------------------------------------------- run
def build_config():
    from transformers.models.glm5_next.configuration_glm5_next import (
        Glm5NextTextConfig,
    )

    tc = json.loads((Path(MODEL) / "config.json").read_text())["text_config"]
    cfg = Glm5NextTextConfig(**tc)
    cfg._attn_implementation = "eager"
    return cfg


def patch_cpu_conv1d():
    """CPU ``F.conv1d`` overflows an internal immediate for the long depthwise conv on
    this aarch64 build; replace it with an exact shift-and-sum (from int8check/extract.py)."""
    import transformers.models.glm5_next.modeling_glm5_next as MM
    from transformers.activations import ACT2FN

    def _fn(hidden_states, weight, bias=None, activation=None, **kw):
        _, _, S = hidden_states.shape
        K = weight.shape[-1]
        x = hidden_states.to(weight.dtype)
        xp = F.pad(x, (K - 1, 0))
        out = sum(weight[..., j].reshape(1, -1, 1) * xp[:, :, j:j + S] for j in range(K))
        if bias is not None:
            out = out + bias[None, :, None]
        if activation is not None:
            out = ACT2FN[activation](out)
        return out.to(hidden_states.dtype)

    MM.causal_conv1d_fn = _fn


def run_to_layer(cfg, input_ids, layer_id: int, ck: Checkpoint, verbose=True):
    """Stream layers 0..layer_id, return the tensor ``layers[layer_id].mlp`` receives
    plus the materialised layer itself (still holding fp32 weights)."""
    from transformers import DynamicCache
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextDecoderLayer,
    )

    B, S = input_ids.shape
    emb = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
    emb.weight.data = ck.get("embed_tokens.weight").float()
    hidden = emb(input_ids).unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    del emb
    gc.collect()

    attn_mask = torch.ones(B, S, dtype=torch.bool)
    position_ids = torch.arange(S).unsqueeze(0)
    cache = DynamicCache(config=cfg)

    captured = {}
    topk_indices = None
    layer = None
    for li in range(layer_id + 1):
        t0 = time.time()
        with torch.device("meta"):
            layer = Glm5NextTextDecoderLayer(cfg, li)
        layer = layer.to_empty(device="cpu").float().eval()
        load_layer_(layer, li, ck)
        t1 = time.time()
        if li == layer_id:
            layer.mlp.register_forward_hook(
                lambda m, inp, out: captured.update(x=inp[0].detach().clone(),
                                                    y=out.detach().clone())
            )
        hidden, topk_indices = layer(
            hidden,
            attention_mask=attn_mask,
            position_ids=position_ids,
            position_embeddings=None,
            past_key_values=cache,
            use_cache=True,
            prev_topk_indices=topk_indices,
        )
        if verbose:
            print(f"  layer {li:>2} ({cfg.layer_types[li][:6]}/"
                  f"{cfg.mlp_layer_types[li]}): load {t1-t0:.1f}s "
                  f"fwd {time.time()-t1:.1f}s", flush=True)
        if li != layer_id:
            del layer
            layer = None
            gc.collect()
    assert "x" in captured, "the mlp hook never fired"
    return captured["x"], captured["y"], layer


# --------------------------------------------------------------------------- MoE ref
def moe_parts(mlp, x):
    """``Glm5NextTextMoE.forward`` decomposed, so the pieces can be scored separately.

    Same calls in the same order as the real module -- only the addition at the end is
    unpacked, because a single fused number cannot say whether the routed half or the
    shared half is wrong.
    """
    assert x.dim() == 2, "pass the MoE input flattened to [tokens, hidden]"
    router_logits, topk_w, topk_i = mlp.gate(x)
    routed = mlp.experts(x, topk_i, topk_w)
    shared = mlp.shared_experts(x)
    dense_w = torch.zeros(x.shape[0], mlp.gate.num_experts, dtype=torch.float32)
    dense_w.scatter_(1, topk_i.long(), topk_w.float())
    return {
        "moe_out": (routed + shared).float(),
        "routed_out": routed.float(),
        "shared_out": shared.float(),
        "router_logits": router_logits.float(),
        "topk_weight_dense": dense_w,
    }, topk_i.long()


# --------------------------------------------------------------------------- defects
def _router_head(scores_for_choice, scores, cfg, num_experts):
    """HF ``Glm5NextTextTopkRouter`` from the sigmoid onwards. n_group == 1 for GLM, so
    the group mask is a no-op, but keep it: a config change must not silently pass."""
    ng, tg = cfg.n_group, cfg.topk_group
    group_scores = (
        scores_for_choice.view(-1, ng, num_experts // ng).topk(2, dim=-1)[0].sum(dim=-1)
    )
    gi = torch.topk(group_scores, k=tg, dim=-1, sorted=False)[1]
    gm = torch.zeros_like(group_scores).scatter_(1, gi, 1)
    mask = gm.unsqueeze(-1).expand(-1, ng, num_experts // ng).reshape(-1, num_experts)
    sfc = scores_for_choice.masked_fill(~mask.bool(), float("-inf"))
    idx = torch.topk(sfc, k=cfg.num_experts_per_tok, dim=-1, sorted=False)[1]
    w = scores.gather(1, idx)
    if cfg.norm_topk_prob:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return idx, w * cfg.routed_scaling_factor


def set_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean |A ∩ B| / k over rows. Set equality, not index equality -- topk is
    ``sorted=False`` and ties at the k-th place are legitimately either way."""
    k = a.shape[1]
    inter = (a.sort(-1).values.unsqueeze(-1) == b.sort(-1).values.unsqueeze(-2)).any(-1)
    return inter.sum(-1).float().mean().item() / k


def row_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Per-token relative error. A global L2 hides the shape of the damage: a rare
    outlier token that is badly wrong and a uniform small drift give the same number."""
    d = (a.double() - b.double()).norm(dim=-1)
    n = b.double().norm(dim=-1).clamp_min(1e-30)
    r = (d / n)
    q = torch.quantile(r, torch.tensor([0.5, 0.99], dtype=torch.float64))
    return {
        "rows_changed": int((d > 0).sum()),
        "rows": int(r.numel()),
        "p50": q[0].item(),
        "p99": q[1].item(),
        "max": r.max().item(),
    }


def probe_defects(mlp32, x32, cfg, ref32, ref16, floors):
    """Numbers for the two PLAN §4 defects.

    Each defect is scored against *its own* two-reference floor rather than the case
    floor, so the comparison holds one thing constant and varies exactly one:

      defect 1 (clamp)  routing is frozen; clamp on/off x fp32/bf16 arithmetic.
      defect 2 (router) arithmetic is frozen at fp32; the router GEMM dtype varies.

    Everything is CPU/torch: what is being measured is a modelling difference, and
    device noise would only blur it.
    """
    out = {}
    flat32 = x32
    x16 = x32.to(torch.bfloat16)
    n_exp = mlp32.gate.num_experts
    experts = mlp32.experts
    limit = float(cfg.swiglu_limit)
    shared32 = ref32["shared_out"]

    # ---- routing, computed the two ways -------------------------------------------
    # Both sides get the SAME bf16 activations -- that is what a bf16 serve feeds the
    # router -- so the only variable is the dtype the GEMM is done and stored in.
    W = mlp32.gate.weight
    bias = mlp32.gate.e_score_correction_bias.float()
    lg32 = F.linear(x16.float(), W.float())
    lg16 = F.linear(x16, W.to(torch.bfloat16)).float()
    i32, w32 = _router_head(lg32.sigmoid() + bias, lg32.sigmoid(), cfg, n_exp)
    i16, w16 = _router_head(lg16.sigmoid() + bias, lg16.sigmoid(), cfg, n_exp)

    # The fp32 expert weights are the master copy and are never cast in place: a
    # bf16 round trip through the parameters would make every later fp32 run measure
    # its own rounding instead of the thing under test.
    gu32, dn32 = experts.gate_up_proj.data, experts.down_proj.data

    def run_experts(idx, w, dtype, clamp: bool):
        orig = experts._apply_gate
        if not clamp:
            def unclamped(gu):
                g, u = gu.chunk(2, dim=-1)
                return F.silu(g) * u
            experts._apply_gate = unclamped
        if dtype is not torch.float32:
            experts.gate_up_proj.data = gu32.to(dtype)
            experts.down_proj.data = dn32.to(dtype)
        try:
            return experts(x32.to(dtype), idx, w.to(torch.float32)).float()
        finally:
            experts._apply_gate = orig
            experts.gate_up_proj.data = gu32
            experts.down_proj.data = dn32

    # ---- defect 1: routed experts silently lose swiglu_limit ----------------------
    a = run_experts(i32, w32, torch.float32, clamp=True)    # reference
    b = run_experts(i32, w32, torch.float32, clamp=False)   # the defect, noise-free
    c = run_experts(i32, w32, torch.bfloat16, clamp=True)   # correct impl in bf16
    d = run_experts(i32, w32, torch.bfloat16, clamp=False)  # what DeepEP+ascend does
    floor_c = rel_err(c, a)
    n_touch = n_over = 0
    mx = 0.0
    for e in range(n_exp):
        rows = (i32 == e).any(-1).nonzero().flatten()
        if rows.numel() == 0:
            continue
        gu = F.linear(flat32[rows], gu32[e])
        g, u = gu.chunk(2, dim=-1)
        n_over += int((g > limit).sum()) + int((u.abs() > limit).sum())
        n_touch += gu.numel()
        mx = max(mx, g.max().item(), u.abs().max().item())
    out["clamp"] = {
        "limit": limit,
        "clipped_elems": n_over,
        "total_elems": n_touch,
        "max_abs_preact": mx,
        "fp32_rel_err": rel_err(b, a),
        "bf16_floor": floor_c,
        "bf16_rel_err": rel_err(d, a),
        "budget": max(floor_c * SLACK, ABS_MIN),
        "moe_out_fp32_rel_err": rel_err(b + shared32, a + shared32),
        "moe_out_bf16_floor": rel_err(c + shared32, a + shared32),
        "moe_out_bf16_rel_err": rel_err(d + shared32, a + shared32),
        "rows_fp32": row_stats(b, a),
        "rows_bf16": row_stats(d, a),
        "rows_floor": row_stats(c, a),
    }

    # ---- defect 2: router GEMM in bf16 while the config asks for float32 -----------
    ov = set_overlap(i32, i16)
    swapped = ~(i32.sort(-1).values == i16.sort(-1).values).all(-1)
    diff_rows = int(swapped.sum())
    e_bf16route = run_experts(i16, w16, torch.float32, clamp=True)
    # Compare the weights densely. Positional comparison is meaningless here: topk is
    # ``sorted=False``, so slot j of the two runs is not the same expert.
    dw32 = torch.zeros(i32.shape[0], n_exp).scatter_(1, i32, w32.float())
    dw16 = torch.zeros(i16.shape[0], n_exp).scatter_(1, i16, w16.float())
    out["router"] = {
        "logits_rel_err": rel_err(lg16, lg32),
        "logits_floor": floors["router_logits"],
        "expert_overlap": ov,
        "swapped_slots": int(round((1 - ov) * i32.numel())),
        "total_slots": int(i32.numel()),
        "rows_with_any_swap": diff_rows,
        "rows": int(i32.shape[0]),
        "weight_rel_err": rel_err(dw16, dw32),
        "routed_rel_err": rel_err(e_bf16route, a),
        "bf16_floor": floor_c,
        "budget": max(floor_c * SLACK, ABS_MIN),
        "moe_out_rel_err": rel_err(e_bf16route + shared32, a + shared32),
        "moe_out_floor": out["clamp"]["moe_out_bf16_floor"],
        "rows_router": row_stats(e_bf16route, a),
        "rows_router_swapped": (row_stats(e_bf16route[swapped], a[swapped])
                                if diff_rows else None),
    }
    return out


def _fmt_rows(r):
    return (f"rows changed {r['rows_changed']}/{r['rows']}  "
            f"per-token rel err p50={r['p50']:.2e} p99={r['p99']:.2e} "
            f"max={r['max']:.2e}")


def print_defects(d):
    c, r = d["clamp"], d["router"]
    print("\n=== PLAN §4 defect 1: DeepEP routed experts lose swiglu_limit ===")
    print(f"  clamp limit                       {c['limit']}")
    print(f"  pre-activations outside limit     {c['clipped_elems']} / {c['total_elems']}"
          f"  ({100.0*c['clipped_elems']/max(c['total_elems'],1):.5f} %)")
    print(f"  max |pre-activation|              {c['max_abs_preact']:.4g}")
    print(f"  routed_out, fp32, no-clamp        rel_err={c['fp32_rel_err']:.3e}"
          f"   (noise-free: this is the modelling error alone)")
    print(f"  routed_out, bf16, WITH clamp      rel_err={c['bf16_floor']:.3e}"
          f"   <- this case's floor, budget {c['budget']:.3e}")
    print(f"  routed_out, bf16, NO clamp        rel_err={c['bf16_rel_err']:.3e}"
          f"   {c['bf16_rel_err']/max(c['budget'],1e-30):.2f}x budget  "
          f"-> {'FAILS' if c['bf16_rel_err'] > c['budget'] else 'within'} the gate")
    print(f"  moe_out (routed+shared)           no-clamp={c['moe_out_bf16_rel_err']:.3e}"
          f"  floor={c['moe_out_bf16_floor']:.3e}")
    print(f"    fp32, clamp on/off (noise-free)  {_fmt_rows(c['rows_fp32'])}")
    print(f"    bf16, no clamp   {_fmt_rows(c['rows_bf16'])}")
    print(f"    bf16, floor      {_fmt_rows(c['rows_floor'])}")
    print("\n=== PLAN §4 defect 2: router GEMM in bf16 (config says float32) ===")
    print(f"  router_logits bf16 vs fp32        rel_err={r['logits_rel_err']:.3e}"
          f"   floor={r['logits_floor']:.3e}"
          f"   {r['logits_rel_err']/max(r['logits_floor'],1e-30):.1f}x floor")
    print(f"  top-8 expert-set overlap          {r['expert_overlap']:.6f}"
          f"  ({r['swapped_slots']} of {r['total_slots']} slots differ)")
    print(f"  tokens with >=1 swapped expert    {r['rows_with_any_swap']} / {r['rows']}"
          f"  ({100.0*r['rows_with_any_swap']/max(r['rows'],1):.2f} %)")
    print(f"  topk weights rel err              {r['weight_rel_err']:.3e}")
    print(f"  routed_out, fp32 arith, bf16 route rel_err={r['routed_rel_err']:.3e}"
          f"   budget={r['budget']:.3e}"
          f"   {r['routed_rel_err']/max(r['budget'],1e-30):.2f}x budget"
          f"  -> {'FAILS' if r['routed_rel_err'] > r['budget'] else 'within'} the gate")
    print(f"  moe_out (routed+shared)           rel_err={r['moe_out_rel_err']:.3e}"
          f"  floor={r['moe_out_floor']:.3e}")
    print(f"    all tokens      {_fmt_rows(r['rows_router'])}")
    if r["rows_router_swapped"]:
        print(f"    swapped tokens  {_fmt_rows(r['rows_router_swapped'])}")



# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--defects", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)
    patch_cpu_conv1d()

    cfg = build_config()
    assert cfg.mlp_layer_types[args.layer] == "sparse", (
        f"layer {args.layer} is {cfg.mlp_layer_types[args.layer]}, not a MoE layer"
    )
    ck = Checkpoint(MODEL)
    ids = make_input_ids(args.seq)
    print(f"model={MODEL}\nlayer={args.layer} seq={args.seq} "
          f"experts={cfg.n_routed_experts} top_k={cfg.num_experts_per_tok} "
          f"swiglu_limit={cfg.swiglu_limit} "
          f"moe_router_dtype={getattr(cfg, 'moe_router_dtype', None)}", flush=True)

    t0 = time.time()
    x_seq, y_hook, layer = run_to_layer(cfg, ids, args.layer, ck)
    x32 = x_seq.reshape(-1, x_seq.shape[-1]).contiguous()
    print(f"reached layer {args.layer} in {time.time()-t0:.0f}s; "
          f"mlp input {tuple(x32.shape)} {x32.dtype}", flush=True)

    mlp = layer.mlp
    ref32, ids32 = moe_parts(mlp, x32)
    agree = rel_err(ref32["moe_out"], y_hook.float().reshape_as(ref32["moe_out"]))
    assert agree == 0.0, f"decomposed MoE disagrees with the module's own output ({agree:.3e})"


    # bf16 reference: same code, bf16 storage. The router keeps HF's internal fp32
    # upcast (that is the reference definition); only its *input* is bf16, so the
    # floor for router_logits is exactly the input-rounding noise, and the NPU's
    # extra bf16 output rounding shows up as excess over it.
    bias32 = mlp.gate.e_score_correction_bias.clone()
    mlp16 = mlp.to(torch.bfloat16)
    mlp16.gate.register_buffer("e_score_correction_bias", bias32)
    ref16, ids16 = moe_parts(mlp16, x32.to(torch.bfloat16))

    floors = {k: noise_floor(ref32[k], ref16[k]) for k in ref32}
    print("\n=== case noise floors (fp32 ref vs bf16 ref) ===")
    for k, v in floors.items():
        print(f"  {k:<20} {v:.3e}   budget {max(v*SLACK, ABS_MIN):.3e}")
    print(f"  {'top-8 expert set':<20} overlap {set_overlap(ids32, ids16):.6f}")

    case = Case(
        name=f"moe_layer{args.layer:02d}_s{args.seq}",
        inputs={"hidden_states": x32},
        ref_fp32=ref32,
        ref_bf16=ref16,
        meta={
            "model": MODEL,
            "layer": args.layer,
            "seq_len": args.seq,
            "n_routed_experts": cfg.n_routed_experts,
            "top_k": cfg.num_experts_per_tok,
            "n_shared_experts": cfg.n_shared_experts,
            "moe_intermediate_size": cfg.moe_intermediate_size,
            "hidden_size": cfg.hidden_size,
            "swiglu_limit": cfg.swiglu_limit,
            "routed_scaling_factor": cfg.routed_scaling_factor,
            "norm_topk_prob": cfg.norm_topk_prob,
            "scoring_func": cfg.scoring_func,
            "n_group": cfg.n_group,
            "topk_group": cfg.topk_group,
            "moe_router_dtype": getattr(cfg, "moe_router_dtype", None),
            "topk_ids_fp32": ids32.to(torch.int32),
            "topk_ids_bf16": ids16.to(torch.int32),
            "expert_set_overlap_floor": set_overlap(ids32, ids16),
        },
    )
    if args.defects:
        # The bf16 pass cast the weights in place; restore the exact checkpoint values
        # rather than upcasting the rounded ones, or the probe would measure its own
        # round trip. Re-reading is cheap: the shards are in page cache by now.
        layer.float()
        load_layer_(layer, args.layer, ck)
        mlp32 = layer.mlp
        mlp32.gate.register_buffer("e_score_correction_bias", bias32)
        d = probe_defects(mlp32, x32, cfg, ref32, ref16, floors)
        print_defects(d)
        case.meta["defects"] = d

    case.save(args.out)
    print(f"\nsaved {args.out}  ({args.out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
