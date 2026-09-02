#!/usr/bin/env python
"""Stage A (CPU, .venv-ref): produce the fp32 reference tensors for the GLM-5.3-Flash
layer-3 kpool indexer.

Runs embed + decoder layers 0..2 for real on a real tokenized prompt, then takes
layer 3's attention-site hyper-connection + input_layernorm to get the exact tensor
the indexer sees.  Emits, in fp32 and *unrotated* (the Hadamard is applied by stage B
so both sides use one implementation):

  pooled_key   [P, 128]        softmax-weighted pool of 4, P = seq // 4
  q_rows       [L, R, 32, 128]  wq_b(q_resid) for the L test rows at each seq_len
  w_rows       [L, R, 32]       weights_proj(x) * n_heads**-0.5
  row_pos      [L, R]           token positions of the test rows
  seq_lens     [L]

Run with:  $ROOT/.venv-ref/bin/python kpool_stage_a_ref.py --out ref.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextDecoderLayer,
    Glm5NextTextHyperConnection,
    Glm5NextTextIndexer,
    Glm5NextTextRMSNorm,
)

CKPT_PREFIX = "model.language_model."


def _patch_conv1d(chunk: int = 2048) -> None:
    """aarch64 CPU F.conv1d raises 'illegal immediate parameter (range error)' on the
    24576-channel x 32771-step depthwise conv.  Same conv, chunked over the sequence --
    each output still comes from the identical 4-tap F.conv1d, so this is not an
    approximation."""
    import transformers.models.glm5_next.modeling_glm5_next as M

    def fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        _, hidden_size, seq_len = hidden_states.shape
        pad = weight.shape[-1] - 1
        x = F.pad(hidden_states.to(weight.dtype), (pad, 0))
        outs = []
        for s in range(0, seq_len, chunk):
            e = min(s + chunk, seq_len)
            outs.append(F.conv1d(x[:, :, s : e + pad], weight.unsqueeze(1), bias,
                                 padding=0, groups=hidden_size))
        out = torch.cat(outs, dim=-1)[:, :, :seq_len]
        if activation is not None:
            out = M.ACT2FN[activation](out)
        return out.to(hidden_states.dtype)

    M.causal_conv1d_fn = fn
    try:  # the class decorator may have captured the original
        M.Glm5NextTextLinearAttention._kernelized_funcs["causal_conv1d_fn"] = fn
    except Exception:  # noqa: BLE001
        pass


class Shards:
    def __init__(self, model_dir: Path):
        self.dir = model_dir
        self.map = json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.handles: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(str(self.dir / shard), framework="pt")
        return self.handles[shard].get_tensor(name)


def kda_layer_state(sh: Shards, layer: int) -> dict[str, torch.Tensor]:
    p = f"{CKPT_PREFIX}layers.{layer}."
    direct = {
        "self_attn.q_proj.weight": "self_attn.q_proj.weight",
        "self_attn.k_proj.weight": "self_attn.k_proj.weight",
        "self_attn.v_proj.weight": "self_attn.v_proj.weight",
        "self_attn.b_proj.weight": "self_attn.b_proj.weight",
        "self_attn.g_a_proj.weight": "self_attn.g_a_proj.weight",
        "self_attn.g_b_proj.weight": "self_attn.g_b_proj.weight",
        "self_attn.o_norm.weight": "self_attn.o_norm.weight",
        "self_attn.o_proj.weight": "self_attn.o_proj.weight",
        "self_attn.A_log": "self_attn.forget_gate.A_log",
        "self_attn.dt_bias": "self_attn.forget_gate.dt_bias",
        "self_attn.f_a_proj.weight": "self_attn.forget_gate.f_a_proj.weight",
        "self_attn.f_b_proj.weight": "self_attn.forget_gate.f_b_proj.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "hc_attn_fn": "attn_hc.fn",
        "hc_attn_base": "attn_hc.base",
        "hc_attn_scale": "attn_hc.scale",
        "hc_ffn_fn": "ffn_hc.fn",
        "hc_ffn_base": "ffn_hc.base",
        "hc_ffn_scale": "ffn_hc.scale",
    }
    state = {hf: sh.get(p + ck).float() for ck, hf in direct.items()}
    state["self_attn.conv1d.weight"] = torch.cat(
        [
            sh.get(p + f"self_attn.{n}_conv1d.weight").float()
            for n in ("q", "k", "v")
        ],
        dim=0,
    )
    return state


def build_prompt_ids(model_dir: Path, seq: int) -> torch.Tensor:
    """Real text, not random ids: concatenate this repo's markdown docs."""
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    repo = Path(__file__).resolve()
    # Sibling SGLang checkout; GLM53_ROOT (or GLM53_WORKSPACE) names the workspace.
    root = Path(os.environ.get("GLM53_ROOT") or os.environ.get("GLM53_WORKSPACE") or "") / "sglang-dllm"
    texts = []
    total = 0
    for p in sorted(root.rglob("*.md")):
        if ".git" in p.parts:
            continue
        try:
            t = p.read_text(errors="ignore")
        except OSError:
            continue
        texts.append(t)
        total += len(t)
        if total > seq * 12:
            break
    blob = "\n\n".join(texts)
    ids = tok(blob, add_special_tokens=False, return_tensors="pt").input_ids[0]
    assert ids.numel() >= seq, f"only {ids.numel()} tokens available, need {seq}"
    return ids[:seq].unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("/mnt/workspace/models/GLM-5.3-Flash-BF16"))
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--seq", type=int, default=32768)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 4096, 8192, 32768])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hf-check-seq", type=int, default=2048,
                    help="cross-check our fp32 pipeline against Glm5NextTextIndexer.forward")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    _patch_conv1d()
    cfg = AutoConfig.from_pretrained(str(args.model)).text_config
    assert cfg.layer_types[args.layer] == "deepseek_sparse_attention"
    sh = Shards(args.model)

    print(f"threads={torch.get_num_threads()}  seq={args.seq}")
    ids = build_prompt_ids(args.model, args.seq)
    mask = torch.ones(1, args.seq, dtype=torch.bool)

    emb_w = sh.get(f"{CKPT_PREFIX}embed_tokens.weight").float()
    h = F.embedding(ids, emb_w)
    del emb_w
    h = h.unsqueeze(2).expand(-1, -1, cfg.hc_mult, -1).contiguous()
    print(f"embed -> {tuple(h.shape)}")

    for i in range(args.layer):
        t0 = time.time()
        layer = Glm5NextTextDecoderLayer(cfg, i).float().eval()
        missing, unexpected = layer.load_state_dict(kda_layer_state(sh, i), strict=True), None
        h, _ = layer(h, attention_mask=mask, position_ids=None, past_key_values=None,
                     use_cache=False, position_embeddings=None)
        del layer
        print(f"layer {i} ({cfg.layer_types[i]}) done in {time.time()-t0:.1f}s "
              f"absmax={h.abs().max():.4f}")

    # --- layer `args.layer`: attention-site hyper-connection + input_layernorm
    p = f"{CKPT_PREFIX}layers.{args.layer}."
    attn_hc = Glm5NextTextHyperConnection(cfg).float().eval()
    attn_hc.load_state_dict({
        "fn": sh.get(p + "hc_attn_fn").float(),
        "base": sh.get(p + "hc_attn_base").float(),
        "scale": sh.get(p + "hc_attn_scale").float(),
    }, strict=True)
    in_ln = Glm5NextTextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).float().eval()
    in_ln.load_state_dict({"weight": sh.get(p + "input_layernorm.weight").float()})

    _post, _comb, x = attn_hc(h)
    del h
    x = in_ln(x)  # [1, S, hidden]
    print(f"indexer input x {tuple(x.shape)} absmax={x.abs().max():.4f}")

    # q_resid = q_a_layernorm(q_a_proj(x))
    q_a_w = sh.get(p + "self_attn.q_a_proj.weight").float()
    q_a_ln = Glm5NextTextRMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps).float().eval()
    q_a_ln.load_state_dict({"weight": sh.get(p + "self_attn.q_a_layernorm.weight").float()})
    q_resid = q_a_ln(F.linear(x, q_a_w))
    del q_a_w

    # --- indexer weights
    ip = p + "self_attn.indexer."
    idx_state = {
        "wq_b.weight": sh.get(ip + "wq_b.weight").float(),
        "wk.weight": sh.get(ip + "wk.weight").float(),
        "k_norm.weight": sh.get(ip + "k_norm.weight").float(),
        "k_norm.bias": sh.get(ip + "k_norm.bias").float(),
        "weights_proj.weight": sh.get(ip + "weights_proj.weight").float(),
        "index_kpool_compress_ape": sh.get(ip + "index_kpool_compress_ape").float(),
        "index_kpool_compress_gate": sh.get(ip + "index_kpool_compress_gate").float(),
    }
    indexer = Glm5NextTextIndexer(cfg, args.layer).float().eval()
    indexer.load_state_dict(idx_state, strict=True)

    n_heads, head_dim, kpool = cfg.index_n_heads, cfg.index_head_dim, cfg.index_kpool
    S = args.seq

    q_all = indexer.wq_b(q_resid).view(1, S, n_heads, head_dim)[0]        # [S, 32, 128]
    k_all = indexer.k_norm(indexer.wk(x)).view(1, S, head_dim)[0]          # [S, 128]
    gate = F.linear(x, indexer.index_kpool_compress_gate)[0]               # [S, 128]
    w_all = (indexer.weights_proj(x).float() * (n_heads ** -0.5))[0]       # [S, 32]

    # --- pooled keys, fp32, exactly HF's get_pooled_states for full pools
    P = S // kpool
    gk = k_all[: P * kpool].view(P, kpool, head_dim)
    gs = gate[: P * kpool].view(P, kpool, head_dim)
    logits = gs + indexer.index_kpool_compress_ape[None]                   # [P, 4, 128]
    prob = logits.softmax(dim=1)
    pooled_key = (prob * gk).sum(dim=1)                                    # [P, 128]
    print(f"pooled_key {tuple(pooled_key.shape)} absmax={pooled_key.abs().max():.5f}")

    # --- test rows: the last `rows` query positions of each seq_len
    seq_lens = list(args.seq_lens)
    R = args.rows
    row_pos = torch.stack([torch.arange(s - R, s) for s in seq_lens])      # [L, R]
    q_rows = torch.stack([q_all[rp] for rp in row_pos])                    # [L, R, 32, 128]
    w_rows = torch.stack([w_all[rp] for rp in row_pos])                    # [L, R, 32]

    out = {
        # Stage A2 additions: the *inputs* the real IndexerKPool.forward_npu needs,
        # so the NPU harness can run the module itself instead of replaying tensors.
        "x_f32": x[0].contiguous(),
        "q_resid_f32": q_resid[0].contiguous(),
        "k_f32": k_all.contiguous(),
        "gate_f32": gate.contiguous(),
        "w_all_f32": w_all.contiguous(),
        "pooled_key": pooled_key.contiguous(),
        "q_rows": q_rows.contiguous(),
        "w_rows": w_rows.contiguous(),
        "row_pos": row_pos.to(torch.int64),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int64),
        "meta": {
            "layer": args.layer, "seq": S, "kpool": kpool, "n_heads": n_heads,
            "head_dim": head_dim, "index_topk": cfg.index_topk,
            "softmax_scale": head_dim ** -0.5,
            "note": "q/pooled_key are UNROTATED fp32; stage B applies Hadamard-128",
        },
    }

    # --- cross-check our fp32 pipeline against the HF module's own forward
    if args.hf_check_seq:
        C = args.hf_check_seq
        got = indexer(hidden_states=x[:, :C], q_resid=q_resid[:, :C],
                      attention_mask=mask[:, :C], past_key_values=None)  # [1, C, topk+3]
        # rebuild the same selection ourselves for the last row
        r = C - 1
        npool = (r + 1) // kpool
        sc = F.relu(torch.einsum("nd,pd->np", q_all[r], pooled_key[:npool]) * (head_dim ** -0.5))
        lg = (w_all[r][:, None] * sc).sum(0)
        k = min(cfg.index_topk // kpool, npool)
        mine = torch.topk(lg, k).indices
        mine_tokens = (mine[:, None] * kpool + torch.arange(kpool)).flatten()
        hf_tokens = got[0, r]
        hf_pool = set((hf_tokens[hf_tokens >= 0][: k * kpool] // kpool).tolist())
        ov = len(hf_pool & set(mine.tolist())) / max(len(hf_pool), 1)
        print(f"[HF cross-check @seq={C}, row {r}] pools k={k} "
              f"hf_pool_count={len(hf_pool)} overlap_with_our_fp32={ov:.6f}")
        out["meta"]["hf_crosscheck_overlap"] = ov
        out["meta"]["hf_crosscheck_seq"] = C

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}  ({args.out.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
