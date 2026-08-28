#!/usr/bin/env python
"""Stage A2 (CPU, .venv-ref, transformers 5.16.1): the attention half of the
GLM-5.3-Flash layer-3 fp32 reference.

Runs the *real* `Glm5NextTextAttention.forward` for layer 3 on the fp32 hidden
state stage A already produced (`x_f32` in ref32k_v2.pt -- embed + layers 0-2
run for real, then layer 3's attn-site hyper-connection + input_layernorm), and
records:
  hf_out_{S}   [R, hidden]  the layer's attention output for the last R rows
  hf_topk_{S}  [R, 2051]    the selection HF itself made for those rows
  kv_a_f32     [S, 512]     kv_a_layernorm(kv_a_proj(x)) -- the KV cache content

It also cross-checks the absorbed CPU reference in ref.py (which stage B uses to
score the NPU) against HF's own expanded forward, so the reference is not itself
taken on faith.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

import ref as R

MODEL = "/mnt/workspace/models/GLM-5.3-Flash-BF16"
P = "model.language_model.layers.{l}."


class Shards:
    def __init__(self, d=MODEL):
        self.dir = Path(d)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.h = {}

    def get(self, name):
        s = self.map[name]
        if s not in self.h:
            self.h[s] = safe_open(str(self.dir / s), framework="pt")
        return self.h[s].get_tensor(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--seqs", type=int, nargs="+", default=[2048, 4096])
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    cfg = AutoConfig.from_pretrained(MODEL).text_config
    cfg._attn_implementation = "sdpa"
    assert cfg.layer_types[args.layer] == "deepseek_sparse_attention"
    sh = Shards()
    d = torch.load(args.ref, map_location="cpu")
    x_all = d["x_f32"]
    print(f"x_f32 {tuple(x_all.shape)}  threads={torch.get_num_threads()}")

    p = P.format(l=args.layer) + "self_attn."
    attn = Glm5NextTextAttention(cfg, args.layer).float().eval()
    sd = {
        "q_a_proj.weight": sh.get(p + "q_a_proj.weight").float(),
        "q_a_layernorm.weight": sh.get(p + "q_a_layernorm.weight").float(),
        "q_b_proj.weight": sh.get(p + "q_b_proj.weight").float(),
        "kv_a_proj_with_mqa.weight": sh.get(p + "kv_a_proj_with_mqa.weight").float(),
        "kv_a_layernorm.weight": sh.get(p + "kv_a_layernorm.weight").float(),
        "kv_b_proj.weight": sh.get(p + "kv_b_proj.weight").float(),
        "o_proj.weight": sh.get(p + "o_proj.weight").float(),
    }
    for n in ("wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias",
              "weights_proj.weight", "index_kpool_compress_ape",
              "index_kpool_compress_gate"):
        sd["indexer." + n] = sh.get(p + "indexer." + n).float()
    missing, unexpected = attn.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    print("HF layer-3 attention loaded")

    lref = R.LayerRef(sh, args.layer, cfg)
    out = {"meta": {"layer": args.layer, "rows": args.rows, "seqs": list(args.seqs),
                    "scaling": lref.scaling, "eps": R.EPS}}

    Rr = args.rows
    for S in args.seqs:
        x = x_all[:S].unsqueeze(0)
        mask = torch.ones(1, S, dtype=torch.bool)
        t0 = time.time()
        hf_out, _, _ = attn(hidden_states=x, attention_mask=mask, past_key_values=None)
        dt = time.time() - t0
        hf_out = hf_out[0]                                    # [S, hidden]
        # HF's own selection, recomputed (the module does not return it unless
        # next_skip_topk); calling the indexer directly is the same code path.
        q_resid = attn.q_a_layernorm(attn.q_a_proj(x))
        topk = attn.indexer(hidden_states=x, q_resid=q_resid,
                            attention_mask=mask, past_key_values=None)[0]  # [S, 2051]

        kv = lref.kv_latent(x_all[:S], torch.float32)
        qno = lref.q_absorbed(x_all[S - Rr : S], torch.float32)
        rows = [topk[t][topk[t] >= 0].long() for t in range(S - Rr, S)]
        mine = lref.attend(qno, kv, rows, torch.float32)
        e = R.rel(mine, hf_out[S - Rr :])
        print(f"[S={S}] HF forward {dt:.1f}s  absmax={hf_out.abs().max():.4f}  "
              f"absorbed-vs-HF rel={e:.3e} cos={R.cos(mine, hf_out[S-Rr:]):.9f}  "
              f"valid/row={float(sum(r.numel() for r in rows))/Rr:.1f}")
        out[f"hf_out_{S}"] = hf_out[S - Rr :].contiguous()
        out[f"hf_topk_{S}"] = topk[S - Rr :].contiguous()
        out["meta"][f"absorbed_vs_hf_rel_{S}"] = e
        del hf_out, topk, kv, qno, mine

    out["kv_a_f32_32768"] = lref.kv_latent(x_all, torch.float32).contiguous()
    torch.save(out, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size/2**20:.1f} MiB)")


if __name__ == "__main__":
    raise SystemExit(main())
