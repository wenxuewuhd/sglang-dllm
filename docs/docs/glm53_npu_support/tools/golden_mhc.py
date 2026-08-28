#!/usr/bin/env python
"""Produce a CPU golden for one GLM-5.3-Flash mHC (hyper-connection) block.

Reference: HuggingFace ``transformers>=5.16.1`` ``Glm5NextTextHyperConnection``.
Run with the reference venv (``$ROOT/.venv-ref``), not the sglang venv.

The block returns ``(post, comb, collapsed)``: ``collapsed`` folds the hc_mult
parallel streams into the sublayer input, while ``post`` and ``comb`` are handed
back for the caller to apply to the sublayer output. Two details the NPU wiring
has to match: the factor 2 in ``post = 2 * sigmoid(...)`` is part of the formula
(this is GLM's ``post_mult_value=2.0``), and ``comb`` is projected onto the
doubly-stochastic manifold by ``hc_sinkhorn_iters`` Sinkhorn-Knopp sweeps.

Emits an fp32 and a bf16 reference for the same input; their difference is this
block's noise floor, which is the bar an NPU implementation has to meet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

STAGES = {"attn": "hc_attn", "ffn": "hc_ffn"}


def load_weights(model_dir: Path, layer: int, stage: str) -> dict[str, torch.Tensor]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.{STAGES[stage]}_"
    handles: dict[str, object] = {}

    def get(suffix: str) -> torch.Tensor:
        name = prefix + suffix
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(str(model_dir / shard), framework="pt")
        return handles[shard].get_tensor(name)

    return {"fn": get("fn"), "base": get("base"), "scale": get("scale")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("/mnt/workspace/models/GLM-5.3-Flash-BF16"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--stage", choices=sorted(STAGES), default="attn")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(str(args.model)).text_config
    state = load_weights(args.model, args.layer, args.stage)
    print(f"layer {args.layer} / {args.stage}: " + ", ".join(
        f"{k}{tuple(v.shape)}" for k, v in state.items()))

    torch.manual_seed(args.seed)
    streams = torch.randn(1, args.tokens, config.hc_mult, config.hidden_size)
    tensors = {"input.hidden_streams": streams}

    for dtype, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        block = Glm5NextTextHyperConnection(config)
        block.load_state_dict({k: v.to(torch.float32) for k, v in state.items()}, strict=True)
        block = block.to(dtype).eval()
        with torch.no_grad():
            post, comb, collapsed = block(streams.to(dtype))
        for name, value in (("post", post), ("comb", comb), ("collapsed", collapsed)):
            tensors[f"output.{tag}.{name}"] = value.to(torch.float32)
        # comb must stay doubly stochastic; a wiring error shows up here first.
        row = comb.float().sum(-1)
        col = comb.float().sum(-2)
        print(f"{tag}: post{tuple(post.shape)} comb{tuple(comb.shape)} collapsed{tuple(collapsed.shape)}"
              f" | comb row-sum {row.min():.5f}..{row.max():.5f}"
              f" col-sum {col.min():.5f}..{col.max():.5f}")

    for name in ("post", "comb", "collapsed"):
        ref = tensors[f"output.fp32.{name}"]
        low = tensors[f"output.bf16.{name}"]
        denom = ref.abs().max().clamp_min(1e-12)
        print(f"noise floor {name:10s} max_abs={(ref - low).abs().max():.3e}"
              f"  rel={(ref - low).abs().max() / denom:.3e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata={
        "layer": str(args.layer), "stage": args.stage, "seed": str(args.seed),
        "source": "transformers Glm5NextTextHyperConnection (CPU)",
    })
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
