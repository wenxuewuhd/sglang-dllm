#!/usr/bin/env python
"""Produce a CPU golden for one GLM-5.3-Flash KDA (linear attention) layer.

The reference is HuggingFace ``transformers>=5.16.1``'s ``Glm5NextTextLinearAttention``,
which is plain PyTorch and therefore runs on CPU. Run this with the reference venv
(``$ROOT/.venv-ref``), NOT the sglang venv -- sglang pins transformers 5.12.1, which
does not know ``glm5_next`` at all.

Two references are emitted per input: one in bfloat16 (the dtype the NPU path will
use, so it is the number to compare against) and one in float32 (the intrinsic
error floor -- a bf16-vs-bf16 mismatch below the bf16-vs-fp32 gap is noise, not a
port bug).

The checkpoint stores the three conv1d weights separately while HF fuses them into
one depthwise conv over ``cat([q, k, v])``, so they are concatenated in that order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextLinearAttention

# checkpoint suffix (under layers.<i>.self_attn.) -> HF parameter name
DIRECT = {
    "q_proj.weight": "q_proj.weight",
    "k_proj.weight": "k_proj.weight",
    "v_proj.weight": "v_proj.weight",
    "b_proj.weight": "b_proj.weight",
    "g_a_proj.weight": "g_a_proj.weight",
    "g_b_proj.weight": "g_b_proj.weight",
    "o_norm.weight": "o_norm.weight",
    "o_proj.weight": "o_proj.weight",
    "A_log": "forget_gate.A_log",
    "dt_bias": "forget_gate.dt_bias",
    "f_a_proj.weight": "forget_gate.f_a_proj.weight",
    "f_b_proj.weight": "forget_gate.f_b_proj.weight",
}
CONV_PARTS = ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")


def load_layer_weights(model_dir: Path, layer: int) -> dict[str, torch.Tensor]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}.self_attn."
    handles: dict[str, object] = {}

    def get(suffix: str) -> torch.Tensor:
        name = prefix + suffix
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(str(model_dir / shard), framework="pt")
        return handles[shard].get_tensor(name)

    state = {hf: get(ckpt) for ckpt, hf in DIRECT.items()}
    # HF applies one depthwise conv over cat([q, k, v]); keep that row order.
    state["conv1d.weight"] = torch.cat([get(p) for p in CONV_PARTS], dim=0)
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("/mnt/workspace/models/GLM-5.3-Flash-BF16"))
    ap.add_argument("--layer", type=int, default=0, help="a linear_attention layer index")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(str(args.model)).text_config
    layer_type = config.layer_types[args.layer]
    if layer_type != "linear_attention":
        raise SystemExit(f"layer {args.layer} is {layer_type!r}, not linear_attention")

    state = load_layer_weights(args.model, args.layer)
    torch.manual_seed(args.seed)
    hidden = torch.randn(args.batch, args.seq, config.hidden_size)
    mask = torch.ones(args.batch, args.seq, dtype=torch.bool)

    tensors = {"input.hidden_states": hidden, "input.attention_mask": mask}
    for dtype, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        module = Glm5NextTextLinearAttention(config, layer_idx=args.layer)
        module.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)
        module = module.to(dtype).eval()
        with torch.no_grad():
            out = module(hidden_states=hidden.to(dtype), cache_params=None, attention_mask=mask)
        out = out[0] if isinstance(out, tuple) else out
        tensors[f"output.{tag}"] = out.to(torch.float32)
        print(f"{tag}: out {tuple(out.shape)} {out.dtype} "
              f"absmax={out.float().abs().max():.4f} mean={out.float().mean():+.6f}")

    ref, low = tensors["output.fp32"], tensors["output.bf16"]
    denom = ref.abs().max().clamp_min(1e-12)
    print(f"\nbf16 vs fp32 (the noise floor for this layer): "
          f"max_abs={(ref - low).abs().max():.3e}  rel={(ref - low).abs().max() / denom:.3e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata={
        "layer": str(args.layer), "seed": str(args.seed),
        "batch": str(args.batch), "seq": str(args.seq),
        "source": "transformers Glm5NextTextLinearAttention (CPU)",
    })
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
