#!/usr/bin/env python
"""Quantize the BF16 checkpoint to compressed-tensors W8A8-INT8, shard by shard.

Which tensors to quantize is not a judgement call: the original FP8 checkpoint
carried a ``weight_scale_inv`` beside every weight it had quantized, and that index
is still on disk even though the FP8 shards are gone. Reading it back gives exactly
the 37338 tensors the vendor chose -- every MoE expert and shared expert, the DSA/MLA
projections, the three dense FFN layers -- and leaves the rest alone: all 34 KDA
layers, the indexer, the hc_* parameters, every norm, the router, embeddings and
lm_head. Re-deriving that set from module-name patterns would be a second source of
truth that could drift from the first.

Activations are per-token dynamic, so there is nothing to calibrate and no forward
pass to run: this is a pure offline transform of the weights. It needs CPU and disk,
not an NPU.

Weights are symmetric per-output-channel:

    scale = absmax(W, dim=1) / 127        fp32, shape [out, 1]
    q     = round(W / scale)              int8, clamped to [-127, 127]

127 rather than 128 keeps the range symmetric, so `q * scale` recovers W with no
offset term. The scale is computed in fp32 from the bf16 weight -- taking the max in
bf16 first would quantize the scale itself.

    $ROOT/.venv-ref/bin/python bf16_to_int8_ct.py \\
        --src /mnt/workspace/models/GLM-5.3-Flash-BF16 \\
        --fp8-index /mnt/workspace/models/GLM-5.3-Flash/model.safetensors.index.json \\
        --dst /mnt/workspace/models/GLM-5.3-Flash-W8A8
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

#: What the compressed-tensors scheme selector in sglang matches on: weights per
#: channel and static, activations per token and dynamic. `_is_dynamic_token_w8a8`
#: requires exactly this combination, and a static activation scheme is rejected.
QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "int-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "int",
                "symmetric": True,
                "strategy": "channel",
                "dynamic": False,
                "observer": "minmax",
            },
            "input_activations": {
                "num_bits": 8,
                "type": "int",
                "symmetric": True,
                "strategy": "token",
                "dynamic": True,
                "observer": None,
            },
            "output_activations": None,
        }
    },
}


def quantize_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8. Returns (int8 weight, fp32 scale [out, 1])."""
    f = w.to(torch.float32)
    absmax = f.abs().amax(dim=1, keepdim=True)
    # An all-zero output channel has absmax 0; any positive scale reproduces it, so
    # pick 1 rather than dividing by zero.
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    q = torch.round(f / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def convert_shard(args) -> tuple[str, dict, int, float]:
    src, dst, names_to_quantize = args
    t0 = time.time()
    out: dict[str, torch.Tensor] = {}
    with safe_open(str(src), framework="pt") as f:
        for name in f.keys():
            w = f.get_tensor(name)
            if name in names_to_quantize:
                q, scale = quantize_channel(w)
                out[name] = q
                out[name[: -len("weight")] + "weight_scale"] = scale
            else:
                out[name] = w
    save_file(out, str(dst), metadata={"format": "pt"})
    sizes = {k: v.numel() * v.element_size() for k, v in out.items()}
    return dst.name, {k: dst.name for k in out}, sum(sizes.values()), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--fp8-index", type=Path, required=True,
                    help="the original FP8 checkpoint's index; its weight_scale_inv "
                         "entries name exactly the tensors to quantize")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="convert only N shards (smoke)")
    args = ap.parse_args()

    fp8_map = json.loads(args.fp8_index.read_text())["weight_map"]
    to_quantize = frozenset(
        k[: -len("_scale_inv")] for k in fp8_map if k.endswith("weight_scale_inv")
    )
    print(f"{len(to_quantize)} tensors to quantize, from the FP8 index", flush=True)

    shards = sorted(args.src.glob("*.safetensors"))
    if args.limit:
        shards = shards[: args.limit]
    args.dst.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = {}
    total_bytes = 0
    t0 = time.time()
    jobs = [(s, args.dst / s.name, to_quantize) for s in shards]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (name, wmap, nbytes, secs) in enumerate(ex.map(convert_shard, jobs), 1):
            weight_map.update(wmap)
            total_bytes += nbytes
            print(f"  [{i}/{len(shards)}] {name} {nbytes / 1024**3:.1f} GiB "
                  f"{secs:.0f}s  (elapsed {time.time() - t0:.0f}s)", flush=True)

    (args.dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2))

    for extra in args.src.glob("*"):
        if extra.suffix != ".safetensors" and extra.name != "model.safetensors.index.json":
            shutil.copy2(extra, args.dst / extra.name)

    cfg = json.loads((args.dst / "config.json").read_text())
    quant = dict(QUANT_CONFIG)
    # The vendor's own ignore list, carried over verbatim rather than re-derived.
    fp8_cfg = json.loads((args.fp8_index.parent / "config.json").read_text())
    quant["ignore"] = fp8_cfg.get("quantization_config", {}).get(
        "modules_to_not_convert", []
    )
    cfg["quantization_config"] = quant
    (args.dst / "config.json").write_text(json.dumps(cfg, indent=2))

    print(f"\nwrote {len(shards)} shards, {total_bytes / 1024**3:.1f} GiB, "
          f"{time.time() - t0:.0f}s -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
