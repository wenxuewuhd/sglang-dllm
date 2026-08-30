#!/usr/bin/env python
"""Quantize the KDA projections of a GLM-5.3-Flash W8A8 checkpoint to INT8.

Why: the vendor's W8A8 checkpoint leaves the 34 linear-attention layers in BF16
(`modules_to_not_convert` in the FP8 source already excluded them, and our
conversion carried that over).  Measured on one A3 die at bs=1, those BF16
projections are the single largest consumer of per-token weight bandwidth --
8.93 GiB of the 20.7 GiB a decode step must read, more than the top-8 routed
experts.  This script exists to answer *how much time* halving those bytes buys,
by making a checkpoint where they are INT8 and measuring it.

*** This is a performance probe, not a deployable quantization. ***  Nothing here
is calibrated and no accuracy was checked.  KDA carries recurrent state, so
weight error there does not behave like weight error in a feed-forward block, and
the vendor's choice not to quantize these layers may well be deliberate.  Treat a
speedup measured this way as an upper bound on what a *correct* KDA quantization
could give, not as a result you can ship.

Quantization: symmetric per-output-channel min-max, which is what
`config_groups.group_0.weights` already declares for every other Linear in this
checkpoint (num_bits 8, type int, symmetric, strategy channel).

    $VENV/bin/python quantize_kda_int8.py \
        --src /var/tmp/glm53/GLM-5.3-Flash-W8A8-e16 \
        --dst /var/tmp/glm53/GLM-5.3-Flash-W8A8-e16-kdaint8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Only these four.  f_a/f_b/g_a/g_b/b_proj are already small (6.5 MiB per layer
# against 256 MiB for q/k/v/o) so quantizing them buys nothing measurable, and
# every module left alone is one less thing that can explain a result.
KDA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.self_attn\.(\w+)\.weight$")


def quantize_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8.  Returns (int8 weight, fp32 scale)."""
    wf = w.to(torch.float32)
    amax = wf.abs().amax(dim=1, keepdim=True)
    # A dead output channel would give scale 0; keep it finite and quantize to 0.
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.clamp(torch.round(wf / scale), -127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--shard-bytes", type=int, default=8 * 1024**3)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    with open(os.path.join(args.src, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    with open(os.path.join(args.src, "config.json")) as f:
        cfg = json.load(f)
    kda_layers = set(cfg["text_config"]["linear_attn_config"]["kda_layers"])

    def is_target(name: str) -> bool:
        m = LAYER_RE.match(name)
        return (
            m is not None
            and int(m.group(1)) in kda_layers
            and m.group(2) in KDA_TARGETS
        )

    names = sorted(weight_map, key=lambda n: (weight_map[n], n))
    out_map: dict[str, str] = {}
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    shards: list[tuple[str, list[str]]] = []
    total_bytes = 0
    n_quant = 0
    before = after = 0
    t0 = time.time()
    open_file = None
    handle = None

    def flush() -> None:
        nonlocal buf, buf_bytes
        if not buf:
            return
        tmp = f"model-{len(shards) + 1:05d}.tmp.safetensors"
        save_file(buf, os.path.join(args.dst, tmp), metadata={"format": "pt"})
        shards.append((tmp, list(buf)))
        print(f"  shard {len(shards)}: {buf_bytes / 1024**3:.2f} GiB, {time.time() - t0:.0f}s")
        buf = {}
        buf_bytes = 0

    for name in names:
        fn = weight_map[name]
        if fn != open_file:
            if handle is not None:
                handle.__exit__(None, None, None)
            handle = safe_open(os.path.join(args.src, fn), framework="pt")
            handle.__enter__()
            open_file = fn
        t = handle.get_tensor(name)
        if is_target(name):
            assert t.dtype == torch.bfloat16, (name, t.dtype)
            before += t.numel() * t.element_size()
            q, s = quantize_channel(t)
            after += q.numel() + s.numel() * 4
            buf[name] = q
            buf[name.replace(".weight", ".weight_scale")] = s
            n_quant += 1
            nbytes = q.numel() + s.numel() * 4
        else:
            buf[name] = t
            nbytes = t.numel() * t.element_size()
        buf_bytes += nbytes
        total_bytes += nbytes
        if buf_bytes >= args.shard_bytes:
            flush()
    flush()
    if handle is not None:
        handle.__exit__(None, None, None)

    n = len(shards)
    for i, (tmp, tnames) in enumerate(shards, start=1):
        final = f"model-{i:05d}-of-{n:05d}.safetensors"
        os.rename(os.path.join(args.dst, tmp), os.path.join(args.dst, final))
        for tn in tnames:
            out_map[tn] = final
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": out_map}, f)

    for fn in os.listdir(args.src):
        if not fn.endswith(".safetensors") and fn != "model.safetensors.index.json":
            shutil.copy2(os.path.join(args.src, fn), os.path.join(args.dst, fn))

    # sglang strips "language_model." from checkpoint names, so the ignore list
    # is written against `model.layers.{L}.self_attn.{proj}`.
    #
    # Dropping q/k/v/o_proj is necessary but NOT sufficient, and this cost a
    # measurement to find out.  sglang builds the three KDA projections as one
    # fused `qkv_proj`, and should_ignore_layer() only expands a fused name back
    # to its components `if proj_name in fused_mapping and layer_name not in
    # ignore`.  The vendor's ignore list names the fused module *itself*
    # ("model.layers.0.self_attn.qkv_proj"), so that guard fails, the expansion
    # never happens, and the layer stays BF16 no matter what you do to q/k/v.
    # Symptom: o_proj quantizes, qkv does not, and the profile still shows
    # MatMulV2 DT_BF16 on "1,4096;24576,4096".
    fused_aliases = ("qkv_proj",)
    drop = {
        f"model.layers.{L}.self_attn.{p}"
        for L in kda_layers
        for p in KDA_TARGETS + fused_aliases
    }
    ig = cfg["quantization_config"]["ignore"]
    kept = [x for x in ig if x not in drop]
    removed = len(ig) - len(kept)
    cfg["quantization_config"]["ignore"] = kept
    cfg["_kda_int8_probe"] = True
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(
        f"quantized {n_quant} tensors ({before / 1024**3:.2f} -> "
        f"{after / 1024**3:.2f} GiB), removed {removed} ignore entries "
        f"(of {len(drop)} candidate ignore names), total {total_bytes / 1024**3:.2f} GiB, "
        f"{time.time() - t0:.0f}s -> {args.dst}"
    )


if __name__ == "__main__":
    main()
