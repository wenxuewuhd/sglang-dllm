#!/usr/bin/env python
"""Write a copy of the GLM-5.3-Flash W8A8 checkpoint that keeps only the first
N routed experts, so the model fits on a single 64 GB A3 die.

Why a pruned checkpoint instead of a loader flag: the loader reads every shard
it is given, so keeping the full 306 GiB checkpoint on disk costs a 306 GiB
read on *every* server start.  A pruned copy is read once per start and is
~31 GiB at N=16.

What is dropped
  model.language_model.layers.{L}.mlp.experts.{E}.*   for E >= N

What is rewritten (not just copied)
  ...mlp.gate.weight                  [288, 4096] -> [N, 4096]
  ...mlp.gate.e_score_correction_bias [288]       -> [N]
  config.json: text_config.n_routed_experts = N

Everything else -- shared expert, dense FFN, KDA/DSA attention, embeddings,
the vision tower, the MTP layer -- is copied verbatim.

*** The routing this produces is NOT the model's routing. ***  top-8 of N is a
different function from top-8 of 288.  This checkpoint is for performance work
only; any accuracy number taken from it is meaningless.  See the report for
which performance conclusions this does and does not perturb.
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

EXPERT_RE = re.compile(r"\.mlp\.experts\.(\d+)\.")
GATE_W = ".mlp.gate.weight"
GATE_B = ".mlp.gate.e_score_correction_bias"

AUX_FILES = [
    "config.json",
    "configuration.json",
    "generation_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "LICENSE",
]


def plan(src: str, num_experts: int):
    """Return (kept tensor names in checkpoint order, name -> source file)."""
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    kept = []
    for name in weight_map:
        m = EXPERT_RE.search(name)
        if m is not None and int(m.group(1)) >= num_experts:
            continue
        kept.append(name)
    # Group by source file so each shard is opened once, in order.
    kept.sort(key=lambda n: (weight_map[n], n))
    return kept, weight_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/workspace/models/GLM-5.3-Flash-W8A8")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument(
        "--shard-bytes", type=int, default=8 * 1024**3, help="target output shard size"
    )
    args = ap.parse_args()

    n = args.experts
    os.makedirs(args.dst, exist_ok=True)
    kept, weight_map = plan(args.src, n)
    print(f"keeping {len(kept)} of {len(weight_map)} tensors (experts 0..{n - 1})")

    out_map: dict[str, str] = {}
    total_bytes = 0
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    shard_id = 0
    shards: list[tuple[str, dict]] = []
    open_file = None
    handle = None
    t0 = time.time()

    def flush() -> None:
        nonlocal buf, buf_bytes, shard_id
        if not buf:
            return
        shard_id += 1
        # Name is provisional; renamed once the shard count is known.
        tmp = f"model-{shard_id:05d}.tmp.safetensors"
        save_file(buf, os.path.join(args.dst, tmp), metadata={"format": "pt"})
        shards.append((tmp, list(buf)))
        print(
            f"  shard {shard_id}: {len(buf)} tensors, "
            f"{buf_bytes / 1024**3:.2f} GiB, {time.time() - t0:.0f}s elapsed"
        )
        buf = {}
        buf_bytes = 0

    for name in kept:
        fn = weight_map[name]
        if fn != open_file:
            if handle is not None:
                handle.__exit__(None, None, None)
            handle = safe_open(os.path.join(args.src, fn), framework="pt")
            handle.__enter__()
            open_file = fn
        t = handle.get_tensor(name)
        if name.endswith(GATE_W) or name.endswith(GATE_B):
            assert t.shape[0] == 288, (name, t.shape)
            t = t[:n].clone()
        buf[name] = t
        nbytes = t.numel() * t.element_size()
        buf_bytes += nbytes
        total_bytes += nbytes
        if buf_bytes >= args.shard_bytes:
            flush()
    flush()
    if handle is not None:
        handle.__exit__(None, None, None)

    total_shards = len(shards)
    for i, (tmp, names) in enumerate(shards, start=1):
        final = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        os.rename(os.path.join(args.dst, tmp), os.path.join(args.dst, final))
        for name in names:
            out_map[name] = final

    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total_bytes}, "weight_map": out_map}, f)

    for fn in AUX_FILES:
        s = os.path.join(args.src, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(args.dst, fn))

    cfg_path = os.path.join(args.dst, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert cfg["text_config"]["n_routed_experts"] == 288
    cfg["text_config"]["n_routed_experts"] = n
    cfg["_pruned_from"] = args.src
    cfg["_pruned_experts_kept"] = n
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(
        f"done: {total_shards} shards, {total_bytes / 1024**3:.2f} GiB, "
        f"{time.time() - t0:.0f}s -> {args.dst}"
    )


if __name__ == "__main__":
    main()
