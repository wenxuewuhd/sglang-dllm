#!/usr/bin/env python
"""Per-kernel roofline: measured device time against the bytes the kernel moves.

The profiler's `Input Shapes` / `Output Shapes` columns are filled in even with
`record_shapes=False`, so every row carries enough to compute its own traffic:

    traffic = sum(input tensor bytes) + sum(output tensor bytes)
    floor   = traffic / 1.25 TB/s        (measured A3 read+write)

A kernel moving less than ~16 MB is dominated by the ~13.5 us fixed launch cost
and its ratio to that floor says nothing about efficiency -- those rows are
labelled `launch` rather than given a meaningless multiple.

⚠ Two kinds of row where the declared shape overstates the traffic, and the
script corrects them from OVERRIDE rather than pretending:

  * `GroupedMatmul` is handed the whole `[E, ...]` expert tensor but only reads
    the experts named in group_list (8 of 16 at bs=1, top-k of E in general).
  * `SparseFlashAttention` / `LightningIndexer` are handed the whole paged KV
    cache but read only the selected pages.

Anything still overstated is marked `[decl]` so a reader knows the ratio is a
lower bound on efficiency, not a verdict.

    $VENV/bin/python kernel_roofline.py --profile /var/tmp/glm53/prof/bs1 --steps 20
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import statistics

BW = 1.25e12  # bytes/s, measured read+write on this A3 die
LAUNCH_US = 13.5
LAUNCH_BOUND_BYTES = 16 * 1024**2

DTYPE_BYTES = {
    "DT_BF16": 2, "FLOAT16": 2, "FLOAT": 4, "DOUBLE": 8,
    "INT8": 1, "BOOL": 1, "INT32": 4, "INT64": 8,
}

#: (Type, Input Shapes) -> (bytes actually moved, why).  For kernels handed a
#: whole cache or a whole expert table but reading only part of it.
OVERRIDE = {
    ("GroupedMatmul", '"8,4096;16,4096,4096;;16,4096;;;;16;8"'):
        (8 * 4096 * 4096, "8 of 16 experts, gate_up"),
    ("GroupedMatmul", '"8,2048;16,2048,4096;;16,4096;;;;16;8"'):
        (8 * 2048 * 4096, "8 of 16 experts, down"),
    # Handed the whole paged KV cache; reads at most index_topk=2048 tokens of
    # kv_lora_rank=512 in bf16.  At bs=1 on a short prompt it is far less.
}

#: Same idea, but keyed by kernel Type alone: these are handed the whole paged KV
#: cache and read only the pages the indexer selected (at most index_topk=2048
#: tokens), so the declared shape overstates traffic by two orders of magnitude.
OVERRIDE_BY_TYPE = {
    "SparseFlashAttention": (2 * 2048 * 512 * 2, "<=2048 selected tokens, k+v"),
    "LightningIndexer": (2048 * 128 * 2, "<=2048 selected tokens"),
}

SHAPE_RE = re.compile(r"[-\d]+(?:,[-\d]+)*")


def traffic_bytes(shapes: str, dtypes: str) -> tuple[int, bool]:
    """Bytes implied by a shapes/dtypes pair, and whether anything was unparseable."""
    dims = [
        [int(x) for x in g.split(",")]
        for g in shapes.strip('"').split(";")
        if g.strip() and SHAPE_RE.fullmatch(g.strip())
    ]
    types = [t.strip() for t in dtypes.split(";") if t.strip()]
    total = 0
    unknown = False
    for i, d in enumerate(dims):
        n = 1
        for x in d:
            n *= max(x, 1)
        t = types[i] if i < len(types) else None
        if t in DTYPE_BYTES:
            total += n * DTYPE_BYTES[t]
        else:
            unknown = True
    return total, unknown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    path = sorted(
        glob.glob(os.path.join(args.profile, "**", "kernel_details.csv"), recursive=True)
    )[-1]
    rows = list(csv.DictReader(open(path)))

    g: dict[tuple, dict] = collections.defaultdict(
        lambda: {"dur": [], "in": 0, "out": 0, "unk": False, "core": ""}
    )
    for r in rows:
        k = (r["Type"], r["Input Shapes"])
        a = g[k]
        a["dur"].append(float(r["Duration(us)"]))
        a["core"] = r["Accelerator Core"]
        if len(a["dur"]) == 1:
            bi, ui = traffic_bytes(r["Input Shapes"], r["Input Data Types"])
            bo, uo = traffic_bytes(r["Output Shapes"], r["Output Data Types"])
            a["in"], a["out"], a["unk"] = bi, bo, ui or uo

    total = sum(sum(a["dur"]) for a in g.values()) / args.steps
    items = sorted(g.items(), key=lambda kv: -sum(kv[1]["dur"]))

    print(
        f"=== bs=1 decode step: {total/1000:.3f} ms device time, "
        f"{len(rows)/args.steps:.0f} kernel launches, floor {BW/1e12:.2f} TB/s ==="
    )
    hdr = (
        f"{'kernel':<24}{'n':>4}{'us/step':>9}{'%':>6}{'cum%':>6}"
        f"{'us/call':>9}{'MiB':>10}{'floor us':>10}{'x floor':>9}  bound"
    )
    print(hdr)
    print("-" * len(hdr))
    cum = 0.0
    for (ktype, shapes), a in items[: args.top]:
        us = sum(a["dur"]) / args.steps
        cum += us
        n = len(a["dur"]) / args.steps
        med = statistics.median(a["dur"])
        note = ""
        if (ktype, shapes) in OVERRIDE or ktype in OVERRIDE_BY_TYPE:
            byts, why = OVERRIDE.get((ktype, shapes)) or OVERRIDE_BY_TYPE[ktype]
            note = f" [{why}]"
        else:
            byts = a["in"] + a["out"]
            if a["unk"]:
                note = " [decl?]"
        floor_us = byts / BW * 1e6
        if byts < LAUNCH_BOUND_BYTES:
            bound = "launch" if med < 2 * LAUNCH_US else "compute"
            ratio = "—"
        else:
            bound = "bandwidth" if med < 1.5 * floor_us else "above floor"
            ratio = f"{med/floor_us:.2f}x"
        print(
            f"{ktype[:24]:<24}{n:>4.0f}{us:>9.1f}{100*us/total:>5.1f}%{100*cum/total:>5.1f}%"
            f"{med:>9.1f}{byts/1024**2:>10.1f}{floor_us:>10.1f}{ratio:>9}  {bound}{note}"
        )
    print(f"... 其余 {len(items)-args.top} 组: {total-cum:.1f} us/step "
          f"({100*(total-cum)/total:.1f}%)")

    # Roll every group up by what limits it, so the tail is accounted for too.
    roll: collections.Counter = collections.Counter()
    rolln: collections.Counter = collections.Counter()
    for (ktype, shapes), a in items:
        us = sum(a["dur"]) / args.steps
        med = statistics.median(a["dur"])
        byts = (OVERRIDE.get((ktype, shapes)) or OVERRIDE_BY_TYPE.get(ktype)
                or (a["in"] + a["out"], ""))[0]
        floor_us = byts / BW * 1e6
        if byts >= LAUNCH_BOUND_BYTES:
            key = "带宽受限（>=16 MB）" if med < 1.5 * floor_us else "高于地板（>=16 MB）"
        elif med >= 2 * LAUNCH_US:
            key = "compute/固定成本主导（<16 MB，>27 us/call）"
        else:
            key = "launch 主导（<16 MB）"
        roll[key] += us
        rolln[key] += len(a["dur"]) / args.steps
    print()
    print(f"{'按限制因素汇总':<38}{'ms/step':>9}{'%':>7}{'kernel/step':>13}")
    for k, v in roll.most_common():
        print(f"  {k:<36}{v/1000:>9.3f}{100*v/total:>6.1f}%{rolln[k]:>13.0f}")


if __name__ == "__main__":
    main()
