#!/usr/bin/env python
"""Attribute every kernel in a decode profile to a layer family.

The network replays as one captured graph, so the profiler cannot tell us which
module a kernel came from.  What it *can* tell us is how many times each
(kernel, input-shape) pair runs per step, and GLM-5.3-Flash has a distinct count
for every layer family:

    34 KDA layers | 11 DSA layers | 42 MoE layers | 3 dense FFN | 45 mHC sites x2

So a group that runs 34 times a step is a KDA kernel, one that runs 42 times is a
MoE kernel, and so on.  Where a count is ambiguous -- 90 is both 45x2 (mHC) and
68+22 (KDA+DSA sharing one shape) -- the shape decides, and those cases are
listed explicitly in SHAPE_RULES rather than guessed.

    $VENV/bin/python attribute_kernels.py --profile /var/tmp/glm53/prof/bs1 --steps 20

Anything the rules cannot place lands in `unclassified`, which is printed with its
share so the reader can judge how much of the table rests on the fallback.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os

# Family sizes in this model, from config.json.
KDA, DSA, MOE, DENSE, ALL_LAYERS = 34, 11, 42, 3, 45

# (kernel Type, exact Input Shapes) -> family.  Only for groups whose call count
# alone is ambiguous, or that are split across two families.
SHAPE_RULES = {
    # 90/step = 68 KDA (f_a, g_a) + 22 DSA (indexer wk, kpool compress gate).
    # Same shape, same cost, so the split is by count.
    ("MatMulV2", '"1,4096;128,4096"'): ("split", {"KDA": 68, "DSA": 22}),
    ("MatMulV2", '"1,4096;154880,4096"'): ("head", None),
    ("HcPre", None): ("mHC", None),
    ("HcPost", None): ("mHC", None),
}

# Fallback: how many times a family's kernels run per step.  First match wins,
# so the more specific counts come first.
COUNT_TO_FAMILY = [
    (KDA, "KDA"),
    (DSA, "DSA"),
    (MOE, "MoE"),
    (ALL_LAYERS, "mHC/per-layer"),
    (DENSE, "dense FFN"),
]


def family_of(ktype: str, shapes: str, per_step: float):
    rule = SHAPE_RULES.get((ktype, shapes)) or SHAPE_RULES.get((ktype, None))
    if rule is not None:
        return rule
    n = round(per_step, 1)
    for base, name in COUNT_TO_FAMILY:
        if abs(n - base) < 0.15:
            return (name, None)
        # exact small multiples: 68 = 2 per KDA layer, 84 = 2 per MoE layer, ...
        for mult in (2, 3, 4, 5, 6):
            if abs(n - base * mult) < 0.15:
                return (name, None)
    if n <= 1.2:
        return ("head/global", None)
    return ("unclassified", None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--top", type=int, default=8, help="rows to show per family")
    args = ap.parse_args()

    hits = glob.glob(
        os.path.join(args.profile, "**", "kernel_details.csv"), recursive=True
    )
    if not hits:
        raise SystemExit(f"no kernel_details.csv under {args.profile}")
    rows = list(csv.DictReader(open(sorted(hits)[-1])))

    groups: dict[tuple, list[float]] = collections.defaultdict(list)
    for r in rows:
        groups[(r["Type"], r["Name"], r["Input Shapes"], r["Accelerator Core"])].append(
            float(r["Duration(us)"])
        )

    fam_us: collections.Counter = collections.Counter()
    fam_calls: collections.Counter = collections.Counter()
    fam_rows: dict[str, list] = collections.defaultdict(list)
    total = 0.0
    for (ktype, name, shapes, core), durs in groups.items():
        us = sum(durs) / args.steps
        per_step = len(durs) / args.steps
        total += us
        fam, split = family_of(ktype, shapes, per_step)
        if fam == "split":
            for f, k in split.items():
                share = us * k / per_step
                fam_us[f] += share
                fam_calls[f] += k
                fam_rows[f].append((share, k, ktype, shapes, core))
            continue
        fam_us[fam] += us
        fam_calls[fam] += per_step
        fam_rows[fam].append((us, per_step, ktype, shapes, core))

    print(
        f"=== decode step: {total / 1000:.3f} ms of device time, "
        f"{len(rows) / args.steps:.0f} kernel launches ===\n"
    )
    print(f"{'family':<16}{'ms/step':>9}{'%':>7}{'kernels/step':>14}{'us/kernel':>11}")
    print("-" * 57)
    for fam, us in fam_us.most_common():
        n = fam_calls[fam]
        print(
            f"{fam:<16}{us / 1000:>9.3f}{100 * us / total:>6.1f}%"
            f"{n:>14.0f}{us / max(n, 1):>11.1f}"
        )
    print("-" * 57)
    print(
        f"{'TOTAL':<16}{total / 1000:>9.3f}{100:>6.1f}%"
        f"{sum(fam_calls.values()):>14.0f}{total / sum(fam_calls.values()):>11.1f}"
    )

    for fam, _ in fam_us.most_common():
        print(f"\n--- {fam} ---")
        for us, n, ktype, shapes, core in sorted(fam_rows[fam], reverse=True)[: args.top]:
            print(
                f"  {us:>8.1f} us/step  n={n:>5.0f}  {us / n:>7.1f} us  "
                f"{ktype[:24]:<24} {core[:10]:<10} {shapes[:52]}"
            )


if __name__ == "__main__":
    main()
