#!/usr/bin/env python
"""Check a standalone bench against what that layer family costs in the full network.

Totals matching is not the check.  A replay that launches a different set of
kernels can land on the right total by accident, and a replay that is missing a
kernel will simply look fast.  So this compares the *inventory*: which op groups
ran, at which shapes, how many times per layer, and only then the time.

Reference is `reference_inventory_cfgI.json`, extracted from the same profile the
report quotes, using `tools/attribute_kernels.py`'s own family_of/split_of --
one attribution in this project, not a second one.  Note the grouping key has to
match that tool exactly: adding dtype to the key splits a group in two, and
since the call count *is* the attribution rule, the halves can land in different
families (measured: DSA reads 902 instead of 891 that way).

Usage:
  python regress_against_network.py --family KDA --profile /var/tmp/glm53/prof/oplab_kda --steps 30
"""
from __future__ import annotations

import argparse, collections, csv, glob, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference_inventory_cfgI.json")


def inventory(profile: str, steps: int) -> dict:
    """Group a profile the way attribute_kernels does, per *iteration*."""
    f = glob.glob(os.path.join(profile, "**", "kernel_details.csv"), recursive=True)
    if not f:
        sys.exit(f"no kernel_details.csv under {profile}")
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(f[0])):
        out[(r["Type"], r["Input Shapes"])].append(float(r["Duration(us)"]))
    return {k: (len(v) / steps, sum(v) / steps, statistics.median(v)) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("KDA", "DSA"), required=True)
    ap.add_argument("--profile", required=True, help="profile dir of the standalone bench")
    ap.add_argument("--steps", type=int, required=True, help="replays captured in that profile")
    ap.add_argument("--layers", type=int, default=None, help="layers the bench replays (default 1)")
    args = ap.parse_args()

    ref = json.load(open(REF))
    fam, summary = args.family, json.load(open(REF))["summary"][args.family]
    n_layers = summary["layers"]
    per_layer = args.layers or 1

    want = collections.defaultdict(lambda: [0.0, 0.0])
    for op in ref["ops"][fam]:
        w = want[(op["type"], op["shapes"])]
        w[0] += op["per_step"] / n_layers * per_layer      # calls per replay
        w[1] += op["us_per_step"] / n_layers * per_layer   # us per replay
    got = inventory(args.profile, args.steps)

    print(f"=== {fam}: standalone ({per_layer} layer(s)) vs in-network per-layer ===\n")
    print(f"{'op':<26}{'shapes':<34}{'calls ref/got':>16}{'us ref':>9}{'us got':>9}{'x':>7}")
    print("-" * 101)
    missing, extra, tot_r, tot_g = [], [], 0.0, 0.0
    for k in sorted(want, key=lambda k: -want[k][1]):
        t, sh = k
        r_n, r_us = want[k]
        tot_r += r_us
        if k not in got:
            missing.append((t, sh, r_n, r_us))
            continue
        g_n, g_us, _ = got[k]
        tot_g += g_us
        flag = "" if abs(g_n - r_n) < 0.15 else "  <-- CALL COUNT"
        print(f"{t[:25]:<26}{sh[:33]:<34}{r_n:8.1f}/{g_n:<7.1f}{r_us:9.1f}{g_us:9.1f}"
              f"{g_us/r_us if r_us else 0:7.2f}{flag}")
    for k, (g_n, g_us, _) in got.items():
        if k not in want:
            extra.append((k[0], k[1], g_n, g_us))
            tot_g += g_us

    print("-" * 101)
    print(f"{'TOTAL':<60}{tot_r:9.1f}{tot_g:9.1f}{tot_g/tot_r if tot_r else 0:7.2f}")
    if missing:
        print(f"\n⚠ {len(missing)} group(s) in the network but NOT in the bench -- the bench is "
              f"incomplete, and a missing kernel just looks fast:")
        for t, sh, n, us in missing:
            print(f"    {t[:25]:<26}{sh[:40]:<42}{n:6.1f} calls  {us:7.1f} us")
    if extra:
        print(f"\n⚠ {len(extra)} group(s) in the bench but NOT in the network -- the bench is "
              f"doing something the model does not:")
        for t, sh, n, us in sorted(extra, key=lambda x: -x[3])[:12]:
            print(f"    {t[:25]:<26}{sh[:40]:<42}{n:6.1f} calls  {us:7.1f} us")

    scaled = tot_g / per_layer * n_layers / 1000
    print(f"\n  bench x {n_layers} layers = {scaled:.3f} ms   vs in-network {summary['ms_per_step']} ms"
          f"   ({100*(scaled/summary['ms_per_step']-1):+.1f}%)")
    print("  ⚠ the totals agreeing is not the check; the inventory agreeing is. A replay with the "
          "wrong\n    kernels can hit the right total by accident.")
    return 1 if (missing or extra) else 0


if __name__ == "__main__":
    sys.exit(main())
