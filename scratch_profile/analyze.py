"""Kernel-level breakdown of an Ascend profile directory.

    python3 analyze.py <profile_dir> [--top N]

Normalizes to per-forward cost by counting GroupedMatmul calls (38 per forward
for LLaDA2-mini: 19 MoE layers x w13/w2), so a capture spanning several denoise
forwards still reports a per-forward number.

Reports, per kernel: time share, calls, and the AI Core pipe counters that say
*why* it costs what it does --
  mac_ratio  cube (matmul) pipe busy
  mte2_ratio HBM -> L1/L0 load pipe busy   (high = bandwidth bound)
  vec_ratio  vector pipe busy
  scalar_r   scalar pipe busy              (high = loop/address overhead)
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

GMM_PER_FORWARD = 38


def find_csv(path):
    hits = glob.glob(os.path.join(path, "**", "kernel_details.csv"), recursive=True)
    if not hits:
        sys.exit(f"no kernel_details.csv under {path}")
    return max(hits, key=os.path.getmtime)


def num(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return 0.0


def classify(name):
    if "GroupedMatmul" in name:
        return "MoE GroupedMatmul"
    if any(
        k in name for k in ("FusedInferAttention", "IncreFlashAttention", "PromptFlash")
    ):
        return "attention"
    if "atMul" in name or "GEMM" in name:
        return "dense matmul"
    if any(
        k in name
        for k in (
            "MoeInitRouting",
            "MoeFinalizeRouting",
            "MoeGatingTopK",
            "MoeCompute",
            "SwiGlu",
        )
    ):
        return "MoE glue"
    if any(k in name for k in ("Softmax", "ArgMax", "Max2", "Exp", "LogSumExp")):
        return "denoise reduction"
    if "RmsNorm" in name or "rope" in name or "Rope" in name:
        return "norm/rope"
    return "other elementwise"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    csv_path = find_csv(args.path)
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        sys.exit(f"{csv_path} is empty")
    dur_key = next(k for k in rows[0] if "Duration" in k)

    n_gmm = sum(1 for r in rows if "GroupedMatmul" in r["Name"])
    n_fwd = max(1.0, n_gmm / GMM_PER_FORWARD)
    total = sum(num(r, dur_key) for r in rows)

    print(f"file      : {csv_path}")
    print(f"kernels   : {len(rows)}, GroupedMatmul {n_gmm} -> forwards ~= {n_fwd:.1f}")
    print(
        f"total     : {total/1000:.1f} ms  ->  {total/1000/n_fwd:.2f} ms per forward\n"
    )

    cat = defaultdict(lambda: [0, 0.0])
    for r in rows:
        c = cat[classify(r["Name"])]
        c[0] += 1
        c[1] += num(r, dur_key)
    print("== by category (ms per forward) ==")
    for name, (n, d) in sorted(cat.items(), key=lambda x: -x[1][1]):
        print(
            f"  {name:22s} {d/1000/n_fwd:7.2f} ms  {d/total*100:5.1f}%  ({n/n_fwd:.0f} calls/fwd)"
        )

    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0, ""])
    for r in rows:
        key = (r["Name"][:36], r.get("Input Shapes", "").replace('"', "")[:40])
        a = agg[key]
        a[0] += 1
        a[1] += num(r, dur_key)
        a[2] += num(r, "aic_mac_ratio")
        a[3] += num(r, "aic_mte2_ratio")
        a[4] += num(r, "aiv_vec_ratio")
        a[5] += num(r, "aic_scalar_ratio") + num(r, "aiv_scalar_ratio")

    print(f"\n== top {args.top} kernels ==")
    hdr = f"{'kernel':36s} {'ms/fwd':>7} {'n/fwd':>6} {'us':>7} {'mac':>5} {'mte2':>5} {'vec':>5} {'scal':>5}  shapes"
    print(hdr)
    for (name, shp), (n, d, mac, mte2, vec, scal, _) in sorted(
        agg.items(), key=lambda x: -x[1][1]
    )[: args.top]:
        print(
            f"{name:36s} {d/1000/n_fwd:7.2f} {n/n_fwd:6.1f} {d/n:7.0f} "
            f"{mac/n:5.2f} {mte2/n:5.2f} {vec/n:5.2f} {scal/n:5.2f}  {shp}"
        )

    gmm = [r for r in rows if "GroupedMatmul" in r["Name"]]
    if gmm:
        print("\n== MoE GroupedMatmul roofline ==")
        by_w = defaultdict(list)
        for r in gmm:
            parts = r.get("Input Shapes", "").replace('"', "").split(";")
            by_w[parts[1] if len(parts) > 1 else "?"].append(r)
        for w, rs in sorted(by_w.items(), key=lambda x: -len(x[1])):
            dims = [int(x) for x in re.findall(r"\d+", w)]
            if len(dims) != 3:
                continue
            E, N, K = dims
            M = int(re.findall(r"\d+", rs[0]["Input Shapes"])[0])
            dur = sum(num(r, dur_key) for r in rs) / len(rs)
            wb = E * N * K * 2
            fl = 2 * M * N * K
            print(
                f"  w={w:>18} M={M:>7} {dur:7.0f}us  "
                f"{fl/1e12/(dur/1e6):6.0f} TFLOPS ({fl/1e12/(dur/1e6)/320*100:3.0f}% of 320T)  "
                f"weights {wb/1e9:.2f}GB -> {wb/1e9/(dur/1e6)/1000:.2f} TB/s  "
                f"t/expert={M//E}"
            )


if __name__ == "__main__":
    main()
