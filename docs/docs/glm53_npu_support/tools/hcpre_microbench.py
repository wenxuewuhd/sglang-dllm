#!/usr/bin/env python
"""HcPre / npu_hc_pre microbenchmark -- a standalone characterisation, no server.

WHY THIS EXISTS
---------------
On GLM-5.3-Flash INT8 TP1 (one Atlas A3 die, Ascend910_9362, NPU graph on),
``HcPre`` is the single largest kernel-level line item that is not bandwidth
bound: **90 calls/step (45 layers x 2 sites) x 33.0 us = 2.969 ms, 8.9% of the
33.348 ms decode step.**  It moves 1.5 MiB, whose 1.25 TB/s floor is 1.3 us.
The e2e report could only say "fixed cost of the operator, ask the vendor".
This script says *which* fixed cost, measured, with no server and no checkpoint.

WHAT IT MEASURES
----------------
  envelope   which shapes/dtypes the operator actually accepts
  m          cost vs M (tokens).  The question that decides whether HcPre is
             still a problem at larger batch.
  pipes      the profiler's PipeUtilization breakdown: which pipe is busy
  split      HcPre vs (HcPreInvRms + stock matmul + HcPreSinkhorn), i.e. where
             inside the fused kernel the time sits
  refs       stock kernels of the same shape / the same traffic, to separate
             "this operator is slow" from "this shape is slow on this hardware"
  iters      cost vs hc_sinkhorn_iters (cross-check against a known result)
  ceiling    what batching all 90 call sites into one grouped matmul would cost
  l2         same weight 90x vs 90 distinct weights (is it cache?)
  native     CANN's own aclnnMhcPreSinkhorn, called through ctypes, head to head

HOW TO RUN
----------
    source <repo>/env/env.sh                     # CANN + custom_ops + venv
    ASCEND_RT_VISIBLE_DEVICES=<die> TASK_QUEUE_ENABLE=1 \
        $VENV/bin/python hcpre_microbench.py --sections all

Needs only ``torch``, ``torch_npu`` and ``custom_ops`` (the vendor op package
that registers ``torch.ops.custom.npu_hc_pre``).  sglang is NOT imported.
Pass ``--sections m,pipes`` to run a subset; ``--help`` lists them.

WHAT YOU SHOULD SEE  (measured 2026-08-30, idle A3 die, TASK_QUEUE_ENABLE=1,
eager mode; the profiler's device Duration(us), p50 of 30 calls)
--------------------------------------------------------------------------
  M=1     28.4 us     M=32     35.7        the cost is FLAT to M=16 and only
  M=2     29.1        M=64     44.5        starts moving past M~32.  A linear
  M=4     29.2        M=256    95.5        fit over the two largest points is
  M=8     30.3        M=1024  249.0          T(M) = 19.7 us + 0.224*M us
  M=16    30.5        M=4096  936.9        break-even at M=88.  M=1 is 93% of
                                           the cost of M=16 and 64% of M=64.
                                           Run-to-run spread is about +-1 us.
  pipes @ M=1: aic_scalar_ratio 0.62, aiv_scalar_ratio 0.26, aic_mac_ratio 0.08,
               aiv_vec_ratio 0.011, aic+aiv_mte2 ~0.20, cube_utilization 56%.
               The kernel is scalar-unit bound.  The vector unit is idle (1%)
               and the MAC does 1.35 us of work inside a 29.6 us kernel.
  split: HcPreInvRms 3.8 + MatMulV3 (1,16384)x(16384,24) 25.2 + HcPreSinkhorn
         10.6 = 39.5 us unfused vs 27.8 us fused.  **The 25 us GEMV is the op.**
  refs:  a *stock* aclnn matmul of exactly that shape costs 24.8-26.3 us, i.e.
         the same.  A pure read of the same 1.5 MiB (fn.sum()) costs 8.1 us
         (195 GB/s = 16% of peak).  A trivial elementwise kernel costs 1.4 us.
         => HcPre is not slower than this hardware's own matmul at this shape.
  iters: 1 -> 23.8, 20 -> 28.1, 40 -> 32.6 us; 0.224 us per iteration, so the
         deployed 20 iterations are 4.5 us = 16% of the call.  Independently
         reproduces the served A/B (hc_sinkhorn_iters 20->1 moved the served
         HcPre median 30.34 -> 26.42 us, REPORT.md 7.5).
  ceiling: 90 separate (1,16384)x(16384,24) = 90 x 25.0 = 2252 us;
           one bmm of 90 such groups = 247 us.  **9.1x**, an upper bound only.
  l2:    same fn 90x -> 27.70 us, 90 distinct fn (135 MiB) -> 27.84 us.  Not
         a cache effect.
  native: aclnnMhcPreSinkhorn agrees with npu_hc_pre to 5e-7 (fp32 outputs) but
          costs 88.6 / 97.0 / 270.6 us at M=1/16/1024 against 29.0 / 30.5 /
          247.3.  The vendor custom op is already the fast one; CANN's own
          in-tree implementation of the same operator is 3.1x slower at M=1.

CAVEATS
-------
* Eager mode.  Host enqueue in eager is ~37-50 us/call, which is *larger* than
  the device time, so wall clock here is host bound; the numbers to read are
  the profiler's device ``Duration(us)``.  In the served run HcPre is inside an
  NPU graph and the host cost is gone.
* The microbench's 28-30 us at M=1 is below the 33.0 us seen in the real decode
  step.  The gap is co-tenancy with the rest of the step, not cache (see `l2`).
* Every number is a p50 of >=30 calls after >=10 warmup calls.  The first call
  to a new shape on Ascend pays compilation and is never in the statistics.
"""

from __future__ import annotations

import argparse
import collections
import csv
import ctypes
import glob
import os
import shutil
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401  (registers the npu device)

try:
    import custom_ops  # noqa: F401  (registers torch.ops.custom.*)
except ImportError:  # pragma: no cover
    sys.exit(
        "custom_ops is not importable; source the env that sets up the vendor "
        "op package (it is what registers torch.ops.custom.npu_hc_pre)"
    )

DEV = "npu:0"
#: GLM-5.3-Flash deployed values.
HC_MULT, HIDDEN, ITERS = 4, 4096, 20
MIX = (2 + HC_MULT) * HC_MULT  # 24 = n^2 + 2n
#: measured on this die, read+write
BW_GBPS = 1250.0
#: what the served step does: 45 layers x 2 sites
SITES_PER_STEP = 90
PROF_ROOT = os.environ.get("HCPRE_PROF_DIR", "/var/tmp/glm53/hcpre_microbench")


# --------------------------------------------------------------------------
# measurement primitives
# --------------------------------------------------------------------------
def make_inputs(m: int, hc_mult: int = HC_MULT, hidden: int = HIDDEN):
    """The four tensors npu_hc_pre takes, in the dtypes serving uses.

    x    (1, M, n, hidden)  bf16   the n residual streams
    fn   (n^2+2n, n*hidden) fp32   the mixing weights -- this is the 1.5 MiB
    scale(3,)               fp32
    base (n^2+2n,)          fp32
    """
    mix = (2 + hc_mult) * hc_mult
    return (
        torch.randn(1, m, hc_mult, hidden, dtype=torch.bfloat16, device=DEV),
        torch.randn(mix, hc_mult * hidden, dtype=torch.float32, device=DEV) * 0.01,
        torch.ones(3, dtype=torch.float32, device=DEV),
        torch.zeros(mix, dtype=torch.float32, device=DEV),
    )


def hc_pre_runner(m, hc_mult=HC_MULT, hidden=HIDDEN, iters=ITERS):
    t = make_inputs(m, hc_mult, hidden)

    def go(k=1):
        x, fn, sc, ba = t
        for _ in range(k):
            torch.ops.custom.npu_hc_pre(
                x, fn, sc, ba, hc_mult=hc_mult, hc_sinkhorn_iters=iters,
                norm_eps=1e-5, hc_eps=1e-6,
            )

    return go


def profile(cases, nrep=30, warmup=10, tag="p"):
    """Run each case `nrep` times inside one profiler session.

    Returns (rows, nrep, walls).  `rows` are kernel_details.csv rows in device
    start-time order, so the k-th block of `nrep` rows of a given kernel name
    belongs to the k-th case that produced it.  Warmup is done for *every* case
    before the session opens: on Ascend the first call to a shape pays
    compilation and tiling selection, and one such call would swamp the median.
    """
    for _, fn in cases:
        fn(warmup)
    torch.npu.synchronize()

    out = os.path.join(PROF_ROOT, tag)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    exp = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
    )
    walls = []
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=True,
        experimental_config=exp,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out),
    ) as p:
        for lab, fn in cases:
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn(nrep)
            torch.npu.synchronize()
            walls.append((lab, (time.perf_counter() - t0) * 1e6 / nrep))
        p.step()

    hits = sorted(glob.glob(os.path.join(out, "**", "kernel_details.csv"), recursive=True))
    if not hits:
        raise SystemExit(f"profiler wrote no kernel_details.csv under {out}")
    rows = list(csv.DictReader(open(hits[-1], newline="")))
    rows.sort(key=lambda r: float(r["Start Time(us)"].strip()))
    return rows, nrep, walls


def blocks(rows, name, nrep, ncase):
    """Split the rows of one kernel into `ncase` consecutive blocks of nrep."""
    sel = [r for r in rows if r["Name"] == name]
    if len(sel) != nrep * ncase:
        print(f"  ! expected {nrep*ncase} `{name}` rows, got {len(sel)}")
    return [sel[i * nrep : (i + 1) * nrep] for i in range(ncase)]


def dur(block):
    return sorted(float(r["Duration(us)"]) for r in block)


def med(block, col):
    v = [float(r[col]) for r in block if r.get(col) not in (None, "", "N/A")]
    return statistics.median(v) if v else float("nan")


def host_enqueue(fn, n=300, warmup=50):
    """Host-side cost of issuing the call, separated from the device."""
    fn(warmup)
    torch.npu.synchronize()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    fn(n)
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) * 1e6 / n, (t2 - t0) * 1e6 / n


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def sec_envelope():
    print("\n=== envelope: what npu_hc_pre actually accepts ===")
    print("(each line is a real call; a rejection prints the operator's own message)")
    trials = [
        ("hc_mult", [2, 3, 4, 6, 8], lambda v: hc_pre_runner(1, hc_mult=v)),
        ("hidden", [1024, 2048, 4096, 7168, 8192], lambda v: hc_pre_runner(1, hidden=v)),
        ("sinkhorn_iters", [0, 1, 20, 100], lambda v: hc_pre_runner(1, iters=v)),
    ]
    for knob, vals, mk in trials:
        ok, bad = [], []
        for v in vals:
            try:
                mk(v)(1)
                torch.npu.synchronize()
                ok.append(v)
            except Exception as exc:  # the operator's own assertion
                bad.append((v, str(exc).strip().splitlines()[0][:90]))
        print(f"  {knob:<16} accepted: {ok}")
        for v, msg in bad:
            print(f"  {'':<16} rejected {v}: {msg}")
    x, fn, sc, ba = make_inputs(1)
    for lab, kw in [
        ("fn as bf16", dict(fn=fn.to(torch.bfloat16))),
        ("x as fp16", dict(x=x.to(torch.float16))),
        ("x as fp32", dict(x=x.to(torch.float32))),
    ]:
        a = dict(x=x, fn=fn, sc=sc, ba=ba)
        a.update(kw)
        try:
            torch.ops.custom.npu_hc_pre(
                a["x"], a["fn"], a["sc"], a["ba"], hc_mult=HC_MULT,
                hc_sinkhorn_iters=ITERS, norm_eps=1e-5, hc_eps=1e-6)
            torch.npu.synchronize()
            print(f"  {lab:<16} accepted")
        except Exception as exc:
            print(f"  {lab:<16} rejected: {str(exc).strip().splitlines()[0][:90]}")


M_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]


def sec_m():
    print("\n=== M sweep: does HcPre amortise over tokens? ===")
    cases = [(f"M={m}", hc_pre_runner(m)) for m in M_GRID]
    rows, nrep, walls = profile(cases, tag="m")
    bl = blocks(rows, "HcPre", nrep, len(cases))
    hdr = (f"{'M':>6} {'p50 us':>8} {'p10':>7} {'p90':>7} {'wall us':>8} "
           f"{'us/token':>9} {'x floor':>8}   traffic MiB")
    print(hdr)
    print("-" * len(hdr))
    pts = []
    for (lab, _), b, (_, w) in zip(cases, bl, walls):
        m = int(lab[2:])
        d = dur(b)
        mib = (m * HC_MULT * HIDDEN * 2 + MIX * HC_MULT * HIDDEN * 4
               + m * HIDDEN * 2 + m * HC_MULT * 4 + m * HC_MULT ** 2 * 4) / 2**20
        floor = mib * 2**20 / (BW_GBPS * 1e9) * 1e6
        p50 = statistics.median(d)
        pts.append((m, p50))
        print(f"{m:>6} {p50:>8.2f} {d[len(d)//10]:>7.2f} {d[int(len(d)*.9)]:>7.2f} "
              f"{w:>8.2f} {p50/m:>9.3f} {p50/floor:>8.1f}x {mib:>12.2f}")
    # linear fit over the two largest points -- fixed vs marginal
    (m1, t1), (m2, t2) = pts[-2], pts[-1]
    slope = (t2 - t1) / (m2 - m1)
    inter = t2 - slope * m2
    print(f"\n  fit over M={m1}..{m2}:  T(M) = {inter:.1f} us + {slope:.3f}*M us")
    print(f"  break-even (marginal == fixed) at M = {inter/slope:.0f}")
    p = dict(pts)
    print(f"  M=1 {p[1]:.1f} us -> M=16 {p[16]:.1f} us ({100*p[1]/p[16]:.0f}% of it) "
          f"-> M=64 {p[64]:.1f} us ({100*p[1]/p[64]:.0f}%)")
    print(f"  at 90 sites/step: {SITES_PER_STEP*p[1]/1000:.3f} ms/step at M=1, "
          f"{SITES_PER_STEP*p[16]/1000:.3f} ms/step at M=16, "
          f"{SITES_PER_STEP*p[1024]/1000:.3f} ms/step at M=1024.")
    print("  => HcPre is a per-STEP constant across the whole deployed decode range.")
    print("     It amortises per *token* (16x at bs=16) but never leaves the step.")


def sec_pipes():
    print("\n=== pipes: which unit is busy inside HcPre ===")
    cases = [(f"M={m}", hc_pre_runner(m)) for m in (1, 16, 1024)]
    rows, nrep, _ = profile(cases, tag="pipes")
    bl = blocks(rows, "HcPre", nrep, len(cases))
    keys = ["aicore_time(us)", "aiv_time(us)", "aic_scalar_ratio", "aic_mac_ratio",
            "aic_mte2_ratio", "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio",
            "cube_utilization(%)"]
    hdr = f"{'case':<8} {'core':<9} {'blk':>7} {'dur us':>8} " + " ".join(
        f"{k.split('(')[0][-13:]:>14}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for (lab, _), b in zip(cases, bl):
        print(f"{lab:<8} {b[0]['Accelerator Core']:<9} "
              f"{b[0]['Block Num']+'/'+b[0]['Mix Block Num']:>7} "
              f"{statistics.median(dur(b)):>8.2f} "
              + " ".join(f"{med(b,k):>14.3f}" for k in keys))
    b = bl[0]
    d = statistics.median(dur(b))
    print(f"\n  at M=1: {d:.1f} us wall on device, of which")
    print(f"    scalar   {med(b,'aic_scalar_time(us)'):5.2f} (cube side) + "
          f"{med(b,'aiv_scalar_time(us)'):5.2f} (vector side) us")
    print(f"    MAC      {med(b,'aic_mac_time(us)'):5.2f} us")
    print(f"    MTE2 in  {med(b,'aic_mte2_time(us)'):5.2f} + "
          f"{med(b,'aiv_mte2_time(us)'):5.2f} us")
    print(f"    VEC      {med(b,'aiv_vec_time(us)'):5.2f} us")
    print(f"    neither core busy: {d - max(med(b,'aicore_time(us)'), med(b,'aiv_time(us)')):5.2f} us")
    print("  -> the operator is scalar/control bound, not memory and not MAC")


def sec_split():
    print("\n=== split: HcPre vs its own three-op decomposition ===")
    x, fn, sc, ba = make_inputs(1)
    xf = x.reshape(1, 1, HC_MULT * HIDDEN).float().contiguous()
    fnt = fn.t().contiguous()

    def mono(k=1):
        for _ in range(k):
            torch.ops.custom.npu_hc_pre(x, fn, sc, ba, hc_mult=HC_MULT,
                                        hc_sinkhorn_iters=ITERS, norm_eps=1e-5, hc_eps=1e-6)

    def split(k=1):
        for _ in range(k):
            r = torch.ops.custom.npu_hc_pre_inv_rms(x, epsilon=1e-5)
            m = torch.matmul(xf, fnt)
            torch.ops.custom.npu_hc_pre_sinkhorn(m, r, sc, ba, x, HC_MULT, ITERS, 1e-6)

    # numerical agreement, so the comparison is between two ways to compute the
    # same thing and not between two different computations
    y, post, comb = torch.ops.custom.npu_hc_pre(
        x, fn, sc, ba, hc_mult=HC_MULT, hc_sinkhorn_iters=ITERS, norm_eps=1e-5, hc_eps=1e-6)
    r = torch.ops.custom.npu_hc_pre_inv_rms(x, epsilon=1e-5)
    o = torch.ops.custom.npu_hc_pre_sinkhorn(
        torch.matmul(xf, fnt), r, sc, ba, x, HC_MULT, ITERS, 1e-6)
    print(f"  agreement split-vs-fused: y {(o[0].float()-y.float()).abs().max():.2e} "
          f"post {(o[1]-post).abs().max():.2e} comb {(o[2]-comb).abs().max():.2e} "
          f"(|y| up to {y.float().abs().max():.2f})")

    rows, nrep, walls = profile([("fused", mono), ("split", split)], tag="split")
    agg = collections.defaultdict(list)
    for rr in rows:
        agg[(rr["Name"][:46], rr["Input Shapes"][:34])].append(float(rr["Duration(us)"]))
    print(f"\n  {'kernel':<48} {'shapes':<36} {'n':>4} {'p50 us':>8}")
    for k, v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {k[0]:<48} {k[1]:<36} {len(v):>4} {statistics.median(v):>8.2f}")
    for lab, w in walls:
        print(f"  wall (host bound, eager) {lab:<10} {w:8.2f} us/call")


def sec_refs():
    print("\n=== refs: what else costs this much on this die ===")
    x, fn, sc, ba = make_inputs(1)
    a32 = x.reshape(1, HC_MULT * HIDDEN).float().contiguous()
    a16 = x.reshape(1, HC_MULT * HIDDEN).contiguous()
    f16 = fn.to(torch.bfloat16).contiguous()
    tiny = torch.randn(1, device=DEV)
    xp = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=DEV)
    res = torch.randn(1, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=DEV)
    po = torch.randn(1, HC_MULT, dtype=torch.float32, device=DEV)
    cb = torch.randn(1, HC_MULT, HC_MULT, dtype=torch.float32, device=DEV)

    def loop(f):
        return lambda k=1: [f() for _ in range(k)] and None

    cases = [
        ("HcPre", hc_pre_runner(1)),
        ("matmul fp32 (1,16384)x(16384,24)", loop(lambda: torch.matmul(a32, fn.t()))),
        ("matmul bf16 (1,16384)x(16384,24)", loop(lambda: torch.matmul(a16, f16.t()))),
        ("fn.sum() -- pure 1.5 MiB read", loop(lambda: fn.sum())),
        ("fn.clone() -- 1.5 MiB read+write", loop(lambda: fn.clone())),
        ("HcPost M=1", loop(lambda: torch.ops.custom.npu_hc_post(xp, res, po, cb))),
        ("torch.add on 1 element", loop(lambda: torch.add(tiny, 1.0))),
    ]
    rows, nrep, walls = profile(cases, tag="refs")
    agg = collections.defaultdict(list)
    for rr in rows:
        agg[(rr["Name"][:44], rr["Input Data Types"][:20], rr["Input Shapes"][:30])].append(
            float(rr["Duration(us)"]))
    mib = MIX * HC_MULT * HIDDEN * 4 / 2**20
    print(f"  the fn weight is {mib:.2f} MiB; its 1.25 TB/s floor is "
          f"{mib*2**20/(BW_GBPS*1e9)*1e6:.2f} us\n")
    print(f"  {'kernel':<46} {'dtypes':<20} {'shapes':<32} {'n':>4} {'p50 us':>8}")
    for k, v in sorted(agg.items(), key=lambda kv: -statistics.median(kv[1])):
        print(f"  {k[0]:<46} {k[1]:<20} {k[2]:<32} {len(v):>4} {statistics.median(v):>8.2f}")
    print("\n  host-side enqueue cost, eager (this is NOT device time; graph mode removes it):")
    for lab, fn_ in [("HcPre", hc_pre_runner(1)), ("torch.add", cases[-1][1])]:
        enq, wall = host_enqueue(fn_)
        print(f"    {lab:<12} enqueue {enq:6.2f} us/call   wall {wall:6.2f} us/call")


def sec_iters():
    print("\n=== iters: how much of HcPre is Sinkhorn? ===")
    grid = [1, 2, 5, 10, 20, 40]
    cases = [(f"it={i}", hc_pre_runner(1, iters=i)) for i in grid]
    rows, nrep, _ = profile(cases, tag="iters")
    bl = blocks(rows, "HcPre", nrep, len(cases))
    print(f"  {'iters':>6} {'p50 us':>8} {'p10':>7} {'p90':>7} {'aicore':>8} {'aiv':>7}")
    got = []
    for i, b in zip(grid, bl):
        d = dur(b)
        got.append(statistics.median(d))
        print(f"  {i:>6} {statistics.median(d):>8.2f} {d[len(d)//10]:>7.2f} "
              f"{d[int(len(d)*.9)]:>7.2f} {med(b,'aicore_time(us)'):>8.2f} "
              f"{med(b,'aiv_time(us)'):>7.2f}")
    per = (got[-1] - got[0]) / (grid[-1] - grid[0])
    print(f"\n  {per:.3f} us per Sinkhorn iteration; the deployed 20 iterations are "
          f"{20*per:.1f} us = {100*20*per/got[grid.index(20)]:.0f}% of the call")
    print("  aicore_time/aiv_time barely move with iters: the extra time is not core-busy")


def sec_ceiling():
    print("\n=== ceiling: what if all 90 call sites were one call? ===")
    print("  UPPER BOUND ONLY.  The two sites in a layer are sequentially dependent")
    print("  (site 2's input is hc_post of the attention output), and layers are a")
    print("  chain, so this is not reachable without changing the model.")
    L = SITES_PER_STEP
    xs = torch.randn(L, 1, HC_MULT * HIDDEN, dtype=torch.float32, device=DEV)
    ws = torch.randn(L, HC_MULT * HIDDEN, MIX, dtype=torch.float32, device=DEV)
    a = xs[0]
    w0 = ws[0]

    def one(k=1):
        for _ in range(k):
            torch.matmul(a, w0)

    def bmm(k=1):
        for _ in range(k):
            torch.bmm(xs, ws)

    rows, nrep, _ = profile([("single", one), ("bmm90", bmm)], tag="ceiling")
    agg = collections.defaultdict(list)
    for rr in rows:
        agg[(rr["Name"][:44], rr["Input Shapes"][:30])].append(float(rr["Duration(us)"]))
    single = grouped = None
    for k, v in agg.items():
        p = statistics.median(v)
        print(f"  {k[0]:<46} {k[1]:<32} p50={p:8.2f} us")
        if "BatchMatMul" in k[0]:
            grouped = p
        else:
            single = p
    if single and grouped:
        print(f"\n  {L} separate GEMVs: {L*single/1000:.3f} ms")
        print(f"  1 grouped GEMV of {L} groups: {grouped/1000:.3f} ms")
        print(f"  ceiling on batching the GEMV: {L*single/grouped:.1f}x "
              f"({(L*single-grouped)/1000:.3f} ms/step)")


def sec_l2():
    print("\n=== l2: is the cost a cold weight read? ===")
    x, _, sc, ba = make_inputs(1)
    one = torch.randn(MIX, HC_MULT * HIDDEN, dtype=torch.float32, device=DEV) * 0.01
    many = [torch.randn(MIX, HC_MULT * HIDDEN, dtype=torch.float32, device=DEV) * 0.01
            for _ in range(SITES_PER_STEP)]
    mib = SITES_PER_STEP * MIX * HC_MULT * HIDDEN * 4 / 2**20

    def hot(k=1):
        for _ in range(k):
            for _ in range(SITES_PER_STEP):
                torch.ops.custom.npu_hc_pre(x, one, sc, ba, hc_mult=HC_MULT,
                                            hc_sinkhorn_iters=ITERS, norm_eps=1e-5, hc_eps=1e-6)

    def cold(k=1):
        for _ in range(k):
            for f in many:
                torch.ops.custom.npu_hc_pre(x, f, sc, ba, hc_mult=HC_MULT,
                                            hc_sinkhorn_iters=ITERS, norm_eps=1e-5, hc_eps=1e-6)

    rows, nrep, _ = profile([("same fn x90", hot), (f"90 distinct fn ({mib:.0f} MiB)", cold)],
                            nrep=10, warmup=3, tag="l2")
    sel = [r for r in rows if r["Name"] == "HcPre"]
    half = len(sel) // 2
    for lab, g in [("same fn x90 (L2 hot)", sel[:half]),
                   (f"90 distinct fn, {mib:.0f} MiB (L2 cold)", sel[half:])]:
        d = sorted(float(r["Duration(us)"]) for r in g)
        print(f"  {lab:<34} n={len(g):>5} p50={statistics.median(d):7.2f} "
              f"p10={d[len(d)//10]:7.2f} p90={d[int(len(d)*.9)]:7.2f}  "
              f"90 calls = {statistics.mean(d)*SITES_PER_STEP/1000:.3f} ms")


# --------------------------------------------------------------------------
# CANN's own operator, through ctypes -- there is no torch_npu binding for it
# --------------------------------------------------------------------------
_ACL_DT = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 27}


def _load_aclnn():
    root = os.path.join(os.environ.get("ASCEND_HOME_PATH", ""), "lib64")
    if not os.path.isdir(root):
        root = "/home/developer/Ascend/ascend-toolkit/latest/lib64"
    base = ctypes.CDLL(os.path.join(root, "libnnopbase.so"), mode=ctypes.RTLD_GLOBAL)
    api = ctypes.CDLL(os.path.join(root, "libopapi.so"), mode=ctypes.RTLD_GLOBAL)
    base.aclCreateTensor.restype = ctypes.c_void_p
    base.aclCreateTensor.argtypes = [
        ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_void_p]
    gws = api.aclnnMhcPreSinkhornGetWorkspaceSize
    gws.restype = ctypes.c_int
    gws.argtypes = ([ctypes.c_void_p] * 4
                    + [ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
                       ctypes.c_double, ctypes.c_bool]
                    + [ctypes.c_void_p] * 8
                    + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)])
    run = api.aclnnMhcPreSinkhorn
    run.restype = ctypes.c_int
    run.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
    return base, gws, run


def sec_native():
    print("\n=== native: CANN's own aclnnMhcPreSinkhorn, head to head ===")
    print("  CANN ships MhcPre / MhcSinkhorn / MhcPreSinkhorn (aclnn_ops_infer).")
    print("  torch_npu binds npu_mhc_pre / npu_mhc_sinkhorn but those have no")
    print("  ascend910_93 kernel binary here (they fail with 561103); the fused")
    print("  MhcPreSinkhorn does have one but no torch binding, so it is called")
    print("  through ctypes below.")
    try:
        base, gws, run = _load_aclnn()
    except OSError as exc:
        print(f"  cannot load the aclnn libraries ({exc}); skipping")
        return

    def acl(t):
        dims = (ctypes.c_int64 * t.dim())(*t.shape)
        st = (ctypes.c_int64 * t.dim())(*t.stride())
        sd = (ctypes.c_int64 * 1)(t.numel())
        return ctypes.c_void_p(base.aclCreateTensor(
            dims, t.dim(), _ACL_DT[t.dtype], st, 0, 2, sd, 1,
            ctypes.c_void_p(t.data_ptr())))

    def build(m):
        x, fn, sc, ba = make_inputs(m)
        n = HC_MULT
        outs = [
            torch.empty(1, m, HIDDEN, dtype=torch.bfloat16, device=DEV),   # hIn
            torch.empty(1, m, n, dtype=torch.float32, device=DEV),         # hPost
            torch.empty(1, m, n * n, dtype=torch.float32, device=DEV),     # hRes
            torch.empty(1, m, n, dtype=torch.float32, device=DEV),         # hPre
            torch.empty(1, m, MIX, dtype=torch.float32, device=DEV),       # hcBeforeNorm
            torch.empty(1, m, dtype=torch.float32, device=DEV),            # invRms
            torch.empty(1, m, n, n, dtype=torch.float32, device=DEV),      # sumOut
            torch.empty(1, m, n, n, dtype=torch.float32, device=DEV),      # normOut
        ]
        w = ctypes.c_uint64(0)
        e = ctypes.c_void_p()
        rc = gws(acl(x), acl(fn), acl(sc), acl(ba), n, ITERS, 1e-6, 1e-5, False,
                 *[acl(t) for t in outs], ctypes.byref(w), ctypes.byref(e))
        if rc != 0:
            raise RuntimeError(f"aclnnMhcPreSinkhornGetWorkspaceSize -> {rc}")
        buf = torch.empty(max(w.value, 1), dtype=torch.uint8, device=DEV)
        stream = torch.npu.current_stream().npu_stream

        def native(k=1):
            for _ in range(k):
                ww = ctypes.c_uint64(0)
                ee = ctypes.c_void_p()
                gws(acl(x), acl(fn), acl(sc), acl(ba), n, ITERS, 1e-6, 1e-5, False,
                    *[acl(t) for t in outs], ctypes.byref(ww), ctypes.byref(ee))
                run(ctypes.c_void_p(buf.data_ptr()), ww.value, ee, ctypes.c_void_p(stream))

        def vendor(k=1):
            for _ in range(k):
                torch.ops.custom.npu_hc_pre(x, fn, sc, ba, hc_mult=n,
                                            hc_sinkhorn_iters=ITERS,
                                            norm_eps=1e-5, hc_eps=1e-6)

        return native, vendor, outs, (x, fn, sc, ba), w.value

    try:
        nat, ven, outs, ins, wsz = build(1)
    except RuntimeError as exc:
        print(f"  {exc}; skipping")
        return
    nat(1)
    torch.npu.synchronize()
    y, post, comb = torch.ops.custom.npu_hc_pre(
        *ins, hc_mult=HC_MULT, hc_sinkhorn_iters=ITERS, norm_eps=1e-5, hc_eps=1e-6)
    print(f"  agreement vs npu_hc_pre: hIn {(outs[0].float()-y.float()).abs().max():.2e} "
          f"hPost {(outs[1]-post).abs().max():.2e} "
          f"hRes {(outs[2].reshape(comb.shape)-comb).abs().max():.2e}")
    print(f"  workspace aclnn asks for: {wsz/2**20:.0f} MiB")

    cases = []
    for m in (1, 16, 1024):
        n_, v_, _, _, _ = build(m)
        cases += [(f"npu_hc_pre M={m}", v_), (f"MhcPreSinkhorn M={m}", n_)]
    rows, nrep, walls = profile(cases, tag="native")
    agg = collections.defaultdict(list)
    for rr in rows:
        agg[(rr["Name"][:44], rr["Input Shapes"][:30])].append(float(rr["Duration(us)"]))
    print(f"\n  {'kernel':<46} {'shapes':<32} {'n':>4} {'p50 us':>8}")
    for k, v in sorted(agg.items()):
        print(f"  {k[0]:<46} {k[1]:<32} {len(v):>4} {statistics.median(v):>8.2f}")


SECTIONS = {
    "envelope": sec_envelope,
    "m": sec_m,
    "pipes": sec_pipes,
    "split": sec_split,
    "refs": sec_refs,
    "iters": sec_iters,
    "ceiling": sec_ceiling,
    "l2": sec_l2,
    "native": sec_native,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", default="m,pipes,split,refs,iters",
                    help="comma-separated, or 'all': " + ", ".join(SECTIONS))
    args = ap.parse_args()
    want = list(SECTIONS) if args.sections == "all" else args.sections.split(",")
    bad = [s for s in want if s not in SECTIONS]
    if bad:
        raise SystemExit(f"unknown section(s) {bad}; pick from {list(SECTIONS)}")

    print(f"device        : {torch.npu.get_device_name(0)}")
    print(f"visible dies  : {os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '(all)')}")
    print(f"TASK_QUEUE_ENABLE={os.environ.get('TASK_QUEUE_ENABLE', '(unset)')}  "
          f"torch {torch.__version__}  torch_npu {torch_npu.__version__}")
    print(f"deployed shape: x(1,M,{HC_MULT},{HIDDEN}) bf16, fn({MIX},{HC_MULT*HIDDEN}) fp32 "
          f"= {MIX*HC_MULT*HIDDEN*4/2**20:.2f} MiB, sinkhorn_iters={ITERS}")
    print("mode          : eager.  Read the profiler's device Duration(us); the")
    print("                wall-clock columns are host bound and graph mode removes them.")
    for s in want:
        SECTIONS[s]()
    print(f"\nprofiler output under {PROF_ROOT} (delete when done)")


if __name__ == "__main__":
    main()
