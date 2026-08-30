#!/usr/bin/env python
"""One GLM-5.3-Flash KDA layer, decode, bs=1, in an NPU graph -- no server.

WHAT THIS IS
------------
The 34 KDA (linear-attention) layers of GLM-5.3-Flash INT8 cost **10.357 ms of
the 31.274 ms decode step** on one Atlas A3 die at bs=1 with the NPU graph on.
That is 272 kernel launches, **8 per layer, 304.6 us per layer** -- or really 9
per layer; see README 5.1, the ninth is filed under `unclassified`.  This script
rebuilds them, in order, with the deployed shapes and dtypes, and
times them inside a captured NPU graph.  Nothing here imports a model, loads a
checkpoint, or starts a server.

    device us/layer  x34  =  what the served step spends in KDA

The reference (config I, `../int8_singlecard/data/kernel_attribution_cfgI.txt`):

     us/step   n  us/call  operator                      input shapes
    --------------------------------------------------------------------------
      5877.2  34    172.9  MatMulV2                      "1,4096;24896,4096"
      2034.1  34     59.8  MatMulV2                      "1,8192;4096,8192"
      1191.3  34     35.0  fused_sigmoid_gating_delta_rule
       519.8  34     15.3  causal_conv1d_4
       416.2  34     12.2  BatchMatMulV2                 "2,1,128;2,8192,128"
       146.1  34      4.3  ClipByValueV2                 "1;;"
       121.6  34      3.6  layer_norm_gated_fwd_kernel
        50.7  34      1.5  Cast                          "2"
    --------------------------------------------------------------------------
     10356.9 272           = 10.357 ms/step, 304.6 us/layer

THE SEQUENCE (glm5_next.py Glm5NextLinearAttention.forward, fused path)

    x [1,4096] bf16
      -> fused_qkvbfg_a_proj   W[24896,4096] bf16        MatMulV2
         split [24576 qkv | 64 beta | 256 f_a,g_a]
      -> fused_fg_b_proj       bmm [2,1,128]x[2,8192,128] BatchMatMulV2
         -> forget_gate [1,8192], norm_gate [1,8192]
      -> torch.ops.npu.causal_conv1d(qkv, w[4,24576], conv_states[S,3,24576],
                                     run_mode=1, activation_mode=1)   causal_conv1d_4
      -> split q,k,v [1,1,64,128] each
      -> fused_sigmoid_gating_delta_rule_update(..., is_kda=True)
      -> FusedRMSNormGated(128, "sigmoid")               layer_norm_gated_fwd_kernel
      -> o_proj               W[4096,8192] bf16          MatMulV2

WHY 57% OF IT IS ONE GEMM AND YOU CANNOT HAVE IT
------------------------------------------------
`[1,4096] x [24896,4096]` bf16 reads 194.6 MiB of weight.  At the 1.25 TB/s this
die actually reaches that is a **163.2 us floor** against a measured 172.9 us --
**1.07x, already against the wall.**  Same story for the o_proj at 1.02x.  The
`refs` section proves this rather than asserting it: for each operator it also
times a stock kernel of the same shape, a pure read of the same bytes, a
read+write of the same bytes, and a trivial elementwise kernel.  Four controls
turn "N times the floor" into a verdict instead of an alarm.

SEQUENCE LENGTH
---------------
**KDA is recurrent: its decode-time state is fixed size and nothing in the layer
depends on the sequence length.**  The `sweep` section demonstrates this rather
than assuming it -- it prints the shape fingerprint of every tensor at each
length (they are byte-identical) and then measures all four anyway, so the
spread you see is this machine's noise floor, not a length effect.  The knob
that *does* move KDA is the number of state slots (`slots` section), i.e. concurrency,
because the state pool is 4.19 MiB per layer per slot and competes for L2.

HOW TO RUN
----------
    source <repo>/env/env.sh
    ASCEND_RT_VISIBLE_DEVICES=<die> python bench_kda_layer.py

    --sections layer,sweep,slots,refs,eager  (default all but eager)
    --seq-lens 1024,4096,32768,131072
    --slots 17            state-pool slots (served: max-running-requests+1)
    --reps 30 --warmup 10

Needs `torch`, `torch_npu`, `sgl_kernel_npu` and two Triton kernels that live in
the sglang tree (see README "what this imports").  It does NOT import any sglang
model, backend, scheduler or memory-pool code.

READ THE PROFILER, NOT THE CLOCK
--------------------------------
Every number printed as `us` is the profiler's device `Duration(us)`.  Wall clock
is reported only in the `eager` section and only to show why graph mode is the
only mode worth measuring.  p50 / p90 / max are printed for every kernel: on this
machine the distribution is the evidence.  A change that moved one GEMM's median
by 2.4% moved its p90 from 338 to 183 us -- the mean would have hidden that.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import itertools
import os
import shutil
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401  registers the npu device and torch.ops.npu

try:
    import sgl_kernel_npu  # noqa: F401  loads libsgl_kernel_npu.so -> torch.ops.npu.causal_conv1d
except ImportError:
    sys.exit("sgl_kernel_npu is not importable; source the project env.sh first")

try:
    from sglang.kernels.ops.attention.fla.fused_norm_gate import layer_norm_gated_fwd
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"cannot import the two Triton kernels ({exc}).\n"
        "They are kernels, not model code -- put the sglang tree on PYTHONPATH:\n"
        "  export PYTHONPATH=<worktree>/python:$PYTHONPATH"
    )

DEV = "npu:0"
BF = torch.bfloat16

# ---- GLM-5.3-Flash deployed values (config.json text_config, TP=1) ----------
HIDDEN = 4096
H = 64  # linear_attn_config.num_heads
D = 128  # linear_attn_config.head_dim
CONV_K = 4  # linear_attn_config.short_conv_kernel_size
LOWER_BOUND = -5.0  # linear_attn_config.gate_lower_bound
RMS_EPS = 1e-5
PROJ = H * D  # 8192   q, k and v each
QKV = 3 * PROJ  # 24576  conv channels
FUSED = QKV + H + 2 * D  # 24896  q|k|v | beta(64) | f_a(128), g_a(128)
KDA_LAYERS = 34

#: measured read+write bandwidth on this die, GB/s (REPORT.md 3)
BW_GBPS = 1250.0
#: L2 per die, MiB (REPORT.md 6.1).  Big enough to hold a whole KDA weight,
#: which is why every bandwidth measurement here needs a cold control.
L2_MIB = 168.0

#: config I, per call, device us.  keyed by (profiler Type, input shapes) where
#: the shapes disambiguate the two MatMulV2 rows.
REF = {
    ("MatMulV2", "1,4096;24896,4096"): 172.9,
    ("MatMulV2", "1,8192;4096,8192"): 59.8,
    ("fused_sigmoid_gating_delta_rule", None): 35.0,
    ("causal_conv1d_4", None): 15.3,
    ("BatchMatMulV2", "2,1,128;2,8192,128"): 12.2,
    ("ClipByValueV2", "1;;"): 4.3,
    ("layer_norm_gated_fwd_kernel", None): 3.6,
    ("Cast", "2"): 1.5,
}
REF_LAYER_US = 304.6
REF_STEP_MS = 10.357

PROF_ROOT = os.environ.get("OPLAB_PROF_DIR", "/var/tmp/glm53/oplab/kda")


# ---------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------
class KDALayer:
    """Weights, state pool and the 9-kernel decode body of one KDA layer.

    Every tensor is allocated once and reused, because an NPU graph replays the
    addresses it captured.  `body()` writes only into buffers it owns.
    """

    def __init__(self, slots: int = 17, slot: int = 0, seed: int = 0):
        g = torch.Generator(device="cpu").manual_seed(seed)

        def rnd(*shape, dtype=BF, scale=0.05):
            return (torch.randn(*shape, generator=g) * scale).to(dtype).to(DEV)

        # -- weights, in the dtypes the W8A8 checkpoint actually loads --------
        # All eight KDA projections are in the checkpoint's `ignore` list, so
        # they load bf16 and produce MatMulV2, not QuantBatchMatmulV3.  This is
        # the author's decision, not an omission (REPORT.md 6.4).
        self.w_qkvbfg = rnd(FUSED, HIDDEN)  # [24896, 4096] bf16
        self.w_fg_b = rnd(2, PROJ, D)  # [2, 8192, 128]  f_b | g_b
        self.w_o = rnd(HIDDEN, PROJ)  # [4096, 8192]
        # conv weight is fp32 on the parameter and cast once per layer to the
        # [width, channels] bf16 the AOT operator demands; the served code
        # caches exactly this tensor (_get_conv_weights_t).
        self.conv_w = rnd(QKV, CONV_K).transpose(0, 1).to(BF).contiguous()
        self.conv_bias = None  # GLM-5.3 has none
        self.norm_w = torch.ones(D, device=DEV, dtype=BF)
        self.A_log = (torch.randn(1, 1, H, 1, generator=g) * 0.1).float().to(DEV)
        self.dt_bias = (torch.randn(PROJ, generator=g) * 0.1).float().to(DEV)

        # -- state pools ------------------------------------------------------
        # conv:  [slots, kernel-1, channels] bf16, window-major.  The AOT
        #        operator requires this layout and rejects fp32 outright.
        # ssm:   [slots, HV, V, K] fp32 = 4.19 MiB per slot per layer.
        self.conv_state = torch.zeros(slots, CONV_K - 1, QKV, device=DEV, dtype=BF)
        self.ssm_state = torch.zeros(slots, H, D, D, device=DEV, dtype=torch.float32)

        # -- per-step metadata, int32 (int64 silently miscomputes) ------------
        self.cache_idx = torch.full((1,), slot, dtype=torch.int32, device=DEV)
        self.qsl = torch.tensor([0, 1], dtype=torch.int32, device=DEV)

        self.x = rnd(1, HIDDEN)
        self.slots, self.slot = slots, slot

    def fingerprint(self):
        """Every tensor that enters the layer, so a sweep can prove nothing moved."""
        return tuple(
            (n, tuple(t.shape), str(t.dtype))
            for n, t in sorted(vars(self).items())
            if isinstance(t, torch.Tensor)
        )

    def body(self):
        """One decode step of one KDA layer.  No host syncs, no data-dependent
        shapes -- nothing here can break graph capture."""
        # 1. fused_qkvbfg_a_proj            MatMulV2 "1,4096;24896,4096"
        fused = torch.matmul(self.x, self.w_qkvbfg.t())
        qkv, _beta, fg_a = torch.split(fused, [QKV, H, 2 * D], dim=-1)
        beta = _beta.unsqueeze(0)  # [1,1,64]

        # 2. fused_fg_b_proj                BatchMatMulV2 "2,1,128;2,8192,128"
        fg = torch.bmm(
            fg_a.view(-1, 2, D).transpose(0, 1), self.w_fg_b.transpose(-1, -2)
        )
        forget_gate, norm_gate = fg[0], fg[1]  # [1,8192] each

        # 3. index guard on the conv slot   ClipByValueV2 "1;;"
        #    Padded graph rows carry -1 and would wrap to the last mamba slot.
        #    Numerically a no-op at bs=1; kept because config I has it.
        idx = torch.clamp(self.cache_idx, min=0)

        # 4. depthwise causal conv, one packed call over all 24576 channels
        #    causal_conv1d_4 (+ an internal Cast of query_start_loc)
        qkv = torch.ops.npu.causal_conv1d(
            qkv.contiguous(),  # last-dim prefix slice: contiguous only at bs=1
            self.conv_w,
            conv_states=self.conv_state,
            bias=self.conv_bias,
            query_start_loc=self.qsl,
            cache_indices=idx,
            activation_mode=1,  # silu
            pad_slot_id=-1,
            run_mode=1,  # decode
        )

        # 5. the recurrence                 fused_sigmoid_gating_delta_rule
        q, k, v = qkv.split([PROJ, PROJ, PROJ], dim=-1)
        q = q.unflatten(-1, (-1, D)).unsqueeze(0)  # [1,1,64,128]
        k = k.unflatten(-1, (-1, D)).unsqueeze(0)
        v = v.unflatten(-1, (-1, D)).unsqueeze(0)
        core = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            q=q,
            k=k,
            v=v,
            a=forget_gate,
            b=beta,
            initial_state_source=self.ssm_state,
            initial_state_indices=self.cache_idx,
            cu_seqlens=self.qsl,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            lower_bound=LOWER_BOUND,
        )

        # 6. gated RMSNorm per head         layer_norm_gated_fwd_kernel
        y, _, _, _ = layer_norm_gated_fwd(
            core.reshape(-1, D),
            norm_gate.reshape(-1, D),
            self.norm_w,
            None,
            activation="sigmoid",
            eps=RMS_EPS,
            is_rms_norm=True,
        )

        # 7. o_proj                         MatMulV2 "1,8192;4096,8192"
        return torch.matmul(y.reshape(1, PROJ), self.w_o.t())


# ---------------------------------------------------------------------------
# measurement primitives
# ---------------------------------------------------------------------------
def capture(fn, warmup=10):
    """Warm up, then capture `fn` into an NPU graph and return a replay closure.

    Warmup is not optional: the first call to a new shape on Ascend pays kernel
    compilation and tiling selection, and Triton kernels JIT on first call.  A
    graph capture that ran cold would capture the compile too.
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    g = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(g):
        fn()
    torch.npu.synchronize()
    g.replay()
    torch.npu.synchronize()

    def replay(k=1):
        for _ in range(k):
            g.replay()

    return replay, g


#: numel of the sentinel tensor that separates cases inside one profiler
#: session.  It must be a shape nothing else in the benchmark uses.
MARKER_NUMEL = 1357


def profile(cases, nrep=30, tag="p"):
    """Profile each (label, callable) in one session; return kernel rows per case.

    Cases are separated by a **marker kernel**, not by assuming that every case
    launches the same number of kernels.  Slicing the row list into equal blocks
    is wrong the moment two cases differ in launch count, and it fails silently:
    it attributes one case's kernels to its neighbour.  (Measured here before the
    fix -- it reported a 68 MiB read as 7.06 us, i.e. 0.12x that read's own
    bandwidth floor, which is not a thing that can happen.)

    Every case is warmed before the session opens: the first call to a new shape
    on Ascend pays kernel compilation and tiling selection, and one such call
    would swamp a 30-sample median.
    """
    for _, fn in cases:
        fn(5)
    torch.npu.synchronize()
    mk = torch.zeros(MARKER_NUMEL, device=DEV)

    def mark():
        mk.add_(1.0)

    mark()
    torch.npu.synchronize()

    out = os.path.join(PROF_ROOT, tag)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    exp = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
    )
    labels, walls = [], []
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
            mark()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn(nrep)
            torch.npu.synchronize()
            walls.append((lab, (time.perf_counter() - t0) * 1e6 / nrep))
            labels.append(lab)
        mark()
        torch.npu.synchronize()
        p.step()

    hits = sorted(glob.glob(os.path.join(out, "**", "kernel_details.csv"), recursive=True))
    if not hits:
        raise SystemExit(f"profiler wrote no kernel_details.csv under {out}")
    rows = list(csv.DictReader(open(hits[-1], newline="")))
    rows.sort(key=lambda r: float(r["Start Time(us)"].strip()))

    tag_m = str(MARKER_NUMEL)
    cuts = [i for i, r in enumerate(rows) if tag_m in r["Input Shapes"]]
    if len(cuts) != len(cases) + 1:
        raise SystemExit(
            f"marker kernel appeared {len(cuts)} times, expected {len(cases)+1}. "
            "Case attribution would be wrong, so refusing to report numbers."
        )
    blocks = [rows[cuts[i] + 1 : cuts[i + 1]] for i in range(len(cases))]
    return list(zip(labels, blocks)), walls


def tabulate(rows, nrep, per_layer_ref=None, layers=KDA_LAYERS):
    """Aggregate kernel rows by (Type, Input Shapes) and print p50/p90/max."""
    agg = collections.defaultdict(list)
    for r in rows:
        key = (r["Type"] or r["Name"], r["Input Shapes"].strip('" '))
        agg[key].append(float(r["Duration(us)"]))
    items = sorted(agg.items(), key=lambda kv: -sum(kv[1]))

    print(
        f"  {'operator':<34} {'shapes':<24} {'n/rep':>5} "
        f"{'p50':>8} {'p90':>8} {'max':>8} {'ref':>7} {'p50/ref':>8}"
    )
    total = 0.0
    for (typ, shapes), v in items:
        v = sorted(v)
        n_per_rep = len(v) / nrep
        p50 = statistics.median(v)
        p90 = v[min(len(v) - 1, int(len(v) * 0.9))]
        ref = REF.get((typ, shapes)) or REF.get((typ, None))
        total += p50 * n_per_rep
        rs = f"{ref:7.1f}" if ref else "      -"
        rr = f"{p50/ref:8.2f}x" if ref else "        "
        print(
            f"  {typ[:34]:<34} {shapes[:24]:<24} {n_per_rep:5.1f} "
            f"{p50:8.2f} {p90:8.2f} {v[-1]:8.2f} {rs} {rr}"
        )
    nk = sum(len(v) for v in agg.values()) / nrep
    print(f"  {'-'*34} {'-'*24} {'-'*5} {'-'*8}")
    print(f"  {'per layer':<34} {'':<24} {nk:5.1f} {total:8.2f} us")
    if per_layer_ref:
        print(
            f"  {'x %d layers' % layers:<34} {'':<24} {nk*layers:5.0f} "
            f"{total*layers/1000:8.3f} ms   target {per_layer_ref*layers/1000:.3f} ms"
            f"   ({100*(total/per_layer_ref-1):+.1f}%)"
        )
    return total, agg


def contention_warning(agg, refs):
    """Refuse to let a shared die be read as an operator result.

    Two independent tests, because on this machine they fail at different times:

      * **median** against a fixed, bandwidth-bound shape whose cost is known.
        Catches a die that is busy for the whole run.
      * **dispersion** (p90/p50) over every reference shape.  Catches a die that
        is busy in bursts -- which is the common case, and which the median can
        miss entirely.  Measured live on 2026-08-31: a DSA layer came out 79.5%
        over target while its biggest GEMM's median was only 1.12x, because the
        damage was all in the tail (p50 71 us, p90 223 us).

    The project's own most important piece of evidence was a distribution shape,
    not a mean.  This is the same lesson, wired into the harness.
    """
    hits = []
    for key, v in agg.items():
        # `refs` keys some operators by (Type, shapes) and some by name alone.
        ref = refs.get(key) or refs.get((key[0], None))
        if not ref or not v:
            continue
        v = sorted(v)
        p50 = statistics.median(v)
        p90 = v[min(len(v) - 1, int(len(v) * 0.9))]
        hits.append((key, p50 / ref, p90 / p50 if p50 else 1.0))
    if not hits:
        return
    worst_ratio = max(h[1] for h in hits)
    worst_spread = max(h[2] for h in hits)
    if worst_ratio <= 1.25 and worst_spread <= 1.25:
        return
    print(
        f"\n  !! THIS DIE IS SHARED -- do not report these numbers as operator costs.\n"
        f"     worst median vs config I : {worst_ratio:.2f}x   (clean die: <= 1.15x)\n"
        f"     worst p90/p50 dispersion : {worst_spread:.2f}x   (clean die: <= 1.10x)"
    )
    for key, r, sp in sorted(hits, key=lambda h: -max(h[1], h[2]))[:4]:
        print(f"       {key[0][:24]:<24} {str(key[1])[:26]:<26} p50/ref {r:5.2f}x  p90/p50 {sp:5.2f}x")
    print(
        "     Run `npu-smi info`.  A busy die inflates everything by roughly 1.7x,\n"
        "     which lands inside the '20% is fine / 2x means wrong shape' gap and is\n"
        "     exactly the kind of number that gets mistaken for a real regression."
    )


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def sec_layer(args):
    print("\n=== layer: the kernels of one KDA layer, captured and replayed ===")
    print("  config I attributes 8 per layer.  This runs 9: torch.ops.npu.causal_conv1d")
    print("  emits BOTH a Cast \"2\" (query_start_loc) and a Cast \"1\" (cache_indices),")
    print("  and the count-based attribution tool filed the second one under")
    print("  `unclassified` because 59 calls/step share that shape.  See README 5.1.")
    lay = KDALayer(slots=args.slots)
    replay, _g = capture(lay.body, warmup=args.warmup)
    blocks, walls = profile([("kda", replay)], nrep=args.reps, tag="layer")
    _, rows = blocks[0]
    print(
        f"  bs=1, T=1, {H} heads x {D}, hidden {HIDDEN}, fused proj {FUSED}, "
        f"conv width {CONV_K} over {QKV} channels, {args.slots} state slots"
    )
    print(f"  graph replay, {args.reps} reps after {args.warmup} warmups\n")
    total, agg = tabulate(rows, args.reps, per_layer_ref=REF_LAYER_US)
    print(f"  wall clock per replay (host, for reference only): {walls[0][1]:.1f} us")
    contention_warning(agg, REF)
    return total


def sec_sweep(args):
    print("\n=== sweep: does anything in a KDA layer depend on sequence length? ===")
    print("  KDA is recurrent.  Its decode state is a fixed [64,128,128] fp32")
    print("  matrix per slot plus a [3,24576] conv window; the sequence length")
    print("  enters no shape, no stride and no loop bound.  First we prove that,")
    print("  then we measure all four lengths anyway.\n")

    lays = {n: KDALayer(slots=args.slots) for n in args.seq_lens}
    base = None
    for n, lay in lays.items():
        fp = lay.fingerprint()
        if base is None:
            base, base_n = fp, n
            print(f"  n={n:>8}: {len(fp)} tensors  (reference fingerprint)")
        else:
            same = "IDENTICAL" if fp == base else "*** DIFFERS ***"
            print(f"  n={n:>8}: {same} to n={base_n}")
            if fp != base:
                for a, b in zip(base, fp):
                    if a != b:
                        print(f"      {a} -> {b}")
    print(
        "\n  So any spread below is this machine's noise floor, not an n effect.\n"
        "  (An honest O(1) claim has to be structural.  Measuring four identical\n"
        "   workloads and finding them equal proves nothing on its own.)\n"
    )

    cases = []
    for n in args.seq_lens:
        replay, _ = capture(lays[n].body, warmup=args.warmup)
        cases.append((f"n={n}", replay))
    blocks, _ = profile(cases, nrep=args.reps, tag="sweep")
    print(f"  {'seq len':>10} {'us/layer':>10} {'x34 (ms)':>10} {'vs shortest':>12}")
    first = None
    for lab, rows in blocks:
        agg = collections.defaultdict(list)
        for r in rows:
            agg[(r["Type"] or r["Name"], r["Input Shapes"])].append(float(r["Duration(us)"]))
        tot = sum(statistics.median(v) * len(v) / args.reps for v in agg.values())
        first = first or tot
        print(
            f"  {lab:>10} {tot:10.2f} {tot*KDA_LAYERS/1000:10.3f} "
            f"{100*(tot/first-1):+11.1f}%"
        )

def sec_family(args):
    """All 34 KDA layers of one decode step, chained, in one graph.

    The point of this section is the assumption the whole deliverable rests on:
    that "one layer x 34" is the same thing as "34 layers".  It is not obviously
    true.  A single layer replayed 30 times reads the *same* 259 MiB of weight
    every time and this die's L2 is ~168 MiB, so the single-layer number is
    partly L2-warm.  A real decode step streams 34 distinct layers past L2 and
    every weight read is cold.  Building all 34 for real is the only way to know
    how much of the gap that accounts for.

    Costs ~11 GB of HBM.  Use --family-layers to shrink it if the die is shared.
    """
    n = args.family_layers
    print(f"\n=== family: {n} distinct KDA layers chained in one graph ===")
    need = n * (FUSED * HIDDEN + HIDDEN * PROJ + 2 * PROJ * D) * 2 / 2**30
    need += n * (args.slots * H * D * D * 4 + args.slots * (CONV_K - 1) * QKV * 2) / 2**30
    print(f"  allocating ~{need:.1f} GB of weights and state ({n} x 34ths of a step)")
    try:
        lays = [KDALayer(slots=args.slots, seed=i) for i in range(n)]
    except RuntimeError as exc:
        print(f"  ! could not allocate ({str(exc).splitlines()[0][:90]})")
        print("    lower --family-layers, or wait for the die to free up.")
        return

    def chain():
        h = lays[0].x
        for lay in lays:
            lay.x.copy_(h)
            h = lay.body()
        return h

    replay, _g = capture(chain, warmup=max(3, args.warmup // 3))
    blocks, walls = profile([("family", replay)], nrep=max(10, args.reps // 3), tag="family")
    rows = blocks[0][1]
    nrep = max(10, args.reps // 3)
    total, agg = tabulate(rows, nrep, per_layer_ref=None)
    scaled = total * KDA_LAYERS / n
    print(
        f"  {n} layers = {total/1000:.3f} ms;  scaled to {KDA_LAYERS} layers = "
        f"{scaled/1000:.3f} ms   target {REF_STEP_MS:.3f} ms   "
        f"({100*(scaled/1000/REF_STEP_MS-1):+.1f}%)"
    )
    print(
        "  Compare against the `layer` section's x34.  If this one is closer to\n"
        "  the served number, the difference is cache residency, not the operators."
    )
    contention_warning(agg, REF)


def sec_slots(args):
    print("\n=== slots: the knob that does move KDA is concurrency, not length ===")
    print("  4.19 MiB of fp32 ssm state per slot per layer.  The served TP1 recipe")
    print("  runs 16 slots = 2.34 GB across 34 layers, and it competes for L2.")
    print("  Slot 0 is always the one read, so only the pool size varies: this")
    print("  measures cache pressure, not a different amount of work.\n")
    print(f"  {'slots':>8} {'ssm MiB':>10} {'us/layer':>10} {'x34 (ms)':>10}")
    for s in args.slot_sweep:
        # One capture at a time.  Holding several graphs plus several state
        # pools alive at once has produced an aicore fault (507015) on this
        # stack; the sweep does not need them simultaneously.
        lay = KDALayer(slots=s, slot=0)
        replay, g = capture(lay.body, warmup=args.warmup)
        blocks, _ = profile([(f"slots={s}", replay)], nrep=args.reps, tag=f"slots{s}")
        rows = blocks[0][1]
        agg = collections.defaultdict(list)
        for r in rows:
            agg[(r["Type"] or r["Name"], r["Input Shapes"])].append(float(r["Duration(us)"]))
        tot = sum(statistics.median(v) * len(v) / args.reps for v in agg.values())
        mib = s * H * D * D * 4 / 2**20
        print(f"  {s:>8} {mib:10.1f} {tot:10.2f} {tot*KDA_LAYERS/1000:10.3f}")
        del replay, g, lay
        torch.npu.empty_cache()


def sec_refs(args):
    print("\n=== refs: is this operator slow, or is this shape slow on this die? ===")
    print("  For each suspect: a stock kernel of the same shape, a pure read of")
    print("  the same bytes, a read+write of the same bytes, and a trivial")
    print("  elementwise kernel.  Those four turn a scary ratio into a verdict.\n")

    lay = KDALayer(slots=args.slots)
    tiny = torch.randn(1, device=DEV)
    w1, w2 = lay.w_qkvbfg, lay.w_o

    def loop(f):
        return lambda k=1: [f() for _ in range(k)] and None

    # L2 on this die is ~168 MB.  Hammering ONE weight 30 times in a row leaves
    # it resident, so a warm measurement can beat its own HBM floor -- the
    # o_proj weight is 64 MiB and fits entirely.  That is not the served
    # condition: a real step streams 34 layers' worth of distinct weights past
    # L2 and every read is cold.  So each GEMM is measured twice, warm and cold,
    # cold meaning "rotate over enough distinct copies to exceed 2x L2".
    x8 = torch.zeros(1, PROJ, device=DEV, dtype=BF)
    xb = torch.zeros(2, 1, D, device=DEV, dtype=BF)
    # 4x L2, not 2x.  At 2x, the 194.5 MiB weight needs only two copies and
    # 389 MiB against a 168 MiB L2 still leaves a lot of reuse -- it reported
    # 1.39 TB/s where a 1.19 GiB read (bench_dsa_layer.py refs, far beyond any
    # reuse) reports 1.16 TB/s.  The rotation has to be big enough that the
    # answer stops moving when you make it bigger.
    n1 = max(2, int(4 * L2_MIB / (FUSED * HIDDEN * 2 / 2**20)) + 1)
    n2 = max(2, int(4 * L2_MIB / (HIDDEN * PROJ * 2 / 2**20)) + 1)
    w1s = [w1] + [torch.empty_like(w1) for _ in range(n1 - 1)]
    w2s = [w2] + [torch.empty_like(w2) for _ in range(n2 - 1)]
    c1 = itertools.cycle(w1s)
    c2 = itertools.cycle(w2s)

    cases = [
        ("qkvbfg GEMM   [1,4096]x[24896,4096]", loop(lambda: torch.matmul(lay.x, w1.t()))),
        (f"  cold, {n1} distinct weights (>4x L2)",
         loop(lambda: torch.matmul(lay.x, next(c1).t()))),
        ("  same bytes, pure read  w.sum()", loop(lambda: w1.sum())),
        ("  same bytes, read+write w.clone()", loop(lambda: w1.clone())),
        ("o_proj GEMM   [1,8192]x[4096,8192]", loop(lambda: torch.matmul(x8, w2.t()))),
        (f"  cold, {n2} distinct weights (>4x L2)",
         loop(lambda: torch.matmul(x8, next(c2).t()))),
        ("  same bytes, pure read  w.sum()", loop(lambda: w2.sum())),
        ("fg bmm        [2,1,128]x[2,8192,128]",
         loop(lambda: torch.bmm(xb, lay.w_fg_b.transpose(-1, -2)))),
        ("  same bytes, pure read  w.sum()", loop(lambda: lay.w_fg_b.sum())),
        ("conv state    read [17,3,24576]", loop(lambda: lay.conv_state.sum())),
        ("ssm state     read [17,64,128,128]", loop(lambda: lay.ssm_state.sum())),
        ("ssm state     read 1 slot [64,128,128]", loop(lambda: lay.ssm_state[0].sum())),
        ("trivial       torch.add on 1 element", loop(lambda: torch.add(tiny, 1.0))),
    ]
    blocks, _ = profile(cases, nrep=args.reps, tag="refs")

    def mib(t):
        return t.numel() * t.element_size() / 2**20

    print(f"  {'case':<40} {'MiB':>8} {'floor':>8} {'p50':>8} {'p90':>8} {'/floor':>8}")
    bytes_of = {
        0: mib(w1), 1: mib(w1), 2: mib(w1), 3: 2 * mib(w1),
        4: mib(w2), 5: mib(w2), 6: mib(w2),
        7: mib(lay.w_fg_b), 8: mib(lay.w_fg_b),
        9: mib(lay.conv_state), 10: mib(lay.ssm_state),
        11: mib(lay.ssm_state[0]), 12: 0.0,
    }
    for i, (lab, rows) in enumerate(blocks):
        d = sorted(float(r["Duration(us)"]) for r in rows)
        if not d:
            continue
        p50 = statistics.median(d)
        p90 = d[min(len(d) - 1, int(len(d) * 0.9))]
        m = bytes_of[i]
        floor = m * 2**20 / (BW_GBPS * 1e9) * 1e6 if m else 0.0
        fs = f"{floor:8.2f}" if floor else "       -"
        rr = f"{p50/floor:7.2f}x" if floor else "        "
        print(f"  {lab:<40} {m:8.2f} {fs} {p50:8.2f} {p90:8.2f} {rr}")
    print(
        "\n  Read this as: a GEMM at ~1.0x the COLD floor of its own weight is\n"
        "  finished.  There is no kernel work left to win there -- only fewer bytes\n"
        "  (a smaller dtype) or fewer calls would move it, and both are model\n"
        "  changes.  Compare the warm and cold rows before believing any ratio\n"
        f"  below 1.0: L2 here is ~{L2_MIB:.0f} MiB and it will happily hold a whole\n"
        "  weight if you keep asking for the same one."
    )


def sec_eager(args):
    print("\n=== eager: why the graph is not optional ===")
    lay = KDALayer(slots=args.slots)
    for _ in range(args.warmup):
        lay.body()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.reps):
        lay.body()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    enq = (t1 - t0) * 1e6 / args.reps
    wall = (t2 - t0) * 1e6 / args.reps
    replay, _ = capture(lay.body, warmup=args.warmup)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    replay(args.reps)
    torch.npu.synchronize()
    g_wall = (time.perf_counter() - t0) * 1e6 / args.reps
    print(f"  eager: host enqueue {enq:8.1f} us/layer, wall {wall:8.1f} us/layer")
    print(f"  graph: wall            {g_wall:8.1f} us/layer")
    print(
        "  In eager the host is ahead of or behind the device by more than the\n"
        "  device cost itself, so an eager wall clock measures the host.  The\n"
        "  served runtime captures decode into a graph; so does this script."
    )


SECTIONS = {
    "layer": sec_layer,
    "sweep": sec_sweep,
    "slots": sec_slots,
    "family": sec_family,
    "refs": sec_refs,
    "eager": sec_eager,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sections", default="layer,sweep,slots,refs")
    ap.add_argument("--seq-lens", default="1024,4096,32768,131072")
    ap.add_argument("--slots", type=int, default=17)
    ap.add_argument("--slot-sweep", default="2,17,65,129")
    ap.add_argument("--family-layers", type=int, default=KDA_LAYERS,
                    help="how many distinct layers the `family` section builds")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    args.seq_lens = [int(v) for v in args.seq_lens.split(",")]
    args.slot_sweep = [int(v) for v in args.slot_sweep.split(",")]
    want = list(SECTIONS) if args.sections == "all" else args.sections.split(",")
    bad = [s for s in want if s not in SECTIONS]
    if bad:
        raise SystemExit(f"unknown section(s) {bad}; pick from {list(SECTIONS)}")

    print(f"device        : {torch.npu.get_device_name(0)}")
    print(f"visible dies  : {os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '(all)')}")
    print(f"torch {torch.__version__}  torch_npu {torch_npu.__version__}")
    print(
        f"target        : {REF_LAYER_US} us/layer x {KDA_LAYERS} layers "
        f"= {REF_STEP_MS} ms/step (config I, bs=1, graph on, one A3 die)"
    )
    print("mode          : NPU graph capture + replay; numbers are profiler device time")
    for s in want:
        SECTIONS[s](args)
    print(f"\nprofiler output under {PROF_ROOT} (delete when done)")


if __name__ == "__main__":
    main()
