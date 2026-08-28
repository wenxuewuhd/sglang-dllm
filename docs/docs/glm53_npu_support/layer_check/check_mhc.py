#!/usr/bin/env python
"""Stage B for mHC: drive the real sglang path on NPU against a stage-A case.

    source $ROOT/env.sh
    PYTHONPATH=$REPO/python $VENV/bin/python check_mhc.py \
        --case $ROOT/goldens/mhc_attn_layer01.pt $ROOT/goldens/mhc_ffn_layer20.pt

Nothing here re-implements mHC.  The candidate is produced by
``sglang.kernels.ops.layernorm.mhc.hc_pre`` / ``hc_post`` -- the exact two
functions ``models/glm5_next.py`` calls -- which route through
``_mhc_pre_dispatch`` / ``_mhc_post_dispatch`` and land on
``torch.ops.custom.npu_hc_pre`` / ``npu_hc_post``.  The dispatch is *verified* to
have taken the NPU branch (the ops are wrapped with a call counter), because the
dispatchers fall back to pure torch silently and a silent fallback would make this
whole check vacuous.

Four blocks come out:

1. **dispatch (the gate).**  ``hc_pre`` then ``hc_post`` chained, exactly as the
   decoder layer runs them, scored by ``harness.check`` -- the two-reference
   method, no invented threshold.  ``post.out.isolated`` re-runs only the post
   half on the *reference* ``post``/``comb`` so a pre-half error cannot be
   mistaken for a post-half one.
2. **device fp32 torch.**  The same case through ``_mhc_pre_torch`` /
   ``_mhc_post_torch`` on NPU in fp32.  This is what the device's own arithmetic
   can do without the fused kernel, i.e. the floor under block 1.
3. **host fp32 torch.**  The same, on CPU.  This separates "sglang's formula
   disagrees with HF" from "the NPU kernel is imprecise"; if block 3 fails, the
   problem is not the device.
4. **Sinkhorn sweep.**  ``npu_hc_pre`` is re-run with ``hc_sinkhorn_iters`` from
   1 to ``iters`` and each result compared with the reference's own iterate at
   that count (stage A saved the whole trace).  This is the only way to see
   inside the fused kernel: the intermediate iterates are not returned, but the
   iteration count is an argument.  It answers both "does the error accumulate
   over the 20 sweeps" and "does the kernel really run the iteration count it is
   told to".

Then the gate is re-run at every deployed M (see ``DEPLOYED_SHAPES``), at M = 0,
and with an ``out_norm`` passed in -- the three shapes/paths the serving code hits
that a single prefill-shaped run does not.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Case, check, report  # noqa: E402
from reference.tolerance import rel_err  # noqa: E402

# --- deployed shapes (read off $ROOT/run/launch_glm_bf16.sh and the server_args
# line of $ROOT/run/glm_bf16.log; nothing here is assumed) -------------------
#
# tp_size 16, dp_size 1, enable_dp_attention False, enable_prefill_cp False.
# That fixes every mHC shape:
#
# * **mHC is not TP-sharded.** `hc_{attn,ffn}_{fn,base,scale}` are plain
#   `nn.Parameter`s on the decoder layer (models/glm5_next.py), not Column/Row
#   parallel, so every rank holds the full (24, hc_mult*4096) `fn`.
# * **Tokens are not sharded either.** `MHCLayerCommunicator.prepare_attn` runs
#   `mhc.attn_split` *before* `_communicate_simple_fn`, i.e. on the tensor in
#   `layer_input_mode`. With no DP attention and no prefill CP,
#   `ScatterMode.model_input_output()` is TP_ATTN_FULL and `_compute_mlp_mode`
#   returns FULL (moe_a2a_backend none), so `layer_input_mode` is TP_ATTN_FULL
#   for every layer -- the whole batch, on every rank.
#
# So the only shape that varies is M = tokens in the batch:
#
#   decode   M <= 16      (--max-running-requests 16)
#   prefill  M <= 8192    (chunked_prefill_size 8192, from the server_args log)
#   batched  M <= 16384   (max_prefill_tokens 16384)
#
# and hidden is always 4096, hc_mult always 4.
DEPLOYED_SHAPES = "1,2,4,8,16,2048,4096,8192"

DEV = "npu"


class _CallCounter:
    """Wraps a ``torch.ops.custom`` entry so we can prove the kernel really ran."""

    def __init__(self, fn):
        self.fn = fn
        self.n = 0

    def __call__(self, *args, **kwargs):
        self.n += 1
        return self.fn(*args, **kwargs)


_COUNTERS: dict = {}


def install_counters():
    """Wrap the two custom ops once per process; later calls reuse the wrappers."""
    if _COUNTERS:
        for c in _COUNTERS.values():
            c.n = 0
        return _COUNTERS

    counters = _COUNTERS
    for name in ("npu_hc_pre", "npu_hc_post"):
        real = getattr(torch.ops.custom, name)
        counters[name] = _CallCounter(real)
        try:
            setattr(torch.ops.custom, name, counters[name])
        except Exception as exc:  # pragma: no cover - namespace is not writable
            counters.clear()
            raise SystemExit(
                f"cannot instrument torch.ops.custom.{name} ({exc}); without it a "
                "silent fallback to the torch path would pass this check for the "
                "wrong reason"
            )
    return counters


def make_dev_tensors(case: Case, m: dict) -> dict:
    """Everything the candidate needs, on device, in the dtypes serving uses."""
    n = m["hc_mult"]
    s = case.inputs["hidden_streams"].shape[0]
    return {
        # bf16 is the serving dtype. The mixing weights stay fp32: `fn` is bf16 in
        # the checkpoint but sglang holds it as an fp32 Parameter, and the
        # dispatcher casts to fp32 anyway -- passing fp32 avoids a second rounding.
        "streams_bf16": case.inputs["hidden_streams"].to(DEV, torch.bfloat16).contiguous(),
        "x_bf16": case.inputs["sublayer_out"].to(DEV, torch.bfloat16).contiguous(),
        "fn": case.inputs["weight.fn"].to(DEV, torch.float32).contiguous(),
        "base": case.inputs["weight.base"].to(DEV, torch.float32).contiguous(),
        "scale": case.inputs["weight.scale"].to(DEV, torch.float32).contiguous(),
        "ref_post": case.ref_fp32["pre.post"].to(DEV, torch.float32).reshape(s, n).contiguous(),
        "ref_comb": case.ref_fp32["pre.comb"].to(DEV, torch.float32).reshape(s, n * n).contiguous(),
    }


def slice_case(case: Case, idx: torch.Tensor) -> Case:
    """A derived case holding only the given token rows.

    mHC is per-token (the only cross-element reduction, Sinkhorn, is inside one
    token's hc_mult x hc_mult matrix), so slicing rows is exact -- the reference
    for a subset is the subset of the reference.  This is how the decode shape
    gets covered without a second stage-A run: serving runs mHC with M = 1 on
    every decode step, and NPU matmul tiling is shape-dependent.
    """
    take = lambda t: t.index_select(0, idx).contiguous()
    return Case(
        name=f"{case.name}[M={idx.numel()}]",
        inputs={
            k: (take(v) if k in ("hidden_streams", "sublayer_out") else v)
            for k, v in case.inputs.items()
        },
        ref_fp32={k: take(v) for k, v in case.ref_fp32.items()},
        ref_bf16={k: take(v) for k, v in case.ref_bf16.items()},
        meta=case.meta,
    )


def build_candidate_dispatch(case: Case, m: dict, dev_tensors: dict) -> dict:
    """The real path: `hc_pre` -> `hc_post`, the two functions glm5_next.py calls."""
    from sglang.kernels.ops.layernorm.mhc import hc_post, hc_pre

    n, d = m["hc_mult"], m["hidden_size"]
    flat = dev_tensors["streams_bf16"].reshape(-1, n * d).contiguous()

    layer_input, h_res, h_post, norm_fused = hc_pre(
        x=flat,
        hc_fn=dev_tensors["fn"],
        hc_scale=dev_tensors["scale"],
        hc_base=dev_tensors["base"],
        hc_mult=n,
        rms_eps=m["rms_norm_eps"],
        hc_eps=m["hc_eps"],
        sinkhorn_iters=m["hc_sinkhorn_iters"],
        post_mult_value=m["post_mult_value"],
        hc_norm_weight=None,
        out_norm_weight=None,
        out_norm_eps=None,
    )
    assert not norm_fused, "npu_hc_pre must not claim to have folded the output norm"

    out = hc_post(
        x=dev_tensors["x_bf16"],
        residual=flat,
        h_post=h_post,
        h_res=h_res,
        hc_mult=n,
    )
    # Post half on the *reference* mixes, so the two halves are separable.
    out_iso = hc_post(
        x=dev_tensors["x_bf16"],
        residual=flat,
        h_post=dev_tensors["ref_post"],
        h_res=dev_tensors["ref_comb"],
        hc_mult=n,
    )
    s = flat.shape[0]
    return {
        "pre.post": h_post.reshape(s, n),
        "pre.comb": h_res.reshape(s, n, n),
        "pre.collapsed": layer_input,
        "post.out": out.reshape(s, n, d),
        "post.out.isolated": out_iso.reshape(s, n, d),
    }


def build_candidate_torch(case: Case, m: dict, streams: torch.Tensor, x: torch.Tensor,
                          fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
                          ref_post: torch.Tensor, ref_comb: torch.Tensor) -> dict:
    """sglang's own pure-torch mHC, run wherever the tensors already live."""
    from sglang.kernels.ops.layernorm.mhc import _mhc_post_torch, _mhc_pre_torch

    post_mix, comb_mix, layer_input = _mhc_pre_torch(
        residual=streams,
        fn=fn,
        hc_scale=scale,
        hc_base=base,
        rms_eps=m["rms_norm_eps"],
        hc_pre_eps=m["hc_eps"],
        hc_sinkhorn_eps=m["hc_eps"],
        hc_post_mult_value=m["post_mult_value"],
        sinkhorn_repeat=m["hc_sinkhorn_iters"],
    )
    out = _mhc_post_torch(x, streams, post_mix, comb_mix)
    out_iso = _mhc_post_torch(x, streams, ref_post, ref_comb)
    s, n, _ = streams.shape
    return {
        "pre.post": post_mix.reshape(s, n),
        "pre.comb": comb_mix,
        "pre.collapsed": layer_input,
        "post.out": out,
        "post.out.isolated": out_iso,
    }


def sinkhorn_sweep(case: Case, m: dict, dev_tensors: dict) -> None:
    """Two questions about the 20 Sinkhorn sweeps, both answered from outside.

    The fused kernel does not hand back its intermediate ``comb`` -- but
    ``hc_sinkhorn_iters`` is an *input*, so re-running it with k = 1..iters gives
    the same sequence.  Reference iterate ``trace[k-1]`` is the reference run with
    ``hc_sinkhorn_iters = k`` (HF does one normalisation before the ``iters - 1``
    alternating sweeps).

    1. **Accumulation.**  ``err(kernel@k, ref32@k)`` as a function of k.  A rising
       curve means the iteration is amplifying device error; a flat one means
       Sinkhorn is contracting it, which is the outcome you want to be able to
       state rather than assume.
    2. **Identification.**  The shipped call uses k = ``iters``; comparing that one
       output against *every* reference iterate says which iterate it really is.
       This is the test that catches an off-by-one or an early convergence exit --
       and it has to be done this way round, because comparing kernel@k with
       ref@k for each k is flat to within noise (all late iterates are close to
       each other), so an argmin over that curve means nothing.
    """
    trace32 = case.inputs["sinkhorn.trace.fp32"]
    trace16 = case.inputs["sinkhorn.trace.bf16"]
    iters = m["hc_sinkhorn_iters"]
    n, d = m["hc_mult"], m["hidden_size"]
    x4 = dev_tensors["streams_bf16"].reshape(1, -1, n, d).contiguous()

    def kernel_comb(k: int) -> torch.Tensor:
        _, _, comb = torch.ops.custom.npu_hc_pre(
            x4,
            dev_tensors["fn"],
            dev_tensors["scale"],
            dev_tensors["base"],
            hc_mult=n,
            hc_sinkhorn_iters=k,
            norm_eps=m["rms_norm_eps"],
            hc_eps=m["hc_eps"],
        )
        return comb.squeeze(0).to("cpu", torch.float32)

    print(f"\n=== {case.name}: sinkhorn accumulation over {iters} iterations ===")
    print("   k  kernel-vs-ref32   ref floor (bf16)  row-sum span          col-sum span")
    combs = {}
    for k in range(1, iters + 1):
        comb = combs[k] = kernel_comb(k)
        err = rel_err(comb, trace32[k - 1])
        floor = rel_err(trace16[k - 1], trace32[k - 1])
        row, col = comb.sum(-1), comb.sum(-2)
        print(
            f"  {k:>2}  {err:.6e}      {floor:.3e}         "
            f"{row.min():.6f}..{row.max():.6f}  {col.min():.6f}..{col.max():.6f}"
        )
    errs = [rel_err(combs[k], trace32[k - 1]) for k in range(1, iters + 1)]
    print(
        f"  -> error vs iteration: first {errs[0]:.3e}, last {errs[-1]:.3e}, "
        f"max {max(errs):.3e} at k={errs.index(max(errs)) + 1}; "
        f"{'no accumulation' if errs[-1] <= errs[0] * 1.5 else 'GROWING'}"
    )

    print(f"\n=== {case.name}: which iterate is the shipped k={iters} output? ===")
    shipped = combs[iters]
    dists = [rel_err(shipped, trace32[j]) for j in range(iters)]
    best = min(range(iters), key=lambda j: dists[j])
    near = sorted(range(iters), key=lambda j: dists[j])[:4]
    for j in sorted(near):
        print(f"  vs ref iterate {j + 1:>2} (0-based {j}): {dists[j]:.6e}")
    if best == iters - 1:
        margin = min(dists[j] for j in range(iters - 1)) / max(dists[best], 1e-30)
        print(
            f"  -> matches reference iterate {iters} (the last one), and the next "
            f"best iterate is {margin:.0f}x further away"
        )
    else:
        print(
            f"  !! the k={iters} output is closest to reference iterate {best + 1}, "
            f"not {iters} -- off by {best + 1 - iters}"
        )


def run_case(
    path: Path, sweep: bool, shapes: tuple[int, ...], diag_max_m: int
) -> int:
    case = Case.load(path)
    m = case.meta
    n, d = m["hc_mult"], m["hidden_size"]
    streams = case.inputs["hidden_streams"]  # (S, n, d) fp32
    x = case.inputs["sublayer_out"]  # (S, d) fp32
    fn = case.inputs["weight.fn"]
    base = case.inputs["weight.base"]
    scale = case.inputs["weight.scale"]
    s = streams.shape[0]

    head = (
        f"{case.name}: layer {m['layer']} ({m['layer_type']} / {m['mlp_type']} mlp), "
        f"stage {m['stage']}, {s} tokens, hc_mult={n}, "
        f"sinkhorn_iters={m['hc_sinkhorn_iters']}, hc_eps={m['hc_eps']}, "
        f"rms_norm_eps={m['rms_norm_eps']}"
    )

    dev_tensors = make_dev_tensors(case, m)

    counters = install_counters()
    cand = build_candidate_dispatch(case, m, dev_tensors)
    torch.npu.synchronize()
    if counters["npu_hc_pre"].n == 0 or counters["npu_hc_post"].n == 0:
        raise SystemExit(
            f"the NPU branch was not taken (npu_hc_pre {counters['npu_hc_pre'].n} "
            f"calls, npu_hc_post {counters['npu_hc_post'].n}); the dispatcher fell "
            "back to torch and this check would have been vacuous"
        )
    rc = report(
        f"{case.name}  [dispatch: hc_pre/hc_post -> npu_hc_pre/npu_hc_post]",
        check(case, cand),
        extra=head
        + f"\n  npu_hc_pre x{counters['npu_hc_pre'].n}, "
        f"npu_hc_post x{counters['npu_hc_post'].n}",
    )

    # Diagnostics: not gates, reference points for reading the block above.
    # `_mhc_post_torch` materialises an (M, n, n, hidden) fp32 intermediate --
    # 4.3 GB at M=16384 -- so at the deployed prefill sizes the diagnostics run on
    # a prefix. The gate above always runs at the full M.
    dcase = case if s <= diag_max_m else slice_case(case, torch.arange(diag_max_m))
    ds = dcase.inputs["hidden_streams"].shape[0]
    dnote = "" if ds == s else f" [first {ds} of {s} rows: the fp32 torch path needs O(M*n*n*h) memory]"
    dfn, dscale, dbase = fn, scale, base
    dev32 = build_candidate_torch(
        dcase,
        m,
        dcase.inputs["hidden_streams"].to(DEV, torch.float32),
        dcase.inputs["sublayer_out"].to(DEV, torch.float32),
        dev_tensors["fn"],
        dev_tensors["scale"],
        dev_tensors["base"],
        dcase.ref_fp32["pre.post"].to(DEV, torch.float32).reshape(ds, n, 1),
        dcase.ref_fp32["pre.comb"].to(DEV, torch.float32),
    )
    report(
        f"{dcase.name}  [diagnostic: sglang _mhc_*_torch, fp32, on NPU]",
        check(dcase, dev32),
        extra="what the device's own arithmetic does without the fused kernel" + dnote,
    )

    host32 = build_candidate_torch(
        dcase, m, dcase.inputs["hidden_streams"], dcase.inputs["sublayer_out"],
        dfn, dscale, dbase,
        dcase.ref_fp32["pre.post"].reshape(ds, n, 1),
        dcase.ref_fp32["pre.comb"],
    )
    report(
        f"{dcase.name}  [diagnostic: sglang _mhc_*_torch, fp32, on CPU]",
        check(dcase, host32),
        extra="a failure here is a formula disagreement with HF, not a device problem"
        + dnote,
    )

    if sweep:
        sinkhorn_sweep(case, m, dev_tensors)

    # Decode shapes. The gate is the same two-reference method on the same
    # reference rows; only the M the kernel is handed changes.
    skipped = [c for c in shapes if c > s]
    for count in shapes:
        if count > s:
            continue
        sub = slice_case(case, torch.arange(count))
        sub_dev = make_dev_tensors(sub, m)
        before = counters["npu_hc_pre"].n
        sub_cand = build_candidate_dispatch(sub, m, sub_dev)
        torch.npu.synchronize()
        assert counters["npu_hc_pre"].n > before, "NPU branch skipped for M=%d" % count
        # NPU bf16 matmul is not batch-shape invariant, so the same rows can come
        # out differently at a different M. Measure that, do not assert on it.
        drift = {
            k: (v.to("cpu", torch.float32) - cand[k][:count].to("cpu", torch.float32))
            .abs()
            .max()
            .item()
            for k, v in sub_cand.items()
        }
        worst = max(drift.values())
        rc |= report(
            f"{sub.name}  [dispatch, deployed shape M={count}]",
            check(sub, sub_cand),
            extra=f"first {count} row(s) of the same case; max |drift| over all 5 "
            f"tensors vs the same rows in the M={s} run: {worst:.3e}",
        )
    if skipped:
        print(
            f"\n  note: shapes {skipped} exceed this case's M={s} and were not run; "
            f"regenerate with reference_mhc.py --tokens/--repeats to cover them"
        )

    from sglang.kernels.ops.layernorm.mhc import hc_post, hc_pre

    # The serving path (`MHCState.attn_split`) hands `hc_pre` the layer's
    # input_layernorm and then applies it *itself* unless `norm_fused` comes back
    # True. The NPU branch returns before it ever reads `out_norm_weight`, so it
    # must return the same numbers and `norm_fused=False` whether or not a norm is
    # passed. If that ever stops holding, the layernorm gets applied twice (or not
    # at all) and nothing else in this file would notice.
    flat_full = dev_tensors["streams_bf16"].reshape(s, n * d).contiguous()
    li_a, hr_a, hp_a, nf_a = hc_pre(
        x=flat_full, hc_fn=dev_tensors["fn"], hc_scale=dev_tensors["scale"],
        hc_base=dev_tensors["base"], hc_mult=n, rms_eps=m["rms_norm_eps"],
        hc_eps=m["hc_eps"], sinkhorn_iters=m["hc_sinkhorn_iters"],
        post_mult_value=m["post_mult_value"],
        out_norm_weight=torch.ones(d, device=DEV, dtype=torch.bfloat16),
        out_norm_eps=m["rms_norm_eps"],
    )
    same = (
        torch.equal(li_a, cand["pre.collapsed"])
        and torch.equal(hp_a.reshape(s, n), cand["pre.post"])
        and torch.equal(hr_a.reshape(s, n, n), cand["pre.comb"])
    )
    print(
        f"\n=== {case.name}: out_norm is not folded on the NPU branch ===\n"
        f"  norm_fused={nf_a} (must be False), outputs unchanged by passing a norm: "
        f"{same}"
    )
    assert not nf_a and same, (
        "the NPU hc_pre branch reacted to out_norm_weight; MHCState applies the "
        "layernorm itself when norm_fused is False, so this would double-apply it"
    )

    # M = 0 is a real serving shape (an idle rank / an empty split). It must not
    # reach the kernel and must not raise.

    empty = dev_tensors["streams_bf16"][:0].reshape(0, n * d)
    li, hr, hp, nf = hc_pre(
        x=empty,
        hc_fn=dev_tensors["fn"],
        hc_scale=dev_tensors["scale"],
        hc_base=dev_tensors["base"],
        hc_mult=n,
        rms_eps=m["rms_norm_eps"],
        hc_eps=m["hc_eps"],
        sinkhorn_iters=m["hc_sinkhorn_iters"],
        post_mult_value=m["post_mult_value"],
    )
    out0 = hc_post(x=dev_tensors["x_bf16"][:0], residual=empty, h_post=hp, h_res=hr, hc_mult=n)
    print(
        f"\n=== {case.name}: M=0 (idle rank) ===\n  hc_pre -> "
        f"{tuple(li.shape)}/{tuple(hr.shape)}/{tuple(hp.shape)} norm_fused={nf}, "
        f"hc_post -> {tuple(out0.shape)}  [ok] no crash"
    )
    return rc



# --- timing ---------------------------------------------------------------
#
# Reported the same way for every module in this directory, so the numbers are
# comparable and can be reconciled against a whole-network latency later:
#
#   shapes   decode bs=16 (@32k context) and prefill chunk 8192 -- the deployed
#            ones. For mHC the context length is irrelevant: mHC never touches
#            the KV cache, so a decode step hands it exactly `bs` token rows and
#            a prefill chunk hands it `chunk` rows. "bs=16 @32k ragged" is
#            therefore M=16 here, and that is not an approximation.
#   warmup   discarded; the *first* call is reported separately, because on this
#            stack it is a different animal (the DSA layer measures 45.3 ms first
#            vs 5.6 ms steady).
#   stat     median of the steady-state calls, each individually synchronised.
#
# Excluded, deliberately, and NOT to be read as a layer latency:
#   * no all-reduce / all-gather -- mHC is replicated and this is a single die,
#     so nothing here pays for the TP16 collectives the real layer pays for;
#   * weights stay resident (one layer, run in a loop), so no weight-load or
#     cache-pressure cost from the other 44 layers;
#   * no KV cache, no other layer types, no cross-layer overlap.
# Multiplying these by 45 does not predict anything.


class _SyncCounter:
    """Counts the host-side stalls (D2H) a call performs.

    A fused kernel that still drags a `.item()` behind it is not fused where it
    matters: every one of these serialises the whole pipeline.
    """

    NAMES = ("item", "tolist", "cpu", "numpy", "nonzero", "__int__", "__float__", "__bool__")

    def __init__(self):
        self.counts = {n: 0 for n in self.NAMES}
        self._saved = {}

    def __enter__(self):
        for n in self.NAMES:
            real = getattr(torch.Tensor, n)
            self._saved[n] = real

            def make(n=n, real=real):
                def fn(t, *a, **k):
                    if t.is_npu:
                        self.counts[n] += 1
                    return real(t, *a, **k)

                return fn

            setattr(torch.Tensor, n, make())
        return self

    def __exit__(self, *_):
        for n, real in self._saved.items():
            setattr(torch.Tensor, n, real)
        return False

    def total(self) -> int:
        return sum(self.counts.values())

    def detail(self) -> str:
        hit = {k: v for k, v in self.counts.items() if v}
        return str(hit) if hit else "none"


def _count_aten(fn):
    """Number of aten calls one invocation makes -- a proxy for kernel launches."""
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except Exception:
        return None

    n = 0

    class _Mode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            nonlocal n
            n += 1
            return func(*args, **(kwargs or {}))

    try:
        with _Mode():
            fn()
    except Exception:
        return None
    return n


def _time(fn, warmup: int, iters: int):
    """(first-call seconds, list of steady-state seconds). Each call synchronised."""
    torch.npu.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.npu.synchronize()
    first = time.perf_counter() - t0
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        out.append(time.perf_counter() - t0)
    return first, out


def _p50(xs):
    return statistics.median(xs) * 1e3  # ms


def _load_avg() -> str:
    """1/5/15-minute load average.

    These runs share 320 cores with the other layer-check tasks. mHC's decode
    shape is launch-bound (see the aten-call counts), so host contention moves the
    numbers: record what the machine was doing rather than pretending it was idle.
    """
    try:
        a, b, c = os.getloadavg()
        return f"{a:.0f}/{b:.0f}/{c:.0f}"
    except OSError:
        return "unavailable"


def bench(case: Case, m: dict, shapes, warmup: int, iters: int, device: int) -> None:
    from sglang.kernels.ops.layernorm.mhc import (
        _mhc_post_torch,
        _mhc_pre_torch,
        hc_post,
        hc_pre,
    )

    n, d = m["hc_mult"], m["hidden_size"]
    src = case.inputs["hidden_streams"]
    src_x = case.inputs["sublayer_out"]
    fn32 = case.inputs["weight.fn"].to(DEV, torch.float32).contiguous()
    scale = case.inputs["weight.scale"].to(DEV, torch.float32).contiguous()
    base = case.inputs["weight.base"].to(DEV, torch.float32).contiguous()

    print(
        f"\n=== {case.name}: timing ===\n"
        f"  npu die {device}; host load average (1/5/15 min) at start: {_load_avg()}\n"
        f"  warmup {warmup} discarded, {iters} timed calls, each synchronised; "
        f"p50 reported.\n"
        f"  EXCLUDES all-reduce (mHC is replicated; single die), weight residency "
        f"(one layer in a loop),\n  and every other layer type. Do NOT multiply by 45."
    )
    for M in shapes:
        reps = -(-M // src.shape[0])
        streams = (
            src.repeat(reps, 1, 1)[:M].to(DEV, torch.bfloat16).contiguous()
        )
        flat = streams.reshape(M, n * d).contiguous()
        xs = src_x.repeat(reps, 1)[:M].to(DEV, torch.bfloat16).contiguous()

        pre_kwargs = dict(
            hc_fn=fn32, hc_scale=scale, hc_base=base, hc_mult=n,
            rms_eps=m["rms_norm_eps"], hc_eps=m["hc_eps"],
            sinkhorn_iters=m["hc_sinkhorn_iters"], post_mult_value=m["post_mult_value"],
        )
        _, h_res, h_post, _ = hc_pre(x=flat, **pre_kwargs)

        do_pre = lambda: hc_pre(x=flat, **pre_kwargs)
        do_post = lambda: hc_post(
            x=xs, residual=flat, h_post=h_post, h_res=h_res, hc_mult=n
        )
        f_pre, t_pre = _time(do_pre, warmup, iters)
        f_post, t_post = _time(do_post, warmup, iters)

        # Fused vs not: the same two halves through sglang's pure-torch path.
        s3 = streams
        pre_t = lambda: _mhc_pre_torch(
            residual=s3, fn=fn32, hc_scale=scale, hc_base=base,
            rms_eps=m["rms_norm_eps"], hc_pre_eps=m["hc_eps"],
            hc_sinkhorn_eps=m["hc_eps"], hc_post_mult_value=m["post_mult_value"],
            sinkhorn_repeat=m["hc_sinkhorn_iters"],
        )
        pm, cm, _ = pre_t()
        post_t = lambda: _mhc_post_torch(xs, s3, pm, cm)
        _, tt_pre = _time(pre_t, warmup, max(iters // 5, 5))
        _, tt_post = _time(post_t, warmup, max(iters // 5, 5))

        # Sinkhorn's share of the pre half: the iteration count is an argument, so
        # sweeping it prices the loop directly.
        x4 = streams.reshape(1, M, n, d).contiguous()
        per_iters = {}
        for k in (1, 5, 10, 20):
            call = lambda k=k: torch.ops.custom.npu_hc_pre(
                x4, fn32, scale, base, hc_mult=n, hc_sinkhorn_iters=k,
                norm_eps=m["rms_norm_eps"], hc_eps=m["hc_eps"],
            )
            _, ts = _time(call, warmup, iters)
            per_iters[k] = _p50(ts)

        with _SyncCounter() as sc_pre:
            do_pre()
            torch.npu.synchronize()
        with _SyncCounter() as sc_post:
            do_post()
            torch.npu.synchronize()
        a_pre, a_post = _count_aten(do_pre), _count_aten(do_post)
        at_pre, at_post = _count_aten(pre_t), _count_aten(post_t)

        p_pre, p_post = _p50(t_pre), _p50(t_post)
        tp_pre, tp_post = _p50(tt_pre), _p50(tt_post)
        slope = (per_iters[20] - per_iters[1]) / 19
        print(
            f"\n  M={M} ({'decode bs=16' if M <= 16 else 'prefill chunk'}), "
            f"hidden={d}, hc_mult={n}\n"
            f"    fused   pre  first {f_pre * 1e3:8.3f} ms   p50 {p_pre:8.3f} ms   "
            f"aten calls {a_pre}\n"
            f"    fused   post first {f_post * 1e3:8.3f} ms   p50 {p_post:8.3f} ms   "
            f"aten calls {a_post}\n"
            f"    fused   pre+post p50 {p_pre + p_post:8.3f} ms\n"
            f"    torch   pre  p50 {tp_pre:8.3f} ms   post p50 {tp_post:8.3f} ms   "
            f"pre+post {tp_pre + tp_post:8.3f} ms   aten calls {at_pre}/{at_post}\n"
            f"    speedup pre {tp_pre / p_pre:5.2f}x  post {tp_post / p_post:5.2f}x  "
            f"both {(tp_pre + tp_post) / (p_pre + p_post):5.2f}x\n"
            f"    sinkhorn iters=1 {per_iters[1]:.3f}  5 {per_iters[5]:.3f}  "
            f"10 {per_iters[10]:.3f}  20 {per_iters[20]:.3f} ms  "
            f"=> {slope * 1e3:.1f} us/iter, the 19 extra sweeps are "
            f"{(per_iters[20] - per_iters[1]) / per_iters[20] * 100:.0f}% of pre\n"
            f"    D2H syncs  pre {sc_pre.total()} {sc_pre.detail()}   "
            f"post {sc_post.total()} {sc_post.detail()}\n"
            f"    host load at end of this shape: {_load_avg()}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=Path, nargs="+", required=True)
    # Dies 10 and 11 are the ones allocated to this task; 8/9 and 0-3 belong to the
    # other two layer checks running concurrently.
    ap.add_argument("--device", type=int, default=10)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument(
        "--bench",
        action="store_true",
        help="report timings instead of checking. Run it in its own process: the "
        "first-call number is only meaningful if nothing ran before it.",
    )
    ap.add_argument("--bench-shapes", default="16,8192")
    ap.add_argument("--bench-warmup", type=int, default=20)
    ap.add_argument("--bench-iters", type=int, default=100)
    ap.add_argument(
        "--diag-max-m",
        type=int,
        default=4096,
        help="cap on M for the two fp32-torch diagnostic blocks (they need an "
        "O(M*hc_mult^2*hidden) fp32 intermediate). The gate is never capped.",
    )
    ap.add_argument(
        "--shapes",
        default=DEPLOYED_SHAPES,
        help="extra token counts to re-run the gate at. The default is the "
        "deployed set: decode M<=16, prefill chunk 8192 (see DEPLOYED_SHAPES)",
    )
    args = ap.parse_args()

    import torch_npu  # noqa: F401
    import custom_ops  # noqa: F401  -- registers torch.ops.custom.npu_hc_*

    torch.npu.set_device(args.device)

    if args.bench:
        shapes = tuple(int(v) for v in args.bench_shapes.split(",") if v.strip())
        for path in args.case:
            case = Case.load(path)
            bench(
                case,
                case.meta,
                shapes,
                args.bench_warmup,
                args.bench_iters,
                args.device,
            )
        return 0

    rc = 0
    for path in args.case:
        shapes = tuple(int(v) for v in args.shapes.split(",") if v.strip())
        rc |= run_case(
            path,
            sweep=not args.no_sweep,
            shapes=shapes,
            diag_max_m=args.diag_max_m,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
