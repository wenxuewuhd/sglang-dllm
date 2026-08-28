"""One timing method for every module check in this directory.

Written once so the KDA, MoE, mHC and dense-FFN numbers are comparable and so nobody
re-derives a different (and quietly wrong) protocol per module.  Depends only on torch
and the stdlib, like `harness.py`.

--------------------------------------------------------------------------------
NEVER multiply a single-layer number by the layer count
--------------------------------------------------------------------------------

A per-layer figure from this file is **not** a latency contribution.  "11 DSA layers x
5.6 ms = 62 ms/token" is wrong, and the error is not small: it drops the all-reduce
after every TP layer, the other layer types, the host-side scheduler, and every overlap
the runtime achieves between compute and communication.  A per-layer number is for
comparing an implementation against itself -- before vs after a change, one shape vs
another.  End-to-end latency comes from an end-to-end run, and from nothing else.

--------------------------------------------------------------------------------
Why the first call is reported on its own
--------------------------------------------------------------------------------

On Ascend the first call to a given kernel/shape pays for JIT compilation, tiling
selection and workspace allocation.  Measured on the DSA layer of this model: first
call 45.3 ms, steady state 5.6 ms -- **8x**.  A mean over a run that includes the first
call is a number with no meaning at all, so `measure()` reports `first_ms` separately
and never lets it into the steady-state statistics.  Steady state is reported as the
median (p50), with p10/p90 so a bimodal distribution is visible instead of averaged
away.

`first_ms` is the first call **of this `measure()` invocation**, which is a true cold
start only if that kernel and shape have not already run in this process.  Run an
accuracy check on the same shape first and the compilation is already paid for, and
`first_ms` comes out at or below `p50`.  `render()` says so explicitly when it sees
`first_ms < p50_ms` rather than letting a meaningless "0.3x" stand as a result.

--------------------------------------------------------------------------------
Host synchronisations are a first-class metric here
--------------------------------------------------------------------------------

For the bookkeeping-heavy layers in this model, wall time is not explained by data
movement.  Worked example from the DSA layer: ~95 MB of traffic for a single layer on a
single rank, which is 50-60 us at one A3 die's HBM bandwidth, against 5.6 ms measured --
about **100x above the bandwidth floor**.  The prefill breakdown pointed the same way:
the largest single item was `expand+tail`, which is pure bookkeeping, at 6.3 ms.  So the
cost function for these layers is very likely the number of times the host has to wait
for the device, not the number of bytes.

`count_syncs()` therefore counts them, with the source line of each one.  It counts by
wrapping the tensor methods that force a device-to-host wait, in a **separate pass** from
the timed one so the wrapper overhead never lands in a reported time.  The count is a
**lower bound**: it sees the operations in `SYNC_OPS` and nothing else, so a sync inside
a fused custom op is invisible to it.  `torch.npu.set_sync_debug_mode` is asked for as a
cross-check when it exists.

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------

    import timing

    def one_call(t):                       # t is a timing.Timer
        with t.phase("gate_up"):
            gate_up, _ = mlp.gate_up_proj(x)
        with t.phase("act"):
            act = activation(gate_up)
        with t.phase("down"):
            out, _ = mlp.down_proj(act)
        return out

    r = timing.measure(one_call, label="dense FFN", shape="M=16, tp=16")
    s = timing.count_syncs(one_call)
    print(timing.render([r], {r.label: s}))

`render()` always emits the exclusions block; there is no way to print a number from
this module without it.
"""

from __future__ import annotations

import statistics
import traceback
from contextlib import contextmanager
from typing import Callable, Dict, List, NamedTuple, Optional

import torch

#: Tensor methods that make the host wait for the device. Not exhaustive -- see the
#: module docstring: the reported count is a lower bound.
SYNC_OPS = (
    "item",
    "tolist",
    "numpy",
    "cpu",
    "nonzero",
    "masked_select",
    "unique",
    "bincount",
    "__int__",
    "__float__",
    "__bool__",
    "__index__",
)

#: Printed by `render()` on every report. These are the things a single-layer,
#: single-device measurement structurally cannot contain.
EXCLUSIONS = (
    "no collective of any kind -- one device, one process, no all-reduce",
    "one layer run repeatedly, so its weights stay resident and hot",
    "no KV / recurrent-state cache pressure from the other layers",
    "no scheduler, sampling or tokenizer work",
)


class Timing(NamedTuple):
    label: str
    shape: str
    first_ms: float
    p50_ms: float
    p10_ms: float
    p90_ms: float
    iters: int
    #: phase name -> median ms across iterations, in first-seen order
    phases: Dict[str, float]


class SyncSite(NamedTuple):
    op: str
    where: str
    count: int


class SyncReport(NamedTuple):
    total: int
    sites: List[SyncSite]
    note: str = ""


def _device_module(device: str):
    mod = getattr(torch, device, None)
    if mod is None or not hasattr(mod, "Event"):
        raise SystemExit(f"torch.{device} has no Event API; cannot time on it")
    return mod


class Timer:
    """Handed to the measured callable so it can mark phases.

    Phase boundaries are recorded with device events, not host clocks, so marking a
    phase does not insert a synchronisation and therefore does not change what is being
    measured.
    """

    def __init__(self, device: str):
        self._dev = _device_module(device)
        self._pairs: List[tuple] = []
        self.enabled = True

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        start = self._dev.Event(enable_timing=True)
        end = self._dev.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pairs.append((name, start, end))

    def _drain(self) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for name, start, end in self._pairs:
            out.setdefault(name, []).append(start.elapsed_time(end))
        self._pairs.clear()
        return out


def prime_device(device: str = "npu") -> None:
    """Pay the process-level runtime init before anything is timed.

    In a fresh process the very first device op also pays for ACL context creation and
    allocator setup, which has nothing to do with the shape under test. Calling this
    first is what makes `measure()`'s `first_ms` a *per-shape* compile cost rather than
    that plus a one-off tax the deployment pays exactly once at startup.
    """
    dev = _device_module(device)
    x = torch.ones(8, 8, device=device, dtype=torch.bfloat16)
    (x @ x).sum()
    dev.synchronize()


def measure(
    fn: Callable[[Timer], object],
    *,
    label: str,
    shape: str = "",
    warmup: int = 5,
    iters: int = 20,
    device: str = "npu",
) -> Timing:
    """Time `fn` once cold, then `iters` times warm, on device events.

    `fn` takes a `Timer` and does one call's worth of work. Its return value is
    discarded but kept alive until after the synchronise, so a lazily-evaluated result
    cannot escape the timed region.
    """
    dev = _device_module(device)
    timer = Timer(device)

    # The cold call: JIT compile, tiling search, workspace allocation. Reported, never
    # averaged in.
    timer.enabled = False
    first_start, first_end = dev.Event(enable_timing=True), dev.Event(enable_timing=True)
    first_start.record()
    keep = fn(timer)
    first_end.record()
    dev.synchronize()
    first_ms = first_start.elapsed_time(first_end)
    del keep

    for _ in range(warmup):
        fn(timer)
    dev.synchronize()
    timer._drain()
    timer.enabled = True

    totals: List[float] = []
    for _ in range(iters):
        start, end = dev.Event(enable_timing=True), dev.Event(enable_timing=True)
        start.record()
        keep = fn(timer)
        end.record()
        totals.append((start, end))
        del keep
    dev.synchronize()
    totals = [s.elapsed_time(e) for s, e in totals]
    phases = {k: statistics.median(v) for k, v in timer._drain().items()}

    ordered = sorted(totals)
    return Timing(
        label=label,
        shape=shape,
        first_ms=first_ms,
        p50_ms=statistics.median(ordered),
        p10_ms=ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        p90_ms=ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        iters=iters,
        phases=phases,
    )


@contextmanager
def _patched_sync_ops(hits: List[tuple]):
    """Wrap the device-to-host operations so each call records op and call site."""
    originals = {}

    def wrap(name, orig):
        def counted(self, *a, **kw):
            if self.is_cpu:  # a CPU tensor never waits on a device
                return orig(self, *a, **kw)
            hits.append((name, _caller()))
            return orig(self, *a, **kw)

        return counted

    for name in SYNC_OPS:
        orig = getattr(torch.Tensor, name, None)
        if orig is None:
            continue
        originals[name] = orig
        setattr(torch.Tensor, name, wrap(name, orig))

    # `.to("cpu")` is a sync too, but `.to()` is on every hot path, so it is wrapped
    # with a cheap device test rather than a site capture on every call.
    to_orig = torch.Tensor.to

    def counted_to(self, *a, **kw):
        if not self.is_cpu:
            target = kw.get("device")
            if target is None and a and isinstance(a[0], (str, torch.device)):
                target = a[0]
            if target is not None and torch.device(target).type == "cpu":
                hits.append(("to(cpu)", _caller()))
        return to_orig(self, *a, **kw)

    torch.Tensor.to = counted_to
    try:
        yield
    finally:
        for name, orig in originals.items():
            setattr(torch.Tensor, name, orig)
        torch.Tensor.to = to_orig


def _caller() -> str:
    """The innermost frame that is neither this file nor inside torch."""
    for frame in reversed(traceback.extract_stack()[:-2]):
        if "/torch/" in frame.filename or frame.filename.endswith("timing.py"):
            continue
        return f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno} in {frame.name}"
    return "<unknown>"


def count_syncs(
    fn: Callable[[Timer], object], *, device: str = "npu", warmup: int = 2
) -> SyncReport:
    """How many host waits one call costs, and where they are.

    Run in its own pass: the wrappers add Python overhead, and a timing measured
    through them would be measuring the wrappers.
    """
    timer = Timer(device)
    timer.enabled = False
    for _ in range(warmup):  # let any first-call bookkeeping happen uncounted
        fn(timer)
    _device_module(device).synchronize()

    hits: List[tuple] = []
    with _patched_sync_ops(hits):
        fn(timer)

    tally: Dict[tuple, int] = {}
    for op, where in hits:
        tally[(op, where)] = tally.get((op, where), 0) + 1
    sites = [
        SyncSite(op, where, n)
        for (op, where), n in sorted(tally.items(), key=lambda kv: -kv[1])
    ]
    note = (
        "lower bound: counts only the ops in timing.SYNC_OPS, so a wait inside a fused "
        "custom op is not visible here"
    )
    return SyncReport(total=len(hits), sites=sites, note=note)


def render(
    timings: List[Timing],
    syncs: Optional[Dict[str, SyncReport]] = None,
    extra_exclusions: tuple = (),
) -> str:
    """The one output format. Always carries the exclusions block."""
    syncs = syncs or {}
    lines: List[str] = []
    for t in timings:
        head = f"{t.label}" + (f"  [{t.shape}]" if t.shape else "")
        lines.append(f"\n=== {head} ===")
        lines.append(
            f"  first call {t.first_ms:8.3f} ms   "
            f"steady p50 {t.p50_ms:8.3f} ms  (p10 {t.p10_ms:.3f} / p90 {t.p90_ms:.3f}, "
            f"n={t.iters})"
        )
        if t.p50_ms > 0:
            ratio = t.first_ms / t.p50_ms
            lines.append(f"  first/steady ratio {ratio:.1f}x")
            if ratio < 1.0:
                lines.append(
                    "    ^ first call was not cold: this shape had already run in this "
                    "process, so the compile/tiling cost is not in this number. For a "
                    "real cold-start figure, measure it in a fresh process."
                )
        if t.phases:
            total = sum(t.phases.values())
            lines.append("  breakdown (median per phase):")
            for name, ms in t.phases.items():
                share = 100.0 * ms / total if total else 0.0
                lines.append(f"    {name:<24} {ms:8.3f} ms  ({share:5.1f}%)")
            lines.append(f"    {'(sum of phases)':<24} {total:8.3f} ms")
        rep = syncs.get(t.label)
        if rep is not None:
            lines.append(f"  host syncs per call: {rep.total}  ({rep.note})")
            for site in rep.sites:
                lines.append(f"    {site.count:>4}x {site.op:<14} {site.where}")

    lines.append("\n  EXCLUDED -- these are not latency numbers:")
    for note in EXCLUSIONS + tuple(extra_exclusions):
        lines.append(f"    - {note}")
    lines.append(
        "    Do NOT multiply a per-layer number by the layer count: that drops the\n"
        "    all-reduce, the other layer types, and every compute/comm overlap."
    )
    return "\n".join(lines)
