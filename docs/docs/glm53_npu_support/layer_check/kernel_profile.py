"""Kernel-level profiling for the module checks in this directory.

`timing.py` answers "how long does this layer take"; this answers "which kernels,
on which core, and how far above their own floor".  Wall clock sees none of it --
an AI CPU fallback, an int64 vector op and a well-tuned cube kernel all just look
like time.

    import profile as prof

    outdir = prof.record(one_call, outdir="/tmp/p_kda", warmup=3, active=3)
    print(prof.summarize(outdir, label="KDA prefill"))

Level1 + PipeUtilization is what produces the `aiv_*` / `aic_*` ratio columns and
the execution-core column; without the experimental config the CSV has neither.
Three active steps is about 3 MB of output, so clean up after reading.

The floor a kernel is measured against is its own traffic, not a global average:
Ascend A3 measured 1.25 TB/s read+write and 1.17 TB/s read-only, with about
13.5 us of fixed launch cost per kernel.  Anything moving less than ~16 MB is
launch-dominated and is not flagged however far above the bandwidth line it sits.
"""

from __future__ import annotations

import csv
import glob
import os
import statistics
from typing import Callable, Dict, List, NamedTuple, Optional

#: Measured on this A3 die, read+write.  See PLAN.md section 3.
BW_RW_GBPS = 1250.0
#: Fixed per-kernel launch cost; below this a kernel's duration says nothing
#: about its efficiency.
LAUNCH_US = 13.5
#: Traffic under which a kernel is launch-dominated by construction.
LAUNCH_BOUND_MB = 16.0


def record(
    fn: Callable[[], object],
    *,
    outdir: str,
    warmup: int = 3,
    active: int = 3,
) -> str:
    """Run `fn` under the Ascend profiler and return the directory written."""
    import torch
    import torch_npu

    exp = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
    )
    os.makedirs(outdir, exist_ok=True)
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=True,
        experimental_config=exp,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(outdir),
    ) as p:
        for _ in range(active):
            fn()
            torch.npu.synchronize()
            p.step()
    return outdir


def _kernel_csv(outdir: str) -> str:
    hits = glob.glob(os.path.join(outdir, "**", "kernel_details.csv"), recursive=True)
    if not hits:
        raise SystemExit(
            f"no kernel_details.csv under {outdir}; the profiler wrote "
            f"{os.listdir(outdir)}"
        )
    return sorted(hits)[-1]


class Kernel(NamedTuple):
    name: str
    core: str
    calls: int
    total_us: float
    avg_us: float
    #: PipeUtilization ratios, when the row carries them.
    ratios: Dict[str, float]


def load(outdir: str) -> List[Kernel]:
    """Aggregate `kernel_details.csv` by (kernel name, execution core)."""
    path = _kernel_csv(outdir)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    cols = {c.lower().strip(): c for c in rows[0]}

    def col(*names: str) -> Optional[str]:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_name = col("name", "op name", "kernel name")
    c_core = col("type", "task type", "accelerator core", "core type")
    c_dur = col("duration(us)", "duration (us)", "task duration(us)")
    ratio_cols = {
        c.lower(): c
        for c in rows[0]
        if "ratio" in c.lower() or "utilization" in c.lower()
    }

    acc: Dict[tuple, dict] = {}
    for r in rows:
        key = (r[c_name], (r.get(c_core) or "").strip())
        a = acc.setdefault(key, {"dur": [], "ratios": {}})
        try:
            a["dur"].append(float(r[c_dur]))
        except (TypeError, ValueError):
            continue
        for low, orig in ratio_cols.items():
            try:
                a["ratios"].setdefault(low, []).append(float(r[orig]))
            except (TypeError, ValueError):
                pass

    out = []
    for (name, core), a in acc.items():
        if not a["dur"]:
            continue
        out.append(
            Kernel(
                name=name,
                core=core,
                calls=len(a["dur"]),
                total_us=sum(a["dur"]),
                avg_us=statistics.median(a["dur"]),
                ratios={
                    k: statistics.median(v) for k, v in a["ratios"].items() if v
                },
            )
        )
    out.sort(key=lambda k: -k.total_us)
    return out


def summarize(
    outdir: str,
    *,
    label: str = "",
    steps: int = 3,
    top: int = 30,
    ratio_keys: tuple = ("aiv_vec_ratio", "aiv_mte2_ratio", "cube_utilization"),
) -> str:
    """One table per profile: where the device time went, and on which core."""
    ks = load(outdir)
    total = sum(k.total_us for k in ks) or 1.0
    lines = [
        f"=== kernels, {label or outdir} ({steps} steps, "
        f"{total/steps/1000:.3f} ms of device time per step) ===",
    ]
    have = [r for r in ratio_keys if any(r in k.ratios for k in ks)]
    head = f"{'kernel':<52} {'core':<10} {'n':>5} {'us/step':>9} {'%':>6} {'us/call':>8}"
    head += "".join(f" {r.replace('_ratio','').replace('_utilization','_util'):>14}" for r in have)
    lines += [head, "-" * len(head)]
    for k in ks[:top]:
        row = (
            f"{k.name[:52]:<52} {k.core[:10]:<10} {k.calls:>5} "
            f"{k.total_us/steps:>9.1f} {100*k.total_us/total:>5.1f}% {k.avg_us:>8.1f}"
        )
        row += "".join(f" {k.ratios.get(r, float('nan')):>14.3f}" for r in have)
        lines.append(row)
    ai_cpu = [k for k in ks if "cpu" in k.core.lower()]
    if ai_cpu:
        lines.append("")
        lines.append(
            f"AI CPU kernels: {len(ai_cpu)} distinct, "
            f"{sum(k.total_us for k in ai_cpu)/steps:.1f} us/step "
            f"({100*sum(k.total_us for k in ai_cpu)/total:.1f}% of device time)"
        )
        for k in ai_cpu:
            lines.append(f"  {k.name[:60]:<60} {k.total_us/steps:>9.1f} us/step")
    lines.append(
        f"floor reference: {BW_RW_GBPS/1000:.2f} TB/s read+write, "
        f"~{LAUNCH_US} us fixed launch; below ~{LAUNCH_BOUND_MB:.0f} MB of traffic a "
        "kernel is launch-dominated and its ratio to the bandwidth line means nothing"
    )
    return "\n".join(lines)
