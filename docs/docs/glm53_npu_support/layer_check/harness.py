"""Shared contract for the per-layer NPU-vs-CPU checks.

Every check in this directory has the same two stages and the same verdict rule, so
that a failure says *which layer, which tensor, by how much* rather than "the model
is wrong".

    stage A  (CPU, $ROOT/.venv-ref)   dump_reference.py
             Runs the real HF model up to the layer under test and saves a *case*:
             the layer's inputs, plus its outputs evaluated twice -- once from fp32
             inputs and once from bf16 inputs.

    stage B  (NPU, $ROOT/.venv-glm53) check_<module>.py
             Feeds the same inputs to the real sglang module on device and compares.

The verdict is the two-reference method from ``operator_handoff/ACCEPTANCE.md``: the
distance between the fp32 and bf16 references *is* the budget. A fixed threshold is
wrong for this model -- the KDA layer-0 golden at seq=64 has an fp32-vs-bf16 relative
error of 1.06e-2, so a 1e-3 gate would reject a bit-perfect implementation.

Imports only torch and the stdlib, because it has to load in both venvs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "operator_handoff"))
from reference.tolerance import (  # noqa: E402
    ABS_MIN,
    SLACK,
    noise_floor,
    rel_err,
)

#: Bumped when the saved layout changes, so a stale golden fails loudly.
CASE_FORMAT = 1


@dataclass
class Case:
    """One layer's inputs and its two reference evaluations."""

    name: str
    inputs: Dict[str, torch.Tensor]
    ref_fp32: Dict[str, torch.Tensor]
    ref_bf16: Dict[str, torch.Tensor]
    meta: Dict[str, object] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": CASE_FORMAT,
                "name": self.name,
                "inputs": self.inputs,
                "ref_fp32": self.ref_fp32,
                "ref_bf16": self.ref_bf16,
                "meta": self.meta,
            },
            path,
        )

    @staticmethod
    def load(path: Path, device: str = "cpu") -> "Case":
        blob = torch.load(path, map_location=device, weights_only=False)
        got = blob.get("format")
        if got != CASE_FORMAT:
            raise SystemExit(
                f"{path}: case format {got}, this harness speaks {CASE_FORMAT}. "
                f"Regenerate with dump_reference.py."
            )
        return Case(
            blob["name"], blob["inputs"], blob["ref_fp32"], blob["ref_bf16"], blob["meta"]
        )


@dataclass
class Result:
    tensor: str
    err: float
    floor: float
    budget: float
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.err <= self.budget

    def __str__(self) -> str:
        mark = "ok  " if self.ok else "FAIL"
        ratio = self.err / self.budget if self.budget else float("inf")
        return (
            f"  [{mark}] {self.tensor:<28} err={self.err:.3e}  "
            f"floor={self.floor:.3e}  budget={self.budget:.3e}  "
            f"({ratio:.2f}x budget){(' ' + self.note) if self.note else ''}"
        )


def check(case: Case, candidate: Dict[str, torch.Tensor]) -> list[Result]:
    """Score every tensor the candidate produced against the case's two references.

    A tensor the case has a reference for but the candidate did not produce is a
    failure, not a skip -- silently checking less than you think is how a wrong
    implementation passes.
    """
    results = []
    for name, ref32 in case.ref_fp32.items():
        got = candidate.get(name)
        if got is None:
            results.append(
                Result(name, float("inf"), 0.0, 0.0, "candidate produced no such tensor")
            )
            continue
        ref16 = case.ref_bf16[name]
        got = got.detach().to("cpu", torch.float32)
        floor = noise_floor(ref32.float(), ref16.float())
        results.append(
            Result(name, rel_err(got, ref32.float()), floor, max(floor * SLACK, ABS_MIN))
        )
    return results


def report(title: str, results: list[Result], extra: str = "") -> int:
    """Print one uniform block. Returns a process exit code."""
    print(f"\n=== {title} ===")
    if extra:
        print(f"  {extra}")
    for r in results:
        print(r)
    bad = [r for r in results if not r.ok]
    print(
        f"  -> {len(results) - len(bad)}/{len(results)} within budget "
        f"(slack {SLACK}, abs floor {ABS_MIN})"
    )
    return 1 if bad else 0


# --- whole-model tracing --------------------------------------------------


def save_trace(path: Path, hidden_fp32: list, hidden_bf16: list, meta: dict) -> None:
    """Per-layer hidden states, for locating the *first* layer that diverges.

    The per-module checks above say whether a module is right in isolation. This says
    where a whole-model run first leaves the reference, which is the question you have
    when the network's output is wrong and you do not yet know why.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CASE_FORMAT,
            "hidden_fp32": [h.to(torch.float32) for h in hidden_fp32],
            "hidden_bf16": [h.to(torch.float32) for h in hidden_bf16],
            "meta": meta,
        },
        path,
    )


def first_divergence(trace_path: Path, hidden_npu: list) -> Optional[int]:
    """The first layer whose NPU hidden state leaves its own noise floor.

    Prints every layer so a slow drift is visible as well as a cliff -- a run that
    degrades gradually and one that breaks at a single layer look identical if you
    only report the first failure.
    """
    blob = torch.load(trace_path, map_location="cpu", weights_only=False)
    ref32, ref16 = blob["hidden_fp32"], blob["hidden_bf16"]
    if len(hidden_npu) != len(ref32):
        raise SystemExit(
            f"trace has {len(ref32)} layers, run produced {len(hidden_npu)}"
        )
    first = None
    print(f"\n=== layer-by-layer divergence ({blob['meta']}) ===")
    for i, (a, b, c) in enumerate(zip(hidden_npu, ref32, ref16)):
        floor = noise_floor(b, c)
        limit = max(floor * SLACK, ABS_MIN)
        err = rel_err(a.detach().to("cpu", torch.float32), b)
        flag = "" if err <= limit else "  <-- FIRST DIVERGENCE" if first is None else "  <--"
        if err > limit and first is None:
            first = i
        print(f"  layer {i:>2}: err={err:.3e}  floor={floor:.3e}{flag}")
    if first is None:
        print("  -> every layer within its noise floor")
    return first
