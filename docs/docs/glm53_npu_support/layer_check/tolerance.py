"""The two-reference (noise floor) acceptance method.

A fixed relative-error threshold is wrong for this model.  Measured on this machine:
the KDA layer-0 golden at seq=64 has an fp32-vs-bf16 relative error of 1.06e-2, so a
1e-3 gate would reject a bit-perfect bf16 implementation.

Instead, for each test case we evaluate the *reference* twice -- once with fp32 inputs
and once with bf16 inputs -- and take the distance between those two results as that
case's noise floor.  A candidate implementation is accepted when its distance from the
fp32 reference is within ``SLACK`` times that floor.

``SLACK`` exists because a candidate may order its reductions differently from the
reference and so land on the other side of the floor.  It is deliberately small and
explicit; raise it only with a recorded reason.
"""

from __future__ import annotations

import os

import torch

#: Multiplier applied to the measured noise floor. Override with GLM53_TOL_SLACK.
SLACK = float(os.environ.get("GLM53_TOL_SLACK", "2.0"))

#: Floor under the floor. When the fp32 and bf16 references agree exactly (integer
#: outputs, or a case with no rounding), the measured floor is 0 and every candidate
#: would have to be bit-identical. This absolute term keeps such cases sane.
ABS_MIN = float(os.environ.get("GLM53_TOL_ABS_MIN", "1e-6"))


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """L2 relative error, computed in fp64 so the metric itself is not the noise."""
    a = actual.detach().to(torch.float64).reshape(-1)
    e = expected.detach().to(torch.float64).reshape(-1)
    assert a.shape == e.shape, f"shape mismatch {tuple(a.shape)} vs {tuple(e.shape)}"
    denom = torch.linalg.vector_norm(e).item()
    if denom == 0.0:
        denom = 1.0
    return torch.linalg.vector_norm(a - e).item() / denom


def noise_floor(ref_fp32: torch.Tensor, ref_bf16: torch.Tensor) -> float:
    """The acceptance budget for one case: distance between the two references."""
    return rel_err(ref_bf16, ref_fp32)


def budget(ref_fp32: torch.Tensor, ref_bf16: torch.Tensor) -> float:
    return max(noise_floor(ref_fp32, ref_bf16) * SLACK, ABS_MIN)


def assert_within_floor(
    actual: torch.Tensor,
    ref_fp32: torch.Tensor,
    ref_bf16: torch.Tensor,
    what: str = "output",
) -> float:
    """Assert ``actual`` is within the noise floor of ``ref_fp32``. Returns the error."""
    err = rel_err(actual, ref_fp32)
    floor = noise_floor(ref_fp32, ref_bf16)
    limit = max(floor * SLACK, ABS_MIN)
    assert err <= limit, (
        f"{what}: rel_err={err:.3e} exceeds budget {limit:.3e} "
        f"(measured noise floor {floor:.3e} x slack {SLACK})"
    )
    return err
