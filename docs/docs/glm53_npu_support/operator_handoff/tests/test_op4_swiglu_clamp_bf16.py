"""OP-4 acceptance tests (OPTIONAL optimisation, NOT a blocker).

bf16-output variant of ``DequantSwigluClampQuant``.

    GLM53_OP_BACKEND=reference python -m pytest tests/test_op4_swiglu_clamp_bf16.py
    GLM53_OP_BACKEND=npu       python -m pytest tests/test_op4_swiglu_clamp_bf16.py
"""

from __future__ import annotations

import pytest
import torch

from reference import backend
from reference.swiglu_clamp import sglang_prelamp_then_swiglu_ref, swiglu_clamp_ref
from reference.tolerance import assert_within_floor

LIMIT = 10.0  # DeepSeek/GLM swiglu_limit; exactly representable in bf16 (see
              # python/sglang/srt/hardware_backend/npu/moe/activation.py:107-109)


def _x(rows: int, inter: int, seed: int, dtype: torch.dtype, scale: float = 6.0):
    g = torch.Generator()
    g.manual_seed(seed)
    return (torch.randn((rows, 2 * inter), generator=g) * scale).to(dtype)


@pytest.mark.parametrize("rows,inter", [(1, 128), (17, 512), (256, 1408)])
def test_bf16_clamped_swiglu(rows, inter):
    x32 = _x(rows, inter, seed=rows + inter, dtype=torch.float32)
    x16 = x32.to(torch.bfloat16)
    ref32 = swiglu_clamp_ref(x32, LIMIT)
    ref16 = swiglu_clamp_ref(x16, LIMIT)
    got = backend.swiglu_clamp_bf16(x16, LIMIT)
    assert got.dtype == torch.bfloat16
    assert got.shape == (rows, inter)
    assert_within_floor(got, ref32, ref16, what=f"swiglu rows={rows} inter={inter}")


def test_matches_sglang_prelamp_path():
    """The vendor mode-1 formula must equal what SGLang computes today on bf16.

    Cross-checks reference/swiglu_clamp.py against the independent restatement of
    apply_swiglu_limit_ + plain swiglu.
    """
    x = _x(64, 256, seed=7, dtype=torch.float32)
    torch.testing.assert_close(
        swiglu_clamp_ref(x, LIMIT, glu_alpha=1.0, glu_bias=0.0, activate_left=True),
        sglang_prelamp_then_swiglu_ref(x, LIMIT),
        rtol=1e-6, atol=1e-6,
    )


def test_clamp_is_actually_applied():
    """Values well outside +/-limit must saturate; guards against clamp_limit ignored."""
    x = torch.full((4, 8), 50.0, dtype=torch.float32)
    got = backend.swiglu_clamp_bf16(x, LIMIT).float()
    expect = swiglu_clamp_ref(x, LIMIT).float()
    torch.testing.assert_close(got, expect, rtol=1e-2, atol=1e-2)
    # silu(10) * 10 ~= 99.9995 ; without the clamp it would be silu(50)*50 = 2500
    assert got.max().item() < 200.0


def test_asymmetric_clamp():
    """The act (silu) half has an UPPER bound only; the gate half is two-sided.

    dequant_swiglu_clamp_quant.h:626-636 -- Mins+Maxs on the gate half, Mins only on
    the act half. A symmetric implementation fails this test.
    """
    inter = 4
    x = torch.zeros((1, 2 * inter), dtype=torch.float32)
    x[0, :inter] = -50.0   # act half, far below -limit: must NOT be clamped to -limit
    x[0, inter:] = 1.0
    got = backend.swiglu_clamp_bf16(x, LIMIT).float()
    # silu(-50) ~= 0 ; silu(-10) ~= -4.5e-4. Both tiny, so compare against the reference
    # rather than a magic number -- but assert the sign structure holds.
    torch.testing.assert_close(got, swiglu_clamp_ref(x, LIMIT).float(), rtol=1e-3, atol=1e-6)
    assert (got <= 0).all()


def test_empty_rows():
    x = torch.zeros((0, 256), dtype=torch.bfloat16)
    got = backend.swiglu_clamp_bf16(x, LIMIT)
    assert got.shape == (0, 128)
