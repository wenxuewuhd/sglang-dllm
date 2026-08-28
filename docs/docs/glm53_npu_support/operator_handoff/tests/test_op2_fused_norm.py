"""OP-2 acceptance tests: the Compressor fused-norm block, RMSNorm and LayerNorm modes.

    GLM53_OP_BACKEND=reference python -m pytest tests/test_op2_fused_norm.py
    GLM53_OP_BACKEND=npu       python -m pytest tests/test_op2_fused_norm.py

The NPU backend targets the standalone test hook requested in
../specs/op2_compressor_layernorm.md (``custom::npu_fused_norm``), which must be the
SAME code path the Compressor uses internally.  Without that hook the norm block cannot
be unit tested at all and acceptance falls back to the end-to-end golden.
"""

from __future__ import annotations

import pytest
import torch

from reference import backend
from reference.fused_norm import (
    NORM_MODE_LAYER,
    NORM_MODE_RMS,
    fused_norm_ref,
    layer_norm_via_torch,
)
from reference.tolerance import assert_within_floor

# index_head_dim for GLM-5.3-Flash (config.json, text_config.index_head_dim).
GLM_HEAD_DIM = 128
EPS = 1e-6


def _inputs(rows: int, dim: int, seed: int, dtype: torch.dtype):
    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.randn((rows, dim), generator=g, dtype=torch.float32)
    w = torch.randn((dim,), generator=g, dtype=torch.float32) * 0.1 + 1.0
    b = torch.randn((dim,), generator=g, dtype=torch.float32) * 0.1
    return x.to(dtype), w, b


@pytest.mark.parametrize("rows", [1, 7, 64, 257])
@pytest.mark.parametrize("dim", [GLM_HEAD_DIM, 64, 512])
def test_layer_norm_mode(rows, dim):
    x32, w, b = _inputs(rows, dim, seed=rows * 31 + dim, dtype=torch.float32)
    x16 = x32.to(torch.bfloat16)

    ref32 = fused_norm_ref(x32, w, b, EPS, NORM_MODE_LAYER)
    ref16 = fused_norm_ref(x16, w, b, EPS, NORM_MODE_LAYER)

    got = backend.fused_norm(x16, w, b, EPS, NORM_MODE_LAYER)
    assert got.dtype == torch.bfloat16
    assert_within_floor(got, ref32, ref16, what=f"layernorm rows={rows} dim={dim}")


@pytest.mark.parametrize("rows", [1, 64])
def test_layer_norm_matches_f_layer_norm(rows):
    """Cross-check the reference against torch's own F.layer_norm (biased variance)."""
    x32, w, b = _inputs(rows, GLM_HEAD_DIM, seed=5, dtype=torch.float32)
    ours = fused_norm_ref(x32, w, b, EPS, NORM_MODE_LAYER)
    theirs = layer_norm_via_torch(x32, w, b, EPS)
    torch.testing.assert_close(ours, theirs, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_rms_norm_mode_is_unchanged(rows):
    """norm_mode=0 must reproduce today's behaviour bit-for-bit on the NPU side.

    On the reference backend this only checks the formula. The real regression gate is
    stated in the spec: with norm_mode=0 and norm_bias absent, the Compressor's cmp_kv
    and state_cache must be BYTE-IDENTICAL to the currently shipped op.
    """
    x32, w, _ = _inputs(rows, GLM_HEAD_DIM, seed=rows, dtype=torch.float32)
    x16 = x32.to(torch.bfloat16)
    ref32 = fused_norm_ref(x32, w, None, EPS, NORM_MODE_RMS)
    ref16 = fused_norm_ref(x16, w, None, EPS, NORM_MODE_RMS)
    got = backend.fused_norm(x16, w, None, EPS, NORM_MODE_RMS)
    assert_within_floor(got, ref32, ref16, what=f"rmsnorm rows={rows}")


def test_layer_norm_differs_from_rms_norm():
    """Guard against a delivery that silently ignores norm_mode.

    With a non-zero row mean the two norms MUST disagree; if they agree, the mode
    switch was not wired up.
    """
    x32, w, b = _inputs(64, GLM_HEAD_DIM, seed=42, dtype=torch.float32)
    x32 = x32 + 3.0  # push the row mean far from zero
    ln = backend.fused_norm(x32, w, b, EPS, NORM_MODE_LAYER)
    rn = fused_norm_ref(x32, w, None, EPS, NORM_MODE_RMS)
    assert not torch.allclose(ln.float(), rn.float(), rtol=1e-2, atol=1e-2)


def test_constant_row_is_finite():
    """A constant row has zero variance: eps must keep the LayerNorm finite."""
    x = torch.full((4, GLM_HEAD_DIM), 2.5, dtype=torch.float32)
    w = torch.ones(GLM_HEAD_DIM)
    b = torch.zeros(GLM_HEAD_DIM)
    got = backend.fused_norm(x, w, b, EPS, NORM_MODE_LAYER)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, torch.zeros_like(got), rtol=0, atol=1e-5)


def test_empty_rows():
    x = torch.zeros((0, GLM_HEAD_DIM), dtype=torch.bfloat16)
    w = torch.ones(GLM_HEAD_DIM)
    b = torch.zeros(GLM_HEAD_DIM)
    got = backend.fused_norm(x, w, b, EPS, NORM_MODE_LAYER)
    assert got.shape == (0, GLM_HEAD_DIM)
