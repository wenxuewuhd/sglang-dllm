"""OP-3 acceptance tests: npu_kv_rmsnorm_rope_cache with rope width 0.

    GLM53_OP_BACKEND=reference python -m pytest tests/test_op3_kv_norm_rope_cache.py
    GLM53_OP_BACKEND=npu       python -m pytest tests/test_op3_kv_norm_rope_cache.py

The REQUIRED contract is rope_dim == 0 (GLM-5.3-Flash has qk_rope_head_dim = 0).
The rope_dim > 0 tests exist to prove the extension did not regress the DeepSeek path;
they are skipped unless GLM53_TEST_ROPE_CONVENTION=1, because this package's rope
convention is a best-effort restatement and is NOT authoritative (see the spec).
"""

from __future__ import annotations

import os

import pytest
import torch

from reference import backend
from reference.kv_norm_rope_cache import kv_norm_rope_cache_ref, rms_norm_ref
from reference.tolerance import assert_within_floor

KV_LORA = 512  # config.json text_config.kv_lora_rank
BLOCK = 128
NUM_BLOCKS = 8
EPS = 1e-5

TEST_ROPE = os.environ.get("GLM53_TEST_ROPE_CONVENTION", "0") == "1"


def _make(tokens: int, rope_dim: int, dtype: torch.dtype, seed: int, index=None):
    g = torch.Generator()
    g.manual_seed(seed)
    kv = torch.randn((tokens, 1, 1, KV_LORA + rope_dim), generator=g).to(dtype)
    gamma = (torch.randn((KV_LORA,), generator=g) * 0.1 + 1.0).to(dtype)
    if rope_dim > 0:
        ang = torch.randn((tokens, 1, 1, rope_dim), generator=g)
        cos, sin = torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)
        k_cache = torch.zeros((NUM_BLOCKS, BLOCK, 1, rope_dim), dtype=dtype)
    else:
        cos = sin = None
        k_cache = None
    ckv_cache = torch.zeros((NUM_BLOCKS, BLOCK, 1, KV_LORA), dtype=dtype)
    if index is None:
        index = torch.arange(tokens, dtype=torch.int64) * 3 + 5
    return kv, gamma, cos, sin, index, k_cache, ckv_cache


@pytest.mark.parametrize("tokens", [1, 8, 63, 256])
def test_rope0_kv_a_and_cache(tokens):
    """rope_dim == 0: the op degenerates to RMSNorm + scatter into ckv_cache."""
    kv, gamma, cos, sin, index, k_cache, ckv = _make(tokens, 0, torch.bfloat16, seed=tokens)
    kv32, gamma32, _, _, _, _, ckv32 = _make(tokens, 0, torch.float32, seed=tokens)

    _, _, _, ref_kv_a32 = kv_norm_rope_cache_ref(
        kv32, gamma32, None, None, index, None, ckv32, EPS, "PA_BNSD", True
    )
    ckv16_ref = torch.zeros_like(ckv)
    _, _, _, ref_kv_a16 = kv_norm_rope_cache_ref(
        kv, gamma, None, None, index, None, ckv16_ref, EPS, "PA_BNSD", True
    )

    k_out, ckv_out, k_pe, kv_a = backend.kv_norm_rope_cache(
        kv, gamma, None, None, index, None, ckv, EPS, "PA_BNSD", True
    )
    assert k_out is None, "rope_dim == 0 must not require or produce a k_rope cache"
    assert k_pe is None or k_pe.numel() == 0, "rope_dim == 0 must produce no k_pe"
    assert kv_a.shape == (tokens, KV_LORA)

    assert_within_floor(kv_a, ref_kv_a32, ref_kv_a16, what=f"kv_a tokens={tokens}")

    # The cache must have been written in place at exactly the requested flat slots.
    flat = ckv_out.reshape(-1, 1, KV_LORA)
    got_rows = flat[index.to(torch.int64)].reshape(tokens, KV_LORA)
    assert_within_floor(got_rows, ref_kv_a32, ref_kv_a16, what="ckv_cache rows")

    # Untouched slots must still be zero.
    mask = torch.ones(NUM_BLOCKS * BLOCK, dtype=torch.bool)
    mask[index.to(torch.int64)] = False
    assert torch.count_nonzero(flat[mask]) == 0, "untouched cache slots were modified"


def test_rope0_matches_plain_rms_norm():
    """Independent cross-check: rope_dim == 0 is exactly RMSNorm on the lora half."""
    kv, gamma, _, _, index, _, ckv = _make(32, 0, torch.float32, seed=1)
    _, _, _, kv_a = kv_norm_rope_cache_ref(
        kv, gamma, None, None, index, None, ckv, EPS, "PA_BNSD", True
    )
    expect = rms_norm_ref(kv.reshape(32, KV_LORA), gamma, EPS)
    torch.testing.assert_close(kv_a, expect, rtol=0, atol=0)


def test_rope0_zero_width_tensors_accepted():
    """The alternative calling convention: 0-width cos/sin/k_cache instead of None.

    Both must be accepted -- the probe script passes 0-width tensors, while SGLang's
    natural call passes None (there is nothing to build cos/sin from when rope==0).
    """
    tokens = 8
    kv, gamma, _, _, index, _, ckv = _make(tokens, 0, torch.bfloat16, seed=2)
    cos = torch.zeros((tokens, 1, 1, 0), dtype=torch.bfloat16)
    sin = torch.zeros((tokens, 1, 1, 0), dtype=torch.bfloat16)
    k_cache = torch.zeros((NUM_BLOCKS, BLOCK, 1, 0), dtype=torch.bfloat16)
    _, _, k_pe, kv_a = backend.kv_norm_rope_cache(
        kv, gamma, cos, sin, index, k_cache, ckv, EPS, "PA_BNSD", True
    )
    assert kv_a.shape == (tokens, KV_LORA)
    assert k_pe is None or k_pe.numel() == 0


def test_empty_batch():
    kv, gamma, _, _, _, _, ckv = _make(0, 0, torch.bfloat16, seed=3)
    index = torch.zeros((0,), dtype=torch.int64)
    _, ckv_out, _, kv_a = backend.kv_norm_rope_cache(
        kv, gamma, None, None, index, None, ckv, EPS, "PA_BNSD", True
    )
    assert kv_a.shape == (0, KV_LORA)
    assert torch.count_nonzero(ckv_out) == 0


def test_padding_sentinel_index_is_skipped():
    """index == -1 marks a padded slot; that token must not be written anywhere.

    NOTE: this is a REQUEST, not an observation -- see the spec. It is how SGLang
    pads a captured-graph batch, and confirming it for rope==0 avoids a second round.
    """
    tokens = 8
    kv, gamma, _, _, _, _, ckv = _make(tokens, 0, torch.bfloat16, seed=4)
    index = torch.arange(tokens, dtype=torch.int64)
    index[3] = -1
    index[6] = -1
    _, ckv_out, _, _ = backend.kv_norm_rope_cache(
        kv, gamma, None, None, index, None, ckv, EPS, "PA_BNSD", True
    )
    flat = ckv_out.reshape(-1, 1, KV_LORA)
    live = index >= 0
    mask = torch.ones(NUM_BLOCKS * BLOCK, dtype=torch.bool)
    mask[index[live]] = False
    assert torch.count_nonzero(flat[mask]) == 0


@pytest.mark.skipif(not TEST_ROPE, reason="set GLM53_TEST_ROPE_CONVENTION=1; see spec")
@pytest.mark.parametrize("rope_dim", [64])
def test_rope_nonzero_regression(rope_dim):
    """DeepSeek's existing shape must keep working after the extension."""
    tokens = 16
    kv, gamma, cos, sin, index, k_cache, ckv = _make(
        tokens, rope_dim, torch.bfloat16, seed=9
    )
    kv32, gamma32, cos32, sin32, _, k32, ckv32 = _make(
        tokens, rope_dim, torch.float32, seed=9
    )
    _, _, ref_pe32, ref_a32 = kv_norm_rope_cache_ref(
        kv32, gamma32, cos32, sin32, index, k32, ckv32, EPS, "PA_BNSD", True
    )
    k16, ckv16 = torch.zeros_like(k_cache), torch.zeros_like(ckv)
    _, _, ref_pe16, ref_a16 = kv_norm_rope_cache_ref(
        kv, gamma, cos, sin, index, k16, ckv16, EPS, "PA_BNSD", True
    )
    _, _, k_pe, kv_a = backend.kv_norm_rope_cache(
        kv, gamma, cos, sin, index, k_cache, ckv, EPS, "PA_BNSD", True
    )
    assert_within_floor(kv_a, ref_a32, ref_a16, what="kv_a rope>0")
    assert_within_floor(
        k_pe.reshape(tokens, rope_dim), ref_pe32.reshape(tokens, rope_dim),
        ref_pe16.reshape(tokens, rope_dim), what="k_pe rope>0",
    )
