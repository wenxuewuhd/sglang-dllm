"""Batch-aware host wrapper for the fused split-qkv + RMSNorm + RoPE NPU kernel.

The upstream ``sgl_kernel_npu`` wrapper pins the launch grid to
``(num_vectorcore // n_cols, n_cols)`` with ``n_cols = kv_hidden // head_dim``
(4 for LLaDA2-mini). Each program then loops ``batch // n_rows`` rows serially,
so at large batch (RL rollout) the per-row scalar/address overhead dominates:
measured 8.0 ms/fwd at 126 blocks, ~4.3x the pure-copy floor, with
``aic_scalar_ratio`` ~0.48.

Collapsing to a single column program (``n_cols = 1``, ``n_rows = num_vectorcore``)
processes each row's full Q/KV width in one pass, cutting the serial iteration
count 4x. Measured 8.0 -> 5.0 ms/fwd (~37%) at 126 blocks, bit-identical Q/V and
K within bf16 rounding (both layouts equidistant from the fp32 reference).

The collapse only pays off once the row loop is long, so it is gated on token
count: below ``wide_grid_min_tokens`` the original per-head grid wins (fewer idle
cores at small batch). This reuses the installed ``@triton.jit`` kernel verbatim
-- only the host-side grid math differs -- so it stays in lockstep with upstream
kernel fixes.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope_pos_cache_half_npu import (
        split_qkv_rmsnorm_rope_half_pos_cache_kernel,
    )
    from sgl_kernel_npu.utils.triton_utils import get_device_properties

    _KERNEL_AVAILABLE = True
except ImportError:
    _KERNEL_AVAILABLE = False


# Below this token count the original per-head grid (n_cols = kv_hidden/head_dim)
# is at least as fast and keeps more cores busy; above it the single-column grid
# wins. 2048 tokens = 64 blocks, safely inside the measured win region (126 blocks
# showed +37%, 64 blocks was within bench noise).
_WIDE_GRID_MIN_TOKENS = 2048


def split_qkv_rmsnorm_rope_pos_cache_half(
    input_tensor: torch.Tensor,  # [B, q_hidden + 2*kv_hidden]
    positions: torch.Tensor,  # [B]
    cos_sin_cache: torch.Tensor,  # [max_seq, rope_dim], packed [cos_half | sin_half]
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: Optional[float] = None,
    q_weight: Optional[torch.Tensor] = None,
    k_weight: Optional[torch.Tensor] = None,
    q_bias: Optional[torch.Tensor] = None,
    k_bias: Optional[torch.Tensor] = None,
    rope_dim: Optional[int] = None,
    cast_norm_to_bf16: bool = True,
    wide_grid_min_tokens: int = _WIDE_GRID_MIN_TOKENS,
):
    """Drop-in for the upstream wrapper with a batch-gated launch grid.

    Semantics and output shapes are identical to
    ``sgl_kernel_npu...split_qkv_rmsnorm_rope_pos_cache_half_npu``; only the grid
    changes, and only above ``wide_grid_min_tokens``.
    """
    _, num_vectorcore = get_device_properties()
    assert input_tensor.dim() == 2
    B, total_hidden = input_tensor.shape

    if rope_dim is None:
        rope_dim = head_dim
    assert rope_dim % 2 == 0 and rope_dim <= head_dim

    expected_total = q_hidden_size + 2 * kv_hidden_size
    assert total_hidden == expected_total
    assert q_hidden_size % kv_hidden_size == 0

    pos = positions
    assert pos.numel() == B, f"positions must be [B], got numel={pos.numel()} B={B}"
    if pos.dtype not in (torch.int32, torch.int64):
        pos = pos.to(torch.int32)
    pos = pos.contiguous()

    cache = cos_sin_cache.contiguous()
    max_seq = cache.shape[0]
    assert max_seq >= 1, "cos_sin_cache must be non-empty"
    stride0 = cache.stride(0)

    head_block = triton.next_power_of_2(head_dim)
    assert head_block == head_dim, "this kernel assumes head_dim is power-of-2"

    # Wide grid: one column program per row (full Q/KV width), n_rows cores. The
    # per-head grid keeps the original head_dim-wide columns.
    if B >= wide_grid_min_tokens:
        kv_block_size = kv_hidden_size
        q_block_size = q_hidden_size
        n_cols = 1
        n_rows = num_vectorcore
    else:
        kv_block_size = head_block
        q_block_size = (q_hidden_size // kv_hidden_size) * head_dim
        n_cols = kv_hidden_size // kv_block_size
        n_rows = (num_vectorcore + n_cols - 1) // n_cols

    q_block_n = q_block_size // head_dim
    k_block_n = kv_block_size // head_dim

    q_out = torch.empty(
        (B, q_hidden_size), device=input_tensor.device, dtype=input_tensor.dtype
    )
    k_out = torch.empty(
        (B, kv_hidden_size), device=input_tensor.device, dtype=input_tensor.dtype
    )
    v_out = torch.empty(
        (B, kv_hidden_size), device=input_tensor.device, dtype=input_tensor.dtype
    )

    bias = q_bias is not None
    norms = eps is not None

    if norms:
        if q_weight is None or q_weight.numel() < head_dim:
            raise ValueError(
                f"RMSNorm needs q_weight with >= head_dim={head_dim} elements, "
                f"got {q_weight.numel() if q_weight is not None else 0}."
            )
        if k_weight is None or k_weight.numel() < head_dim:
            raise ValueError(
                f"RMSNorm needs k_weight with >= head_dim={head_dim} elements, "
                f"got {k_weight.numel() if k_weight is not None else 0}."
            )
    if bias:
        if q_bias is None or q_bias.numel() < head_dim:
            raise ValueError(
                f"bias needs q_bias with >= head_dim={head_dim} elements, "
                f"got {q_bias.numel() if q_bias is not None else 0}."
            )
        if k_bias is None or k_bias.numel() < head_dim:
            raise ValueError(
                f"bias needs k_bias with >= head_dim={head_dim} elements, "
                f"got {k_bias.numel() if k_bias is not None else 0}."
            )

    split_qkv_rmsnorm_rope_half_pos_cache_kernel[(n_rows, n_cols, 1)](
        input_tensor,
        pos,
        cache,
        q_out,
        k_out,
        v_out,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        B,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        total_hidden_size=expected_total,
        eps=eps if eps is not None else 0.0,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        q_block_n=q_block_n,
        k_block_n=k_block_n,
        bias=bias,
        norms=norms,
        head_dim=head_dim,
        rope_dim=rope_dim,
        half_rope_dim=rope_dim // 2,
        cos_sin_stride0=stride0,
        cast_norm_to_bf16=cast_norm_to_bf16,
        max_seq=max_seq,
    )

    return q_out, k_out, v_out


if not _KERNEL_AVAILABLE:
    split_qkv_rmsnorm_rope_pos_cache_half = None  # noqa: F811
