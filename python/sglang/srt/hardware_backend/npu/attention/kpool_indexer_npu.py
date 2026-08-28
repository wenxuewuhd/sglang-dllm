"""Ascend path for GLM-5.3-Flash's kpool DSA indexer.

kpool stores its compressed index keys in fp8, which Atlas A3 cannot express at
all -- ``bishengir-compile`` will not lower the e4m3 conversion and the torch side
faults on device. The four Triton kernels that write the cache fail there, and
only there.

This module takes the other route: keep the compressed key in **bf16**, which is
both the accuracy ceiling and the one dtype ``torch_npu.npu_lightning_indexer``
reads. The compression itself -- a per-dimension softmax-weighted pool of
``index_kpool`` slots, then a Hadamard-128 rotation -- is small enough to write in
torch, so the fp8 kernels are bypassed rather than ported.

The functions here mirror ``_kpool_softmax_rotate_write_cache_kernel`` in
``srt/layers/attention/dsa/kpool_fp8_index.py`` up to the point where that kernel
quantizes: same arithmetic, same order, same two bf16 roundings.
"""

from __future__ import annotations

import torch

def hadamard_transform_npu(x: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """Natural-order Hadamard transform over the last dimension, on any device.

    Stage for stage this is ``_hadamard128`` from ``kpool_fp8_index.py``, widened
    to any power-of-two width: stage ``i`` pairs elements ``stride`` apart, for
    ``stride`` doubling from 1. The matrix it realizes is Sylvester ``H_n``, the
    same one the CUDA ``hadamard_transform`` and the Triton key side realize, so
    a query rotated here and a key rotated there still share a dot product.

    It is written as a butterfly rather than a matmul on purpose: an NPU fp32
    matmul may run in a reduced-precision mode, while these are exact adds.

    ``scale`` defaults to ``n**-0.5``, which makes the transform orthonormal.
    """
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"Hadamard width must be a power of 2, got {n}"
    lead, stride = x.shape[:-1], 1
    while stride < n:
        x = x.reshape(*lead, n // (2 * stride), 2, stride)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(*lead, n)
        stride *= 2
    return x * (n**-0.5 if scale is None else scale)


def rotate_activation_npu(x: torch.Tensor) -> torch.Tensor:
    """``rotate_activation`` for Ascend.

    The shared implementation resolves to a CUDA JIT kernel that rejects
    non-CUDA tensors, so the query side of the indexer has no NPU path at all.
    """
    return hadamard_transform_npu(x.float()).to(x.dtype)


def compress_pool_bf16(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    write_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """One compressed, rotated, bf16 index key per pool.

    ``slot_k`` / ``slot_score`` are ``(n_pools, pool_size, head_dim)`` and ``ape``
    is ``(pool_size, head_dim)``. The gate is **per dimension**, not one scalar
    per slot, so the softmax runs over the pool axis independently for each of
    the 128 dimensions.

    Returns ``(n_pools, head_dim)`` bf16 -- exactly the value the fp8 kernel holds
    the instant before it quantizes.
    """
    score = slot_score.float() + ape.float()
    # Same order as the kernel: exponentiate against the row max, accumulate, then
    # divide -- not softmax-then-weight, which rounds differently.
    prob = torch.exp(score - score.amax(dim=1, keepdim=True))
    x = (slot_k.float() * prob).sum(dim=1) / prob.sum(dim=1)
    if write_mask is not None:
        x = torch.where(write_mask.view(-1, 1), x, torch.zeros_like(x))
    # The kernel rounds to bf16 twice: once before the rotation and once after.
    x = x.to(torch.bfloat16).float()
    return hadamard_transform_npu(x).to(torch.bfloat16)


def visible_pool_runs(
    pool_lens: torch.Tensor, req_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Segment query rows into runs that share one visible-pool count.

    kpool's visibility grows at ``1/pool_size`` the query rate, a slope
    ``sparse_mode=3`` (rightDownCausal, slope 1) cannot express. Runs of rows that
    see the same number of pools can each be one TND "batch" with its own
    ``actual_seq_lengths_key`` under ``sparse_mode=0``, which is exact.

    Segmenting on ``pool_lens`` alone would merge the tail of one request into the
    head of the next whenever they happen to agree, and those two need different
    page tables -- so the run key carries the request index too.

    Returns ``(cu_seqlens_q, key_lens, run_req_index)``: the TND query prefix sum,
    the visible-pool count per run, and which request each run belongs to.
    """
    pool_lens = pool_lens.to(torch.int64)
    key = req_index.to(torch.int64) * (int(pool_lens.max()) + 1) + pool_lens
    _, counts = torch.unique_consecutive(key, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    return (
        ends.to(torch.int32),
        pool_lens[starts].to(torch.int32),
        req_index[starts].to(torch.int64),
    )


def select_pools(
    query: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    pool_lens: torch.Tensor,
    block_table: torch.Tensor,
    group_topk: int,
) -> torch.Tensor:
    """Score the pooled index cache and take the top ``group_topk`` pools.

    ``torch_npu.npu_lightning_indexer`` computes ``sum_h w_h * relu(q_h . k_j)``
    and fuses the top-k, so kpool's own pooled-selection kernel is not needed. It
    returns **logical** positions -- pool ids -- with the valid ones a
    score-ordered prefix and ``-1`` padding after, which is the contract
    ``expand_pooled_groups_to_topk`` already expects.

    ``query`` is ``(T, n_heads, head_dim)`` bf16, ``index_k_cache`` is
    ``(pages, page_size, 1, head_dim)`` bf16 in PA_BSND, and ``weights`` is
    ``(T, n_heads)`` -- pass it in **fp32**: the operator accepts fp32, and a bf16
    gate moves a handful of near-tie pools for no benefit.
    """
    import torch_npu

    return torch_npu.npu_lightning_indexer(
        query=query,
        key=index_k_cache,
        weights=weights,
        actual_seq_lengths_query=cu_seqlens_q,
        actual_seq_lengths_key=pool_lens,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=group_topk,
        # Not mode 3: see visible_pool_runs for why the causal mask is carried by
        # the segmentation instead.
        sparse_mode=0,
    )[0].squeeze(1)


def topk_from_pooled_selection(
    selected_groups: torch.Tensor,
    group_lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    seq_lens: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
    out_rows: int | None = None,
) -> torch.Tensor:
    """``topk_from_pooled_history_logits`` for a selection that is already made.

    The shared version scores, selects, expands and appends the tail in one call,
    and its selection step is CUDA-only. Here the operator has already selected,
    so this picks the composition up from the expand -- both steps below are
    shared code that runs on Ascend as-is.
    """
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
        append_kpool_tail_to_topk,
        expand_pooled_groups_to_topk,
    )

    expanded = expand_pooled_groups_to_topk(
        selected_groups.contiguous(),
        selected_groups >= 0,
        topk=topk,
        pool_size=pool_size,
        page_table=page_table,
        topk_offsets=topk_offsets,
    )
    result = (
        expanded
        if seq_lens is None
        else append_kpool_tail_to_topk(
            expanded,
            seq_lens=seq_lens,
            pool_lens=group_lengths,
            pool_size=pool_size,
            page_table=page_table,
            topk_offsets=topk_offsets,
        )
    )
    if out_rows is None or out_rows == result.shape[0]:
        return result
    padded = torch.full(
        (out_rows, result.shape[1]), -1, dtype=result.dtype, device=result.device
    )
    padded[: result.shape[0]] = result
    return padded
