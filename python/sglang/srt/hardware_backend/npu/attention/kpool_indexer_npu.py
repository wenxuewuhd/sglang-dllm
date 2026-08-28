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


class KPoolNPUIndexerMixin:
    """``forward_npu`` for :class:`IndexerKPool`.

    Deliberately does not go through ``BaseIndexerMetadata``. On Ascend
    ``get_indexer_metadata()`` returns ``None`` -- ``AscendAttnBackend`` does not
    define it -- so none of the kpool metadata the CUDA forward reads exists, and
    the backend that does build it cannot be selected or even constructed here.
    The non-kpool DSA indexer already solved this the same way
    (``dsa/dsa_npu_indexer.py``): read ``forward_metadata`` directly.

    Two differences from the CUDA forward, both consequences of bf16 storage:
    the query is not ``act_quant``-ed, so the head gate carries no ``q_scale``;
    and the selection comes back from the operator instead of being computed from
    logits, so the transform picks up at the expand.
    """

    def _kpool_head_gate_npu(self, x: torch.Tensor) -> torch.Tensor:
        """The per-head gate, in fp32.

        The CUDA path folds ``q_scale`` in here to undo ``act_quant``; with a bf16
        query there is nothing to undo. Kept in fp32 because the operator accepts
        fp32 and a bf16 gate moves a handful of near-tie pools for nothing --
        ``weights_proj`` is an fp32 parameter to begin with.
        """
        weights, _ = self.weights_proj(x.float())
        return (weights * self.n_heads**-0.5 * self.softmax_scale).contiguous()

    def _kpool_compress_write_extend_npu(
        self, key, gate_score, forward_batch, layer_id, block_tables, pool
    ) -> None:
        """Drain whole pools into the cache, and park the remainder in the tail."""
        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            compute_pooled_write_locs,
        )

        kpool, page_size = self.index_kpool, pool.page_size
        offset = 0
        for i in range(forward_batch.batch_size):
            q_len = int(forward_batch.extend_seq_lens_cpu[i])
            if q_len == 0:
                continue
            seq_len = int(forward_batch.seq_lens_cpu[i])
            first_pos = seq_len - q_len
            if first_pos % kpool != 0:
                raise NotImplementedError(
                    "index_kpool_compress extend requires kpool-aligned chunk "
                    "starts. Set chunked_prefill_size % index_kpool == 0 and "
                    "avoid non-aligned prefix reuse."
                )
            key_chunk = key[offset : offset + q_len]
            score_chunk = gate_score[offset : offset + q_len]
            n_pools = q_len // kpool
            n_drain = n_pools * kpool
            if n_pools > 0:
                num_token_pages = (seq_len + page_size - 1) // page_size
                pool_ids = first_pos // kpool + torch.arange(
                    n_pools, dtype=torch.int64, device=key.device
                )
                write_locs = compute_pooled_write_locs(
                    block_tables[i, :num_token_pages].contiguous(), pool_ids, kpool
                )
                pool.set_index_k_bf16(
                    layer_id,
                    write_locs,
                    compress_pool_bf16(
                        key_chunk[:n_drain].view(n_pools, kpool, self.head_dim),
                        score_chunk[:n_drain].view(n_pools, kpool, self.head_dim),
                        self.index_kpool_compress_ape,
                    ),
                )
            pool.set_compress_tail_for_request(
                layer_id=layer_id,
                req_pool_idx=forward_batch.req_pool_indices[i].to(torch.long),
                key_tail=key_chunk[n_drain:],
                score_tail=score_chunk[n_drain:],
                n_remain=q_len - n_drain,
                dst_logical_start=first_pos + n_drain,
            )
            offset += q_len

    def _kpool_extend_rows_npu(self, forward_batch, device):
        """Per-query-row sequence length and owning request, for the segmentation."""
        seq_lens, req_index = [], []
        for i in range(forward_batch.batch_size):
            q_len = int(forward_batch.extend_seq_lens_cpu[i])
            if q_len == 0:
                continue
            seq_len = int(forward_batch.seq_lens_cpu[i])
            seq_lens.append(torch.arange(seq_len - q_len + 1, seq_len + 1))
            req_index.append(torch.full((q_len,), i))
        return (
            torch.cat(seq_lens).to(device=device, dtype=torch.int32),
            torch.cat(req_index).to(device=device, dtype=torch.int32),
        )

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        layer_id: int,
        return_indices: bool = True,
    ) -> torch.Tensor | None:
        import torch.nn.functional as F

        from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
            build_pooled_page_table_64,
        )
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            get_token_to_kv_pool,
        )

        mode = forward_batch.forward_mode
        out_cols = self.index_topk + self.index_kpool - 1
        if mode.is_idle() or len(forward_batch.seq_lens_cpu) == 0:
            return torch.full(
                (x.shape[0], out_cols), -1, dtype=torch.int32, device=x.device
            )
        if not (mode.is_decode_or_idle() or mode.is_extend()):
            raise NotImplementedError(
                f"The Ascend kpool indexer supports decode and extend, got {mode}."
            )

        pool = get_token_to_kv_pool()
        block_tables = get_attn_backend().forward_metadata.block_tables

        query, key, gate_score = self._get_q_k_bf16(
            q_lora, x, positions, enable_dual_stream=False, forward_batch=forward_batch
        )
        if gate_score is None:
            gate_score = F.linear(x, self.index_kpool_compress_gate)

        # Write the cache before scoring: a query sees every pool that closed at
        # or before its own position, its own included.
        if mode.is_decode_or_idle():
            batch = key.shape[0]
            pool.kpool_decode_update_index_cache(
                layer_id=layer_id,
                key=key,
                slot_score=gate_score,
                ape=self.index_kpool_compress_ape,
                block_tables=block_tables,
                req_pool_indices=forward_batch.req_pool_indices[:batch],
                positions=positions[:batch],
                seq_lens=forward_batch.seq_lens[:batch],
                out_cache_loc=forward_batch.out_cache_loc[:batch],
            )
            seq_lens_row = forward_batch.seq_lens[:batch].to(torch.int32)
            req_index_row = torch.arange(batch, device=x.device, dtype=torch.int32)
        else:
            self._kpool_compress_write_extend_npu(
                key, gate_score, forward_batch, layer_id, block_tables, pool
            )
            seq_lens_row, req_index_row = self._kpool_extend_rows_npu(
                forward_batch, x.device
            )

        if not return_indices:
            return None

        pool_lens_row = torch.div(
            seq_lens_row, self.index_kpool, rounding_mode="floor"
        ).to(torch.int32)
        cu_seqlens_q, run_pool_lens, run_req = visible_pool_runs(
            pool_lens_row, req_index_row
        )
        pooled_page_table = build_pooled_page_table_64(
            block_tables, self.index_kpool
        ).contiguous()

        selected = select_pools(
            query=query.contiguous(),
            index_k_cache=pool.get_index_k_with_scale_buffer(layer_id),
            weights=self._kpool_head_gate_npu(x),
            cu_seqlens_q=cu_seqlens_q,
            pool_lens=run_pool_lens,
            block_table=pooled_page_table[run_req].contiguous(),
            group_topk=self.index_topk // self.index_kpool,
        )

        # No page table and no offsets: that yields logical token positions, which
        # is exactly what npu_sparse_flash_attention consumes downstream.
        return topk_from_pooled_selection(
            selected,
            group_lengths=pool_lens_row,
            pool_size=self.index_kpool,
            topk=self.index_topk,
            seq_lens=seq_lens_row,
        )
