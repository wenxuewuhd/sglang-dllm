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

_SYLVESTER_CACHE: dict = {}


def _sylvester(n: int, device: torch.device) -> torch.Tensor:
    """Unscaled natural-order Sylvester ``H_n``, fp32, cached per device."""
    key = (n, str(device))
    h = _SYLVESTER_CACHE.get(key)
    if h is None:
        h = torch.ones(1, 1, device=device, dtype=torch.float32)
        while h.shape[0] < n:
            h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
        _SYLVESTER_CACHE[key] = h
    return h


def hadamard_transform_npu(x: torch.Tensor, scale=None) -> torch.Tensor:
    """Natural-order Hadamard transform over the last dimension, on any device.

    Realizes Sylvester ``H_n`` -- the same matrix the CUDA ``hadamard_transform``
    and the Triton ``_hadamard128`` realize -- so a query rotated here and a key
    rotated there still share a dot product.

    Runs in fp32 whatever the input dtype, because both of those do: the CUDA
    kernel loads into ``float x_vals[..]`` (``hadamard_jit.cuh:150``) and the
    Triton key side butterflies an fp32 accumulator. A bf16 transform rounds
    seven times where they round once, and silently moves the selection -- it
    cost 0.0006 of selection overlap at 32k, measured.

    A matmul, not the butterfly the kernels use: measured on this target an fp32
    matmul against ``H_n`` is full fp32 precision (rel 1.3e-7 against the
    butterfly, no reduced-precision mode) and 16.5x faster. The butterfly's only
    advantage would have been dodging a precision mode that does not exist here.

    ``scale`` defaults to ``n**-0.5``, which makes the transform orthonormal.
    """
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"Hadamard width must be a power of 2, got {n}"
    out = x.float() @ _sylvester(n, x.device)
    return (out * (n**-0.5 if scale is None else scale)).to(x.dtype)


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
    pool_lens: torch.Tensor, req_index: torch.Tensor, max_runs: int | None = None
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

    ``max_runs`` pads the three outputs to a fixed length so their shapes stop
    depending on the data, which is what a graph capture needs. Padding a prefix sum
    means repeating its final value, so the added runs span zero query rows.
    ``npu_lightning_indexer`` treats those as a no-op: measured on device, padding a
    128-run segmentation by 1, 8 and 128 empty runs each gives bit-identical output
    (``probe/p6_a1_padded_runs.py``). Left as ``None``, the output is exactly as long
    as there are runs and the shape is dynamic -- fine for eager, not capturable.
    """
    # Compare the two run keys directly rather than packing them into one integer.
    # The packed form needed `int(pool_lens.max())`, which is a device-to-host wait,
    # and it fed `torch.unique_consecutive`, which on Ascend has no AI Core
    # implementation and falls back to `aclnnUniqueConsecutive` on the AI CPU
    # (measured: 112 us for 8192 rows).
    n = int(pool_lens.shape[0])
    device = pool_lens.device
    zero = torch.zeros(1, dtype=torch.int64, device=device)
    if n == 0:
        width = max_runs or 0
        empty = torch.zeros(width, dtype=torch.int32, device=device)
        return empty, empty.clone(), empty.to(torch.int64)
    # n == 1 needs no special case: `changed` is empty, so the nonzero branch appends
    # nothing and the scatter branch writes nothing, both leaving starts = [0, ...].
    changed = (req_index[1:] != req_index[:-1]) | (pool_lens[1:] != pool_lens[:-1])
    if max_runs is None:
        # nonzero's output length is the run count, so this branch cannot be captured.
        starts = torch.cat([zero, changed.nonzero().flatten() + 1])
    else:
        # Scatter each change position into the slot its rank names, instead of
        # compacting with nonzero. cumsum gives the first change rank 1, so slot 0
        # is never a target and keeps its 0. Rows that are not a boundary are sent
        # to a scratch slot past the end and dropped with the slice.
        rank = changed.cumsum(0)
        pos = torch.arange(1, n, device=device, dtype=torch.int64)
        slot = torch.where(changed, rank, torch.full_like(rank, max_runs))
        # Unused slots hold n, which makes them start where the sequence ends: an
        # empty run, which is exactly what the padding has to look like.
        starts = torch.full((max_runs + 1,), n, dtype=torch.int64, device=device)
        starts[0] = 0
        starts.scatter_(0, slot, pos)
        starts = starts[:max_runs]
    ends = torch.cat(
        [starts[1:], torch.full((1,), n, dtype=torch.int64, device=device)]
    )
    # A padded slot holds n, which is one past the last row; clamp so the gather is
    # in range. The value it picks up is irrelevant -- the run spans no rows.
    gather = starts.clamp(max=max(n - 1, 0))
    return (
        ends.to(torch.int32),
        pool_lens[gather].to(torch.int32),
        req_index[gather].to(torch.int64),
    )


def max_visible_pool_runs(n_rows: int, batch: int, kpool: int) -> int:
    """A static upper bound on the run count, for `visible_pool_runs(max_runs=...)`.

    Within one request the row's key count rises by exactly one per row, so
    ``pool_lens = key_count // kpool`` changes once every ``kpool`` rows and the
    request contributes at most ``ceil(q_len / kpool) + 1`` runs. Summed over the
    batch that is at most ``ceil(n_rows / kpool) + batch``; the extra 1 is slack.
    """
    return -(-n_rows // kpool) + batch + 1


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

    # `selected_groups >= 0` against a Python scalar makes torch_npu widen the
    # int32 selection to int64 and compare there: 162 us for [8192, 512] versus
    # 17 us for the same compare against a 0-dim int32 tensor (measured).
    expanded = expand_pooled_groups_to_topk(
        selected_groups.contiguous(),
        selected_groups.ge(selected_groups.new_zeros(())),
        topk=topk,
        pool_size=pool_size,
        page_table=page_table,
        topk_offsets=topk_offsets,
    )
    if seq_lens is None:
        return expanded
    return append_kpool_tail_to_topk(
        expanded,
        seq_lens=seq_lens,
        pool_lens=group_lengths,
        pool_size=pool_size,
        page_table=page_table,
        topk_offsets=topk_offsets,
    )


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

    @staticmethod
    def _compress_write_plan(extend_seq_lens, seq_lens, n_rows: int, kpool: int):
        """Where every pooled write and every tail row goes, in fixed-size tensors.

        The loop this replaces walked the batch in Python and sliced the query rows
        per request, so every tensor it built had a data-dependent size and a graph
        capture would have baked one forward's values in. Here the pool count is
        bounded by ``n_rows // kpool + batch`` and the tail by ``kpool - 1`` per
        request, both fixed once the capture shape is.

        Returns ``(rows, valid, pool_ids, req_of_pool, tail_rows, tail_valid,
        tail_slots_logical)``:

        * ``rows`` ``[P_max, kpool]`` -- the query rows each pool reduces
        * ``valid`` ``[P_max]`` -- which pools are real; the rest must be steered to
          scratch by the caller, not skipped, because skipping is what needs a
          dynamic shape in the first place
        * ``pool_ids`` ``[P_max]`` -- the logical pooled-K id, for the page lookup
        * ``req_of_pool`` ``[P_max]`` -- which request, for the page table row
        * ``tail_rows`` / ``tail_valid`` / ``tail_slots_logical`` ``[B, kpool-1]``

        Requests contributing no rows fall out on their own: the pool ownership is
        resolved against cumulative *ends*, so an empty request leaves the running
        total unchanged and is stepped over, and its tail is entirely invalid.
        """
        batch = int(extend_seq_lens.shape[0])
        device = extend_seq_lens.device
        n_pools_per = torch.div(extend_seq_lens, kpool, rounding_mode="floor")
        pool_ends = n_pools_per.cumsum(0)
        p_max = n_rows // kpool + batch

        p = torch.arange(p_max, device=device, dtype=torch.int64)
        req_of_pool = (
            (p.unsqueeze(1) >= pool_ends.unsqueeze(0)).sum(1).clamp(max=batch - 1)
        )
        valid = p < pool_ends[-1]
        p_in_req = p - (pool_ends[req_of_pool] - n_pools_per[req_of_pool])

        row_starts = extend_seq_lens.cumsum(0) - extend_seq_lens
        first_pos = seq_lens - extend_seq_lens
        base = row_starts[req_of_pool] + p_in_req * kpool
        last_row = max(n_rows - 1, 0)
        rows = (
            base.unsqueeze(1) + torch.arange(kpool, device=device, dtype=torch.int64)
        ).clamp(max=last_row)
        pool_ids = (
            torch.div(first_pos[req_of_pool], kpool, rounding_mode="floor") + p_in_req
        )

        n_remain = extend_seq_lens - n_pools_per * kpool
        t = torch.arange(kpool - 1, device=device, dtype=torch.int64)
        tail_valid = t.unsqueeze(0) < n_remain.unsqueeze(1)
        tail_rows = (
            (row_starts + n_pools_per * kpool).unsqueeze(1) + t
        ).clamp(max=last_row)
        tail_slots_logical = (first_pos + n_pools_per * kpool).unsqueeze(1) + t
        return (
            rows,
            valid,
            pool_ids,
            req_of_pool,
            tail_rows,
            tail_valid,
            tail_slots_logical,
        )

    def _kpool_compress_write_extend_npu(
        self, key, gate_score, forward_batch, layer_id, block_tables, pool
    ) -> None:
        """Drain whole pools into the cache, and park the remainder in the tail.

        One batched scatter rather than a Python loop over requests. Invalid slots
        are steered to reserved rows instead of being dropped, because dropping them
        is exactly the dynamic shape a graph capture cannot take.

        Safe as a scatter only because **physical pages are disjoint across live
        requests** -- the allocator's invariant. Were two requests to share a page,
        two pools could target one cache row and the scatter order would be
        undefined, where the loop's later request simply won.
        """
        kpool, page_size = self.index_kpool, pool.page_size
        n_rows = int(key.shape[0])
        if n_rows == 0:
            return
        extend_seq_lens = forward_batch.extend_seq_lens.to(torch.int64)
        seq_lens = forward_batch.seq_lens.to(torch.int64)

        # The alignment this needs is guaranteed upstream: chunked_prefill_size is
        # asserted to be a multiple of page_size (64), radix prefix matches are
        # floored to page multiples, and 64 is a multiple of index_kpool. Kept as a
        # check because an unaligned start would corrupt the cache silently.
        #
        # Read from the host-side lengths, not the device ones. `bool(t.any())` on a
        # device tensor is a device-to-host wait -- the very thing this rewrite is
        # removing -- and it would throw 107027 under capture besides.
        if any(
            (int(seq) - int(q)) % kpool
            for seq, q in zip(
                forward_batch.seq_lens_cpu, forward_batch.extend_seq_lens_cpu
            )
        ):
            raise NotImplementedError(
                "index_kpool_compress extend requires kpool-aligned chunk "
                "starts. Set chunked_prefill_size % index_kpool == 0 and "
                "avoid non-aligned prefix reuse."
            )

        (
            rows,
            valid,
            pool_ids,
            req_of_pool,
            tail_rows,
            tail_valid,
            tail_slots,
        ) = self._compress_write_plan(extend_seq_lens, seq_lens, n_rows, kpool)

        pooled = compress_pool_bf16(
            key[rows], gate_score[rows], self.index_kpool_compress_ape
        )
        # The same addressing `compute_pooled_write_locs` does, but per pool against a
        # 2-D page table instead of one request's 1-D slice, and written the way the
        # decode path had to be: a flat index_select rather than block_tables[a, b].
        # Two-tensor advanced indexing has no AI Core implementation and falls back to
        # aclnnIndex on the AI CPU, which cost that path 37.5% of its device time.
        slots_per_page, block_k = pool.slots_per_page, block_tables.shape[1]
        page_col = (
            torch.div(pool_ids, slots_per_page, rounding_mode="floor") * kpool
        ).clamp(0, block_k - 1)
        page = block_tables.reshape(-1).index_select(
            0, req_of_pool * block_k + page_col
        )
        locs = page.to(torch.int64) * pool.page_size + torch.remainder(
            pool_ids, slots_per_page
        )
        pool.set_index_k_bf16(
            layer_id,
            torch.where(valid, locs, torch.full_like(locs, pool.scratch_loc)),
            pooled,
        )

        req_idx = forward_batch.req_pool_indices.to(torch.long)
        pool.set_compress_tail_batched(
            layer_id=layer_id,
            req_pool_idx=req_idx.unsqueeze(1).expand_as(tail_rows).reshape(-1),
            key_tail=key[tail_rows.reshape(-1)],
            score_tail=gate_score[tail_rows.reshape(-1)],
            slots_logical=tail_slots.reshape(-1),
            valid=tail_valid.reshape(-1),
        )

    @staticmethod
    def _extend_rows(extend_seq_lens, seq_lens, n_rows: int):
        """Per-query-row sequence length and owning request, from device tensors.

        Row ``r`` of request ``i`` (0-based within the request) sees
        ``seq_len_i - q_len_i + r + 1`` keys, which is what the old host-side
        ``arange(seq_len - q_len + 1, seq_len + 1)`` per request produced.

        Everything here has a shape fixed by ``n_rows`` and the batch width, and
        nothing is read back to the host. That is the point: the host-side version
        rebuilt these two tensors in Python every forward and copied them over, and
        a graph capture would have baked one forward's values in permanently.

        ``ends`` is the cumulative row count, so ``(pos >= ends).sum()`` is the
        index of the first request whose rows have not run out at ``pos``. Requests
        contributing zero rows leave ``ends`` unchanged and are therefore skipped,
        which matches the ``q_len == 0: continue`` they used to get. The comparison
        is against ``ends`` rather than starts for exactly that reason -- starts
        would land on the empty request instead of the next one.
        """
        ends = extend_seq_lens.cumsum(0)
        pos = torch.arange(n_rows, device=ends.device, dtype=ends.dtype)
        # [n_rows, batch] of bools: cheap at these sizes (8192 x 128 worst case), and
        # unlike searchsorted it uses only ge and sum, which are ordinary AI Core ops.
        req_index = (pos.unsqueeze(1) >= ends.unsqueeze(0)).sum(1)
        starts = ends - extend_seq_lens
        row_in_req = pos - starts[req_index]
        rows = seq_lens[req_index] - extend_seq_lens[req_index] + row_in_req + 1
        return rows.to(torch.int32), req_index.to(torch.int32)

    def _kpool_extend_rows_npu(self, forward_batch, n_rows: int):
        """Per-query-row sequence length and owning request, for the segmentation.

        ``n_rows`` comes from the caller's query tensor rather than from
        ``extend_seq_lens_cpu``: that field is a Python list, and its length is
        the batch, not the row count.
        """
        return self._extend_rows(
            forward_batch.extend_seq_lens.to(torch.int64),
            forward_batch.seq_lens.to(torch.int64),
            n_rows,
        )

    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        layer_id: int,
        layer_scatter_modes=None,
        dynamic_scale: torch.Tensor = None,
        return_indices: bool = True,
    ) -> torch.Tensor | None:
        # Positions 6 and 7 exist to match the *NPU* indexer call convention:
        # forward_dsa_prepare_npu passes (layer_scatter_modes, dynamic_scale)
        # positionally, as DSANPUIndexerMixin.forward_npu declares them. The
        # CUDA forward has `return_indices` in slot 6 instead, so it stays a
        # keyword here; forward_mha_prepare_npu already passes it by keyword.
        # Neither extra is usable on this path: the scatter modes only matter to
        # the all-gather-after-qlora variant (not wired for kpool), and
        # dynamic_scale is the MLAPO quantized-q scale, which a bf16 index cache
        # has nothing to undo.
        assert (
            dynamic_scale is None
        ), "kpool indexer reads a bf16 query; a dynamic_scale would be dropped"
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
                forward_batch, x.shape[0]
            )

        if not return_indices:
            return None

        pool_lens_row = torch.div(
            seq_lens_row, self.index_kpool, rounding_mode="floor"
        ).to(torch.int32)
        if mode.is_decode_or_idle():
            # Decode has one query row per request, so every row is already its
            # own run. Skipping the segmentation skips its `int(...max())` and
            # its data-dependent output shape -- both fatal to graph capture, and
            # decode is the mode that gets captured.
            cu_seqlens_q = torch.arange(
                1, pool_lens_row.shape[0] + 1, device=x.device, dtype=torch.int32
            )
            run_pool_lens, run_req = pool_lens_row, req_index_row.long()
        else:
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
