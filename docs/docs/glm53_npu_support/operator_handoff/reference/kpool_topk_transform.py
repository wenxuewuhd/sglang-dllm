"""OP-1 reference: fused kpool group-top-k + pool->raw expand + tail append.

This is a line-by-line restatement of the CUDA kernel
``python/sglang/kernels/jit/csrc/dsa/kpool_topk_transform.cuh``
(``kpool_topk_transform_kernel``, lines 244-305), which is the semantics the Ascend
kernel must reproduce.  The decomposed torch path in
``python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py``
(``expand_pooled_groups_to_topk`` at :379, ``append_kpool_tail_to_topk`` at :421)
agrees with it and was used as a cross-check.

See ../specs/op1_kpool_topk_transform.md for the prose spec.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

PAD = -1


def select_groups_ref(
    score_row: torch.Tensor,
    length: int,
    group_topk: int,
) -> torch.Tensor:
    """The pool-level selection for one row. Returns int64 group ids, ascending.

    ``score_row`` is already windowed: it is ``score[b, row_start : row_start+length]``.
    Group ids are therefore 0-based *relative to row_start*.

    When ``length <= group_topk`` every group is selected and the CUDA kernel skips
    the top-k entirely (`.cuh:272`), emitting groups in ascending id order.  When
    ``length > group_topk`` the kernel runs a radix top-k whose output order is
    unspecified.  We return ascending ids in both cases; ORDER IS NOT PART OF THE
    CONTRACT (see the spec).
    """
    assert score_row.numel() == length
    if length <= group_topk:
        return torch.arange(length, dtype=torch.int64)
    vals = score_row.to(torch.float32)
    idx = torch.topk(vals, group_topk, largest=True, sorted=True).indices
    return torch.sort(idx.to(torch.int64)).values


def kpool_topk_transform_ref(
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation. Returns int32 ``[B, topk + tail_cols]``."""
    assert score.dim() == 2, "score must be [B, S]"
    assert score.dtype == torch.float32, "score must be float32"
    assert lengths.dim() == 1 and lengths.shape[0] == score.shape[0]
    assert pool_size > 1, "pool_size must be > 1 (.cuh:369)"
    assert topk % pool_size == 0, "topk must be a multiple of pool_size (.cuh:381)"
    assert not (page_table is not None and topk_indices_offset is not None), (
        "page_table and topk_indices_offset are mutually exclusive (.cuh:370)"
    )
    assert page_table_row_index is None or page_table is not None, (
        "page_table_row_index requires page_table (.cuh:373)"
    )

    batch = score.shape[0]
    group_topk = topk // pool_size
    tail_cols = (pool_size - 1) if seq_lens is not None else 0
    out_cols = topk + tail_cols
    out = torch.full((batch, out_cols), PAD, dtype=torch.int32)

    for b in range(batch):
        length = int(lengths[b])
        row_start = int(row_starts[b]) if row_starts is not None else 0
        assert length >= 0
        assert row_start + length <= score.shape[1], (
            f"row {b}: row_start({row_start}) + length({length}) exceeds "
            f"score columns ({score.shape[1]})"
        )

        selected = select_groups_ref(
            score[b, row_start : row_start + length], length, group_topk
        )

        # .cuh:268-270
        history_len = min(length * pool_size, topk)
        tail_count = (int(seq_lens[b]) % pool_size) if seq_lens is not None else 0

        for col in range(out_cols):
            if col < history_len:
                group_rank = col // pool_size
                slot = col % pool_size
                raw = int(selected[group_rank]) * pool_size + slot
            elif seq_lens is not None and col < history_len + tail_count:
                raw = length * pool_size + (col - history_len)
            else:
                continue  # already PAD
            out[b, col] = _transform(
                raw, b, page_table, page_table_row_index, topk_indices_offset
            )
    return out


def _transform(
    raw: int,
    b: int,
    page_table: Optional[torch.Tensor],
    page_table_row_index: Optional[torch.Tensor],
    topk_indices_offset: Optional[torch.Tensor],
) -> int:
    """.cuh:230-242 -- transform_kpool_token."""
    if page_table is not None:
        row = int(page_table_row_index[b]) if page_table_row_index is not None else b
        return int(page_table[row, raw])
    if topk_indices_offset is not None:
        return raw + int(topk_indices_offset[b])
    return raw


# ---------------------------------------------------------------------------
# Checker helpers -- used by the tests to compare two implementations without
# demanding index-by-index equality (ties at the k-th boundary legitimately differ).
# ---------------------------------------------------------------------------


def history_len_of(length: int, pool_size: int, topk: int) -> int:
    return min(length * pool_size, topk)


def decode_row(
    row: torch.Tensor,
    length: int,
    pool_size: int,
    topk: int,
    tail_count: int,
    inverse,
) -> Tuple[list, list, list]:
    """Split one output row into (selected group ids, tail raw tokens, pad columns).

    ``inverse(value) -> raw_token`` undoes whatever index transform the case used.
    Also verifies the structural contract: each of the first ``history_len`` columns
    must be the ``slot``-th token of the group at ``col // pool_size``.
    """
    history_len = history_len_of(length, pool_size, topk)
    groups: list = []
    for rank in range(history_len // pool_size):
        raws = [inverse(int(row[rank * pool_size + s])) for s in range(pool_size)]
        gid, rem = divmod(raws[0], pool_size)
        assert rem == 0, (
            f"history group {rank} does not start on a pool boundary: raw={raws[0]}"
        )
        for s in range(pool_size):
            assert raws[s] == gid * pool_size + s, (
                f"history group {rank} is not a contiguous pool expansion: {raws}"
            )
        groups.append(gid)

    tail = [
        inverse(int(row[history_len + i])) for i in range(tail_count)
    ]
    pads = [int(v) for v in row[history_len + tail_count :]]
    return groups, tail, pads


# ---------------------------------------------------------------------------
# Independent second reference, built from the DECOMPOSED torch path in
# python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py -- fast_topk_v2 (:624)
# + expand_pooled_groups_to_topk (:379) + append_kpool_tail_to_topk (:421).
# Vectorised, and written from the sglang source rather than from the .cuh, so that
# agreeing with kpool_topk_transform_ref above is real evidence and not a tautology.
#
# Caveat carried over from sglang: the decomposed path asserts page_table_row_index is
# None (kpool_fp8_index.py:618-620). We support it here anyway, because the fused
# operator must (see the spec).
# ---------------------------------------------------------------------------


def kpool_topk_transform_decomposed_ref(
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch, cols = score.shape
    group_topk = topk // pool_size

    # --- fast_topk_v2 equivalent: per-row windowed top-k over `length` groups -------
    selected = torch.zeros((batch, group_topk), dtype=torch.int64)
    for b in range(batch):
        length = int(lengths[b])
        row_start = int(row_starts[b]) if row_starts is not None else 0
        window = score[b, row_start : row_start + length].to(torch.float32)
        if length == 0:
            continue
        k = min(group_topk, length)
        idx = torch.topk(window, k, largest=True, sorted=True).indices
        selected[b, :k] = torch.sort(idx.to(torch.int64)).values

    # --- expand_pooled_groups_to_topk (kpool_fp8_index.py:379-418) -----------------
    rank = torch.arange(group_topk, dtype=torch.int32)
    max_valid_groups = min(cols, group_topk)
    valid_counts = torch.minimum(
        lengths.to(torch.int32), torch.full_like(lengths.to(torch.int32), max_valid_groups)
    )
    group_valid = rank.unsqueeze(0) < valid_counts.unsqueeze(1)

    offsets = torch.arange(pool_size, dtype=torch.int64)
    token_ids = selected.unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(batch, topk)
    valid = group_valid.unsqueeze(-1).expand(-1, -1, pool_size).reshape(batch, topk)

    if page_table is not None:
        rows = (
            page_table_row_index.to(torch.int64)
            if page_table_row_index is not None
            else torch.arange(batch, dtype=torch.int64)
        )
        safe = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        expanded = torch.gather(page_table[rows], dim=1, index=safe).to(torch.int32)
    elif topk_indices_offset is not None:
        expanded = (
            token_ids + topk_indices_offset.to(torch.int64).unsqueeze(1)
        ).to(torch.int32)
    else:
        expanded = token_ids.to(torch.int32)
    expanded = torch.where(valid, expanded, torch.full_like(expanded, PAD))

    if seq_lens is None:
        return expanded

    # --- append_kpool_tail_to_topk (kpool_fp8_index.py:421-552) --------------------
    out_cols = topk + pool_size - 1
    out = torch.full((batch, out_cols), PAD, dtype=torch.int32)
    for b in range(batch):
        length = int(lengths[b])
        tail_start = length * pool_size
        history_len = min(tail_start, topk)
        tail_count = int(seq_lens[b]) % pool_size
        out[b, :history_len] = expanded[b, :history_len]
        for i in range(tail_count):
            raw = tail_start + i
            out[b, history_len + i] = _transform(
                raw, b, page_table, page_table_row_index, topk_indices_offset
            )
    return out
