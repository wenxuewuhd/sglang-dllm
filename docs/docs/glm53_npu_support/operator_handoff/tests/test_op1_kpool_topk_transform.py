"""OP-1 acceptance tests: kpool fused group-top-k + expand + tail append.

Run against the torch reference (default) or the delivered NPU operator:

    GLM53_OP_BACKEND=reference python -m pytest tests/test_op1_kpool_topk_transform.py
    GLM53_OP_BACKEND=npu       python -m pytest tests/test_op1_kpool_topk_transform.py

ACCEPTANCE IS SET-BASED, NOT INDEX-BY-INDEX.  Ties at the k-th boundary legitimately
differ between implementations, and the radix top-k in the CUDA kernel emits its
selection in an unspecified order.  What is checked:

  1. structure  -- each history slot is the s-th token of the group at col//pool_size
  2. selection  -- the sorted MULTISET of selected scores equals the reference's
                   (uniquely determined by k and the score vector, ties or not)
  3. tail + pad -- compared index-by-index, exactly
"""

from __future__ import annotations

import pytest
import torch

from reference import backend
from reference.kpool_topk_transform import (
    PAD,
    decode_row,
    history_len_of,
    kpool_topk_transform_decomposed_ref,
    kpool_topk_transform_ref,
)

POOL_SIZE = 4  # index_kpool for GLM-5.3-Flash


def _rng(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _identity_inverse(_b):
    return lambda v: v


def _offset_inverse(offsets):
    return lambda b: (lambda v: v - int(offsets[b]))


def _page_table_inverse(page_table, page_table_row_index):
    def make(b):
        row = int(page_table_row_index[b]) if page_table_row_index is not None else b
        inv = {int(v): i for i, v in enumerate(page_table[row].tolist())}
        return lambda v: inv[v]

    return make


def _check(
    got: torch.Tensor,
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    inverse_factory,
    row_starts=None,
    seq_lens=None,
    expected=None,
):
    """Compare ``got`` against ``expected`` (a reference output) tie-insensitively."""
    batch, out_cols = got.shape
    assert got.dtype == torch.int32
    assert out_cols == topk + (pool_size - 1 if seq_lens is not None else 0)
    assert expected.shape == got.shape

    for b in range(batch):
        length = int(lengths[b])
        row_start = int(row_starts[b]) if row_starts is not None else 0
        tail_count = (int(seq_lens[b]) % pool_size) if seq_lens is not None else 0
        inverse = inverse_factory(b)

        g_groups, g_tail, g_pad = decode_row(
            got[b], length, pool_size, topk, tail_count, inverse
        )
        e_groups, e_tail, e_pad = decode_row(
            expected[b], length, pool_size, topk, tail_count, inverse
        )

        # (1) structure is verified inside decode_row; also: no duplicate groups,
        #     every group id in range.
        assert len(set(g_groups)) == len(g_groups), f"row {b}: duplicate groups"
        assert all(0 <= gid < length for gid in g_groups), f"row {b}: group id OOR"
        assert len(g_groups) == len(e_groups) == history_len_of(
            length, pool_size, topk
        ) // pool_size

        # (2) selection: sorted multiset of selected scores must match exactly.
        window = score[b, row_start : row_start + length].to(torch.float64)
        g_scores = sorted(float(window[gid]) for gid in g_groups)
        e_scores = sorted(float(window[gid]) for gid in e_groups)
        assert g_scores == e_scores, (
            f"row {b}: selected-score multiset differs\n"
            f"  sum(impl)={sum(g_scores):.9g} sum(ref)={sum(e_scores):.9g}"
        )

        # (3) tail and padding are exact.
        assert g_tail == e_tail, f"row {b}: tail differs {g_tail} vs {e_tail}"
        assert g_pad == e_pad, f"row {b}: padding differs"
        assert all(v == PAD for v in g_pad), f"row {b}: padding must be {PAD}"


def _run_case(
    lengths_list,
    group_topk,
    *,
    pool_size=POOL_SIZE,
    seed=0,
    use_offsets=False,
    use_page_table=False,
    use_page_table_row_index=False,
    use_row_starts=False,
    use_seq_lens=True,
    tie_score=False,
    cols=None,
):
    g = _rng(seed)
    batch = len(lengths_list)
    topk = group_topk * pool_size
    lengths = torch.tensor(lengths_list, dtype=torch.int32)
    row_starts = None
    if use_row_starts:
        row_starts = torch.tensor(
            [3 * i for i in range(batch)], dtype=torch.int32
        )
    max_needed = max(
        (int(lengths[b]) + (int(row_starts[b]) if row_starts is not None else 0))
        for b in range(batch)
    ) if batch else 0
    n_cols = cols if cols is not None else max(max_needed, 1)

    if tie_score:
        # Everything equal -> every boundary is a tie.
        score = torch.zeros((batch, n_cols), dtype=torch.float32)
    else:
        score = torch.randn((batch, n_cols), generator=g, dtype=torch.float32)

    seq_lens = None
    if use_seq_lens:
        seq_lens = torch.tensor(
            [int(lengths[b]) * pool_size + (b % pool_size) for b in range(batch)],
            dtype=torch.int32,
        )

    topk_offsets = None
    page_table = None
    page_table_row_index = None
    inverse_factory = _identity_inverse

    if use_offsets:
        topk_offsets = torch.tensor(
            [1000 * (b + 1) for b in range(batch)], dtype=torch.int32
        )
        inverse_factory = _offset_inverse(topk_offsets)
    elif use_page_table:
        # An injective (permutation) page table so the test can invert the transform.
        max_seq = int(max([int(s) for s in seq_lens])) if seq_lens is not None else topk
        pt_cols = max(max_seq, topk) + pool_size
        n_rows = batch + 3 if use_page_table_row_index else batch
        page_table = torch.stack(
            [
                torch.randperm(pt_cols, generator=g).to(torch.int32) + r * 100_000
                for r in range(n_rows)
            ]
        )
        if use_page_table_row_index:
            page_table_row_index = torch.tensor(
                [(b * 2 + 1) % n_rows for b in range(batch)], dtype=torch.int32
            )
        inverse_factory = _page_table_inverse(page_table, page_table_row_index)

    kwargs = dict(
        score=score,
        lengths=lengths,
        pool_size=pool_size,
        topk=topk,
        page_table=page_table,
        topk_indices_offset=topk_offsets,
        row_starts=row_starts,
        seq_lens=seq_lens,
        page_table_row_index=page_table_row_index,
    )

    expected = kpool_topk_transform_ref(**kwargs)
    got = backend.kpool_topk_transform(**kwargs)
    _check(
        got, score, lengths, pool_size, topk, inverse_factory,
        row_starts=row_starts, seq_lens=seq_lens, expected=expected,
    )
    return kwargs, expected


# ---------------------------------------------------------------------------
# The reference itself, cross-checked against the decomposed sglang path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opts",
    [
        dict(),
        dict(use_offsets=True, use_row_starts=True),
        dict(use_page_table=True),
        dict(use_page_table=True, use_page_table_row_index=True),
        dict(use_seq_lens=False),
    ],
    ids=["plain", "ragged-offsets", "paged", "paged-shared-table", "no-tail"],
)
def test_reference_agrees_with_decomposed_path(opts):
    kwargs, expected = _run_case([0, 1, 7, 8, 9, 20], group_topk=8, seed=11, **opts)
    other = kpool_topk_transform_decomposed_ref(**kwargs)
    # Both references select the same groups in the same (ascending) order, so here we
    # CAN demand exact equality -- and we do, because any difference would be a bug in
    # our restatement of the semantics rather than an implementation tie.
    assert torch.equal(expected, other), (
        "fused-kernel restatement disagrees with the decomposed sglang path"
    )


# ---------------------------------------------------------------------------
# Shape / transform coverage.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opts",
    [
        dict(),
        dict(use_offsets=True),
        dict(use_offsets=True, use_row_starts=True),
        dict(use_page_table=True),
        dict(use_page_table=True, use_page_table_row_index=True),
        dict(use_seq_lens=False),
    ],
    ids=["plain", "ragged", "ragged-rowstarts", "paged", "paged-shared-table", "no-tail"],
)
def test_transforms(opts):
    _run_case([5, 8, 13, 31], group_topk=8, seed=3, **opts)


@pytest.mark.parametrize("length", [0, 1, 7, 8, 9, 63, 64, 65])
def test_length_around_group_topk_boundary(length):
    """group_topk=8: below, at, and above the `length <= K` fast path (.cuh:272)."""
    _run_case([length], group_topk=8, seed=length + 1)


def test_all_tied_scores():
    """Every score equal: the k-th boundary is entirely a tie.

    A conforming implementation may return ANY size-k subset here. The selected-score
    multiset check passes for all of them, which is exactly why it is the criterion.
    """
    _run_case([40], group_topk=8, seed=5, tie_score=True)


def test_partial_tie_at_boundary():
    """Distinct high scores, then a block of equal scores straddling the k-th place."""
    pool_size, group_topk = POOL_SIZE, 8
    topk = group_topk * pool_size
    length = 20
    score = torch.zeros((1, length), dtype=torch.float32)
    score[0, :5] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0])
    score[0, 5:] = 1.0  # 15 tied candidates for the remaining 3 slots
    lengths = torch.tensor([length], dtype=torch.int32)
    seq_lens = torch.tensor([length * pool_size + 2], dtype=torch.int32)
    kwargs = dict(
        score=score, lengths=lengths, pool_size=pool_size, topk=topk,
        page_table=None, topk_indices_offset=None, row_starts=None,
        seq_lens=seq_lens, page_table_row_index=None,
    )
    expected = kpool_topk_transform_ref(**kwargs)
    got = backend.kpool_topk_transform(**kwargs)
    _check(got, score, lengths, pool_size, topk, _identity_inverse,
           seq_lens=seq_lens, expected=expected)


def test_empty_batch():
    score = torch.zeros((0, 4), dtype=torch.float32)
    lengths = torch.zeros((0,), dtype=torch.int32)
    seq_lens = torch.zeros((0,), dtype=torch.int32)
    out = backend.kpool_topk_transform(
        score=score, lengths=lengths, pool_size=POOL_SIZE, topk=32,
        page_table=None, topk_indices_offset=None, row_starts=None,
        seq_lens=seq_lens, page_table_row_index=None,
    )
    assert out.shape == (0, 32 + POOL_SIZE - 1)
    assert out.dtype == torch.int32


def test_all_lengths_zero():
    """Zero history, tail only. history_len=0, so every column is tail or padding."""
    _run_case([0, 0, 0], group_topk=8, seed=7)


def test_glm_production_shape():
    """The shape that actually runs: index_topk=2048, index_kpool=4 -> group_topk=512.

    out_cols = 2048 + 3 = 2051, deliberately NOT a multiple of 16/32.
    """
    _run_case([300, 512, 513, 900], group_topk=512, seed=17, cols=1024)


def test_glm_production_shape_paged():
    _run_case(
        [300, 512, 513, 900], group_topk=512, seed=19, cols=1024,
        use_page_table=True, use_page_table_row_index=True,
    )
