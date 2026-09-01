#!/usr/bin/env python
"""The batched extend compress-write must land the same bytes as the loop it replaces.

The loop walked the batch in Python, sliced the query rows per request, and called the
pool once per request. Every tensor in it had a data-dependent length, so a graph
capture would have baked one forward's values in permanently -- the last thing on this
path, besides the registry work, keeping prefill out of the graph.

This does not check indices in isolation. It models the index cache and the tail ring
as plain tensors and compares their **contents** after both forms have written, so a
wrong write location shows up as a differing row rather than passing unnoticed.

What it cannot check, and what still needs a machine: that the operators the real
version calls (`npu_scatter_nd_update_`, `compress_pool_bf16`) behave as the model
assumes, and that the flat `index_select` really keeps the page-table gather off the
AI CPU.

    $ROOT/.venv-ref/bin/python check_compress_write_plan.py
"""

import ast
import pathlib
import random

import torch

MODULE = pathlib.Path(
    "${GLM53_ROOT}/sglang-dllm/python/sglang/srt/"
    "hardware_backend/npu/attention/kpool_indexer_npu.py"
)
KPOOL, PAGE, SLOTS_PER_PAGE, DIM = 4, 64, 64, 8


def shipped_plan():
    """`_compress_write_plan` out of the module, without importing it (needs torch_npu)."""
    src = MODULE.read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "_compress_write_plan":
            ns = {"torch": torch}
            exec(ast.get_source_segment(src, node), ns)  # noqa: S102
            return ns["_compress_write_plan"]
    raise SystemExit("_compress_write_plan not found; was it renamed?")


def pooled(key_rows, score_rows):
    """Stand-in for compress_pool_bf16: any deterministic reduction over the pool."""
    return (key_rows * score_rows).sum(dim=1)


def loop_form(key, score, q_lens, seq_lens, req_idx, block_tables, cache, tail_k, tail_s):
    """What the code did before: one pass per request."""
    offset = 0
    for i, q_len in enumerate(q_lens):
        if q_len == 0:
            continue
        seq_len = seq_lens[i]
        first_pos = seq_len - q_len
        key_chunk, score_chunk = key[offset : offset + q_len], score[offset : offset + q_len]
        n_pools = q_len // KPOOL
        n_drain = n_pools * KPOOL
        if n_pools > 0:
            pool_ids = first_pos // KPOOL + torch.arange(n_pools, dtype=torch.int64)
            col = (pool_ids // SLOTS_PER_PAGE) * KPOOL
            page = block_tables[i].index_select(0, col)
            locs = page * PAGE + pool_ids % SLOTS_PER_PAGE
            cache[locs] = pooled(
                key_chunk[:n_drain].view(n_pools, KPOOL, DIM),
                score_chunk[:n_drain].view(n_pools, KPOOL, DIM),
            )
        n_remain = q_len - n_drain
        if n_remain > 0:
            slots = (
                torch.arange(n_remain, dtype=torch.long) + first_pos + n_drain
            ) % tail_k.shape[1]
            tail_k[req_idx[i], slots] = key_chunk[n_drain:]
            tail_s[req_idx[i], slots] = score_chunk[n_drain:]
        offset += q_len


def batched_form(plan, key, score, q_t, s_t, req_idx, block_tables, cache, tail_k,
                 tail_s, n_rows, scratch_loc, scratch_row):
    """What the code does now, using the shipped plan verbatim."""
    rows, valid, pool_ids, req_of_pool, tail_rows, tail_valid, tail_slots = plan(
        q_t, s_t, n_rows, KPOOL
    )
    block_k = block_tables.shape[1]
    col = ((pool_ids // SLOTS_PER_PAGE) * KPOOL).clamp(0, block_k - 1)
    page = block_tables.reshape(-1).index_select(0, req_of_pool * block_k + col)
    locs = page * PAGE + pool_ids % SLOTS_PER_PAGE
    cache[torch.where(valid, locs, torch.full_like(locs, scratch_loc))] = pooled(
        key[rows], score[rows]
    )

    width = tail_k.shape[1]
    dest = torch.where(
        tail_valid.reshape(-1),
        req_idx.unsqueeze(1).expand_as(tail_rows).reshape(-1),
        torch.full((tail_rows.numel(),), scratch_row, dtype=torch.long),
    )
    flat_slots = tail_slots.reshape(-1) % width
    tail_k.view(-1, DIM)[dest * width + flat_slots] = key[tail_rows.reshape(-1)]
    tail_s.view(-1, DIM)[dest * width + flat_slots] = score[tail_rows.reshape(-1)]


def main() -> int:
    plan = shipped_plan()
    random.seed(0)
    torch.manual_seed(0)
    n_cache, n_slots, tail_w = 60000, 40, 8
    checked = 0

    for trial in range(1500):
        batch = random.randint(1, 12)
        q_lens = [random.choice([0, KPOOL, random.randint(1, 60)]) for _ in range(batch)]
        # first_pos must be kpool-aligned; upstream guarantees it and the code asserts it
        seq_lens = [q + KPOOL * random.randint(0, 20) for q in q_lens]
        q_t = torch.tensor(q_lens, dtype=torch.int64)
        s_t = torch.tensor(seq_lens, dtype=torch.int64)
        n_rows = int(q_t.sum())
        if n_rows == 0:
            continue
        key, score = torch.randn(n_rows, DIM), torch.randn(n_rows, DIM)
        req_idx = torch.randperm(n_slots - 1)[:batch] + 1  # slot 0 stays the scratch row
        pages = max((max(seq_lens) + PAGE - 1) // PAGE + 2, 4)
        # Physical pages are disjoint across live requests -- the allocator's invariant,
        # and what makes a batched scatter safe. Shared pages would let two requests
        # target one cache row, where the loop's later request simply won.
        block_tables = torch.randperm(batch * pages).view(batch, pages).to(torch.int64)

        want = (
            torch.zeros(n_cache + 1, DIM),
            torch.zeros(n_slots, tail_w, DIM),
            torch.zeros(n_slots, tail_w, DIM),
        )
        got = tuple(t.clone() for t in want)
        loop_form(key, score, q_lens, seq_lens, req_idx, block_tables, *want)
        batched_form(plan, key, score, q_t, s_t, req_idx, block_tables, *got,
                     n_rows=n_rows, scratch_loc=n_cache, scratch_row=0)

        assert torch.equal(want[0][:n_cache], got[0][:n_cache]), (
            f"index cache differs, trial {trial}: q={q_lens} seq={seq_lens}"
        )
        assert torch.equal(want[1][1:], got[1][1:]), f"tail key differs, trial {trial}"
        assert torch.equal(want[2][1:], got[2][1:]), f"tail score differs, trial {trial}"
        checked += 1

    print(f"{checked} trials: the batched form writes the same index cache and tail "
          f"ring as the per-request loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
