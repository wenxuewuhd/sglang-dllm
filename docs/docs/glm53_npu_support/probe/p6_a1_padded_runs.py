"""A1: does `npu_lightning_indexer` tolerate zero-length runs?

`visible_pool_runs` segments query rows into runs and hands the operator one
`actual_seq_lengths_query` entry per run. The number of runs depends on the data, so
its output shape is dynamic -- and a graph capture needs static shapes. That is the
last thing keeping prefill out of the graph.

The obvious fix is to pad to a fixed run count. Padding a *prefix sum* means repeating
the final value, which makes the extra runs span zero query rows. Whether the operator
treats such a run as a no-op or as something else is not documented and cannot be read
off the source: it has to be asked.

This asks it. Same inputs twice, once segmented exactly and once with the run arrays
padded out with empty runs, and compares the outputs.

    source $ROOT/env.sh && npy probe/p6_a1_padded_runs.py
"""

import torch
import torch_npu

N_HEADS, HEAD_DIM, PAGE, KPOOL = 32, 128, 64, 4
DEV = "npu"
TOPK = 512


def run_indexer(query, key, weights, cu_seqlens_q, pool_lens, block_table):
    return torch_npu.npu_lightning_indexer(
        query=query,
        key=key,
        weights=weights,
        actual_seq_lengths_query=cu_seqlens_q,
        actual_seq_lengths_key=pool_lens,
        block_table=block_table,
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=TOPK,
        sparse_mode=0,
    )[0].squeeze(1)


def main() -> int:
    torch.manual_seed(0)
    n_rows, n_pools, n_pages = 512, 600, 32

    query = torch.randn(n_rows, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    weights = torch.rand(n_rows, N_HEADS, dtype=torch.float32, device=DEV)
    key = torch.randn(n_pages, PAGE, 1, HEAD_DIM, dtype=torch.bfloat16, device=DEV)

    # One run every KPOOL rows, which is what the real segmentation produces: pool
    # visibility grows at 1/kpool the query rate.
    n_runs = n_rows // KPOOL
    cu = torch.arange(1, n_runs + 1, device=DEV, dtype=torch.int32) * KPOOL
    lens = torch.clamp(
        torch.arange(1, n_runs + 1, device=DEV, dtype=torch.int32), max=n_pools
    )
    block_table = torch.arange(n_pages, device=DEV, dtype=torch.int32).repeat(
        n_runs, 1
    )

    exact = run_indexer(query, key, weights, cu, lens, block_table)
    print(f"exact:  {n_runs} runs -> out {tuple(exact.shape)} {exact.dtype}")

    # 1 / 8 / n_runs were the original three. The fix to `max_visible_pool_runs`
    # (ceil-of-sum was not sum-of-ceils) adds one `batch` more padded runs than
    # before, so the values a real extend batch now produces are +batch: 8 at the
    # measured serving shape, 128 at max_running_requests. Asking the operator
    # directly is the only reproducible way to ask -- through the server, eight
    # concurrent requests group into prefill batches nondeterministically, and an
    # A/B of the two bounds came back with max|dlp| up to 2.7 that reproduced at
    # 1.9 when the SAME build was run against itself (measured 2026-08-30).
    for extra in (1, 8, 16, 128, n_runs):
        # Padding a prefix sum means repeating the last value: the added runs start
        # where they end, so they span no query rows.
        cu_p = torch.cat([cu, cu[-1:].repeat(extra)])
        lens_p = torch.cat([lens, lens[-1:].repeat(extra)])
        bt_p = torch.cat([block_table, block_table[-1:].repeat(extra, 1)])
        try:
            padded = run_indexer(query, key, weights, cu_p, lens_p, bt_p)
        except Exception as exc:  # noqa: BLE001
            print(f"+{extra:<5} runs: RAISED {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if padded.shape != exact.shape:
            print(f"+{extra:<5} runs: shape changed {tuple(padded.shape)}")
            continue
        same = torch.equal(padded, exact)
        diff = (padded.float() - exact.float()).abs().max().item()
        print(f"+{extra:<5} runs: out {tuple(padded.shape)}  "
              f"bit-identical={same}  max|d|={diff:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
