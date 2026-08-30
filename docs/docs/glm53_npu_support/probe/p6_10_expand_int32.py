"""P6.10: expand_pooled_groups_to_topk materialises 134 MB of int64 to cast it away.

    offsets   = arange(pool_size, dtype=int64)
    token_ids = group_ids.to(int64).unsqueeze(-1) * pool_size + offsets   # [rows, groups, 4]
    ...
    output = token_ids.to(torch.int32)

At the deployment shape -- 8192 rows, 512 groups, pool 4 -- that intermediate is
8192*512*4*8 = 134 MB, and on the NPU path (page_table and topk_offsets both None) it
exists only to be narrowed to int32 immediately.

int32 holds the values with room to spare: a pool id is at most context_length /
pool_size, so a token id is at most context_length, ~32768 here and ~2^20 even on a
1M-context model, against int32's 2.1e9. Ascend also emulates int64 vector arithmetic,
so the wide type costs twice: traffic and instruction count.

The page_table branch has to stay int64 -- torch.gather requires an int64 index -- so
this compares the branch we actually take.

    source $ROOT/env.sh
    ASCEND_RT_VISIBLE_DEVICES=14 PYTHONPATH=$REPO/python:$PYTHONPATH npy probe/p6_10_expand_int32.py
"""

import time

import torch


def expand_int64(group_ids, group_valid, topk, pool_size):
    """Today's arithmetic, verbatim (the page_table / topk_offsets branches removed)."""
    device = group_ids.device
    offsets = torch.arange(pool_size, device=device, dtype=torch.int64)
    token_ids = group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(group_ids.shape[0], topk)
    valid = (
        group_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(group_ids.shape[0], topk)
    )
    output = token_ids.to(torch.int32)
    return torch.where(valid, output, torch.full_like(output, -1))


def expand_int32(group_ids, group_valid, topk, pool_size):
    device = group_ids.device
    offsets = torch.arange(pool_size, device=device, dtype=torch.int32)
    token_ids = group_ids.to(torch.int32).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(group_ids.shape[0], topk)
    valid = (
        group_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(group_ids.shape[0], topk)
    )
    return torch.where(valid, token_ids, torch.full_like(token_ids, -1))


def timed(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> int:
    torch.manual_seed(0)
    dev, pool_size, topk = "npu", 4, 2048
    groups = topk // pool_size
    for rows in (8192, 4096, 1024, 16):
        # -1 marks an unselected group; the arithmetic still runs on it, so the
        # narrow type has to survive negatives too.
        group_ids = torch.randint(-1, 8192, (rows, groups), dtype=torch.int32, device=dev)
        group_valid = group_ids.ge(group_ids.new_zeros(()))
        a = expand_int64(group_ids, group_valid, topk, pool_size)
        b = expand_int32(group_ids, group_valid, topk, pool_size)
        same = torch.equal(a, b)
        t_a = timed(lambda: expand_int64(group_ids, group_valid, topk, pool_size))
        t_b = timed(lambda: expand_int32(group_ids, group_valid, topk, pool_size))
        mb = rows * groups * pool_size * 8 / 1024**2
        print(f"rows={rows:>5}: identical={same}  int64={t_a:7.3f} ms  int32={t_b:7.3f} ms  "
              f"speedup={t_a / t_b:5.2f}x   (int64 intermediate {mb:6.1f} MB)")
        if not same:
            d = (a != b).nonzero()
            print(f"   first diff at {d[0].tolist()}: {a[tuple(d[0])]} vs {b[tuple(d[0])]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
