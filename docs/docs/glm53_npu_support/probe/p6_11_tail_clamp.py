"""P6.11: is the clamped history gather what makes the tail kernel scalar-bound?

`_append_kpool_tail_to_topk_kernel` profiles at 4.73 ms with `aiv_vec_ratio` 0.027
and `aiv_mte2_ratio` 0.0 -- it neither computes nor moves anything, so the time is
addressing. The suspect is:

    safe_history_cols = tl.minimum(cols, N_COLS - 1)
    tl.load(topk_ptr + row * s0 + safe_history_cols * s1, mask=mask & is_history, ...)

The clamp cannot change a lane the mask keeps: `is_history` is `cols < history_len`
and `history_len <= N_COLS`, so wherever the value is used, `cols == safe_history_cols`
already. What it does change is the address expression -- non-affine in `cols`, so a
contiguous vector load becomes per-element addressing.

Dropping it relies on Triton's contract that a masked-off lane is not accessed. That
holds on CUDA; whether triton-ascend honours it is the thing to find out here, which
is why this probe runs the two variants side by side and compares element for element
before looking at any timing.

    source $ROOT/env.sh
    ASCEND_RT_VISIBLE_DEVICES=14 PYTHONPATH=$REPO/python:$PYTHONPATH npy probe/p6_11_tail_clamp.py
"""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(topk_ptr, seq_lens_ptr, pool_lens_ptr, out_ptr,
            topk_stride_0, topk_stride_1, out_stride_0, out_stride_1,
            N_COLS: tl.constexpr, OUT_COLS: tl.constexpr, POOL_SIZE: tl.constexpr,
            CLAMP: tl.constexpr, BLOCK_COLS: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLS)
    mask = cols < OUT_COLS
    seq_len = tl.load(seq_lens_ptr + row).to(tl.int32)
    pool_len = tl.load(pool_lens_ptr + row).to(tl.int32)
    tail_start = pool_len * POOL_SIZE
    history_len = tl.minimum(tail_start, N_COLS)
    tail_count = seq_len % POOL_SIZE
    is_history = cols < history_len
    if CLAMP:
        src = tl.minimum(cols, N_COLS - 1)
    else:
        src = cols
    history_value = tl.load(
        topk_ptr + row * topk_stride_0 + src * topk_stride_1,
        mask=mask & is_history, other=-1,
    )
    tail_offset = cols - history_len
    is_tail = (tail_offset >= 0) & (tail_offset < tail_count)
    value = tl.where(is_history, history_value, -1)
    value = tl.where(is_tail, tail_start + tail_offset, value)
    tl.store(out_ptr + row * out_stride_0 + cols * out_stride_1, value, mask=mask)


def run(topk, seq_lens, pool_lens, pool_size, clamp):
    rows, n_cols = topk.shape
    out_cols = n_cols + pool_size - 1
    out = torch.empty((rows, out_cols), dtype=torch.int32, device=topk.device)
    _kernel[(rows,)](
        topk, seq_lens, pool_lens, out,
        topk.stride(0), topk.stride(1), out.stride(0), out.stride(1),
        N_COLS=n_cols, OUT_COLS=out_cols, POOL_SIZE=pool_size,
        CLAMP=clamp, BLOCK_COLS=triton.next_power_of_2(out_cols),
    )
    return out


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
    dev = "npu"
    pool_size, n_cols = 4, 2048
    for rows, seq_hi in ((8192, 32768), (4096, 16384), (1024, 4096), (16, 32768)):
        topk = torch.randint(0, 1 << 20, (rows, n_cols), dtype=torch.int32, device=dev)
        seq_lens = torch.randint(1, seq_hi, (rows,), dtype=torch.int32, device=dev)
        pool_lens = torch.div(seq_lens, pool_size, rounding_mode="floor").to(torch.int32)

        a = run(topk, seq_lens, pool_lens, pool_size, clamp=True)
        b = run(topk, seq_lens, pool_lens, pool_size, clamp=False)
        same = torch.equal(a, b)
        t_a = timed(lambda: run(topk, seq_lens, pool_lens, pool_size, True))
        t_b = timed(lambda: run(topk, seq_lens, pool_lens, pool_size, False))
        print(f"rows={rows:>5} n_cols={n_cols}: identical={same}  "
              f"clamped={t_a:7.3f} ms  unclamped={t_b:7.3f} ms  "
              f"speedup={t_a / t_b:5.2f}x")
        if not same:
            d = (a != b).nonzero()
            print(f"   first differing element: {d[0].tolist()} -> {a[tuple(d[0])]} vs {b[tuple(d[0])]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
