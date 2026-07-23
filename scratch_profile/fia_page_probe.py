"""Probe the FIA kernel's INTERNAL KV tile granularity on 910B NPU.

Question: sglang passes block_size=32 to npu_fused_infer_attention_score, but the
op doc says block_size must be >=128. Does the kernel internally retile KV into
128-token blocks regardless (rounding work up to ceil(ctx/128)*128), or does it
honor the 32 granularity (ceil(ctx/32)*32)?

Method: fix block_size, use a UNIFORM context length, and sweep ctx in steps of
32 across several 128 boundaries. A flash kernel loads/compute KV in whole tiles,
so per-call latency rounds UP to the tile granularity:
  - internal tile = 32  -> latency rises at EVERY 32-step (96,128,160,192,...)
  - internal tile = 128 -> latency is FLAT within (128k,128(k+1)] and jumps only
                           just past each 128 multiple (129,257,385,...)

So at block_size=32: if 160/192/224/256 are ~flat and all sit above 128, then a
128-multiple jump pattern => the kernel really works in 128 internally. If they
climb monotonically per 32-step => it honors 32.

Run:
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ASCEND_RT_VISIBLE_DEVICES=<free> python fia_page_probe.py --pairs   # discriminator
  ASCEND_RT_VISIBLE_DEVICES=<free> python fia_page_probe.py           # full staircase

RESULT (910B3, block_size=32, device-timed, bs=128, reproducible over 2 seeds):
  ctx 129->192->256 (real KV +60%) : 226 / 220 / 230 us  -> FLAT across the band
  ctx 257 (next 128-band)          : 275 us              -> jumps
  ctx 384 (128-aligned)            : 293 us; 383 -> 328  -> aligned is a minimum
  Latency tracks ceil(ctx/128), NOT ceil(ctx/32). => the FIA kernel computes
  attention at 128-token KV granularity INTERNALLY even when block_size=32 is
  passed; exact-128-multiples are fastest, just-below (224/255/383) pay a
  partial-tile mask penalty. So block_size=32 does not buy finer compute -- it
  only adds ~4x block_table indirection (133 vs 34 page entries per 4.2K req),
  which is the ~11% seen when moving 32->128 in fia_decode_bench. The KV N-tile
  (128) is fine; the real inefficiency is the thin query M=32 (cube underfill,
  mac=0.24) + online-softmax serialization.
"""

import argparse
import math
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

Q_HEADS, KV_HEADS, HEAD_DIM = 16, 4, 128
Q_BLOCK = 32
DTYPE = torch.bfloat16


def time_fia(bs, ctx, page_size, iters, warmup, seed):
    dev = "npu"
    T = bs * Q_BLOCK
    npg_per_req = math.ceil(ctx / page_size)
    num_pages = bs * npg_per_req
    query = torch.randn(T, Q_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)
    kv_feat = KV_HEADS * HEAD_DIM
    k = torch.randn(num_pages, page_size, kv_feat, dtype=DTYPE, device=dev)
    v = torch.randn(num_pages, page_size, kv_feat, dtype=DTYPE, device=dev)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_pages).astype(np.int32)
    block_table = torch.from_numpy(perm.reshape(bs, npg_per_req)).to(dev)
    actual_seq_lengths = (
        torch.cumsum(torch.tensor([Q_BLOCK] * bs, dtype=torch.int32), dim=0)
        .int()
        .tolist()
    )
    actual_seq_lengths_kv = [ctx] * bs
    scale = 1.0 / math.sqrt(HEAD_DIM)

    def run():
        o, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            k,
            v,
            block_table=block_table,
            block_size=page_size,
            num_heads=Q_HEADS,
            num_key_value_heads=KV_HEADS,
            input_layout="TND",
            atten_mask=None,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )
        return o

    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    # device-side timing to strip host jitter
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # ms->us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument(
        "--page-size",
        type=int,
        default=32,
        help="block_size passed to the op (probe target = 32)",
    )
    ap.add_argument("--lo", type=int, default=96)
    ap.add_argument("--hi", type=int, default=544)
    ap.add_argument("--step", type=int, default=32)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--pairs",
        action="store_true",
        help="just run the discriminating ctx pairs, not the full sweep",
    )
    args = ap.parse_args()

    if args.pairs:
        print(
            f"block_size={args.page_size}  bs={args.bs}  (device-timed, "
            f"{args.iters} iters)"
        )
        print("discriminator: ctx just-over-128k vs the next 128-multiple.")
        print("  internal tile=128 -> the pair is ~EQUAL (both = ceil/128 tiles)")
        print(
            "  internal tile=32  -> the just-over one is ~(ceil32*32)/(128*k) faster\n"
        )
        print(f"{'ctx':>6} {'ceil/32*32':>11} {'ceil/128*128':>13} {'us':>9}")
        for ctx in [129, 160, 256, 257, 384, 385, 512]:
            us = time_fia(
                args.bs, ctx, args.page_size, args.iters, args.warmup, args.seed
            )
            print(
                f"{ctx:>6} {math.ceil(ctx/32)*32:>11} {math.ceil(ctx/128)*128:>13} {us:>9.2f}"
            )
        return

    print(
        f"block_size={args.page_size}  bs={args.bs}  q_block={Q_BLOCK}  "
        f"GQA {Q_HEADS}/{KV_HEADS}  head_dim={HEAD_DIM}"
    )
    print(f"{'ctx':>6} {'ceil/32':>7} {'ceil/128':>8} {'us':>9} {'d_us':>8}  marker")
    prev = None
    for ctx in range(args.lo, args.hi + 1, args.step):
        us = time_fia(args.bs, ctx, args.page_size, args.iters, args.warmup, args.seed)
        d = "" if prev is None else f"{us - prev:+8.1f}"
        c32, c128 = math.ceil(ctx / 32), math.ceil(ctx / 128)
        mark = (
            "  <-128 boundary" if ctx % 128 <= args.step - 1 and ctx > args.lo else ""
        )
        # flag the first ctx in each new 128-band
        band = (
            "  [new 128-band]"
            if (
                prev is not None
                and math.ceil(ctx / 128) != math.ceil((ctx - args.step) / 128)
            )
            else ""
        )
        print(f"{ctx:>6} {c32:>7} {c128:>8} {us:>9.1f} {d:>8}{band}")
        prev = us
    print(
        "\nRead: if d_us jumps only on [new 128-band] rows and is ~0 between, "
        "the kernel tiles KV at 128 internally. If d_us is ~constant every "
        "32-step, it honors block_size=32."
    )


if __name__ == "__main__":
    main()
