"""Standalone FusedInferAttentionScore (FIA) decode micro-benchmark for the
Ascend kernel team.

Reproduces, in isolation, the exact npu_fused_infer_attention_score call that
sglang's ascend backend issues on the dLLM decode path (forward_dllm in
python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py), at the
LLaDA2.1-mini 4K/1.5K steady-state operating point where this kernel was
profiled as the single largest and least roofline-efficient op on 910B3.

Profiled operating point (bs=72, ~4.2K context, page_size=32):
  - trace kernel: FusedInferAttentionScore
  - trace shapes: query [2304,16,128]; k/v paged [13697,32,512]
  - measured: 1232 us/call, mac=0.24, mte2=0.93
  - roofline: bandwidth-region (AI=128), floor ~390 us/call @1.6 TB/s
  - i.e. running at ~0.51 TB/s effective = 32% of the 1.6 TB/s spec roofline.
    Both cube (mac) and load (bandwidth) pipes are underutilized -> the target
    for optimization is the paged-KV gather + online-softmax cube/vector
    serialization, not more raw bandwidth.

Run (on a free 910B card):
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python fia_decode_bench.py                 # profiled default
  python fia_decode_bench.py --contiguous    # contiguous pages (gather upper bound)
  python fia_decode_bench.py --uniform-ctx 4231

Reproduced standalone on 910B3 (this bench):
  default (scattered pages, variable ctx) : 1092 us  36% of roofline  0.57 TB/s
  --contiguous                            : 1064 us  37% of roofline  0.59 TB/s
  --uniform-ctx 4231                      : 1069 us  36% of roofline  0.58 TB/s
The three are within 3% -> the loss is INTRINSIC to the kernel (online-softmax
cube/vector serialization leaving both pipes idle), NOT the paged-KV gather
pattern nor the length distribution. Optimizing the gather won't help; the win
is in the QK->softmax->AV pipelining. Standalone ~1092 us vs ~1232 us in-model
(the extra ~140 us is contention with the rest of the forward + slightly
deeper context at the profiled sample).

--page-size sweep (block_size passed to the op; sglang uses 32):
  32 : 1088 us  36%     64 : 970 us  40%     128: 970 us  40%
  256:  967 us  41%     512: 961 us  41%
block_size=32 is below the op's documented 128-512 range and costs ~11%; >=128
plateaus at ~40% of roofline. So the page size is a real but minor lever -- the
remaining ~60% gap is the thin-Q_S (32-row) PromptFlash tiling + online-softmax.
"""

import argparse
import math
import time

import numpy as np
import torch
import torch_npu  # noqa: F401  (registers torch.ops.npu.*)

# ---- LLaDA2.1-mini attention config (from the model config + server args) ----
Q_HEADS = 16  # num_attention_heads
KV_HEADS = 4  # num_key_value_heads  (GQA 4:1)
HEAD_DIM = 128  # head_dim
Q_BLOCK = 32  # dLLM decode block: 32 query tokens per request per forward
# KV page/block size. sglang uses 32 (== dllm_block_size), but the op doc says
# block_size must be 128-512 in steps of 128 -> --page-size sweeps this.
DTYPE = torch.bfloat16
SPEC_BW = 1.6e12  # 910B3 HBM bandwidth (bytes/s)
SPEC_TFLOPS = 320e12  # 910B3 bf16


def build_context_lengths(bs, avg_ctx, spread, uniform, seed):
    if uniform:
        return [avg_ctx] * bs
    rng = np.random.default_rng(seed)
    lo, hi = avg_ctx - spread, avg_ctx + spread
    ctx = rng.integers(lo, hi + 1, size=bs)
    return [int(x) for x in ctx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=72)
    ap.add_argument(
        "--avg-ctx",
        type=int,
        default=4231,
        help="mean per-request KV context length (profiled ~4231)",
    )
    ap.add_argument(
        "--spread",
        type=int,
        default=700,
        help="+/- range of per-request context (variable-length TND)",
    )
    ap.add_argument(
        "--uniform-ctx",
        type=int,
        default=0,
        help="if >0, all requests use this exact context length",
    )
    ap.add_argument(
        "--num-pages",
        type=int,
        default=13697,
        help="physical pages in the KV pool (trace showed 13697)",
    )
    ap.add_argument(
        "--contiguous",
        action="store_true",
        help="assign contiguous physical pages per request "
        "(best-case gather); default scatters pages like a real pool",
    )
    ap.add_argument(
        "--page-size",
        type=int,
        default=32,
        help="KV page/block_size passed to the op (sglang=32; "
        "doc says 128-512 in steps of 128)",
    )
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    PAGE_SIZE = args.page_size
    dev = "npu"
    bs = args.bs
    uniform = args.uniform_ctx > 0
    avg_ctx = args.uniform_ctx if uniform else args.avg_ctx
    ctx = build_context_lengths(bs, avg_ctx, args.spread, uniform, args.seed)
    sum_ctx = sum(ctx)
    max_pages_per_req = max(math.ceil(c / PAGE_SIZE) for c in ctx)
    total_pages_used = sum(math.ceil(c / PAGE_SIZE) for c in ctx)
    num_pages = max(args.num_pages, total_pages_used)

    T = bs * Q_BLOCK  # packed query tokens

    # --- query: TND packed [T, Q_HEADS, HEAD_DIM] ---
    query = torch.randn(T, Q_HEADS, HEAD_DIM, dtype=DTYPE, device=dev)

    # --- paged KV pool: [num_pages, PAGE_SIZE, KV_HEADS*HEAD_DIM] (matches trace) ---
    kv_feat = KV_HEADS * HEAD_DIM  # 512
    k_cache = torch.randn(num_pages, PAGE_SIZE, kv_feat, dtype=DTYPE, device=dev)
    v_cache = torch.randn(num_pages, PAGE_SIZE, kv_feat, dtype=DTYPE, device=dev)

    # --- block_table [bs, max_pages_per_req]: physical page id per logical page ---
    rng = np.random.default_rng(args.seed + 1)
    if args.contiguous:
        perm = np.arange(num_pages)
    else:
        perm = rng.permutation(num_pages)  # scattered like a churned KV pool
    block_table = torch.zeros(bs, max_pages_per_req, dtype=torch.int32)
    cursor = 0
    for i, c in enumerate(ctx):
        npg = math.ceil(c / PAGE_SIZE)
        block_table[i, :npg] = torch.from_numpy(
            perm[cursor : cursor + npg].astype(np.int32)
        )
        cursor += npg
    block_table = block_table.to(dev)

    # --- TND actual-seq metadata, exactly as forward_dllm builds it ---
    # Q: cumsum of the per-request 32-token block; KV: per-request context length.
    actual_seq_lengths = (
        torch.cumsum(torch.tensor([Q_BLOCK] * bs, dtype=torch.int32), dim=0)
        .int()
        .tolist()
    )
    actual_seq_lengths_kv = list(ctx)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    def run():
        out, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            k_cache,
            v_cache,
            block_table=block_table,
            block_size=PAGE_SIZE,
            num_heads=Q_HEADS,
            num_key_value_heads=KV_HEADS,
            input_layout="TND",
            atten_mask=None,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )
        return out

    for _ in range(args.warmup):
        run()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        run()
    torch.npu.synchronize()
    us = (time.perf_counter() - t0) / args.iters * 1e6

    # --- roofline ---
    kv_bytes = sum_ctx * KV_HEADS * HEAD_DIM * 2 * 2  # K+V, bf16
    flops = 4 * Q_BLOCK * Q_HEADS * HEAD_DIM * sum_ctx  # QK+AV, madd
    ai = flops / kv_bytes
    bw_floor_us = kv_bytes / SPEC_BW * 1e6
    comp_floor_us = flops / SPEC_TFLOPS * 1e6
    floor_us = max(bw_floor_us, comp_floor_us)
    eff_bw = kv_bytes / (us * 1e-6)

    print(f"=== FIA decode micro-bench (1 layer, 1 call) ===")
    print(
        f"bs={bs}  q_block={Q_BLOCK}  packed_T={T}  heads={Q_HEADS}/{KV_HEADS}  "
        f"head_dim={HEAD_DIM}  page={PAGE_SIZE}  dtype=bf16"
    )
    print(
        f"context: sum_kv={sum_ctx}  avg={sum_ctx//bs}  "
        f"{'uniform' if uniform else f'variable +/-{args.spread}'}  "
        f"pages_used={total_pages_used}/{num_pages}  "
        f"pages={'contiguous' if args.contiguous else 'scattered'}"
    )
    print(
        f"query      [{T},{Q_HEADS},{HEAD_DIM}]   kv paged [{num_pages},{PAGE_SIZE},{kv_feat}]"
    )
    print()
    print(f"latency        : {us:8.1f} us/call")
    print(
        f"roofline floor : {floor_us:8.1f} us/call   "
        f"({'bandwidth' if bw_floor_us > comp_floor_us else 'compute'}-bound, "
        f"AI={ai:.0f})"
    )
    print(
        f"  bw floor     : {bw_floor_us:8.1f} us  (kv {kv_bytes/1e6:.0f} MB @ {SPEC_BW/1e12:.1f} TB/s)"
    )
    print(
        f"  compute floor: {comp_floor_us:8.1f} us  ({flops/1e9:.0f} GFLOP @ {SPEC_TFLOPS/1e12:.0f} T)"
    )
    print(
        f"efficiency     : {floor_us/us*100:5.1f}% of roofline   "
        f"eff-BW {eff_bw/1e12:.2f} TB/s ({eff_bw/SPEC_BW*100:.0f}% of spec)"
    )


if __name__ == "__main__":
    main()
