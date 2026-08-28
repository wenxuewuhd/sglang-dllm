# OP-1 — kpool index-K cache (WITHDRAWN)

**Status: WITHDRAWN. No operator work is requested.**

The open question this request rested on — §5, *what computes the indexer logits on
Ascend* — has an answer: `torch_npu.npu_lightning_indexer`, which takes GLM's 32 index
heads and ships in the target runtime. It reads **bf16** keys, so the index cache moves
fp8 → **bf16**, which is a repo-side dtype change behind no vendor operator at all.
See §5 and `../../PLAN.md` §2.6.

The int8 measurements in §2 stand and are not retracted — int8 really is more accurate
than the fp8 it would have replaced. They are kept as the fallback record for P5 memory
pressure (`PLAN.md` §2.7). What retired them from phase 1 is that **nothing on Ascend can
read an int8 index cache**: `npu_lightning_indexer` rejects int8, and
`npu_quant_lightning_indexer` cannot express 32 query heads.

**Everything below is the withdrawn request, kept for the record.**

Reference implementation: [`../reference/kpool_topk_transform.py`](../reference/kpool_topk_transform.py)
Tests: [`../tests/test_op1_kpool_topk_transform.py`](../tests/test_op1_kpool_topk_transform.py)

---

## 1. Why the shape of this changed

The original request was a fused group-top-k plus pool→raw expand plus tail append. It is
not needed. Measured on target (triton-ascend 3.2.2 / Ascend910_9362), **7 of the 10 kpool
Triton kernels compile, run, and are bit-identical** to torch references transcribed from
the kernel bodies — including the top-k, the expand, the tail append, and the plan/layout
machinery. That code already works on this hardware.

The four that fail all fail in the same place, and it is not a Triton problem:

```
[ConvertLinalgRToBinary] LLVM ERROR: unsupported datatype for arith::TruncFOp to hfusion
```

**Atlas A3 cannot express the fp8 e4m3 conversion at all.** `x.to(torch.float8_e4m3fn)`
faults on device too. kpool stores its compressed index keys in fp8, and that single fact
is what blocks the path. Substitute bf16 for the fp8 store and three of the four compile,
run, and match the reference across the whole computation.

So the work is a **storage-format change**, not a new kernel.

## 2. int8 is the route, and it is measured

DeepSeek-V4 on Ascend already stores its index keys as int8 (`ascend_dsv4_backend.py:685`,
`:469-470`, `:597-598`), so the precedent exists. The question was whether GLM's indexer
still selects the same tokens. It does — and int8 is **better than the fp8 it replaces**.

Simulated on CPU with the HF `Glm5NextTextIndexer`, real layer-3 weights, real hidden
states (embed + layers 0–2 run for real), faithful pipeline
(`pooled_key → bf16 → Hadamard128 → bf16 → quant → dequant`), a real `torch.float8_e4m3fn`
cast for the incumbent, 512 query rows per length:

| seq_len | fp8 e4m3 + ue8m0 (incumbent) | **int8 absmax/127** |
|---|---|---|
| ≤ 2048 | 1.000 | 1.000 (selection does not bind) |
| 4096 | 0.9912 | **0.9977** |
| 8192 | 0.9852 | **0.9965** |
| 32768 | 0.9653 | **0.9918** |

Overlap of the selected pool set against a bf16 ceiling, mean over rows. Score mass
retained at 32k: fp8 0.99928, int8 0.99996. Key reconstruction error (rel-L2): fp8 0.0265,
**int8 0.0063** — 4.2× better, on all 11 DSA layers, kurtosis 2.81–3.17 throughout.

Swaps are between near-ties and int8's are the smaller ones: the worst token fp8 drops
sits 0.084 of the top-k score span above the cut; int8's worst is 0.020. Neither ever drops
a high-scoring token.

## 3. Conditions — all three must hold

**3.1 Exact `absmax/127`. Do not carry over the `ue8m0` scale rounding.**
kpool forces `ROUND_SCALE=True` (`deepseek_v2.py:1877` sets `scale_fmt="ue8m0"`, consumed
at `dsa_indexer_kpool.py:248, 295, 424, 1526`). Rounding the scale up to a power of two is
free for a float format and costs int8 a real bit: overlap at 32k drops 99.18% → 98.84%.

**3.2 Keep the Hadamard-128 rotation** (`kpool_fp8_index.py:862-869`). This is *why* int8
wins: post-rotation kurtosis is 2.8–3.2, i.e. near-Gaussian, which is the regime where a
uniform grid beats e4m3's 3-bit mantissa. Drop the rotation and the result above does not
transfer. Note this contradicts an earlier conclusion of ours that the rotation was
removable — it is removable only for a *bf16* indexer, where it genuinely does not change
the dot product. Under quantization it changes the error.

**3.3 Symmetric, scale-only, no zero point**, keeping the `1e-4` absmax floor. **fp16 scale
storage is safe**: measured over 8192 pooled rows at 32k, scales span 2.0e-2 to 4.8e-2, a
>300× margin over fp16's min normal, no subnormals, no rows at the epsilon floor, and
fp16-vs-fp32 scales produced identical selections. That is structural — `k_norm` is a
LayerNorm so the key norm is bounded, and the Hadamard equalises it.

## 4. What changes in the code

Quantization happens on the **compressed pooled key** — one 128-vector per cache slot, one
symmetric absmax scale per slot. Pooling, the APE add, the softmax-weighted compression and
the Hadamard all happen upstream in fp32/bf16 and are untouched by the dtype. The tail ring
buffer is bf16 in a separate per-request tensor and is never quantized.

Every fp8 assumption is in the storage and the consumer, none in the math:

| what | where |
|---|---|
| `448.0` / `-448.0` / `1.0/448.0` literals | `kpool_fp8_index.py:950-963, 1108-1121, 1162-1169`; `kernels/ops/attention/dsa/triton_kernel.py:31-33` |
| `torch.float8_e4m3fn` dtypes and views (~14 sites) | `kpool_fp8_index.py:709, 716, 721, 813, 1289, 1687`; `dsa_indexer_kpool.py:221, 1003, 1113, 1315, 1344` |
| packed page layout `page_size*(128+4)`, scale region at byte offset `page_size*128` | `mem_cache/index_key_cache.py:33-38`; `index_buf_accessor.py:305`; `kpool_fp8_index.py:70, 749, 844, 1315, 1716` |
| `// 4` byte→fp32 offset arithmetic, 5 kernels | `kpool_fp8_index.py:101-103, 975-977, 1140-1142, 1248-1250, 1629-1631` |
| `head_dim_with_sf = 132` | `dsa_indexer_kpool.py:876` |
| `payload_bytes = head_dim + 4` (CP all-gather) | `kpool_fp8_index.py:1487, 1425-1428, 1459-1462` |
| four kernels cast to fp8 implicitly, by storing into an fp8-viewed pointer — there is no `.to(tl.float8e4nv)` to grep for | `_kpool_softmax_rotate_write_cache_kernel:874`, `_kpool_decode_update_and_maybe_write_cache_kernel:992`, `_kpool_assemble_softmax_rotate_write_cache_kernel:1174`, `_kpool_write_tail_and_maybe_compress_kernel:1531` |

**A layout decision comes with this.** kpool packs key and fp32 scale into one uint8 page;
DeepSeek-V4 on Ascend uses two buffers with an fp16 scale
(`npu/dsv4/dsv4_memory_pool.py:169-178` and `:180-190`). Following DeepSeek-V4 retires the
packed page and with it rows 3 through 6 of that table, rather than porting them.

## 5. This section was the open question — it is now closed

**What computes kpool's scoring on Ascend: `torch_npu.npu_lightning_indexer`.**

The candidate considered here was `npu_quant_lightning_indexer`, and it is indeed
unusable: its metadata op **only accepts `num_heads_q=64`** while GLM's `index_n_heads`
is **32** (measured twice, independently — 16, 32 and 128 all fail). That remains true.

What the search missed is that `npu_lightning_indexer` — no `quant_` — is a *different*
operator, already used by DeepSeek-V4's non-kpool path at
`srt/layers/attention/dsa/dsa_npu_indexer.py:25`. Measured on target, probe at
`../../probe/p3_4_lightning_indexer.py`:

- accepts `num_heads_q` ∈ {16, 32, 64}; 128 fails
- computes exactly `Σ_h w_h · relu(q_h · k_j)` — top-k sets identical to a torch
  reference at `n_heads=32`, prefill and decode
- keys are **bf16 only**: fp16 and int8 are both rejected, *despite the operator doc
  claiming float16 support*
- returns **logical** sequence positions, not physical cache slots (verified against a
  shuffled block table) — which is what kpool's pool→token expand consumes
- pads with `-1` when `actual_seq_lengths_key < sparse_count`
- fuses the top-k, so kpool's pooled selection kernel is not needed either

kpool's visibility grows at 1/4 the query rate (`floor(seq_len/kpool)` pools), which
`sparse_mode=3` (rightDownCausal, slope 1) cannot express. Expressing it as **runs of
query rows sharing one visible-pool count**, one TND batch per run, with `sparse_mode=0`,
is exact — verified at 256/1024/4096-token chunks; 4096 query rows against 8192 pools in
3.6 ms.

This closes the request: the dtype question dissolves with it, because the operator's
input dtype decides the storage format, and it is bf16.

## 6. Verification

Everything in §2 is CPU simulation against the HF reference — no NPU run and no accuracy
evaluation. §1's kernel results are on-hardware. Layers 7–43 were checked by running
layer-3 hidden states through each layer's own indexer weights, which rules out a
pathological `k_norm`/`wk` in any layer but does not prove the overlap numbers transfer
verbatim.
