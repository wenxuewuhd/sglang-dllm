# OP-1 — kpool fused group-top-k + pool→raw expand + tail append

**Status:** NEW kernel. **Priority: highest.**
**Why:** this runs on every decode step, on all 11 DSA (`deepseek_sparse_attention`)
layers of GLM-5.3-Flash. There is no NPU equivalent today, and the decomposed fallback
would cost a full `topk` over 2048-wide logits plus two gather passes per layer per step.

Reference implementation: [`../reference/kpool_topk_transform.py`](../reference/kpool_topk_transform.py)
Tests: [`../tests/test_op1_kpool_topk_transform.py`](../tests/test_op1_kpool_topk_transform.py)

---

## 1. What the operator does, in one paragraph

The DSA indexer scores *pools* of keys, not individual keys. Each pool covers
`pool_size = index_kpool = 4` consecutive tokens. For each query row we pick the best
`group_topk = index_topk / index_kpool = 2048 / 4 = 512` pools by score, expand each
selected pool into its `pool_size` token indices, map those indices through either a
page table or a per-row offset, and finally append the *tail* — the 0…`pool_size-1`
most recent tokens that do not yet form a complete pool. Everything that is not written
is filled with the padding sentinel `-1`. The whole thing is one kernel on CUDA; it must
be one kernel on Ascend too, because splitting it re-materialises a `[B, 512]` index
tensor per layer per step.

## 2. Source of truth

| What | Where |
|---|---|
| The kernel to reproduce | `python/sglang/kernels/jit/csrc/dsa/kpool_topk_transform.cuh:244-305` (`kpool_topk_transform_kernel`) |
| Index transform | same file, `:230-242` (`transform_kpool_token`) |
| Host-side arg validation | same file, `:337-460` (`KpoolTopKTransformKernel::transform`) |
| Python wrapper | `python/sglang/kernels/ops/moe/kpool_topk_transform.py:31-80` (`fast_kpool_topk_transform_fused`) |
| Dispatcher / decomposed path | `python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py:555-663` (`topk_from_pooled_history_logits`) |
| Clearest statement of the semantics (pure torch/triton) | same file, `:379-418` (`expand_pooled_groups_to_topk`), `:421-552` (`append_kpool_tail_to_topk`) |
| Call sites | `python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py:922, :1035, :1144, :1384` |

## 3. Proposed interface

**Op type:** `KpoolTopkTransform` **aclnn:** `aclnnKpoolTopkTransform`
**Torch binding:** `torch.ops.custom.npu_kpool_topk_transform`

```
npu_kpool_topk_transform(
    Tensor  score,                          # [B, S]  float32
    Tensor  lengths,                        # [B]     int32
    int     pool_size,
    int     topk,
    *,
    Tensor? page_table            = None,   # [B, P] or [R, P]  int32
    Tensor? topk_indices_offset   = None,   # [B]     int32
    Tensor? row_starts            = None,   # [B]     int32
    Tensor? seq_lens              = None,   # [B]     int32
    Tensor? page_table_row_index  = None,   # [B]     int32
) -> Tensor                                 # [B, out_cols]  int32
```

If a workspace-style aclnn signature is preferred, the output tensor may be passed in
instead of returned; SGLang allocates it either way. `out_cols` is fully determined by
the inputs (§5), so an out-variant needs no extra attribute.

### 3.1 Inputs

| Name | Shape | dtype | Req. | Meaning |
|---|---|---|---|---|
| `score` | `[B, S]` | float32 | yes | Pool-level logits. **Row-major, last dim contiguous, row stride `S` may exceed the used width.** For row `b` only `score[b, row_starts[b] : row_starts[b] + lengths[b]]` is read. `B` is the number of query rows in the batch (decode: one per sequence; extend: one per query token). |
| `lengths` | `[B]` | int32 | yes | Number of *valid pool groups* for row `b`. `0 <= lengths[b]`, and `row_starts[b] + lengths[b] <= S`. |
| `pool_size` | scalar | int | yes | Tokens per pool. **`> 1`** (`.cuh:369`). GLM: `4`. |
| `topk` | scalar | int | yes | Token-level budget. Must be a multiple of `pool_size` (`.cuh:381`). GLM: `2048`. `group_topk = topk / pool_size` = **512** for GLM. |
| `page_table` | `[B, P]` or `[R, P]` | int32 | no | Raw-token → physical-token map. Row-major, last dim contiguous, row stride `P`. `[B, P]` when `page_table_row_index` is absent; `[R, P]` (a shared/compact table with `R` unrelated to `B`) when it is present. **Mutually exclusive with `topk_indices_offset`** (`.cuh:370`). |
| `topk_indices_offset` | `[B]` | int32 | no | Per-row constant added to raw tokens (the *ragged* layout). Mutually exclusive with `page_table`. |
| `row_starts` | `[B]` | int32 | no | Column offset **inside** row `b` of `score` where that row's groups begin. Absent ⇒ 0. Note it is applied **in addition to** the `b`-th row stride: the kernel reads `score_base + b*S + row_starts[b] + i`. Selected group ids are 0-based **relative to `row_starts[b]`**, not absolute columns. |
| `seq_lens` | `[B]` | int32 | no | Token-level sequence length. **Presence of this argument is what enables the tail append**, and it also changes `out_cols` (§5). |
| `page_table_row_index` | `[B]` | int32 | no | Which row of `page_table` row `b` should use. **Requires `page_table`** (`.cuh:373`). Absent ⇒ row `b` uses `page_table[b]`. |

### 3.2 Output

| Name | Shape | dtype | Meaning |
|---|---|---|---|
| return | `[B, out_cols]` | int32 | Selected token indices, already transformed. Contiguous (row stride `out_cols`). Unwritten positions are `-1`. |

## 4. Correction to the brief

> "the decomposed branch asserts `page_table_row_index is None`; the paged **decode**
> path DOES pass it"

Half right, and the half that is wrong matters for prioritisation. `page_table_row_index`
is produced at `python/sglang/srt/layers/attention/dsa/kpool_plan.py:414-420`
(`ragged_paged_page_table_row_index = repeat_interleave(local_req_pool_indices, ragged_q_len_t)`,
with `ragged_paged_page_table = req_to_token`) and is consumed only by
`dsa_indexer_kpool.py:1017-1043` inside `_get_topk_ragged_kpool_plan` — the **ragged
extend / chunked-prefill** path, gated on `SGLANG_DSA_FUSE_TOPK` and
`TopkTransformMethod.PAGED`.

The **decode** call site is `dsa_indexer_kpool.py:921-929`; it obtains its mapping from
`_kpool_fused_topk_mapping(metadata)` (`:806-827`) with no extra arguments, so
`paged_page_table_row_index` is `None` there and the third return value is discarded.

The conclusion for you is unchanged: **the operator must support
`page_table_row_index`**, because prefill uses it and prefill uses the same kernel. The
only thing that changes is that it is not on the per-decode-step hot path.

Also note the shape consequence: in that mode `page_table` is the *entire*
`req_to_token` pool. Its row stride is context-scale (~2²⁰ on a 1 M-context model), so
`row * stride` overflows int32 from row ~2048 — the triton fallback promotes to int64
for exactly this reason (`kpool_fp8_index.py:536-545`). **Compute page-table addresses
in 64-bit.**

## 5. Exact semantics

Derived quantities:

```
group_topk = topk / pool_size                       # 512 for GLM
tail_cols  = (seq_lens != null) ? pool_size - 1 : 0 # 3 for GLM
out_cols   = topk + tail_cols                       # 2051 for GLM
```

Per row `b`:

```
length      = lengths[b]
row_start   = row_starts ? row_starts[b] : 0
window      = score[b, row_start : row_start + length]        # length floats

if length <= group_topk:                                      # .cuh:272
    selected = [0, 1, ..., length-1]                          # identity, ascending
else:
    selected = indices of the group_topk largest values of `window`   # ORDER UNSPECIFIED

history_len = min(length * pool_size, topk)                   # .cuh:268-269
tail_count  = seq_lens ? (seq_lens[b] % pool_size) : 0        # .cuh:270

for col in [0, out_cols):
    if col < history_len:
        raw = selected[col / pool_size] * pool_size + (col % pool_size)
    elif seq_lens && col < history_len + tail_count:
        raw = length * pool_size + (col - history_len)
    else:
        out[b, col] = -1;  continue
    out[b, col] = transform(raw)

transform(raw) =
    page_table          ? page_table[ page_table_row_index ? page_table_row_index[b] : b ][raw]
  : topk_indices_offset ? raw + topk_indices_offset[b]
  :                       raw
```

Read the three branches of `transform` in that priority order. `page_table` and
`topk_indices_offset` are mutually exclusive so the order only documents intent.

### 5.1 Things that are deliberately NOT part of the contract

* **The order of the selected groups.** The CUDA kernel writes them in radix-scan
  completion order when `length > group_topk`, and in ascending id order when
  `length <= group_topk`. Emit whatever order is cheapest; downstream is a gather into
  sparse attention, which is permutation-invariant over keys.
* **Which of several tied groups is picked** at the `group_topk`-th boundary. See §7.

### 5.2 Things that ARE part of the contract

* `history_len` uses `min(length*pool_size, topk)` — **not** `length*pool_size`. When
  `length > group_topk` the history is truncated to exactly `topk` columns.
* The tail starts at column `history_len`, *not* at column `topk`. When
  `length*pool_size < topk` the tail sits in the middle of the row and the padding
  follows it.
* `tail_count` is `seq_lens[b] % pool_size` — a value in `[0, pool_size)`. This is why
  `tail_cols = pool_size - 1` suffices.
* Everything past `history_len + tail_count` is `-1`, including the whole row when
  `length == 0` and `seq_lens[b] % pool_size == 0`.
* Output dtype is int32 even though `page_table` values can be large; SGLang guarantees
  they fit (the KV pool is int32-indexed).

## 6. Edge cases and error behaviour

| Case | Required behaviour |
|---|---|
| `B == 0` | No-op. Return an empty `[0, out_cols]` int32 tensor. Must not fault. |
| `lengths[b] == 0` | `history_len = 0`. Row is tail (if any) then all `-1`. |
| `lengths[b] == 1` | Legal. Do not assume the top-k path. |
| `lengths[b] == group_topk` | Takes the identity path (`length <= K`). |
| `lengths[b] == group_topk + 1` | Takes the top-k path. This boundary is tested. |
| Ragged `lengths` across the batch | Normal. Rows are independent; nothing is batched over `length`. |
| `pool_size <= 1` | Reject (`.cuh:369`). |
| `topk % pool_size != 0` | Reject (`.cuh:381`). |
| both `page_table` and `topk_indices_offset` | Reject (`.cuh:370`). |
| `page_table_row_index` without `page_table` | Reject (`.cuh:373`). |
| Padding sentinel | `-1`, int32, everywhere. Never `0`, never a large sentinel. |
| Out-of-range `raw` into `page_table` | **Cannot happen** given a valid batch: `raw < length*pool_size <= seq_lens[b] <= P` for the history part and `raw <= seq_lens[b]-1` for the tail. The CUDA kernel does not bounds-check (`.cuh:236`); you need not either, but clamping instead of faulting is acceptable. |
| Rows beyond the real batch | SGLang pads the *output* itself (`kpool_fp8_index.py:610-616, :657-663`); the operator never sees padded rows. Do not implement `out_rows`. |

## 7. Known deviation in the CUDA kernel (do not reproduce)

`fast_topk_cuda_tl_impl` keeps at most `SMEM_INPUT_SIZE = 4096` threshold-bin candidates
per radix round and silently drops the rest (`.cuh:149`, `.cuh:168`, `.cuh:215`). When
more than 4096 groups land in the same 8-bit bin the CUDA result is **not** the exact
top-k. Do not replicate this; produce the exact top-k. The acceptance criterion (§8) is
written so that both are accepted as long as the selected *score multiset* matches, and
an exact implementation will always match the exact reference.

## 8. Acceptance criteria

Index-by-index equality is the wrong gate: order is unspecified (§5.1) and ties at the
`k`-th boundary legitimately differ. There is also **no CUDA device on this machine**,
so nobody can produce reference indices from the original kernel. The gate is therefore:

1. **Structure.** For every `rank < history_len / pool_size` and every
   `s < pool_size`, `out[b, rank*pool_size + s]` must be `transform(g*pool_size + s)`
   for a single group id `g` — i.e. each pool expands contiguously and in slot order.
2. **Selection.** Let `G` be the group ids recovered in (1). Then
   `sorted([window[g] for g in G])` must equal, **element by element and exactly**, the
   same list computed from the reference. The multiset of selected *values* is uniquely
   determined by `k` and the score vector even when ties exist, so this is an exact
   test that is nevertheless blind to which tied index was taken. (The looser
   "sum of selected scores" form is implied by it.)
3. **No duplicates**, and every `g` in `[0, length)`.
4. **Tail and padding, exactly.** Columns `[history_len, history_len + tail_count)` and
   `[history_len + tail_count, out_cols)` are compared index-by-index with `==`.

This is what `tests/test_op1_kpool_topk_transform.py::_check` implements. There is no
floating-point tolerance anywhere in OP-1: the operator returns integers.

## 9. Shapes and alignment on the target

For GLM-5.3-Flash on Atlas A3 (`Ascend910_9362`, compile for `ascend910_93`):

| Quantity | Decode | Extend (chunked prefill) |
|---|---|---|
| `pool_size` | 4 | 4 |
| `topk` | 2048 | 2048 |
| `group_topk` | 512 | 512 |
| `out_cols` | 2051 | 2051 |
| `B` | batch size (1…max concurrency) | number of query tokens in the chunk |
| `S` | `ceil(max_seq_len / 4)` | length of the concatenated ragged pool buffer |
| `lengths[b]` | `seq_len[b] / 4` | per-query pooled prefix length |

**`out_cols = 2051` is not a multiple of 16 or 32.** The trailing 3 columns are exactly
the tail region, so the unaligned part of the store is the part that is hardest to make
regular. Plan for it rather than requiring an aligned `out_cols`.

`group_topk` should be a **runtime** value if that is free; the CUDA side makes it a
compile-time constant and instantiates `{128, 160, 192, 224, 256, 512}`
(`python/sglang/kernels/ops/moe/kpool_topk_transform.py:13`). **512 is the only value
GLM needs**; the others are DeepSeek-V3.2 configurations.

## 10. Not pinned down

* Whether SGLang will ever call this with `pool_size != 4`. Everything in the spec is
  written generically; the only value exercised in this project is 4.
* Whether an aclgraph / captured-graph decode path will require the operator to be
  capturable with fixed shapes. Assume it will: **no host-side reads of tensor contents,
  no dynamic allocation whose size depends on tensor values.** `lengths`, `seq_lens`
  and `row_starts` must be consumed on device.
