# GLM-5.3-Flash on Ascend — operator handoff package

This directory is a **self-contained work order for an Ascend kernel team**. It
specifies two operators, gives a pure-torch reference
implementation of each that *is* the definition of correct, ships an executable test
suite, and states the acceptance criteria. It is meant to be implementable without
asking us anything.

Everything here is new; nothing outside this directory was changed.

> The surrounding project documents ([`../PLAN.md`](../PLAN.md),
> [`../SETUP.md`](../SETUP.md)) are in Chinese. This package is in English; the code and
> the specs are the parts that matter and they are language-neutral.

---

## 0. OP-1 has been re-scoped — read this before starting it

**Do not build OP-1 as written.** OP-2 and OP-3 are unaffected; start with those. What
follows replaces OP-1's premise; the spec body has not been rewritten yet.

The kpool epilogue OP-1 asks for exists as **Triton** kernels, not CUDA ones, and Triton
runs on this hardware through `triton-ascend`. We measured all of them on the target
machine:

| outcome | count | detail |
|---|---|---|
| compile, run, **bit-exact** vs a torch transcription of the kernel body | **7 of 10** | including the ragged layout, the write plan, the tail append, the tail scatter, and the pool-slot pack/select |
| fail to compile — **fp8** | 4 | `bishengir-compile` rejects the e4m3 conversion outright; `x.to(torch.float8_e4m3fn)` faults on device too. **Atlas A3 cannot express fp8.** With the fp8 store stood in for bf16, three of these compile, run and match the reference across the whole computation |
| **triton-ascend codegen defect** | 1 | `_hadamard128`'s 7-stage rotation faults with a UB out-of-bounds in some contexts. It runs correctly *inside* three of the kernels and faults standalone, so it is a lowering bug, not a missing language feature |

So the top-k, the expand, the tail append and the plan/layout machinery **already work on
this hardware**. Two things do not, and neither is the operator this document specified.

**The real gap is the index-K cache dtype.** kpool stores compressed index keys in fp8,
and A3 has no fp8. **DeepSeek-V4 already solved this on Ascend by using int8 instead** —
see `ascend_dsv4_backend.py:685`, which sets `compressor.li_kv_dtype = "int8"`, and the
int8 branches at `:469-470` and `:597-598`. Whether GLM's kpool can follow the same route
is the question to settle before commissioning anything.

**The Hadamard defect is probably moot for us.** The 128-point rotation is an orthonormal
matrix applied to both q and k, so the dot product it feeds is unchanged and it can be
dropped entirely on a bf16 indexer. If it is dropped, the codegen bug stops mattering; it
is still worth reporting upstream to triton-ascend.

**Scoring is separate and does not come along.** `deep_gemm.fp8_paged_mqa_logits` and its
tilelang variant are CUDA, not Triton. `npu_quant_lightning_indexer` may already cover it
(§2a below).

We will rewrite OP-1 once the int8 question is settled.

---

## 1. Context for someone who has never seen this project

**SGLang** is an LLM inference server. We are bringing up **GLM-5.3-Flash** on
**Atlas A3** (SoC `Ascend910_9362`, 16 die). The model is a 45-layer hybrid: 34 layers of
linear attention plus **11 layers of "DeepSeek sparse attention" (DSA)**.

DSA is the part that creates the work. Instead of attending to every key, each DSA layer
runs a small **indexer** that scores the keys and picks the best `index_topk = 2048` of
them; attention then runs only over those. GLM's indexer scores keys in **pools of
`index_kpool = 4`**, so the selection is really "pick the best `2048 / 4 = 512` pools,
then expand each pool back into its 4 token indices". That expansion, plus a few
bookkeeping steps around it, is **OP-1** — and it runs on all 11 DSA layers on **every
decode step**, so it is the one that matters.

The other two are smaller shape/feature gaps where an operator that already exists on
Ascend does not quite cover GLM's configuration:

* GLM's index-K normalisation is a **LayerNorm** (mean-subtract + bias); the vendor's
  fused `compressor` only knows RMSNorm. → **OP-2**
* GLM has **`qk_rope_head_dim = 0`** — no rotary half at all — and the fused MLA
  cache-write op rejects a zero-width rope. → **OP-3**

Relevant model config (`/mnt/workspace/models/GLM-5.3-Flash-BF16/config.json`, under
`text_config`; the weights are already dequantised to BF16):

```
index_topk = 2048        index_kpool = 4          index_n_heads = 32
index_head_dim = 128     index_kpool_compress = true
index_kpool_always_select_tail = true
qk_nope_head_dim = 256   qk_rope_head_dim = 0     v_head_dim = 256
kv_lora_rank = 512       hidden_size = 4096       45 layers (34 linear + 11 DSA)
```

## 2. The operators

| # | Operator | Kind | Priority | One-line rationale |
|---|---|---|---|---|
| **1** | [kpool pool→raw expand + tail append (the top-k epilogue)](specs/op1_kpool_topk_transform.md) | **NEW kernel** | **highest** | The DSA indexer's inner loop. 11 layers, every decode step. **Scope narrowed — read §2a below before starting:** the pooled scoring and the group top-k already exist on Ascend and are in DeepSeek-V4 production. What is missing is the epilogue. |
| **2** | [`compressor` with a LayerNorm variant](specs/op2_compressor_layernorm.md) | extend a vendor op | high | The vendor op fuses RMSNorm only; GLM's index-K norm is a true LayerNorm. Blocks the compressed index-K path on the same 11 layers. |
| **3** | [`npu_kv_rmsnorm_rope_cache` accepting rope width 0](specs/op3_kv_norm_rope_cache_rope0.md) | extend a torch_npu op | medium | GLM has no rotary half. Measured: rope=0 raises on both v1 and v2. A software fallback exists but costs an extra kernel + an extra pass over `[T,512]` on the 11 sparse-attention layers. |
| ~~4~~ | ~~bf16-output `DequantSwigluClampQuant`~~ | — | **WITHDRAWN — do not build** | `torch_npu.npu_clipped_swiglu` already ships in the target runtime, supports A3, and takes bf16 in → bf16 out. Measured on device: with `alpha=1.0, limit=10.0, bias=0.0, interleaved=False` it is **bit-exact** with the reference. Its defaults are gpt-oss values, but every one is a parameter. See [`specs/op4_optional_swiglu_bf16.md`](specs/op4_optional_swiglu_bf16.md). |

**Do them in that order.** OP-1 is most of the value; OP-2 and OP-3 are small deltas on
code that already exists.

### 2a. OP-1's scope is narrower than a from-scratch top-k

`torch.ops.custom.npu_quant_lightning_indexer(..., cmp_ratio=4, sparse_count=..., sparse_mode=3)`
is already in DeepSeek-V4 production (`ascend_dsv4_backend.py:729-750`), and
`aclnnQuantLightningIndexer` exposes `cmpRatio` / `sparseCount` / `sparseIndicesOut` /
`sparseValuesOut`. Its own bf16 reference path (`ascend_dsv4_backend.py:618-682`) scores
over `seq // ratio` pooled entries and calls `.topk(min(index_topk, seq // ratio))`, then
pads with `-1`. So **at `cmp_ratio=4` the pooled MQA scoring and the group top-k both
already exist on Ascend and return pool-level indices with `-1` padding.**

What is genuinely missing is the **epilogue**: expand each selected pool by `index_kpool`
in slot order, truncate to `min(length * P, topk)`, append the visible tail, apply the
page-table / offset map, and fill with `-1`. Scope the work as that epilogue and as
composing it with the existing indexer — not as a new radix-select top-k.

Two consequences the spec body has not yet been rewritten for:

- The score tensor that the current OP-1 interface takes as input is produced today only
  by CUDA-only kernels (`deep_gemm.fp8_paged_mqa_logits` / the tilelang variant,
  `dsa_indexer_kpool.py:895-920`). On Ascend there is no producer for it, and the Ascend
  op that *could* produce it fuses scoring and selection and emits indices rather than
  scores. **Settle the composition with us before implementing to the current signature.**
- On the decode path the logits are passed **uncleaned** (`clean_logits=False`,
  `dsa_indexer_kpool.py:907` and `:918`; extend passes `True`). Values outside
  `[row_starts[b], row_starts[b] + lengths[b])` are garbage and may be NaN. Any
  implementation that vectorises to an aligned tile **must not let the padding reach a
  max-reduction or a mask**.

## 3. How the pieces fit together

```
                       ┌──────────────────────── one DSA layer, one decode step ───┐
   hidden states ──►   │  wk projection                                            │
                       │        │                                                  │
                       │        ▼                                                  │
                       │   LayerNorm(128)  ◄── OP-2: the vendor `compressor` fuses  │
                       │        │               this norm, but RMSNorm only        │
                       │        ▼                                                  │
                       │   pool-compress + fp8 quant + write index cache           │
                       │        │                                                  │
                       │        ▼                                                  │
                       │   pooled MQA logits  ->  score[B, S]  (one score per pool) │
                       │        │                                                  │
                       │        ▼                                                  │
                       │  ┌──────────────────────────────────────────────────────┐ │
                       │  │ OP-1  kpool_topk_transform                           │ │
                       │  │   pick best 512 pools  ->  expand x4  ->  map through│ │
                       │  │   page table / offset  ->  append tail  ->  pad -1   │ │
                       │  └──────────────────────────────────────────────────────┘ │
                       │        │  int32 [B, 2051]                                 │
                       │        ▼                                                  │
                       │   sparse attention gathers exactly these keys             │
                       └───────────────────────────────────────────────────────────┘

   separately, on the 11 sparse-attention layers:
        MLA latent cache write ──► OP-3: RMSNorm + RoPE + scatter, fused.
                                   GLM's rope half is 0 wide; the op rejects that.
```

OP-1's output is consumed directly by sparse attention as a gather index list, which is
**permutation-invariant over keys** — that is why the order of the selected pools is not
part of OP-1's contract, and why its acceptance is set-based (see below).

## 4. What is in this directory

```
README.md                    <- you are here
ENVIRONMENT.md               <- how to build the env and how to run the tests
ACCEPTANCE.md                <- the two acceptance methods, and why a fixed 1e-3 is wrong
run_tests.sh                 <- one-line test driver

specs/
  op1_kpool_topk_transform.md          <- full interface + exact semantics + edge cases
  op2_compressor_layernorm.md
  op3_kv_norm_rope_cache_rope0.md
  op4_optional_swiglu_bf16.md          <- OPTIONAL

reference/                   <- pure torch, CPU, no dependencies beyond torch.
  kpool_topk_transform.py      THESE ARE THE DEFINITION OF CORRECT.
  fused_norm.py
  kv_norm_rope_cache.py
  swiglu_clamp.py
  tolerance.py               <- the two-reference (noise floor) method
  backend.py                 <- the ONE file that names the delivered NPU operators

tests/                       <- pytest; same files target reference or NPU
  test_op1_kpool_topk_transform.py
  test_op2_fused_norm.py
  test_op3_kv_norm_rope_cache.py
  test_op4_swiglu_clamp_bf16.py
```

Every factual claim in the specs cites `file:line` in either this repository or the
installed vendor packages. Where something could not be pinned down, the spec says so
under a "Not pinned down" heading instead of guessing.

## 5. Running the tests

Needs only `torch` (CPU is fine) and `pytest` — no sglang, no torch_npu, no CANN:

```bash
./run_tests.sh                       # against the torch reference
GLM53_OP_BACKEND=npu ./run_tests.sh  # against the delivered operators
```

**Verified**: 60 passed, 1 skipped in 6.3 s under torch 2.7.1+cpu / pytest 9.0.3 on the
project machine. (The skip is an OP-3 case that depends on an unverified RoPE convention;
see [specs/op3](specs/op3_kv_norm_rope_cache_rope0.md) §8.) Details in
[ENVIRONMENT.md](ENVIRONMENT.md).

`reference/backend.py` is the only place the delivered operator names appear. If a
delivered name or signature differs from the spec, change it there — the test bodies are
device-agnostic and need no edits.

## 6. Acceptance, in two sentences

For floating-point outputs we do **not** use a fixed threshold: we run the reference in
fp32 and again in bf16, take the distance between those two results as that case's noise
floor, and accept an implementation that lands within it. (A measured reason: the KDA
layer-0 golden at seq=64 has an fp32-vs-bf16 relative error of **1.06e-2**, so a 1e-3
gate would reject a bit-perfect implementation.)

For OP-1, which returns **indices**, acceptance is **set equality of the selected pools
and exact equality of the selected-score multiset — never index-by-index equality**:
ties at the 512-th boundary legitimately differ between implementations, the selection
order is unspecified, and there is no CUDA device on this machine to produce reference
indices anyway. Full statement in [ACCEPTANCE.md](ACCEPTANCE.md).

## 7. One correction to how this was originally framed

The brief said the paged **decode** path passes `page_table_row_index` to OP-1. It is
actually the ragged **extend / chunked-prefill** path that does
(`dsa_indexer_kpool.py:1017-1043`, fed from `kpool_plan.py:414-420`); the decode call
site at `dsa_indexer_kpool.py:921-929` leaves it `None`. **The operator must still
support it** — prefill uses the same kernel — but it is not on the per-decode-step hot
path. There is a real consequence for the implementation: in that mode the page table is
the entire `req_to_token` pool, whose row stride is context-scale, so page-table
addresses must be computed in 64-bit. Details in
[specs/op1](specs/op1_kpool_topk_transform.md) §4.
