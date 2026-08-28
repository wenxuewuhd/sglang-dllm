# GLM-5.3-Flash on Ascend — operator handoff package

This directory is a **self-contained work order for an Ascend kernel team**. It
specifies three operators (plus one optional optimisation), gives a pure-torch reference
implementation of each that *is* the definition of correct, ships an executable test
suite, and states the acceptance criteria. It is meant to be implementable without
asking us anything.

Everything here is new; nothing outside this directory was changed.

> The surrounding project documents ([`../PLAN.md`](../PLAN.md),
> [`../SETUP.md`](../SETUP.md)) are in Chinese. This package is in English; the code and
> the specs are the parts that matter and they are language-neutral.

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

## 2. The three operators

| # | Operator | Kind | Priority | One-line rationale |
|---|---|---|---|---|
| **1** | [kpool fused group-top-k + pool→raw expand + tail append](specs/op1_kpool_topk_transform.md) | **NEW kernel** | **highest** | The DSA indexer's inner loop. 11 layers, every decode step. No NPU equivalent; splitting it costs a 2048-wide top-k plus two gather passes per layer per step. |
| **2** | [`compressor` with a LayerNorm variant](specs/op2_compressor_layernorm.md) | extend a vendor op | high | The vendor op fuses RMSNorm only; GLM's index-K norm is a true LayerNorm. Blocks the compressed index-K path on the same 11 layers. |
| **3** | [`npu_kv_rmsnorm_rope_cache` accepting rope width 0](specs/op3_kv_norm_rope_cache_rope0.md) | extend a torch_npu op | medium | GLM has no rotary half. Measured: rope=0 raises on both v1 and v2. A software fallback exists but costs an extra kernel + an extra pass over `[T,512]` on all 45 layers. |
| 4 | [bf16-output `DequantSwigluClampQuant`](specs/op4_optional_swiglu_bf16.md) | extend a vendor op | **OPTIONAL — not a blocker** | The op's `swiglu_mode=1` already matches GLM's clamped SwiGLU exactly, but its output is unconditionally int8, so the bf16 path keeps a separate pre-clamp pass. Pure optimisation. |

**Do them in that order.** OP-1 is most of the value and all of the novelty; OP-2 and
OP-3 are small deltas on code that already exists; OP-4 is a nice-to-have.

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

   separately, on all 45 layers:
        MLA latent cache write ──► OP-3: RMSNorm + RoPE + scatter, fused.
                                   GLM's rope half is 0 wide; the op rejects that.
   separately, in the MoE experts:
        gate/up ──► OP-4 (optional): clamped SwiGLU with a bf16 output.
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
