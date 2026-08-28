# GLM-5.3-Flash on Ascend — operator handoff package

**This package is empty: all four requests are withdrawn, and no operator work is
asked for.** It is kept as a record of what was checked and why each request fell.

It was a **self-contained work order for an Ascend kernel team**. It
specified operators, gave a pure-torch reference
implementation of each that *is* the definition of correct, ships an executable test
suite, and states the acceptance criteria. It is meant to be implementable without
asking us anything.

Everything here is new; nothing outside this directory was changed.

> The surrounding project documents ([`../PLAN.md`](../PLAN.md),
> [`../SETUP.md`](../SETUP.md)) are in Chinese. This package is in English; the code and
> the specs are the parts that matter and they are language-neutral.

---

## 0. Where this stands

Four operators were specified. **All four are withdrawn.** Read this section before
anything else — the spec bodies are kept only as a record.

| | status |
|---|---|
| ~~OP-1~~ kpool | **withdrawn** — the index cache moves fp8 → **bf16**, a repo-side dtype change with no operator behind it. `torch_npu.npu_lightning_indexer` reads bf16 keys, takes GLM's 32 index heads, and already ships. See [`specs/op1_kpool_topk_transform.md`](specs/op1_kpool_topk_transform.md). |
| ~~OP-2~~ compressor LayerNorm | **withdrawn** — GLM never calls the vendor `compressor`, and its index-K LayerNorm was never fused into an operator. |
| ~~OP-3~~ `kv_rmsnorm_rope_cache` at rope 0 | **withdrawn** — the operator does reject a zero-width rope, and that measurement stands; GLM simply never calls it. Both call sites are unreachable. See [`specs/op3_kv_norm_rope_cache_rope0.md`](specs/op3_kv_norm_rope_cache_rope0.md). |
| ~~OP-4~~ bf16 swiglu | **withdrawn** — `torch_npu.npu_clipped_swiglu` already ships and is bit-exact once its four parameters are passed. |

Nothing is left. **Atlas A3 has no fp8** — `bishengir-compile` cannot lower the e4m3 conversion and the
torch side faults on device — so kpool's fp8 index cache cannot run as written. That was
OP-1. It is no longer an operator request: storing the cache in **bf16** makes three of
the four failing Triton kernels compile and match, and `npu_lightning_indexer` consumes
bf16 keys directly at GLM's 32 heads, fusing the top-k. The formerly open question —
*what computes the indexer logits on Ascend* — is answered in OP-1 §5.

A note on how this package's hit rate went: **five inferred gaps, five falsified.** Four
operators turned out to already exist on the target, hidden behind default parameters, a
same-named operator in another namespace, or a near-identical operator name
(`npu_lightning_indexer` vs `npu_quant_lightning_indexer`). The fifth, OP-3, was a real
operator limitation attached to a call site GLM never reaches.

Two lessons, pointing opposite ways. **Run the operator before writing the request** --
that is what killed the first four. And **check that the model actually calls it** -- that
is what killed the fifth. A third, from a near miss on OP-3: GLM's checkpoint carries no
`rope_scaling`, which makes the yarn branch look dead and the request look withdrawable
for the wrong reason; sglang's own config class synthesizes a default, so the flag is
true. Reading the config file alone gives the opposite answer to constructing the config.

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

See [§0](#0-where-this-stands) for status and [`specs/`](specs/) for each request.
All four specs are withdrawn; what computes the indexer logits on Ascend is answered in
[`specs/op1_kpool_topk_transform.md`](specs/op1_kpool_topk_transform.md) §5.

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
