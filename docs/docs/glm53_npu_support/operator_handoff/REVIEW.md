# Review findings

An independent pass over this package, plus a sweep of the GLM-5.3 branch for gaps the
package missed. Everything below was either executed on the target hardware or traced to
`file:line`; where it is neither, it says so.

The package body has been corrected for the two items that would have wasted the most
time (OP-4 withdrawn, OP-1's scope narrowed). The remaining items are recorded here
rather than rewritten into the specs, so the specs stay a stable work order.

---

## Corrected in the package

**OP-4 is withdrawn.** `torch_npu.npu_clipped_swiglu` already ships in the target runtime
and is **bit-exact** with the reference at `alpha=1.0, limit=10.0, bias=0.0,
interleaved=False` (verified twice, independently, on device; the same call at its default
parameters differs by 156, which is what made the defaults look like a different
operator). No kernel work is needed for GLM's clamped SwiGLU.

**OP-1 is an epilogue, not a from-scratch top-k.** `npu_quant_lightning_indexer` at
`cmp_ratio=4` already does the pooled scoring and the group top-k and is in DeepSeek-V4
production. See README §2a for what is actually missing and for the two composition
questions to settle with us first.

**OP-3 covers 11 layers, not 45.** Only the sparse-attention layers hold an MLA latent
cache; the 34 linear-attention layers never enter that path.

---

## Open against the specs

- **The decode path passes uncleaned logits.** `clean_logits=False` at
  `dsa_indexer_kpool.py:907` and `:918`; extend passes `True`. Values outside
  `[row_starts[b], row_starts[b] + lengths[b])` are garbage and may be NaN. Recorded in
  README §2a; the OP-1 spec body still only says the window "is read".
- **OP-3's tests use convention A while its spec recommends convention B.** Six of the
  seven OP-3 tests pass `cos=sin=k_cache=None`, and one hard-asserts `k_out is None`. A
  team following the spec's own recommendation fails most of the OP-3 suite. Reconcile
  before the team starts.
- **OP-2 §7 rests on unverified applicability.** It admits we have not confirmed SGLang
  routes GLM's kpool compressor through the vendor op. Resolve that before scheduling it
  as "high priority, blocks 11 layers".
- **`ACCEPTANCE.md`'s ±0.5 pp end-to-end gate is far too tight.** The recorded single-round
  binomial standard error is about ±3.2 pp and the three-round sample SD 1.8–2.8 pp. A
  ±0.5 pp gate would reject correct work roughly half the time. Use the three-round mean
  and a band derived from the measured SD.
- **`run_tests.sh` defaults to the system python**, which carries torch 2.7.1 — a
  different torch_npu major from the documented target — while `.venv-glm53` has no
  pytest and this document forbids installing into it. `GLM53_OP_BACKEND=npu` would
  silently test the wrong runtime. Resolve before anyone runs the suite against hardware.

## Test suite

Sixteen wrong "delivered operators" were injected through `reference/backend.py`; one
survived: implementing LayerNorm as `centered / (sqrt(var) + eps)` instead of
`centered / sqrt(var + eps)`. At unit variance the two differ by ~5e-7, below the bf16
floor, and every LayerNorm case used `randn` inputs.

`test_layer_norm_small_variance_pins_eps_placement` closes it. Verified by injecting that
exact mutant into the backend only: the previous cases pass it (20 passed), the new cases
catch it (4 failed).

Two structural weaknesses remain, both worth knowing when reading a green run:

- In the default backend most tests compare the reference against itself, so "all passed"
  says little about whether the reference is right. Only
  `test_reference_agrees_with_decomposed_path` and `test_rope0_matches_plain_rms_norm`
  carry independent force — and the former shares `torch.topk` tie-breaking and the
  `_transform` helper with the path it checks, so it corroborates the expand/tail
  structure rather than the selection.
- **The combination production always takes is untested**: the ragged extend site passes
  `page_table`, `page_table_row_index` and `row_starts` together
  (`dsa_indexer_kpool.py:1035-1043`). The suite covers them only pairwise. That is also
  where 64-bit page addressing and the windowed score interact.

Also uncovered: `S == 0` with `B > 0`, and garbage outside the score window.

---

## Gaps this package does not cover, ranked

These are things that run as plain torch on the NPU today and cost real time. Only the
first three are worth acting on soon, and **none of the top three is kernel work** — the
operators already exist.

| # | Item | What runs instead | Frequency | Verdict |
|---|---|---|---|---|
| 1 | **mHC pre/post** (`mhc.py:1662-1674`, `:1701-1702`) | ~140 eager kernels per `hc_pre`, ~120 of them a 19-iteration Sinkhorn loop on `[s,4,4]`; `hc_post` materialises a `[s,4,4,4096]` fp32 temp — 16x its own output | **90 calls per forward** (2 x 45 layers), ~12,600 launches | **Largest item, and not kernel work.** `npu_hc_pre` / `npu_hc_post` exist and are wired for DeepSeek-V4 (`deepseek_v4.py:1905-1915`, `:2028-2029`); `mhc.py`'s flat entry points simply have no `_is_npu` branch. |
| 2 | **KDA prefill conv1d** (`ascend_kda_backend.py:326-360`) | `F.conv1d(groups=dim)` over a dense ragged-to-padded buffer, plus a full bf16 to fp32 round-trip of the activation | 102 calls per forward (34 layers x 3) | Repo-side fix: the shared backend makes one packed call; the Ascend split into three is extend-only and unexplained. The existing `npu_fused_causal_conv1d` does **not** fit (weight K fixed at 3, GLM needs 4). |
| 3 | **MoE SwiGLU clamp** (`deepseek_v2.py:464-470`, `npu/moe/activation.py:78-122`) | two strided clamps plus a full-size `cat`, then `npu_swiglu` — four kernels where one suffices | per layer per forward | Same `npu_clipped_swiglu` that retires OP-4. Largest in prefill. |
| 4 | DeepEP-normal device-to-host sync (`moe_runner/ascend.py:270-274`) | `.sum(0).cpu().numpy().tolist()` then straight back to device | 42x per prefill forward; decode is clean | Real but the fix lives in the third-party `deep_ep` wheel. Profile first. |
| 5 | NoPE unfused split + RMSNorm (`deepseek_v2_attention_mla_npu.py:405-433`) | separate splits and two RMSNorms; also a `q.clone()` of `[T,1536]` that appears dead | 11 layers per forward | Same root cause as OP-3; scope together. |

Three of the six env flags the launch script sets to force torch paths are **no-ops on
this model** — `SGLANG_OPT_USE_FUSED_HASH_TOPK` (GLM has no hash layers, and
`hash_topk.py:232` already excludes NPU), `SGLANG_OPT_FP8_WO_A_GEMM` (DeepSeek-V4 only),
`SGLANG_OPT_BF16_FP32_GEMM_ALGO` (its only consumer is unreachable from GLM). They read
as evidence of fallbacks that are not there.

## Two accuracy bugs found in passing

Not performance, but they will block the per-module comparisons:

- **The routed-expert path silently drops `swiglu_limit`.** `moe_runner/ascend.py:114-118`
  reads `config.gemm1_clamp_limit`, which is `None` for GLM, and never
  `config.swiglu_limit`. So on the same layer the shared expert clamps and the routed
  experts do not.
- **The NPU router GEMM runs in bf16** (`deepseek_v2.py:567-568`) where CUDA accumulates
  in fp32, although GLM sets `moe_router_dtype: float32`.
