# OP-3 — `npu_kv_rmsnorm_rope_cache`: accept rope width 0

**Status:** EXTEND an existing torch_npu op. **Priority: medium** (structural, but a
workable software fallback exists — see §7).

Reference implementation: [`../reference/kv_norm_rope_cache.py`](../reference/kv_norm_rope_cache.py)
Tests: [`../tests/test_op3_kv_norm_rope_cache.py`](../tests/test_op3_kv_norm_rope_cache.py)

---

## 1. What is missing

`torch_npu.npu_kv_rmsnorm_rope_cache` is one fused op that does, for MLA:

1. RMSNorm over the `kv_lora_rank` half of the latent cache line,
2. RoPE over the `qk_rope_head_dim` half,
3. a scatter of (1) into `ckv_cache` and (2) into `k_rope_cache` at `out_cache_loc`.

Call site: `python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py:93-108`.

GLM-5.3-Flash has **`qk_rope_head_dim = 0`** (`/mnt/workspace/models/GLM-5.3-Flash-BF16/config.json`,
`text_config`: `qk_nope_head_dim=256`, `qk_rope_head_dim=0`, `v_head_dim=256`,
`kv_lora_rank=512`), so steps (2) and (3b) do not exist and the tensors that carry them
have width 0.

The op's registered schema makes `cos`, `sin` and `k_cache` **non-optional**
(`.venv-glm53/lib/python3.12/site-packages/torch_npu/csrc/aten/npu_native_functions_by_codegen.yaml:186`):

```
npu_kv_rmsnorm_rope_cache(Tensor kv, Tensor gamma, Tensor cos, Tensor sin,
                          Tensor index, Tensor k_cache, Tensor ckv_cache, *,
                          Tensor? k_rope_scale=None, Tensor? c_kv_scale=None,
                          Tensor? k_rope_offset=None, Tensor? c_kv_offset=None,
                          Tensor? v=None, float epsilon=1e-5,
                          str cache_mode='Norm', bool is_output_kv=False)
                          -> (Tensor, Tensor, Tensor, Tensor)
```

`..._v2` and `..._v2_functional` (`:188`, `:190`) have the same three required tensors.

### Measured behaviour

**Recorded result, taken rather than re-measured.** All 16 dies are in use by a running
SGLang server (`npu-smi info` shows ~56 GB/65 GB HBM on every die and a
`sglangschedul` process holding it), so the probe was not re-run for this document.

From `docs/docs/glm53_npu_support/PLAN.md:172` (row C4), produced by
`docs/docs/glm53_npu_support/probe/p0_6_rope0.py` on this machine:

| rope width | `npu_kv_rmsnorm_rope_cache` | `npu_kv_rmsnorm_rope_cache_v2` |
|---|---|---|
| 64 | OK | OK |
| **0** (zero-width `cos`/`sin`/`k_cache`) | **RuntimeError** | **RuntimeError** |

The error surfaces from `aclnnKvRmsNormRopeCache*`.

## 2. Proposed extension

Support `rope_dim == 0`, defined as `kv.shape[-1] - gamma.shape[0] == 0`, under **both**
calling conventions:

| Convention | `cos` / `sin` | `k_cache` | `k_pe` output |
|---|---|---|---|
| **A — omitted** (preferred; this is what SGLang would naturally emit, since there is nothing to build `cos`/`sin` from) | `None` | `None` | `None` |
| **B — zero-width** (what `p0_6_rope0.py` passes) | `[T,1,1,0]` | `[num_blocks, block_size, 1, 0]` | `[T,1,1,0]` |

Convention A needs the schema relaxed to `Tensor? cos, Tensor? sin, Tensor? k_cache`.
If only one convention can be delivered, **deliver B** — it needs no schema change and
SGLang can allocate the zero-width tensors — but say so, because the SGLang-side change
differs.

Everything else is unchanged. `epsilon`, `cache_mode`, `is_output_kv`, the four
scale/offset tensors and `v` keep their meanings.

## 3. Exact semantics at `rope_dim == 0`

```
T    = kv.shape[0]                      # kv is [T, 1, 1, kv_lora_rank]  (BNSD)
L    = gamma.shape[0]                   # kv_lora_rank = 512 for GLM

kv_a = kv[:, 0, 0, :L] / sqrt( mean(kv²) + epsilon ) * gamma      # fp32 accum,
                                                                  # cast back to kv.dtype
ckv_cache.view(-1, 1, L)[index[t]] = kv_a[t]     for every t with index[t] >= 0

k_pe     = None (convention A) or an empty [T,1,1,0] tensor (convention B)
k_cache  untouched / absent
```

`index` is `[T]` int64 and holds **flat** slot ids into the paged cache, i.e.
`block_id * block_size + offset_in_block`. SGLang passes
`forward_batch.out_cache_loc.to(torch.int64)` (`deepseek_v2_attention_mla_npu.py:98`).

Return value stays the 4-tuple `(k_cache, ckv_cache, k_pe, kv_a)` when
`is_output_kv=True`; SGLang destructures it as `_, _, k_pe, kv_a`
(`deepseek_v2_attention_mla_npu.py:93`).

## 4. Cache layout

The reference and tests define **`cache_mode="PA_BNSD"`** only:

* `ckv_cache`: `[num_blocks, block_size, 1, kv_lora_rank]`
* `k_cache`  : `[num_blocks, block_size, 1, rope_dim]`  (absent at `rope_dim == 0`)

`cache_mode="PA_NZ"` is the NZ-format variant SGLang selects when `is_fia_nz()`
(`deepseek_v2_attention_mla_npu.py:106`). **We did not pin down its byte layout**, so the
tests do not cover it — but the extension must apply to it as well, since the same call
site chooses between the two at runtime.

## 5. Edge cases

| Case | Required behaviour |
|---|---|
| `T == 0` | No-op; caches unmodified; `kv_a` is `[0, L]`. Must not fault. |
| `index[t] == -1` | **Skip that token entirely** — no write to either cache. This is how SGLang pads a captured-graph batch. **This is a request, not something we measured**: we did not verify the current op's behaviour on `-1`. If it currently writes to slot `-1` (or faults), say so and we will mask on the SGLang side instead. |
| Duplicate `index` values | Last writer wins; SGLang never produces duplicates. |
| `gamma` dtype | Same as `kv` (bf16 in production). The probe passes bf16. |
| `epsilon` | `kv_a_layernorm.variance_epsilon`; `1e-6` for GLM's `kv_a_layernorm`. |

## 6. Acceptance criteria

1. `rope_dim == 0` must **not raise** — under whichever of conventions A/B is delivered.
2. `kv_a` and the written `ckv_cache` rows must be within the two-reference noise floor
   of `kv_norm_rope_cache_ref` (see [`../ACCEPTANCE.md`](../ACCEPTANCE.md)).
3. Cache slots not named by `index` must be **bit-unchanged**.
4. `rope_dim == 64` must still behave exactly as it does today (regression).

Criterion 4 is `test_rope_nonzero_regression`, which is **skipped by default**: this
package's RoPE convention (`interleaved_rope_ref`, even/odd pairing) is a best-effort
restatement and was **not verified** against the vendor op — see §8. Enable it with
`GLM53_TEST_ROPE_CONVENTION=1` only once that convention is confirmed, or replace the
reference's rope with the confirmed one.

## 7. If this operator cannot be extended

There is a software fallback, already scoped as work item D2/P3.3 in
`docs/docs/glm53_npu_support/PLAN.md:229` and `:492`: split the call into
`torch_npu.npu_rms_norm` plus a hand-written scatter (`reshape_and_cache` /
`index_copy_`). It costs an extra kernel launch and an extra pass over the
`[T, 512]` tensor per layer per step across all 45 layers. That is why this is a
"medium" and not a "low": the fallback is real but it is not free.

## 8. Not pinned down

* **The RoPE convention inside the op.** SGLang applies `npu_interleave_rope` to `q_pe`
  with the same `cos`/`sin` (`deepseek_v2_attention_mla_npu.py:85-89`), which suggests
  the fused op uses interleaved (even/odd pair) RoPE for `k_pe` too, and that is what
  `interleaved_rope_ref` implements. **We did not verify it.** It does not affect the
  `rope_dim == 0` contract, where RoPE is a no-op.
* **`cache_mode="PA_NZ"` layout** (§4).
* **Current behaviour on `index == -1`** (§5).
