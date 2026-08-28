# OP-2 — `compressor`: add a LayerNorm variant to the fused norm

**Status: WITHDRAWN — do not build this.**

GLM-5.3-Flash never calls the vendor `compressor`. That operator is referenced twice in
the tree and both are the DeepSeek-V4 path (`ascend_dsv4_backend.py:401`, and a docstring
in `dsv4_memory_pool.py:407`). GLM's kpool compresses through the Triton kernels in
`kpool_fp8_index.py`, reached via `IndexerKPool._compress_write_*` and the pool's
`kpool_*_update_index_cache`.

The LayerNorm concern does not survive either: GLM's index-K norm is a plain `LayerNorm`
module (`dsa_indexer_kpool.py:146`) applied to the key *before* it enters the compression
path (`:625`, `:645`, `:666`). It was never fused into an operator, so there is nothing to
extend. "The vendor op fuses RMSNorm only" was true of DeepSeek-V4's operator and not a
statement about GLM.

What genuinely blocks GLM's compression is fp8: the compress-and-write Triton kernels do
not compile on A3 because it cannot express the e4m3 conversion. That is the same question
as OP-1's, and it is one question rather than two.

The sections below are kept as a record of what was traced.


**Status:** EXTEND an existing vendor op. **Priority: high** (blocks the GLM DSA
indexer's compressed index-K path; 11 layers).

Reference implementation: [`../reference/fused_norm.py`](../reference/fused_norm.py)
Tests: [`../tests/test_op2_fused_norm.py`](../tests/test_op2_fused_norm.py)

---

## 1. What is missing

The vendor `Compressor` op fuses a norm, but **only RMSNorm**, and it has no bias input.
Verified in the installed package
(`/mnt/workspace/y00359136/work/glm53_dev/env/opp_custom/vendors/custom_transformer/`):

* `op_impl/ai_core/tbe/custom_transformer_impl/ascendc/compressor/arch22/compressor_block_vec_perf.h:1256-1260`
  calls `MultRowRmsNorm(...)` unconditionally — there is no mode switch.
* `.../compressor/arch22/rms_norm.h:37-86` is the entire formula:
  `Mul` → `RowSum` → `Muls(1/N)` → `Adds(eps)` → `Sqrt` → `RowDivs` → `MatMulVec(gamma)`,
  i.e. `y = x / sqrt(mean(x²) + eps) * gamma`. No mean subtraction, no bias.
* `op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json`, op
  `Compressor`: `input5 = norm_weight` (required, float32); there is **no** bias input,
  and the attr list is
  `rope_head_dim,cmp_ratio,coff,norm_eps,rotary_mode,cache_mode,state_cache_stride_dim0`
  — no norm mode.
* `op_proto/inc/compressor_proto.h` confirms the same eight `INPUT` / four
  `OPTIONAL_INPUT` / seven attrs.
* `op_api/include/aclnnop/aclnn_compressor.h` is the matching aclnn signature.

GLM-5.3-Flash's index-K norm is a **true LayerNorm**:

* `python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py:146` —
  `self.k_norm = LayerNorm(self.head_dim, dtype=torch.float32)`, with
  `head_dim = index_head_dim = 128`.
* `python/sglang/srt/layers/layernorm.py:1006-1020` — `forward_native` upcasts to fp32
  and calls `F.layer_norm(x, (hidden_size,), weight=weight, bias=bias, eps=eps)`.
* Applied at `dsa_indexer_kpool.py:625, :645, :666` (`key = self.k_norm(key)`).

DeepSeek-V4 takes the RMSNorm branch instead
(`python/sglang/srt/layers/attention/dsa/dsa_indexer.py:287-295` picks `RMSNorm` when
`config.index_k_norm_type == "rms"`), which is why the vendor op only ever needed one.
The NPU call site that must keep working is
`python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py:401-420`.

## 2. Proposed extension

Two additions to `Compressor`, both backward compatible:

| Kind | Name | Type | Default | Meaning |
|---|---|---|---|---|
| new **optional input** (after `norm_weight`) | `norm_bias` | float32, `[head_dim]`, ND | absent | LayerNorm β. Must be absent when `norm_mode == 0`. |
| new **optional attr** | `norm_mode` | int | `0` | `0` = RMSNorm (today's behaviour, bit-for-bit), `1` = LayerNorm. |

Appending an optional input and an optional attr keeps the existing aclnn call ABI
usable by callers that pass neither; if the toolchain forbids inserting `norm_bias`
mid-list, append it after `start_pos` instead — SGLang binds by keyword either way.

Resulting aclnn signature (delta only):

```c
aclnnCompressorGetWorkspaceSize(
    ..., const aclTensor *normWeight,
         const aclTensor *normBiasOptional,      /* NEW */
    ..., int64_t normMode,                       /* NEW, default 0 */
    ...);
```

Torch binding: `torch.ops.npu.compressor(..., norm_bias=None, ..., norm_mode=0)`.

## 3. Exact math

Let `D = head_dim` (128 for GLM) and let the row be the `D` values the op already feeds
into `MultRowRmsNorm`. Accumulate in fp32 — the op already keeps this tensor in fp32.

```
norm_mode == 0  (unchanged):
    y = x / sqrt( mean(x²) + eps ) * gamma

norm_mode == 1  (new):
    mu  = mean(x)
    xc  = x - mu
    var = mean(xc²)                    # BIASED, denominator D, matching F.layer_norm
    y   = xc / sqrt( var + eps ) * gamma + beta
```

Both use `sqrt` + divide rather than `rsqrt` + multiply, matching `rms_norm.h`'s
`Sqrt` → `RowDivs` sequence, so mode 0 stays bit-identical to today.

`eps` is the existing `norm_eps` attr. SGLang passes
`compressor.norm.variance_epsilon` (`ascend_dsv4_backend.py:417`); for GLM's index
LayerNorm the default is `1e-6` (`python/sglang/srt/layers/layernorm.py:978`).

## 4. Required test hook

The norm block is buried inside a large fused op, so it cannot be unit tested through
`Compressor` alone. **Please also expose the same norm block as a standalone op:**

```
torch.ops.custom.npu_fused_norm(Tensor x, Tensor weight, Tensor? bias,
                                float eps, int norm_mode) -> Tensor
```

* `x`: `[rows, D]`, bf16 / fp16 / fp32, ND, contiguous. Returns the same dtype/shape.
* `weight`: `[D]` float32. `bias`: `[D]` float32, required iff `norm_mode == 1`.
* It **must call the same device code** the `Compressor` calls, not a separate copy.

This is what `../reference/backend.py` targets and what
`tests/test_op2_fused_norm.py` exercises. Without it, OP-2 has no unit-level acceptance
gate and we fall back to end-to-end goldens, which are far slower to bisect.

## 5. Edge cases

| Case | Required behaviour |
|---|---|
| `rows == 0` | No-op; return an empty tensor. Must not fault. |
| Constant row (`var == 0`) | `eps` keeps it finite; result is exactly `beta`. Tested. |
| `norm_mode == 0` with `norm_bias` present | Reject. |
| `norm_mode == 1` with `norm_bias` absent | Reject. |
| `norm_mode` not in `{0, 1}` | Reject. |
| `D` not a multiple of the vector block | Must still work; GLM's `D = 128` is friendly, but `rms_norm.h` already handles a `col % FP32_REPEAT_ELEMENT_NUM` remainder and mode 1 must too. |

## 6. Acceptance criteria

1. **Regression, exact.** With `norm_mode = 0` and `norm_bias` absent, `Compressor`'s
   `cmp_kv` and `state_cache` outputs must be **byte-identical** to the currently
   shipped build on the DeepSeek-V4 path
   (`ascend_dsv4_backend.py:401`). Not "within tolerance" — byte-identical. This is the
   cheapest possible proof that the mode switch did not perturb the existing path.
2. **New mode, two-reference floor.** `npu_fused_norm(..., norm_mode=1)` in bf16 must be
   within the noise floor of `fused_norm_ref` — see [`../ACCEPTANCE.md`](../ACCEPTANCE.md).
3. **Mode is actually wired.** With a row mean far from zero, mode 1 and mode 0 must
   disagree. (`test_layer_norm_differs_from_rms_norm`.) A delivery that accepts
   `norm_mode` and ignores it fails here.

## 7. Second, separable requirement on the same op — `rope_head_dim = 0`

Not in the original brief, but it falls out of the same investigation and you will hit
it: `Compressor` declares `rope_sin` and `rope_cos` as **required** inputs
(`aic-ascend910_93-ops-info.json`, `input6`/`input7`) and `rope_head_dim` as a
**required attr** (`compressor_proto.h`). GLM-5.3-Flash has `qk_rope_head_dim = 0`
(`/mnt/workspace/models/GLM-5.3-Flash-BF16/config.json`, `text_config`), so if SGLang is
to use `Compressor` on the GLM index path at all, the op must accept
`rope_head_dim == 0` with `rope_sin`/`rope_cos` absent or zero-width and skip `CalRope`
(`compressor_block_vec_perf.h:1264-1268`).

**We have not confirmed that SGLang will route GLM's kpool compressor through this
vendor op.** GLM's compression is a softmax over per-pool gate scores plus an APE add
and a Hadamard rotation (`dsa_indexer_kpool.py:240-251, :285-296`;
`kpool_fp8_index.py:666+`), which is structurally the same family as `Compressor`'s
`wkv`/`wgate`/`ape`/`state_cache` fusion but we did not verify term-by-term equivalence.
Treat §7 as "very likely needed, cheap to add while you are in the file", and §1-§6 as
the firm request.

## 8. Not pinned down

* Whether the fp32 `norm_weight` input should stay fp32 or gain a bf16 variant. SGLang
  currently upcasts (`ascend_dsv4_backend.py:459`,
  `compressor._fused_norm_weight_fp32 = compressor.norm.weight.to(torch.float32)`), and
  GLM's `LayerNorm` parameters are already fp32, so **fp32 is fine**.
* Whether `Compressor`'s internal norm operates on `x` in fp32 for fp16 inputs too. The
  `T` type parameter in `compressor_block_vec_perf.h` is fp32 in the paths we read, but
  we did not enumerate every template instantiation.
