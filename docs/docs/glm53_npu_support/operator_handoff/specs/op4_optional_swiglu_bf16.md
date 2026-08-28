# OP-4 (OPTIONAL) — bf16-output variant of `DequantSwigluClampQuant`

**Status:** OPTIONAL optimisation. **This is explicitly NOT a blocker.** GLM-5.3-Flash
runs correctly today without it. Do OP-1, OP-2 and OP-3 first.

Reference implementation: [`../reference/swiglu_clamp.py`](../reference/swiglu_clamp.py)
Tests: [`../tests/test_op4_swiglu_clamp_bf16.py`](../tests/test_op4_swiglu_clamp_bf16.py)

---

## 1. The situation

The existing vendor op `custom::npu_dequant_swiglu_clamp_quant` at `swiglu_mode = 1`
already computes GLM's clamped SwiGLU **exactly**. From the AscendC source
(`opp_custom/vendors/customize/op_impl/ai_core/tbe/customize_impl/ascendc/dequant_swiglu_clamp_quant/dequant_swiglu_clamp_quant.h:606-648`,
`SwiGluGate`):

```
gate_half = clamp(gate_half, -clamp_limit, +clamp_limit) + glu_bias   # Mins, Maxs, Adds
act_half  = min(act_half, clamp_limit)                                # Mins only
act_half  = act_half / (1 + exp(-glu_alpha * act_half))               # Muls, Exp, Adds, Div
out       = gate_half * act_half                                      # Mul
```

`activate_left = true` puts the silu input on the **left** half
(`actOffset_ = actRight * UbFactorDimy`, `dequant_swiglu_clamp_quant.h:155-156`).

With `glu_alpha = 1.0, glu_bias = 0.0, activate_left = true` that is exactly the
reference documented in
`python/sglang/srt/hardware_backend/npu/moe/activation.py:79-98`:

```
up   = clamp(up,   -limit, limit)     # two-sided
gate = clamp(gate,         limit)     # upper bound only — SiLU saturates below
x    = silu(gate) * up
```

Note the clamp is **asymmetric**; a symmetric implementation is wrong.

## 2. Why it cannot be used on the bf16 path

`dst_type` exists as an attr (default `2`), but the op-info JSON
(`opp_custom/vendors/customize/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json`,
op `DequantSwigluClampQuant`) declares `output0.y` as **`int8` in all six dtype slots**
and `output1.scale` as fp32 in all six. So the output is unconditionally int8 and
`dst_type` cannot select bf16. That matches the note already recorded in
`activation.py:96-98`.

The consequence: the bf16 expert path keeps a **separate pre-clamp pass** over the
`[rows, 2*intermediate]` gate/up buffer (`activation.py:118-122`) before a plain fused
SwiGLU. That pass was measured at 589 GB/s (`activation.py:60-66`) — fast, but it is a
whole extra read+write of the largest tensor in the MoE path, on every expert call.

## 3. What would help

Make `dst_type` real for at least `bfloat16`:

```
y  : bfloat16 [rows, intermediate]   when dst_type selects bf16
scale : unused / may be an empty tensor in that case
```

Everything else — inputs, `swiglu_mode`, `clamp_limit`, `glu_alpha`, `glu_bias`,
`activate_left`, `activate_dim` — unchanged. The dequant inputs
(`weight_scale`, `activation_scale`, `bias`, `quant_scale`, `quant_offset`,
`group_index`) are all optional already and the bf16 path passes none of them.

Adding `float16` as well is free if the same template covers it, but bf16 is the one
that matters.

## 4. Acceptance criteria

Two-reference noise floor against `swiglu_clamp_ref` — see
[`../ACCEPTANCE.md`](../ACCEPTANCE.md) — plus:

* **the clamp must actually bite** (`test_clamp_is_actually_applied`), and
* **the clamp must be asymmetric** (`test_asymmetric_clamp`): the silu half has an
  upper bound only.

`clamp_limit = 10.0` is the production value and is exactly representable in bf16, so
this case has an unusually tight noise floor.

## 5. Not pinned down

* Whether `scale` (`output1`) can legally be an empty tensor when the output is bf16, or
  whether the op must still produce a dummy fp32 tensor. Either is fine for SGLang; the
  adapter in `../reference/backend.py` discards it.
* Whether the int8 path would then be able to drop its own pre-clamp too. It already
  can (`activation.py:95-97`): at mode 1 the op reproduces the reference exactly, so
  the int8 path could call it directly today. That is an SGLang-side change, not yours.
