"""OP-4 (OPTIONAL, not a blocker) reference: bf16-output DequantSwigluClampQuant.

The existing vendor op ``custom::npu_dequant_swiglu_clamp_quant`` at ``swiglu_mode=1``
already computes GLM's / DeepSeek's clamped SwiGLU exactly.  Verified in the AscendC
source ``opp_custom/vendors/customize/.../dequant_swiglu_clamp_quant/dequant_swiglu_clamp_quant.h:606-648``
(``SwiGluGate``):

    act  half (silu input) : min(act, clamp_limit)   -- upper bound only
                             act = act / (1 + exp(-glu_alpha * act))
    gate half (multiplier) : clamp(gate, -clamp_limit, +clamp_limit) + glu_bias
    out                    = gate_half * act_half

With ``glu_alpha=1.0, glu_bias=0.0`` that is exactly the reference documented at
``python/sglang/srt/hardware_backend/npu/moe/activation.py:79-98``.
``activate_left=True`` puts the silu input on the LEFT half: ``actOffset_ = actRight *
UbFactorDimy`` (``dequant_swiglu_clamp_quant.h:155-156``), so ``activate_left`` -> offset 0.

Why it still cannot be used on the bf16 path: the op-info JSON lists
``output0.y`` as ``int8`` in ALL SIX dtype slots
(``opp_custom/vendors/customize/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json``,
op ``DequantSwigluClampQuant``), even though ``dst_type`` exists as an attr with default 2.
So ``dst_type`` cannot select a bf16 output, and the bf16 path keeps the separate
pre-clamp at ``activation.py:118-122``.
"""

from __future__ import annotations

import torch


def swiglu_clamp_ref(
    x: torch.Tensor,
    clamp_limit: float,
    glu_alpha: float = 1.0,
    glu_bias: float = 0.0,
    activate_left: bool = True,
) -> torch.Tensor:
    """Clamped SwiGLU over the last dim, split in half. fp32 accum, returns x.dtype."""
    assert x.shape[-1] % 2 == 0
    half = x.shape[-1] // 2
    xf = x.to(torch.float32)
    left, right = xf[..., :half], xf[..., half:]
    act, gate = (left, right) if activate_left else (right, left)

    gate = torch.clamp(gate, min=-clamp_limit, max=clamp_limit) + glu_bias
    act = torch.clamp(act, max=clamp_limit)
    act = act / (1.0 + torch.exp(-glu_alpha * act))
    return (gate * act).to(x.dtype)


def sglang_prelamp_then_swiglu_ref(x: torch.Tensor, clamp_limit: float) -> torch.Tensor:
    """Independent cross-check: the path SGLang runs today on the bf16 side.

    ``apply_swiglu_limit_`` (activation.py:78-122) clamps in place with per-column
    bounds -- ``-inf..limit`` on the gate half, ``-limit..limit`` on the up half -- and
    then a plain fused swiglu computes ``silu(gate) * up``.  Note that SGLang's
    "gate" (silu input) is the LEFT half, matching ``activate_left=True``.
    """
    assert x.shape[-1] % 2 == 0
    half = x.shape[-1] // 2
    xf = x.to(torch.float32)
    gate = torch.clamp(xf[..., :half], max=clamp_limit)
    up = torch.clamp(xf[..., half:], min=-clamp_limit, max=clamp_limit)
    return (torch.nn.functional.silu(gate) * up).to(x.dtype)
