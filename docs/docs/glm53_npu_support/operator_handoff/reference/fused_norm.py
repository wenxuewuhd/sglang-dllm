"""OP-2 reference: the norm block inside the vendor ``Compressor`` op.

The vendor op today fuses RMSNorm only.  Evidence:

* ``opp_custom/vendors/custom_transformer/.../compressor/arch22/compressor_block_vec_perf.h:1256-1260``
  calls ``MultRowRmsNorm`` unconditionally; there is no mode switch.
* ``.../compressor/arch22/rms_norm.h:37-86`` is the whole formula:
  ``dst = (src / sqrt(mean(src^2) + eps)) * gamma`` -- no mean subtraction, no bias.
* The op-info JSON (``op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json``)
  has ``input5 = norm_weight`` (required, float32) and NO bias input, and its attr list is
  ``rope_head_dim,cmp_ratio,coff,norm_eps,rotary_mode,cache_mode,state_cache_stride_dim0``
  -- no norm mode.

GLM-5.3-Flash's index-K norm is a true LayerNorm:
``python/sglang/srt/layers/layernorm.py:1006-1020`` calls ``F.layer_norm`` with a bias,
and ``python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py:146`` constructs it as
``LayerNorm(self.head_dim, dtype=torch.float32)`` (head_dim = index_head_dim = 128).

This module is the reference for the *delta*: a ``norm_mode`` switch and an optional
``norm_bias``.  See ../specs/op2_compressor_layernorm.md.
"""

from __future__ import annotations

from typing import Optional

import torch

NORM_MODE_RMS = 0
NORM_MODE_LAYER = 1


def fused_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    norm_mode: int = NORM_MODE_RMS,
) -> torch.Tensor:
    """Row-wise norm over the last dim. Accumulates in fp32, returns ``x.dtype``.

    ``norm_mode=0`` (RMSNorm) is exactly what the vendor op does today:
        y = x / sqrt(mean(x^2) + eps) * weight
    Note the division by ``sqrt``, not a multiply by ``rsqrt``: that is what
    ``rms_norm.h`` emits (``Sqrt`` then ``RowDivs``), and it is a different rounding.

    ``norm_mode=1`` (LayerNorm) is what GLM needs:
        y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
    with the *biased* (population, denominator N) variance, matching ``F.layer_norm``.
    """
    assert x.dim() >= 2
    dim = x.shape[-1]
    assert weight.shape == (dim,), f"weight must be [{dim}], got {tuple(weight.shape)}"
    out_dtype = x.dtype
    xf = x.to(torch.float32)
    wf = weight.to(torch.float32)

    if norm_mode == NORM_MODE_RMS:
        assert bias is None, "norm_mode=0 (RMSNorm) takes no bias"
        denom = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        y = xf / denom * wf
    elif norm_mode == NORM_MODE_LAYER:
        assert bias is not None, "norm_mode=1 (LayerNorm) requires a bias"
        assert bias.shape == (dim,)
        mean = xf.mean(dim=-1, keepdim=True)
        centered = xf - mean
        var = centered.pow(2).mean(dim=-1, keepdim=True)
        y = centered / torch.sqrt(var + eps) * wf + bias.to(torch.float32)
    else:
        raise ValueError(f"unknown norm_mode {norm_mode}")
    return y.to(out_dtype)


def layer_norm_via_torch(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """Independent cross-check of ``norm_mode=1`` against ``F.layer_norm``.

    Mirrors ``LayerNorm.forward_native`` (python/sglang/srt/layers/layernorm.py:1006-1020),
    which upcasts to ``self.dtype`` (fp32 for this layer) before normalising.
    """
    orig = x.dtype
    return torch.nn.functional.layer_norm(
        x.to(torch.float32), (x.shape[-1],), weight=weight.to(torch.float32),
        bias=bias.to(torch.float32), eps=eps,
    ).to(orig)
