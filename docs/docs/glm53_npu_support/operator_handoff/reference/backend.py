"""Backend selection: run the tests against the torch reference or the NPU operator.

    GLM53_OP_BACKEND=reference   (default) -- test the pure-torch reference against itself
                                              and against its independent cross-checks
    GLM53_OP_BACKEND=npu                   -- test the delivered Ascend operators

The NPU adapters below are the ONLY place that names the delivered operators.  They are
written against the interfaces proposed in ../specs/.  If a delivered operator's name or
signature differs, change it here and nowhere else.

Each adapter takes and returns CPU tensors, so the test bodies are device-agnostic.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

BACKEND = os.environ.get("GLM53_OP_BACKEND", "reference").lower()
NPU_DEVICE = os.environ.get("GLM53_NPU_DEVICE", "npu:0")

_VALID = ("reference", "npu")
if BACKEND not in _VALID:
    raise RuntimeError(f"GLM53_OP_BACKEND must be one of {_VALID}, got {BACKEND!r}")


def is_npu() -> bool:
    return BACKEND == "npu"


def _to_npu(t: Optional[torch.Tensor]):
    return None if t is None else t.to(NPU_DEVICE)


def _import_torch_npu():
    import torch_npu  # noqa: F401  (registers torch.ops.npu / torch.ops.custom)

    return torch_npu


# ---------------------------------------------------------------------------
# OP-1  kpool fused group-top-k + expand + tail append
# ---------------------------------------------------------------------------


def kpool_topk_transform(
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    topk: int,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if BACKEND == "reference":
        from .kpool_topk_transform import kpool_topk_transform_ref

        return kpool_topk_transform_ref(
            score, lengths, pool_size, topk, page_table, topk_indices_offset,
            row_starts, seq_lens, page_table_row_index,
        )

    _import_torch_npu()
    out = torch.ops.custom.npu_kpool_topk_transform(
        _to_npu(score),
        _to_npu(lengths),
        pool_size,
        topk,
        page_table=_to_npu(page_table),
        topk_indices_offset=_to_npu(topk_indices_offset),
        row_starts=_to_npu(row_starts),
        seq_lens=_to_npu(seq_lens),
        page_table_row_index=_to_npu(page_table_row_index),
    )
    return out.cpu()


# ---------------------------------------------------------------------------
# OP-2  fused norm block (RMSNorm / LayerNorm) used inside Compressor
# ---------------------------------------------------------------------------


def fused_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    norm_mode: int,
) -> torch.Tensor:
    if BACKEND == "reference":
        from .fused_norm import fused_norm_ref

        return fused_norm_ref(x, weight, bias, eps, norm_mode)

    _import_torch_npu()
    # Test hook requested in ../specs/op2_compressor_layernorm.md: the same norm block
    # the Compressor uses internally, exposed as a standalone op so it can be unit tested.
    out = torch.ops.custom.npu_fused_norm(
        _to_npu(x), _to_npu(weight), _to_npu(bias), eps, norm_mode
    )
    return out.cpu()


# ---------------------------------------------------------------------------
# OP-3  npu_kv_rmsnorm_rope_cache with rope width 0
# ---------------------------------------------------------------------------


def kv_norm_rope_cache(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    index: torch.Tensor,
    k_cache: Optional[torch.Tensor],
    ckv_cache: torch.Tensor,
    epsilon: float,
    cache_mode: str = "PA_BNSD",
    is_output_kv: bool = True,
):
    if BACKEND == "reference":
        from .kv_norm_rope_cache import kv_norm_rope_cache_ref

        return kv_norm_rope_cache_ref(
            kv, gamma, cos, sin, index, k_cache, ckv_cache, epsilon,
            cache_mode, is_output_kv,
        )

    _import_torch_npu()
    d_kc, d_ckv = _to_npu(k_cache), _to_npu(ckv_cache)
    res = torch.ops.npu.npu_kv_rmsnorm_rope_cache(
        _to_npu(kv), _to_npu(gamma), _to_npu(cos), _to_npu(sin),
        _to_npu(index).to(torch.int64), d_kc, d_ckv,
        epsilon=epsilon, cache_mode=cache_mode, is_output_kv=is_output_kv,
    )
    # Write the device caches back into the caller's CPU tensors so the test sees the
    # in-place effect on the same objects it passed in.
    ckv_cache.copy_(d_ckv.cpu())
    if k_cache is not None:
        k_cache.copy_(d_kc.cpu())
    k_pe = res[2].cpu() if (len(res) > 2 and res[2] is not None) else None
    kv_a = res[3].cpu() if (len(res) > 3 and res[3] is not None) else None
    return k_cache, ckv_cache, k_pe, kv_a


# ---------------------------------------------------------------------------
# OP-4  bf16-output clamped SwiGLU (OPTIONAL)
# ---------------------------------------------------------------------------


def swiglu_clamp_bf16(
    x: torch.Tensor,
    clamp_limit: float,
    glu_alpha: float = 1.0,
    glu_bias: float = 0.0,
    activate_left: bool = True,
) -> torch.Tensor:
    if BACKEND == "reference":
        from .swiglu_clamp import swiglu_clamp_ref

        return swiglu_clamp_ref(x, clamp_limit, glu_alpha, glu_bias, activate_left)

    _import_torch_npu()
    y, _scale = torch.ops.custom.npu_dequant_swiglu_clamp_quant(
        _to_npu(x),
        None, None, None, None, None, None,
        activate_left=activate_left,
        quant_mode="static",
        dst_type=torch.bfloat16,  # the thing that must start working
        round_mode="rint",
        activate_dim=-1,
        swiglu_mode=1,
        clamp_limit=clamp_limit,
        glu_alpha=glu_alpha,
        glu_bias=glu_bias,
    )
    return y.cpu()
