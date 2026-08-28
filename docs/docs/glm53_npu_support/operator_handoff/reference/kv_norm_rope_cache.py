"""OP-3 reference: ``npu_kv_rmsnorm_rope_cache`` with rope width 0.

Call site: ``python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py:93``.
GLM-5.3-Flash has ``qk_rope_head_dim = 0`` (config.json, text_config), so the fused op
must accept a zero-width rope half.

Recorded measurement on this machine (docs/docs/glm53_npu_support/PLAN.md:172, row C4,
produced by docs/docs/glm53_npu_support/probe/p0_6_rope0.py): with rope=64 both
``npu_kv_rmsnorm_rope_cache`` and ``..._v2`` succeed; with rope=0 (zero-width cos/sin
and k_cache) BOTH raise RuntimeError.

The RoPE branch below is only exercised when ``rope_dim > 0`` and is NOT an authoritative
statement of the vendor op's rope convention -- see the spec's "not pinned down" section.
The contract we actually need is ``rope_dim == 0``, where rope is a no-op.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def rms_norm_ref(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.to(torch.float32)
    denom = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf / denom * gamma.to(torch.float32)).to(x.dtype)


def interleaved_rope_ref(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved (even/odd pair) RoPE. UNVERIFIED against the vendor op -- see docstring."""
    if x.shape[-1] == 0:
        return x
    xf = x.to(torch.float32)
    even = xf[..., 0::2]
    odd = xf[..., 1::2]
    c_even, c_odd = cos.to(torch.float32)[..., 0::2], cos.to(torch.float32)[..., 1::2]
    s_even, s_odd = sin.to(torch.float32)[..., 0::2], sin.to(torch.float32)[..., 1::2]
    out = torch.empty_like(xf)
    out[..., 0::2] = even * c_even - odd * s_even
    out[..., 1::2] = odd * c_odd + even * s_odd
    return out.to(x.dtype)


def kv_norm_rope_cache_ref(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    index: torch.Tensor,
    k_cache: Optional[torch.Tensor],
    ckv_cache: torch.Tensor,
    epsilon: float = 1e-5,
    cache_mode: str = "PA_BNSD",
    is_output_kv: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Reference for ``torch.ops.npu.npu_kv_rmsnorm_rope_cache``.

    Args
      kv        : [T, 1, 1, kv_lora_rank + rope_dim], bf16/fp16 (BNSD)
      gamma     : [kv_lora_rank], same dtype as kv
      cos, sin  : [T, 1, 1, rope_dim] or None when rope_dim == 0
      index     : [T] int64, FLAT slot ids into the paged cache
                  (slot = block_id * block_size + offset_in_block); -1 means "skip"
      k_cache   : [num_blocks, block_size, 1, rope_dim] or None when rope_dim == 0
      ckv_cache : [num_blocks, block_size, 1, kv_lora_rank]

    Returns ``(k_cache, ckv_cache, k_pe, kv_a)`` -- the caches are updated IN PLACE and
    also returned, matching the v1 4-tuple used at deepseek_v2_attention_mla_npu.py:93.
    ``k_pe`` is ``None`` when ``rope_dim == 0`` (see spec for the alternative of a
    zero-width tensor).
    """
    assert cache_mode == "PA_BNSD", (
        "this reference defines PA_BNSD only; PA_NZ is a layout variant, see the spec"
    )
    assert kv.dim() == 4 and kv.shape[1] == 1 and kv.shape[2] == 1
    tokens, total_dim = kv.shape[0], kv.shape[3]
    lora = gamma.shape[0]
    rope_dim = total_dim - lora
    assert rope_dim >= 0
    if rope_dim == 0:
        assert cos is None or cos.shape[-1] == 0
        assert sin is None or sin.shape[-1] == 0
        assert k_cache is None or k_cache.shape[-1] == 0
    else:
        assert cos is not None and sin is not None and k_cache is not None

    flat = kv.reshape(tokens, total_dim)
    kv_a = rms_norm_ref(flat[:, :lora], gamma, epsilon)

    if rope_dim > 0:
        k_pe = interleaved_rope_ref(
            flat[:, lora:].reshape(tokens, 1, 1, rope_dim), cos, sin
        ).reshape(tokens, rope_dim)
    else:
        k_pe = None

    idx = index.to(torch.int64)
    ckv_flat = ckv_cache.reshape(-1, ckv_cache.shape[2], ckv_cache.shape[3])
    live = idx >= 0
    ckv_flat[idx[live]] = kv_a[live].reshape(-1, 1, lora).to(ckv_cache.dtype)
    if rope_dim > 0:
        k_flat = k_cache.reshape(-1, k_cache.shape[2], k_cache.shape[3])
        k_flat[idx[live]] = k_pe[live].reshape(-1, 1, rope_dim).to(k_cache.dtype)

    if not is_output_kv:
        return k_cache, ckv_cache, None, None
    return k_cache, ckv_cache, k_pe, kv_a
