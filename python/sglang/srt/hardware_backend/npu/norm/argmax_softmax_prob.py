"""Single-pass fused argmax + softmax-probability-of-argmax kernel (triton-ascend).

The dLLM denoise step needs, per position, the argmax token and its softmax
probability (the confidence compared against the unmask threshold). The portable
implementation in ``dllm/algorithm/base.py`` does this as separate max / sub /
exp / sum ops over ``[tokens, vocab=157184]``, reading the logits 5-6x. This
kernel reads each logit once, keeping running (max, argmax, sumexp) in registers
via the online-softmax recurrence, and returns ``(argmax_id, 1/sumexp)`` -- the
argmax token's logit equals the max, so its ``exp(max-max)=1`` and its softmax
probability is ``1/sumexp``.

Measured vs the chunked reference (bf16 logits, fp32 accumulation): argmax
bit-identical, probability max relative error 3.8e-7, zero threshold-0.5 flips.
Speedup 7.2x at bs=1 (launch-overhead bound), 2.3x at 126 blocks.
"""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - NPU-only path
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _argmax_prob_kernel(
        logits_ptr,  # [B, V]
        argmax_ptr,  # [B] int64
        prob_ptr,  # [B] float32
        B,
        V,
        stride_b,
        BLOCK_V: tl.constexpr,
    ):
        row_pid = tl.program_id(0)
        n_prog = tl.num_programs(0)
        for row in tl.range(row_pid, B, n_prog):
            base = row * stride_b
            m = tl.full((), float("-inf"), tl.float32)
            arg = tl.zeros((), tl.int64)
            s = tl.zeros((), tl.float32)
            for start in range(0, V, BLOCK_V):
                offs = start + tl.arange(0, BLOCK_V)
                mask = offs < V
                x = tl.load(
                    logits_ptr + base + offs, mask=mask, other=float("-inf")
                ).to(tl.float32)
                chunk_max = tl.max(x, axis=0)
                new_m = tl.maximum(m, chunk_max)
                s = s * tl.exp(m - new_m) + tl.sum(tl.exp(x - new_m), axis=0)
                chunk_arg = tl.argmax(x, axis=0).to(tl.int64) + start
                arg = tl.where(chunk_max > m, chunk_arg, arg)
                m = new_m
            tl.store(argmax_ptr + row, arg)
            tl.store(prob_ptr + row, 1.0 / s)


def argmax_softmax_prob_fused(
    logits: torch.Tensor, block_v: int = 16384
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row argmax id and the softmax probability of that argmax token.

    ``logits``: ``[B, V]`` (any float dtype). Returns
    ``(argmax_ids [B] int64, prob [B] float32)``. Accumulates in fp32; never
    materializes a ``[B, V]`` temporary.
    """
    assert logits.dim() == 2
    B, V = logits.shape
    logits = logits.contiguous()
    argmax = torch.empty(B, dtype=torch.int64, device=logits.device)
    prob = torch.empty(B, dtype=torch.float32, device=logits.device)
    try:
        from sgl_kernel_npu.utils.triton_utils import get_device_properties

        _, n_core = get_device_properties()
    except Exception:
        n_core = 40
    grid = (min(n_core, B),)
    _argmax_prob_kernel[grid](
        logits, argmax, prob, B, V, logits.stride(0), BLOCK_V=block_v
    )
    return argmax, prob


if not _TRITON_AVAILABLE:
    argmax_softmax_prob_fused = None  # noqa: F811
