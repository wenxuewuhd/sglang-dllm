# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
# Unit tests for torch_npu.npu_fused_infer_attention_score (CANN FIA).
# Reference: Ascend Extension for PyTorch 7.3.0 API doc and
# python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py call sites.

import unittest

import torch

from sglang.srt.utils import is_npu


def _causal_mask_upper_triangle_flag(seq_len_q: int, seq_len_kv: int) -> torch.Tensor:
    """Bool mask: True = masked (upper triangle). Shape (seq_len_q, seq_len_kv)."""
    # row i can see cols 0..i (causal), so mask j>i
    mask = torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)  # upper triangle = True
    return mask


def _fia_mask_2048_causal() -> torch.Tensor:
    """Causal mask flag for sparse_mode=3: (2048,2048), True = masked (upper)."""
    return _causal_mask_upper_triangle_flag(2048, 2048)


@unittest.skipIf(
    not is_npu(), "NPU not available, skip npu_fused_infer_attention_score tests"
)
class TestNpuFusedInferAttentionScore(unittest.TestCase):
    """Test torch.ops.npu.npu_fused_infer_attention_score (FIA) usage patterns."""

    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("torch.npu not available")
        self.device = "npu:0"
        self.dtype = torch.float16
        self.num_heads = 8
        self.num_kv_heads = 8
        self.head_dim = 128
        self.scale = 1.0 / (self.head_dim**0.5)

    def test_prefill_bsnd_sparse3_single_batch(self):
        """
        Prefill (Q_S > 1), BSND layout, sparse_mode=3 (rightDownCausal).
        Matches ascend_backend FIA extend path: per-request (1, q_len, N, D) with 2048x2048 mask.
        """
        q_len = 64  # must be <= 2048 when using sparse_mode=3
        B, N, D = 1, self.num_heads, self.head_dim
        q = torch.randn(B, q_len, N, D, dtype=self.dtype, device=self.device)
        k = torch.randn(B, q_len, N, D, dtype=self.dtype, device=self.device)
        v = torch.randn(B, q_len, N, D, dtype=self.dtype, device=self.device)
        atten_mask = _fia_mask_2048_causal().to(self.device)

        out, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            num_heads=N,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSND",
            atten_mask=atten_mask.unsqueeze(0),
            sparse_mode=3,
            scale=self.scale,
            next_tokens=0,
        )
        self.assertEqual(out.shape, (B, q_len, N, D))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_prefill_bsnd_sparse3_multiple_q_lens_loop(self):
        """
        Simulate ascend_backend extend loop: multiple requests with different q_len,
        each call (1, q_len, N, D) with sparse_mode=3 and same 2048x2048 mask.
        """
        q_lens = [32, 64, 16]
        N, D = self.num_heads, self.head_dim
        atten_mask = _fia_mask_2048_causal().to(self.device)
        total_tokens = sum(q_lens)
        q = torch.randn(total_tokens, N, D, dtype=self.dtype, device=self.device)
        k = torch.randn(total_tokens, N, D, dtype=self.dtype, device=self.device)
        v = torch.randn(total_tokens, N, D, dtype=self.dtype, device=self.device)

        offset = 0
        outputs = []
        for q_len in q_lens:
            q_b = q[offset : offset + q_len].unsqueeze(0)
            k_b = k[offset : offset + q_len].unsqueeze(0)
            v_b = v[offset : offset + q_len].unsqueeze(0)
            out_b, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_b,
                k_b,
                v_b,
                num_heads=N,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BSND",
                atten_mask=atten_mask.unsqueeze(0),
                sparse_mode=3 if q_len != 1 else 0,
                scale=self.scale,
                next_tokens=0,
            )
            outputs.append(out_b.squeeze(0))
            offset += q_len

        out_cat = torch.cat(outputs, dim=0)
        self.assertEqual(out_cat.shape, (total_tokens, N, D))
        self.assertFalse(torch.isnan(out_cat).any())

    def test_decode_bsnd_single_token_no_block_table(self):
        """
        Decode (Q_S=1): batch of single-query tokens, contiguous KV.
        Shape: q (B, 1, N, D), k/v (B, kv_len, N, D) or contiguous flat with actual_seq_lengths_kv.
        """
        B, kv_len, N, D = 2, 128, self.num_heads, self.head_dim
        q = torch.randn(B, 1, N, D, dtype=self.dtype, device=self.device)
        k = torch.randn(B, kv_len, N, D, dtype=self.dtype, device=self.device)
        v = torch.randn(B, kv_len, N, D, dtype=self.dtype, device=self.device)

        out, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            num_heads=N,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSND",
            atten_mask=None,
            scale=self.scale,
        )
        self.assertEqual(out.shape, (B, 1, N, D))
        self.assertFalse(torch.isnan(out).any())

    def test_prefill_sparse0_custom_q_kv_mask(self):
        """
        Prefill with sparse_mode=0 and explicit (Q_S, KV_S) causal mask.
        Use when Q_S or KV_S > 2048 (sparse_mode=3 requires 2048x2048).
        """
        q_len = 128
        kv_len = 128
        B, N, D = 1, self.num_heads, self.head_dim
        q = torch.randn(B, q_len, N, D, dtype=self.dtype, device=self.device)
        k = torch.randn(B, kv_len, N, D, dtype=self.dtype, device=self.device)
        v = torch.randn(B, kv_len, N, D, dtype=self.dtype, device=self.device)
        mask_flag = _causal_mask_upper_triangle_flag(q_len, kv_len).to(self.device)
        # FIA expects mask True = masked; we use bool upper triangle
        atten_mask = mask_flag.unsqueeze(0)

        out, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            num_heads=N,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSND",
            atten_mask=atten_mask,
            sparse_mode=0,
            scale=self.scale,
            next_tokens=0,
        )
        self.assertEqual(out.shape, (B, q_len, N, D))
        self.assertFalse(torch.isnan(out).any())

    def test_bsnd_gqa(self):
        """GQA: num_heads=8, num_key_value_heads=2."""
        q_len = 64
        B, N, KvN, D = 1, 8, 2, self.head_dim
        q = torch.randn(B, q_len, N, D, dtype=self.dtype, device=self.device)
        k = torch.randn(B, q_len, KvN, D, dtype=self.dtype, device=self.device)
        v = torch.randn(B, q_len, KvN, D, dtype=self.dtype, device=self.device)
        atten_mask = _fia_mask_2048_causal().to(self.device)

        out, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            num_heads=N,
            num_key_value_heads=KvN,
            input_layout="BSND",
            atten_mask=atten_mask.unsqueeze(0),
            sparse_mode=3,
            scale=self.scale,
            next_tokens=0,
        )
        self.assertEqual(out.shape, (B, q_len, N, D))
        self.assertFalse(torch.isnan(out).any())

    def test_tnd_actual_seq_lengths_batch(self):
        """
        TND layout with actual_seq_lengths / actual_seq_lengths_kv (cumsum).
        Matches dllm/forward_dllm style: concatenated batch with variable lengths.
        """
        # Two sequences: len 32 and 64
        lengths = [32, 64]
        total_q = sum(lengths)
        N, D = self.num_heads, self.head_dim
        q = torch.randn(total_q, N, D, dtype=self.dtype, device=self.device)
        k = torch.randn(total_q, N, D, dtype=self.dtype, device=self.device)
        v = torch.randn(total_q, N, D, dtype=self.dtype, device=self.device)
        actual_seq_lengths = torch.cumsum(
            torch.tensor(lengths, dtype=torch.int64), dim=0
        ).tolist()
        actual_seq_lengths_kv = actual_seq_lengths

        out, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            num_heads=N,
            num_key_value_heads=N,
            input_layout="TND",
            atten_mask=None,
            scale=self.scale,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )
        self.assertEqual(out.shape, (total_q, N, D))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
