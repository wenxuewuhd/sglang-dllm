# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
# Test npu_fused_infer_attention_score for chunk prefill (8192) with Page Attention (PA)
# and NPU graph capture. Includes torch golden reference.
# Reference: Ascend Extension for PyTorch 7.3.0 API doc;
# python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py.

import unittest

import torch

from sglang.srt.utils import is_npu

# CANN page attention: block_size 128~512 (128的倍数). We use 512 for 8192 chunk.
# (若需 page_size=2048 的语义，可在非 PA 路径或后续 CANN 支持更大 block 时再测)
PAGE_SIZE = 512
CHUNK_PREFILL_LEN = 8192


def _causal_mask_upper_triangle_flag(seq_len_q: int, seq_len_kv: int) -> torch.Tensor:
    """Bool mask: True = masked (upper triangle). Shape (seq_len_q, seq_len_kv)."""
    mask = torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)
    return mask


def _fia_mask_2048_causal() -> torch.Tensor:
    """Causal mask for sparse_mode=3: (2048,2048), True = masked."""
    return _causal_mask_upper_triangle_flag(2048, 2048)


def torch_golden_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """
    Golden reference: scaled dot-product attention with causal mask.
    q, k, v: (B, S, N, D) or (S, N, D). Returns same shape as q.
    """
    if q.dim() == 3:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    # (B, N, S, D) @ (B, N, D, S) -> (B, N, S, S)
    B, S, N, D = q.shape
    q = q.transpose(1, 2)  # (B, N, S, D)
    k = k.transpose(1, 2)  # (B, N, S, D)
    v = v.transpose(1, 2)  # (B, N, S, D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        mask = _causal_mask_upper_triangle_flag(S, S).to(
            device=scores.device, dtype=torch.bool
        )
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.matmul(attn, v)  # (B, N, S, D)
    out = out.transpose(1, 2)  # (B, S, N, D)
    if squeeze:
        out = out.squeeze(0)
    return out


def build_paged_kv_from_contiguous(
    k_contig: torch.Tensor,
    v_contig: torch.Tensor,
    page_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """
    k_contig, v_contig: (S, N, D). Build paged (block_num, page_size, N*D) on device.
    """
    S, N, D = k_contig.shape
    H_kv = N * D
    num_blocks = (S + page_size - 1) // page_size
    k_paged = torch.zeros(num_blocks, page_size, H_kv, dtype=dtype, device=device)
    v_paged = torch.zeros(num_blocks, page_size, H_kv, dtype=dtype, device=device)
    k_flat = k_contig.reshape(S, -1)  # (S, N*D)
    v_flat = v_contig.reshape(S, -1)
    for b in range(num_blocks):
        start = b * page_size
        end = min((b + 1) * page_size, S)
        length = end - start
        k_paged[b, :length].copy_(k_flat[start:end])
        v_paged[b, :length].copy_(v_flat[start:end])
    return k_paged, v_paged


def build_dummy_block_table(
    batch_size: int,
    seq_len: int,
    page_size: int,
    device: torch.device,
) -> torch.Tensor:
    """block_table: (B, max_blocks_per_seq). Block ids 0..num_blocks-1 per seq."""
    num_blocks = (seq_len + page_size - 1) // page_size
    # One batch: one row of block ids
    block_ids = torch.arange(num_blocks, dtype=torch.int32, device=device)
    return block_ids.unsqueeze(0).expand(batch_size, -1)


@unittest.skipIf(not is_npu(), "NPU not available")
class TestNpuFiaChunkPrefillPaAndGraph(unittest.TestCase):
    """Chunk prefill 8192 + PA + NPU graph capture + torch golden."""

    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("torch.npu not available")
        self.device = "npu:0"
        self.dtype = torch.float16
        self.num_heads = 8
        self.num_kv_heads = 8
        self.head_dim = 128
        self.scale = 1.0 / (self.head_dim**0.5)
        self.page_size = PAGE_SIZE
        self.chunk_len = CHUNK_PREFILL_LEN

    def test_1_torch_golden_causal_attention(self):
        """Sanity: torch golden causal attention (contiguous q,k,v)."""
        B, S, N, D = 1, 256, self.num_heads, self.head_dim
        q = torch.randn(B, S, N, D, dtype=self.dtype, device=self.device)
        k = torch.randn(B, S, N, D, dtype=self.dtype, device=self.device)
        v = torch.randn(B, S, N, D, dtype=self.dtype, device=self.device)
        out = torch_golden_causal_attention(q, k, v, self.scale, causal=True)
        self.assertEqual(out.shape, (B, S, N, D))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_2_fia_extend_8192_chunk_prefill_pa_vs_golden(self):
        """
        Simulate chunk prefill: one forward extend with 8192 tokens.
        - Build dummy paged KV cache (page_size=512), block_table.
        - Run FIA with PA (block_table, block_size, actual_seq_lengths_kv).
        - Compare to torch golden (same q,k,v contiguous, causal).
        """
        S = self.chunk_len
        N, D = self.num_heads, self.head_dim
        # Contiguous q, k, v for this chunk (for golden and for filling PA cache)
        q = torch.randn(1, S, N, D, dtype=self.dtype, device=self.device)
        k = torch.randn(1, S, N, D, dtype=self.dtype, device=self.device)
        v = torch.randn(1, S, N, D, dtype=self.dtype, device=self.device)

        # Golden on contiguous
        golden = torch_golden_causal_attention(q, k, v, self.scale, causal=True)
        golden = golden.squeeze(0)  # (S, N, D)

        # Build paged KV and block_table
        k_contig = k.squeeze(0)  # (S, N, D)
        v_contig = v.squeeze(0)
        k_paged, v_paged = build_paged_kv_from_contiguous(
            k_contig, v_contig, self.page_size, self.device, self.dtype
        )
        block_table = build_dummy_block_table(1, S, self.page_size, self.device)

        # FIA with PA: TND query (S, N, D), KV paged (block_num, page_size, N*D)
        # actual_seq_lengths / actual_seq_lengths_kv: list of per-batch lengths (cumsum for TND = [8192] for B=1)
        actual_seq_lengths = [S]
        actual_seq_lengths_kv = [S]

        query_tnd = q.squeeze(0).contiguous()  # (8192, N, D)
        out_fia, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query_tnd,
            k_paged,
            v_paged,
            block_table=block_table,
            block_size=self.page_size,
            num_heads=N,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            atten_mask=None,
            scale=self.scale,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )
        torch.npu.synchronize()
        self.assertEqual(out_fia.shape, (S, N, D))

        # Compare to golden (allow fp16 tolerance)
        diff = (out_fia - golden).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        self.assertFalse(
            torch.isnan(out_fia).any(),
            "FIA output has NaN",
        )
        self.assertLess(
            max_diff,
            0.5,
            f"FIA vs golden max_diff={max_diff} mean_diff={mean_diff}",
        )

    def test_3_fia_extend_8192_bsnd_sparse3_vs_golden(self):
        """
        Chunk prefill 8192 with BSND + sparse_mode=3 (no PA).
        FIA sparse_mode=3 requires mask (2048,2048); q_len 8192 > 2048 so we split
        into 4 calls of 2048 each and concat, or use one call with q_len<=2048.
        Doc: sparse_mode=3 mask fixed 2048. So for 8192 we use 4 x 2048 calls.
        """
        S = self.chunk_len
        N, D = self.num_heads, self.head_dim
        q = torch.randn(1, S, N, D, dtype=self.dtype, device=self.device)
        k = torch.randn(1, S, N, D, dtype=self.dtype, device=self.device)
        v = torch.randn(1, S, N, D, dtype=self.dtype, device=self.device)
        golden = torch_golden_causal_attention(q, k, v, self.scale, causal=True)
        golden = golden.squeeze(0)

        atten_mask = _fia_mask_2048_causal().to(self.device)
        segment = 2048
        outputs = []
        for start in range(0, S, segment):
            end = min(start + segment, S)
            seg_len = end - start
            q_b = q[:, start:end]
            k_b = k[:, start:end]
            v_b = v[:, start:end]
            out_b, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_b,
                k_b,
                v_b,
                num_heads=N,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BSND",
                atten_mask=atten_mask.unsqueeze(0),
                sparse_mode=3,
                scale=self.scale,
                next_tokens=0,
            )
            outputs.append(out_b.squeeze(0))
        out_fia = torch.cat(outputs, dim=0)
        torch.npu.synchronize()
        self.assertEqual(out_fia.shape, (S, N, D))
        diff = (out_fia - golden).abs()
        self.assertLess(diff.max().item(), 0.5, "FIA BSND sparse3 vs golden")

    def test_4_npu_graph_capture_replay_fia_pa(self):
        """
        Run FIA PA (chunk 8192) inside NPU graph: capture then replay.
        Compare replay output to eager run.
        """
        S = self.chunk_len
        N, D = self.num_heads, self.head_dim
        q = torch.randn(1, S, N, D, dtype=self.dtype, device=self.device)
        k = torch.randn(1, S, N, D, dtype=self.dtype, device=self.device)
        v = torch.randn(1, S, N, D, dtype=self.dtype, device=self.device)

        k_contig = k.squeeze(0)
        v_contig = v.squeeze(0)
        k_paged, v_paged = build_paged_kv_from_contiguous(
            k_contig, v_contig, self.page_size, self.device, self.dtype
        )
        block_table = build_dummy_block_table(1, S, self.page_size, self.device)
        actual_seq_lengths = [S]
        actual_seq_lengths_kv = [S]
        query_tnd = q.squeeze(0).contiguous()

        # Eager run
        out_eager, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query_tnd,
            k_paged,
            v_paged,
            block_table=block_table,
            block_size=self.page_size,
            num_heads=N,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            atten_mask=None,
            scale=self.scale,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )
        torch.npu.synchronize()

        # Staging buffer for capture (same shape as inputs/output)
        query_staging = query_tnd.clone()
        k_paged_staging = k_paged.clone()
        v_paged_staging = v_paged.clone()
        out_staging = torch.empty_like(query_tnd)

        def run_once():
            o, _ = torch.ops.npu.npu_fused_infer_attention_score(
                query_staging,
                k_paged_staging,
                v_paged_staging,
                block_table=block_table,
                block_size=self.page_size,
                num_heads=N,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                atten_mask=None,
                scale=self.scale,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
            )
            out_staging.copy_(o)
            return out_staging

        graph = torch.npu.NPUGraph()
        torch.npu.synchronize()
        # pool= None: minimal test; production may use memory pool.
        with torch.npu.graph(graph, stream=torch.npu.current_stream()):
            capture_out = run_once()
        torch.npu.synchronize()

        # Replay: update staging inputs to new data (optional; same data for match)
        query_staging.copy_(query_tnd)
        k_paged_staging.copy_(k_paged)
        v_paged_staging.copy_(v_paged)
        graph.replay()
        torch.npu.synchronize()

        self.assertEqual(capture_out.shape, out_eager.shape)
        diff = (capture_out - out_eager).abs()
        self.assertLess(
            diff.max().item(),
            1e-2,
            "Graph replay vs eager FIA PA",
        )


if __name__ == "__main__":
    unittest.main()
