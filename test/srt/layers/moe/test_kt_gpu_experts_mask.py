# SPDX-License-Identifier: Apache-2.0
"""KTMoEWrapper takes a per-expert bool mask; num_gpu_experts is the prefix case."""

import unittest

import torch

from sglang.srt.layers.moe.kt_ep_wrapper import build_prefix_gpu_experts_mask


class TestPrefixGpuExpertsMask(unittest.TestCase):
    def test_prefix_is_marked(self):
        self.assertEqual(
            build_prefix_gpu_experts_mask(8, 3).tolist(), [True] * 3 + [False] * 5
        )

    def test_zero_means_all_cpu(self):
        self.assertFalse(bool(build_prefix_gpu_experts_mask(4, 0).any()))

    def test_minus_one_means_no_split(self):
        """-1 is the sentinel FusedMoE's weight loader already tests for."""
        self.assertTrue(bool(build_prefix_gpu_experts_mask(4, -1).all()))

    def test_over_count_is_clamped(self):
        self.assertEqual(build_prefix_gpu_experts_mask(4, 99).tolist(), [True] * 4)

    def test_dtype_and_length(self):
        m = build_prefix_gpu_experts_mask(256, 32)
        self.assertEqual(m.dtype, torch.bool)
        self.assertEqual(m.numel(), 256)
        self.assertEqual(int(m.sum()), 32)


if __name__ == "__main__":
    unittest.main()
