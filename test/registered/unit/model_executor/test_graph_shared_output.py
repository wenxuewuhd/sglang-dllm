import unittest

import torch

from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput


class TestGraphSharedOutput(unittest.TestCase):
    def test_logits_buffers_are_shared_per_dtype(self):
        shared = GraphSharedOutput(device=torch.device("cpu"), max_rows=4)

        fp32 = shared.get_logits_buffer(8, rows=2)
        bf16 = shared.get_logits_buffer(8, rows=2, dtype=torch.bfloat16)

        self.assertEqual(fp32.dtype, torch.float32)
        self.assertEqual(bf16.dtype, torch.bfloat16)
        self.assertNotEqual(fp32.data_ptr(), bf16.data_ptr())
        self.assertEqual(
            bf16.data_ptr(),
            shared.get_logits_buffer(8, rows=4, dtype=torch.bfloat16).data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
