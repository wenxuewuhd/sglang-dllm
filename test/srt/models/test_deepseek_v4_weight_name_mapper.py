# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4's hf_to_sglang_mapper must rewrite the checkpoint's own module
naming into the HF layout that quantization configs are matched against.

No accelerator, no checkpoint: the ignore entries below are verbatim shapes from a
DeepSeek-V4-Flash-W8A8 compressed-tensors config.
"""

import unittest

from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.deepseek_v4_nextn import DeepseekV4ForCausalLMNextN

# The four shapes that make up the 279-entry ignore list, plus the bare head.
TARGET_ENTRIES = [
    "layers.0.attn.wq_a",
    "layers.2.attn.wkv",
    "layers.42.attn.wo_a",
    "layers.7.attn.indexer.weights_proj",
]
MTP_ENTRIES = ["mtp.0.attn.wq_a", "mtp.0.attn.wkv", "mtp.0.attn.wo_a"]


class TestTargetModelMapper(unittest.TestCase):
    def setUp(self):
        self.apply = DeepseekV4ForCausalLM.hf_to_sglang_mapper.apply_list

    def test_layer_entries_reach_hf_module_paths(self):
        got = self.apply(TARGET_ENTRIES)
        self.assertEqual(
            got,
            [
                "model.layers.0.self_attn.wq_a",
                "model.layers.2.self_attn.wkv",
                "model.layers.42.self_attn.wo_a",
                "model.layers.7.self_attn.indexer.weights_proj",
            ],
        )

    def test_ffn_becomes_mlp(self):
        self.assertEqual(
            self.apply(["layers.3.ffn.gate"]), ["model.layers.3.mlp.gate"]
        )

    def test_norms_are_renamed(self):
        self.assertEqual(
            self.apply(["layers.1.attn_norm.weight", "layers.1.ffn_norm.weight"]),
            [
                "model.layers.1.input_layernorm.weight",
                "model.layers.1.post_attention_layernorm.weight",
            ],
        )

    def test_already_hf_shaped_names_are_left_alone(self):
        """A checkpoint that already speaks HF must not be double-rewritten."""
        hf = ["model.layers.0.self_attn.wq_a"]
        self.assertEqual(self.apply(hf), hf)


class TestNextNModelMapper(unittest.TestCase):
    def test_mtp_entries_reach_the_nextn_decoder(self):
        """The MTP layer's modules live under model.decoder in this model, so the
        target model's 'layers.' prefix rule does not apply to them."""
        got = DeepseekV4ForCausalLMNextN.hf_to_sglang_mapper.apply_list(MTP_ENTRIES)
        self.assertEqual(
            got,
            [
                "model.decoder.self_attn.wq_a",
                "model.decoder.self_attn.wkv",
                "model.decoder.self_attn.wo_a",
            ],
        )

    def test_substr_rules_are_shared_with_the_target_model(self):
        self.assertEqual(
            DeepseekV4ForCausalLMNextN.hf_to_sglang_mapper.orig_to_new_substr,
            DeepseekV4ForCausalLM.hf_to_sglang_mapper.orig_to_new_substr,
        )


if __name__ == "__main__":
    unittest.main()
