# SPDX-License-Identifier: Apache-2.0
"""KTConfig construction from server args. No accelerator required."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.moe.kt_ep_wrapper import create_kt_config_from_server_args

NUM_HIDDEN_LAYERS = 43


def _server_args(**overrides):
    args = SimpleNamespace(
        kt_weight_path="/weights/expert_layer{layer_idx}.gguf",
        kt_method="LLAMAFILE",
        kt_cpuinfer=64,
        kt_threadpool_count=2,
        kt_num_gpu_experts=32,
        kt_max_deferred_experts_per_token=0,
        chunked_prefill_size=8192,
    )
    args.__dict__.update(overrides)
    args.get_hf_config = lambda: SimpleNamespace(
        num_hidden_layers=NUM_HIDDEN_LAYERS
    )
    return args


class TestCreateKTConfig(unittest.TestCase):
    def test_returns_none_without_weight_path(self):
        self.assertIsNone(
            create_kt_config_from_server_args(_server_args(kt_weight_path=None), 0)
        )

    def test_target_model_layer_index_is_unchanged(self):
        for layer_idx in (0, 7, NUM_HIDDEN_LAYERS - 1):
            config = create_kt_config_from_server_args(_server_args(), layer_idx)
            self.assertEqual(config.layer_idx, layer_idx)

    def test_nextn_layer_index_is_offset_past_the_target_model(self):
        """An MTP draft layer is built with layer_id=0; without the offset it
        would collide with the target model's layer 0 -- same expert weights,
        and a second KTMoEWrapper for the same index."""
        target = create_kt_config_from_server_args(_server_args(), 0)
        draft = create_kt_config_from_server_args(_server_args(), 0, is_nextn=True)

        self.assertEqual(draft.layer_idx, NUM_HIDDEN_LAYERS)
        self.assertNotEqual(draft.layer_idx, target.layer_idx)

    def test_nextn_offset_is_per_mtp_index(self):
        for mtp_idx in (0, 1):
            config = create_kt_config_from_server_args(
                _server_args(), mtp_idx, is_nextn=True
            )
            self.assertEqual(config.layer_idx, NUM_HIDDEN_LAYERS + mtp_idx)

    def test_nextn_falls_back_without_num_hidden_layers(self):
        """Offsetting needs num_hidden_layers; if the config is unavailable the
        index is left alone rather than silently corrupted."""
        args = _server_args()
        args.get_hf_config = lambda: SimpleNamespace()

        with patch("sglang.srt.layers.moe.kt_ep_wrapper.logger") as mock_logger:
            config = create_kt_config_from_server_args(args, 0, is_nextn=True)

        self.assertEqual(config.layer_idx, 0)
        mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
