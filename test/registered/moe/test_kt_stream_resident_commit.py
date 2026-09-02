"""A streaming prefill that aborts must never leave resident weights and resident masks
describing different expert sets.

The behaviour this pins down: on the depool path each layer's resident weights are rewritten
the moment that layer is streamed, so if the mask update were deferred to the last MoE layer,
an exception at layer L (a convert OOM) would leave every layer < L holding expert top[i]'s
weights in slot i under a mask still claiming slot i is expert i. Wrong expert computed,
nothing raised, and a log line saying "static set kept" that is false for those layers.

Runs on CPU tensors and needs no accelerator: the module's real maybe_streaming_forward /
_streaming_forward / _apply_resident_layer_depool are exercised, and only the weight source
and the NPU-only helpers are stubbed.
"""

import importlib.util
import os
import sys
import types
import unittest

import torch

import sglang.srt.layers.moe.kt_stream_prefill as _uut
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# The module keeps process-global resident-commit state, so every scenario gets a fresh copy
# loaded from the same file the normal import resolved.
MODULE_PATH = _uut.__file__

E, H, I = 16, 8, 4  # tiny stand-ins for 288 / 4096 / 2048
K = 4  # resident slots per layer
MOE_LAYERS = [3, 4, 5, 6, 7]


def _fresh_module():
    spec = importlib.util.spec_from_file_location("kt_stream_prefill_uut", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kt_stream_prefill_uut"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Wrap:
    """Stand-in for KTEPWrapperMethod's resident-routing state."""

    def __init__(self):
        self.num_gpu_experts = K
        self.gpu_experts_mask = torch.zeros(E, dtype=torch.bool)
        self.logical_to_gpu_index = torch.full((E,), -1, dtype=torch.int64)
    """Stand-in for KTEPWrapperMethod's resident-routing state."""

    def __init__(self):
        self.num_gpu_experts = K
        self.gpu_experts_mask = torch.zeros(E, dtype=torch.bool)
        self.logical_to_gpu_index = torch.full((E,), -1, dtype=torch.int64)
        self.wrapper = None
        # static prefix placement: slot i holds expert i
        self.gpu_experts_mask[:K] = True
        self.logical_to_gpu_index[:K] = torch.arange(K, dtype=torch.int64)


class _Layer:
    """Stand-in for the FusedMoE module's resident parameters.

    Expert e's weight is the constant e, so a slot's content names the expert it holds.
    """

    def __init__(self):
        self.w13_weight = torch.nn.Parameter(
            torch.arange(K, dtype=torch.float32).view(K, 1, 1).expand(K, H, 2 * I).clone()
        )
        self.w2_weight = torch.nn.Parameter(
            torch.arange(K, dtype=torch.float32).view(K, 1, 1).expand(K, I, H).clone()
        )
        self.w13_weight_scale = torch.nn.Parameter(
            torch.arange(K, dtype=torch.float32).view(K, 1).expand(K, 2 * I).clone()
        )
        self.w2_weight_scale = torch.nn.Parameter(
            torch.arange(K, dtype=torch.float32).view(K, 1).expand(K, H).clone()
        )


def _full_experts():
    w13 = torch.arange(E, dtype=torch.float32).view(E, 1, 1).expand(E, H, 2 * I).clone()
    w2 = torch.arange(E, dtype=torch.float32).view(E, 1, 1).expand(E, I, H).clone()
    s13 = torch.arange(E, dtype=torch.float32).view(E, 1).expand(E, 2 * I).clone()
    s2 = torch.arange(E, dtype=torch.float32).view(E, 1).expand(E, H).clone()
    return w13, s13, w2, s2


def _topk_output(seed):
    """Routing skewed so the per-layer top-K is emphatically not the prefix set 0..K-1."""
    g = torch.Generator().manual_seed(seed)
    hot = torch.randperm(E, generator=g)[:K]
    ids = hot.repeat(64).view(-1, 1).repeat(1, 4)  # [tokens, top_k], only hot experts
    ids = hot.repeat(64).view(-1, 1).repeat(1, 4)  # [tokens, top_k], only hot experts
    cold = torch.arange(E).view(-1, 1).repeat(1, 4)
    ids = torch.cat([ids, cold], 0)  # every expert seen; hot ones far more often
    out = types.SimpleNamespace()
    out.topk_ids = ids.to(torch.int32)
    out.topk_weights = torch.ones_like(ids, dtype=torch.float32)
    return out

def _inconsistent(mod):
    """Layers whose slot contents disagree with what their mask says the slot holds.

    Checked from the tensors themselves, not from the module's own bookkeeping, so it is a
    genuine check of the fix rather than of its accounting.
    """
    bad = []
    for L in MOE_LAYERS:
        layer, wrap = mod._REGISTRY[L]
        l2g = wrap.logical_to_gpu_index
        for e in range(E):
            slot = int(l2g[e])
            if slot < 0:
                continue
            held = float(layer.w13_weight.data[slot, 0, 0])
            if held != float(e):
                bad.append((L, e, slot, held))
                break
    return bad


class TestKTStreamResidentCommit(CustomTestCase):
    """Three shapes an abort takes; each leaves the pass at a different point."""

    def _drive(self, fails):
        """Run a full streaming pass with the OOM injected where ``fails`` says, and return
        the layers whose slot contents disagree with their own mask."""
        os.environ["KT_PREFILL_STREAM"] = "1"
        mod = _fresh_module()
        mod._KT_PREFILL_STREAM = True
        mod._KT_GGUF_DEDUP = True
        mod._KT_DYN_RESIDENT = True
        mod._T = 8
        mod._CFG.update(E=E, H=H, I=I, num_layers=max(MOE_LAYERS) + 1)
        for L in MOE_LAYERS:
            mod._REGISTRY[L] = (_Layer(), _Wrap())

        w13, s13, w2, s2 = _full_experts()

        def _fake_stream(layer_idx, dev):
            if fails(layer_idx):
                raise torch.OutOfMemoryError("injected: MXFP4 convert OOM")
            return w13, s13, w2, s2

        mod._stream_layer_weights = _fake_stream
        mod._streaming_fused_experts = lambda **kw: torch.zeros(
            kw["hidden_states"].shape[0], H
        )
        mod._is_prefill = lambda: True
        mod._wrapper_dims = lambda q: (E, H, I, max(MOE_LAYERS) + 1)

        qm = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
        qm.tp_rank = 0
        qm.kt_config = types.SimpleNamespace(layer_idx=None)

        x = torch.zeros(64, H)
        for L in MOE_LAYERS:
            qm.kt_config.layer_idx = L
            mod.maybe_streaming_forward(qm, x, _topk_output(L), False)
        return _inconsistent(mod)

    def _assert_consistent(self, fails):
        bad = self._drive(fails)
        self.assertEqual(
            bad,
            [],
            "resident weights and masks describe different expert sets on "
            f"{sorted({L for L, *_ in bad})}; each entry is (layer, expert, slot, "
            "expert whose weights the slot actually holds)",
        )

    def test_persistent_from_middle(self):
        """Over-budget memory stays over budget: every layer from 5 on fails."""
        self._assert_consistent(lambda L: L >= 5)

    def test_last_layer_only(self):
        """Only the last layer fails, so the end-of-pass flush never runs and the abort
        path is the sole commit."""
        self._assert_consistent(lambda L: L == MOE_LAYERS[-1])

    def test_transient_middle(self):
        """Negative control: one layer fails and the pass continues, which the last layer's
        flush repairs even without the commit protocol. Here to show the other two are not
        simply reporting failure everywhere."""
        self._assert_consistent(lambda L: L == 5)


if __name__ == "__main__":
    unittest.main()
