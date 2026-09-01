#!/usr/bin/env python3
"""Regression test: a streaming prefill that aborts must never leave resident weights and
resident masks describing different expert sets.

Black-box behaviour that must not come back: with KT_DYNAMIC_RESIDENT + KT_MXFP4_DEPOOL, the
depool path rewrote each layer's resident weights immediately but deferred every mask update to
the last MoE layer.  An exception at layer L (a convert OOM) therefore left every layer < L with
expert top[i]'s weights in slot i under a mask still claiming slot i is expert i -- wrong output,
no exception, and a log line saying "static set kept".

Runs on CPU tensors: the module's real maybe_streaming_forward / _streaming_forward /
_apply_resident_layer_depool are exercised, only the weight source and the NPU-only helpers are
stubbed.  Pass the module path as argv[1] to check a specific copy of the file.
"""

import importlib.util
import os
import sys
import types

# The module under test imports sglang.srt.*, so the repo's python/ has to be importable.
# Do it here rather than making the caller remember a PYTHONPATH: this file used to live
# under docs/ and needed one, which is a good way to have a regression test nobody runs.
_REPO_PYTHON = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../python")
)
if os.path.isdir(_REPO_PYTHON) and _REPO_PYTHON not in sys.path:
    sys.path.insert(0, _REPO_PYTHON)

import torch

E, H, I = 16, 8, 4  # tiny stand-ins for 288 / 4096 / 2048
K = 4  # resident slots per layer
MOE_LAYERS = [3, 4, 5, 6, 7]

# Two shapes a real abort takes.  "transient" (one layer fails, the rest of the pass runs) is
# self-healing even without the fix, because the last layer still flushes; it is here to show the
# test is not simply reporting failure everywhere.  The other two are the hazard.
SCENARIOS = {
    "persistent-from-middle": lambda L: L >= 5,  # over-budget memory stays over budget
    "last-layer-only": lambda L: L == MOE_LAYERS[-1],  # the flush itself never runs
    "transient-middle": lambda L: L == 5,
}


def _load(path):
    spec = importlib.util.spec_from_file_location("kt_stream_prefill_uut", path)
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


def run(path, scenario):
    fails = SCENARIOS[scenario]
    os.environ["KT_PREFILL_STREAM"] = "1"
    mod = _load(path)
    mod._KT_PREFILL_STREAM = True
    mod._KT_MXFP4_DEPOOL = True
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

    from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod

    qm = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
    qm.tp_rank = 0
    qm.kt_config = types.SimpleNamespace(layer_idx=None)

    x = torch.zeros(64, H)
    strict = os.environ.get("KT_STREAM_STRICT") == "1"
    if strict and hasattr(mod, "_KT_STREAM_STRICT"):
        mod._KT_STREAM_STRICT = True
    raised = None
    for L in MOE_LAYERS:
        qm.kt_config.layer_idx = L
        try:
            mod.maybe_streaming_forward(qm, x, _topk_output(L), False)
        except Exception as e:  # only reachable with KT_STREAM_STRICT=1
            raised = e
            break

    bad = _inconsistent(mod)
    print(f"module      : {path}")
    print(f"scenario    : {scenario}  (OOM injected at layers "
          f"{[L for L in MOE_LAYERS if fails(L)]})")
    print(f"strict      : {strict}  raised: {type(raised).__name__ if raised else None}")
    for L in MOE_LAYERS:
        layer, wrap = mod._REGISTRY[L]
        slots = [float(layer.w13_weight.data[j, 0, 0]) for j in range(K)]
        claim = [int(e) for e in range(E) if int(wrap.logical_to_gpu_index[e]) >= 0]
        print(
            f"  L{L}: slots hold experts {[int(s) for s in slots]}  "
            f"mask claims {claim}  {'OK' if all(int(a)==b for a,b in zip(slots,claim)) else 'MISMATCH'}"
        )
    if bad:
        print(f"RESULT: INCONSISTENT -- {len(bad)} layer(s) compute the wrong experts: {bad}")
        return 1
    print("RESULT: consistent -- every layer's slots match its mask")
    return 0


if __name__ == "__main__":
    p = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../python/sglang/srt/layers/moe/kt_stream_prefill.py",
        )
    )
    rc = 0
    for name in SCENARIOS:
        rc |= run(os.path.normpath(p), name)
        print()
    print("OVERALL:", "FAIL (silent wrong experts)" if rc else "PASS")
    sys.exit(rc)
