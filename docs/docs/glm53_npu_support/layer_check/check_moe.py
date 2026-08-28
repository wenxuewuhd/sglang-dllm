#!/usr/bin/env python
"""Stage B for the MoE layers: drive the real sglang NPU MoE path and score it.

Everything that computes a number here is the production class:

  router GEMM   ``models.deepseek_v2.MoEGate``            (bf16 ``F.linear`` on NPU)
  expert pick   ``layers.moe.topk.TopK``                  -> ``fused_topk_npu``
  dispatch      ``token_dispatcher.ascend_tp.AscendTPDispatcher``
  gmm+act+gmm   ``moe_runner.ascend.AscendRunnerCore``    (+ ``NPUUnquantMoEMethod``)
  combine       the same dispatcher's ``combine``
  shared expert ``models.deepseek_v2.DeepseekV2MLP``

Only the *plumbing* is faked: there is no distributed init on a one-process check, so
``get_parallel()`` is overridden with the shipped topology
(``$ROOT/run/launch_glm_bf16.sh``: ``--tp-size 16``, no ``--dp-size``, no
``--ep-size``, ``--moe-a2a-backend none``).

**TP16 is emulated, not approximated.** With EP=1 the 288 experts are *not* split
across ranks; ``intermediate_size`` is (``fused_moe_triton/layer.py:313``:
``intermediate_size // moe_tp_size`` = 2048/16 = **128**). So every rank runs all 288
experts at intermediate 128 and the 16 partial results are summed by an all-reduce.
This script walks the 16 shards one at a time on a single die and sums them: identical
arithmetic, identical per-kernel shapes, no second device needed. Running one rank with
intermediate 2048 would be a *different kernel shape*, and NPU bf16 gemm is not
shape-invariant (PLAN §2.4), so that shortcut is not taken.

    source $ROOT/env.sh
    PYTHONPATH=$REPO/python $VENV/bin/python check_moe.py \
        --case $ROOT/goldens/moe_layer03_s8192.pt --tokens 16,1024,4096,8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "operator_handoff"))

MODEL = os.environ.get("GLM53_MODEL", "/mnt/workspace/models/GLM-5.3-Flash-BF16")
CKPT_PREFIX = "model.language_model."


# --------------------------------------------------------------------------- weights
def load_experts_host(layer: int, cfg):
    """All 288 experts of one layer, in the [E, 2I, H] / [E, H, I] layout the Ascend
    grouped matmul wants, kept on the host so the per-rank shards can be sliced out."""
    from safetensors import safe_open

    d = Path(MODEL)
    wmap = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def get(name):
        shard = wmap[CKPT_PREFIX + name]
        if shard not in handles:
            handles[shard] = safe_open(str(d / shard), framework="pt")
        return handles[shard].get_tensor(CKPT_PREFIX + name)

    E, H, I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    w13 = torch.empty(E, 2 * I, H, dtype=torch.bfloat16)
    w2 = torch.empty(E, H, I, dtype=torch.bfloat16)
    for e in range(E):
        p = f"layers.{layer}.mlp.experts.{e}."
        w13[e, :I] = get(p + "gate_proj.weight")
        w13[e, I:] = get(p + "up_proj.weight")
        w2[e] = get(p + "down_proj.weight")
    shared = {
        n: get(f"layers.{layer}.mlp.shared_experts.{n}.weight")
        for n in ("gate_proj", "up_proj", "down_proj")
    }
    gate = get(f"layers.{layer}.mlp.gate.weight")
    bias = get(f"layers.{layer}.mlp.gate.e_score_correction_bias")
    return w13, w2, shared, gate, bias


# --------------------------------------------------------------------------- build
def build_config(meta):
    tc = json.loads((Path(MODEL) / "config.json").read_text())["text_config"]
    cfg = types.SimpleNamespace(**tc)
    for k in ("n_routed_experts", "num_experts_per_tok", "hidden_size",
              "moe_intermediate_size", "swiglu_limit", "routed_scaling_factor"):
        assert getattr(cfg, k) == meta[k.replace("num_experts_per_tok", "top_k")
                                       if k == "num_experts_per_tok" else k], (
            f"{k}: case says {meta.get(k)}, checkpoint config says {getattr(cfg, k)}"
        )
    return cfg


def build_router(cfg, gate_w, bias, dev):
    """The real ``MoEGate``. Built under ``bfloat16`` as the default dtype because that
    is what ``set_default_torch_dtype(model dtype)`` does around model construction --
    and the bf16 gate weight is precisely PLAN §4 defect 2."""
    from sglang.srt.models.deepseek_v2 import MoEGate

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        gate = MoEGate(config=cfg, quant_config=None, prefix="gate")
    finally:
        torch.set_default_dtype(old)
    gate.weight.data = gate_w.to(gate.weight.dtype).to(dev)
    gate.e_score_correction_bias.data = bias.to(
        gate.e_score_correction_bias.dtype
    ).to(dev)
    return gate.to(dev).eval()


def build_topk(cfg, gate):
    """Exactly the kwargs ``DeepseekV2MoE.__init__`` builds for this family
    (deepseek_v2.py: the ``noaux_tc`` branch), with the shared expert unfused --
    GLM on ascend_tp has ``num_fused_shared_experts == 0``."""
    from sglang.srt.layers.moe.topk import TopK

    return TopK(
        top_k=cfg.num_experts_per_tok,
        layer_id=0,
        renormalize=cfg.norm_topk_prob,
        use_grouped_topk=True,
        num_expert_group=cfg.n_group,
        num_fused_shared_experts=0,
        topk_group=cfg.topk_group,
        scoring_func=cfg.scoring_func,
        correction_bias=gate.e_score_correction_bias,
        quant_config=None,
        routed_scaling_factor=cfg.routed_scaling_factor,
        # UnquantizedFusedMoEMethod + runner backend auto -> False
        # (fused_moe_triton/layer.py:446). The 2.5x is applied on the output instead,
        # as DeepseekV2MoE.forward does.
        apply_routed_scaling_factor_on_output=False,
    )


def build_moe_runner(cfg, inter_per_rank, layer_id):
    from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerCore
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
    from sglang.srt.layers.quantization.unquant import NPUUnquantMoEMethod

    layer = types.SimpleNamespace()
    layer.w13_kernel = NPUUnquantMoEMethod()
    layer.w2_kernel = NPUUnquantMoEMethod()
    rc = MoeRunnerConfig(
        num_experts=cfg.n_routed_experts,
        num_local_experts=cfg.n_routed_experts,
        hidden_size=cfg.hidden_size,
        intermediate_size_per_partition=inter_per_rank,
        layer_id=layer_id,
        top_k=cfg.num_experts_per_tok,
        num_fused_shared_experts=0,
        params_dtype=torch.bfloat16,
        activation="silu",
        swiglu_limit=cfg.swiglu_limit,
        routed_scaling_factor=cfg.routed_scaling_factor,
        layer=layer,
    )
    return layer, rc, AscendRunnerCore(rc), AscendTPDispatcher(rc)


def build_shared(cfg, inter_per_rank, weights, rank, dev):
    """The real ``DeepseekV2MLP`` at the per-rank shape. ``tp_size=1`` because this
    process *is* one rank's worth of arithmetic; the checkpoint slice supplies what a
    real rank's weight loader would have given it."""
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP

    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        mlp = DeepseekV2MLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=inter_per_rank,
            hidden_act=cfg.hidden_act,
            quant_config=None,
            reduce_results=False,
            prefix="shared_experts",
            tp_rank=0,
            tp_size=1,
            swiglu_limit=cfg.swiglu_limit,
        )
    finally:
        torch.set_default_dtype(old)
    lo, hi = rank * inter_per_rank, (rank + 1) * inter_per_rank
    mlp.gate_up_proj.weight.data = torch.cat(
        [weights["gate_proj"][lo:hi], weights["up_proj"][lo:hi]], 0
    ).to(dev)
    mlp.down_proj.weight.data = weights["down_proj"][:, lo:hi].contiguous().to(dev)
    return mlp.to(dev).eval()


# --------------------------------------------------------------------------- run
def attach_weights(layer, w13, w2):
    """What the weight loader plus ``process_weights_after_loading`` would leave behind."""
    layer.w13_weight = w13
    layer.w2_weight = w2
    layer.w13_kernel.process_weights_after_loading(layer, "w13")
    layer.w2_kernel.process_weights_after_loading(layer, "w2")
    from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

    return AscendQuantInfo(w13_weight=layer.w13_weight, w2_weight=layer.w2_weight)


def run_routed(x_dev, topk_out, quant_info, runner, dispatcher):
    from sglang.srt.layers.moe.moe_runner.ascend import AscendRunnerInput
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPCombineInput

    disp_out = dispatcher.dispatch(x_dev, topk_out)
    ri = AscendRunnerInput(
        hidden_states=disp_out.hidden_states,
        hidden_states_scale=disp_out.hidden_states_scale,
        expert_tokens=disp_out.expert_tokens,
        group_list_type=disp_out.group_list_type,
    )
    out = runner.run(ri, quant_info, {})
    return dispatcher.combine(AscendTPCombineInput(hidden_states=out.hidden_states))


def make_shard(w13_all, w2_all, rank, inter, full_inter):
    lo, hi = rank * inter, (rank + 1) * inter
    w13 = torch.cat([w13_all[:, lo:hi],
                     w13_all[:, full_inter + lo:full_inter + hi]], dim=1).contiguous()
    return w13, w2_all[:, :, lo:hi].contiguous()


# --------------------------------------------------------------------------- bench
class D2HCounter:
    """Count host-side syncs by wrapping the Tensor accessors that cause them.

    Counting beats grepping for the Python layer: a ``.item()`` inside a kernel wrapper
    three call frames down is invisible to a source scan, and the number that matters is
    per forward, not per file. The limit is honest: this sees only Python-level
    ``torch.Tensor`` accessors, so a stream sync issued inside a C++ op or a torch_npu
    dispatch would not be counted. A zero here means "no Python-level D2H", not "no
    sync at all".
    """

    NAMES = ("item", "tolist", "cpu", "numpy", "__int__", "__float__", "__bool__")

    def __init__(self):
        self.counts = {}
        self._orig = {}

    def __enter__(self):
        for n in self.NAMES:
            orig = getattr(torch.Tensor, n)
            self._orig[n] = orig

            def make(n=n, orig=orig):
                def wrapper(self_, *a, **k):
                    # A CPU tensor's .cpu() is a no-op, not a device sync.
                    if not (n == "cpu" and self_.device.type == "cpu"):
                        D2HCounter._active.counts[n] = (
                            D2HCounter._active.counts.get(n, 0) + 1
                        )
                    return orig(self_, *a, **k)

                return wrapper

            setattr(torch.Tensor, n, make())
        D2HCounter._active = self
        return self

    def __exit__(self, *exc):
        for n, orig in self._orig.items():
            setattr(torch.Tensor, n, orig)
        D2HCounter._active = None

    @property
    def total(self):
        return sum(self.counts.values())


def _p50(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def timed(fn, iters, warmup):
    """First call reported separately: on this stack it is dominated by kernel
    compile/tiling selection and is 8x the steady state for the DSA layer (PLAN)."""
    torch.npu.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.npu.synchronize()
    first = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t) * 1e3)
    return first, _p50(samples), min(samples), max(samples)


def synthetic_topk(n_tok, n_exp, k, weights, dev, mode):
    """Same token count, two extreme expert distributions.

    ``npu_grouped_matmul`` splits its work by expert, so the per-group row counts are
    its tiling input. ``uniform`` spreads the T*k rows over all 288 groups (tiny groups,
    many of them); ``concentrated`` puts every token on the same k experts (k fat
    groups). Which is faster is not obvious in advance -- that is the point of measuring
    it -- and real routing is neither.
    """
    if mode == "uniform":
        base = torch.arange(n_tok * k, device=dev, dtype=torch.int32) % n_exp
        ids = base.view(n_tok, k)
    elif mode == "concentrated":
        ids = torch.arange(k, device=dev, dtype=torch.int32).repeat(n_tok, 1)
    else:
        raise ValueError(mode)
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    return StandardTopKOutput(weights[:n_tok].contiguous(), ids, None)


def bench(cfg, gate, topk, layer, runner, dispatcher, shared_mlp, w13, w2,
          x_all, dev, shapes, iters, warmup):
    print("\n=== steady-state latency, ONE rank (one die) of the TP16 layer ===")
    print("  excluded on purpose: the TP all-reduce after the MoE (16-rank collective,"
          "\n  not measurable here); weights stay resident because one layer is replayed,"
          "\n  so no cross-layer HBM pressure and no weight streaming;"
          "\n  no attention, no norm, no hyper-connection.")
    quant_info = attach_weights(layer, w13, w2)
    for n_tok, tag in shapes:
        n_tok = min(n_tok, x_all.shape[0])
        x = x_all[x_all.shape[0] - n_tok:].to(torch.bfloat16).to(dev)
        rl = gate(x)
        tko = topk(x, rl)

        def whole():
            r = run_routed(x, topk(x, gate(x)), quant_info, runner, dispatcher)
            return r * cfg.routed_scaling_factor + shared_mlp(x)

        segs = {
            "router GEMM (MoEGate)": lambda: gate(x),
            "TopK (npu_moe_gating_top_k)": lambda: topk(x, rl),
            "routed experts (dispatch+gmm+act+gmm+combine)":
                lambda: run_routed(x, tko, quant_info, runner, dispatcher),
            "shared expert (DeepseekV2MLP)": lambda: shared_mlp(x),
        }
        first, p50, lo, hi = timed(whole, iters, warmup)
        print(f"\n  [{tag}] tokens={n_tok}")
        print(f"    whole layer            first {first:8.2f} ms   "
              f"p50 {p50:7.3f} ms   (min {lo:.3f} max {hi:.3f})")
        for name, fn in segs.items():
            f1, m, _, _ = timed(fn, iters, warmup)
            print(f"    {name:<46} first {f1:8.2f} ms   p50 {m:7.3f} ms")
        with D2HCounter() as c:
            whole()
        print(f"    host syncs per forward: {c.total}"
              + (f"  {c.counts}" if c.counts else "  (none)"))

        # expert-distribution sensitivity, routed path only
        n_exp, k = cfg.n_routed_experts, cfg.num_experts_per_tok
        w = tko.topk_weights
        real = timed(lambda: run_routed(x, tko, quant_info, runner, dispatcher),
                     iters, warmup)[1]
        line = [f"    routed path vs expert distribution (same {n_tok} tokens): "
                f"real {real:.3f} ms"]
        for mode in ("uniform", "concentrated"):
            st = synthetic_topk(n_tok, n_exp, k, w, dev, mode)
            t = timed(lambda st=st: run_routed(x, st, quant_info, runner, dispatcher),
                      iters, warmup)[1]
            line.append(f"{mode} {t:.3f} ms")
        print("   ".join(line))
        used = int(torch.unique(tko.topk_ids).numel())
        # A bandwidth floor to compare the measured time against: the grouped matmul
        # only touches the experts that got rows.
        per_expert_bytes = (2 * cfg.moe_intermediate_size // 16 * cfg.hidden_size
                            + cfg.hidden_size * cfg.moe_intermediate_size // 16) * 2
        moved = used * per_expert_bytes / 1e6
        print(f"    experts actually selected: {used}/{n_exp}   "
              f"expert weight traffic this rank: {moved:.0f} MB "
              f"(all 288 would be {n_exp * per_expert_bytes / 1e6:.0f} MB)")


# ------------------------------------------------------- DeepEP swiglu_limit check
def verify_deepep_clamp(cfg, gate, topk, layer, dispatcher, w13, w2, x_all, dev):
    """Score the three routed-expert activations on a *real* gmm1 output.

    PLAN §4 defect 1 lives in an activation the shipped GLM recipe never builds
    (``--moe-a2a-backend none`` -> AscendTP -> ``NPUSwiglu``). To measure the defect and
    its fix without a 16-rank DeepEP job, take the gate/up tensor the real dispatcher +
    real w13 gemm produce and hand it to each activation in turn. The gemm and its input
    are identical for all three, so the only variable is the activation.
    """
    from sglang.srt.hardware_backend.npu.moe.activation import (
        NPUSwiglu,
        NPUSwigluDeepEPKernel,
    )
    from reference.tolerance import ABS_MIN, SLACK, noise_floor, rel_err

    n_tok = min(8192, x_all.shape[0])
    x = x_all[x_all.shape[0] - n_tok:].to(torch.bfloat16).to(dev)
    quant_info = attach_weights(layer, w13, w2)
    disp = dispatcher.dispatch(x, topk(x, gate(x)))
    gate_up = layer.w13_kernel.apply(
        quant_info, disp.hidden_states, disp.expert_tokens, pertoken_scale=None,
        output_dtype=torch.bfloat16, weight_prefix="w13",
        group_list_type=disp.group_list_type,
    )
    limit = float(cfg.swiglu_limit)
    g, u = gate_up.float().cpu().chunk(2, dim=-1)
    ref32 = torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)
    g16, u16 = g.to(torch.bfloat16), u.to(torch.bfloat16)
    ref16 = (torch.nn.functional.silu(g16.clamp(max=limit))
             * u16.clamp(-limit, limit)).float()
    floor = noise_floor(ref32, ref16)
    budget = max(floor * SLACK, ABS_MIN)
    over = int((g > limit).sum()) + int((u.abs() > limit).sum())

    variants = {
        "NPUSwiglu (ascend_tp, shipped)": NPUSwiglu(swiglu_limit=limit),
        "NPUSwigluDeepEPKernel, pre-fix (no swiglu_limit)":
            NPUSwigluDeepEPKernel(need_quant=False, alpha=None, limit=None),
        "NPUSwigluDeepEPKernel, fixed (swiglu_limit forwarded)":
            NPUSwigluDeepEPKernel(need_quant=False, alpha=None, limit=None,
                                  swiglu_limit=limit),
    }
    print(f"\n=== routed-expert activation on a real gmm1 output "
          f"({tuple(gate_up.shape)}, {over} elements past the limit) ===")
    print(f"  bf16 noise floor {floor:.3e}, budget {budget:.3e} (slack {SLACK})")
    for name, act in variants.items():
        if isinstance(act, NPUSwigluDeepEPKernel):
            out, _ = act._apply_activation(gate_up.clone(), disp.expert_tokens,
                                           disp.group_list_type)
        else:
            out, _ = act._apply_activation(gate_up.clone())
        err = rel_err(out.float().cpu(), ref32)
        mark = "ok  " if err <= budget else "FAIL"
        print(f"  [{mark}] {name:<54} err={err:.3e}  "
              f"{err / budget:.2f}x budget")


# ------------------------------------------------------------ router dtype check
def verify_router_dtype(cfg, gate, topk, meta, x_all, dev, iters, warmup):
    """What the config asks for vs what the NPU does, measured on device.

    ``moe_router_dtype: float32`` is in GLM-5.3-Flash's config.json and is read by
    nobody in sglang. ``MoEGate.forward`` has a bf16 ``F.linear`` on the non-CUDA
    branch, so the logits are rounded to bf16 before the top-k ever sees them (the NPU
    top-k then upcasts them again, which cannot undo it). This scores both dtypes
    against the fp32 reference's expert set and prices the difference.
    """
    from reference.tolerance import rel_err

    ref_all = meta["topk_ids_fp32"].to(torch.int64)
    floor_all = meta["topk_ids_bf16"].to(torch.int64)
    k = ref_all.shape[1]

    def overlap(a, b):
        hit = (a.sort(-1).values.unsqueeze(-1)
               == b.sort(-1).values.unsqueeze(-2)).any(-1)
        return hit.sum(-1).float().mean().item() / k

    print("\n=== router GEMM dtype: shipped bf16 vs the configured float32 ===")
    for n_tok in (16, min(8192, x_all.shape[0])):
        rows = slice(x_all.shape[0] - n_tok, x_all.shape[0])
        x = x_all[rows].to(torch.bfloat16).to(dev)
        W = gate.weight
        lg16 = torch.nn.functional.linear(x, W)
        lg32 = torch.nn.functional.linear(x.float(), W.float())
        ids16 = topk(x, lg16).topk_ids.to(torch.int64).cpu()
        ids32 = topk(x, lg32).topk_ids.to(torch.int64).cpu()
        ref, flo = ref_all[rows], floor_all[rows]
        t16 = timed(lambda: torch.nn.functional.linear(x, W), iters, warmup)[1]
        t32 = timed(lambda: torch.nn.functional.linear(x.float(), W.float()),
                    iters, warmup)[1]
        print(f"  tokens={n_tok}")
        print(f"    logits rel err vs fp32-on-device : bf16 "
              f"{rel_err(lg16.float().cpu(), lg32.float().cpu()):.3e}")
        print(f"    top-{k} set overlap vs the fp32 CPU reference: "
              f"bf16 {overlap(ids16, ref):.6f}   fp32 {overlap(ids32, ref):.6f}   "
              f"(bf16-reference floor {overlap(flo, ref):.6f})")
        print(f"    router GEMM p50: bf16 {t16:.3f} ms   fp32 {t32:.3f} ms   "
              f"(+{(t32 - t16) * 1e3:.0f} us)")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--tokens", default="", help="comma list; default = the whole case")
    ap.add_argument("--tp", type=int, default=16, help="shipped recipe is --tp-size 16")
    ap.add_argument("--ranks", default="", help="subset of ranks, for a quick smoke run")
    ap.add_argument("--die", type=int, default=int(os.environ.get("DIE", "12")))
    ap.add_argument("--port", type=int, default=29511)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench-iters", type=int, default=21)
    ap.add_argument("--bench-warmup", type=int, default=5)
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--verify-router-dtype", action="store_true",
                    help="bf16 vs fp32 router GEMM: expert sets and cost")
    ap.add_argument("--verify-deepep-clamp", action="store_true",
                    help="score the three swiglu variants on a real gmm1 output")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    import torch_npu  # noqa: F401

    dev = f"npu:{args.die}"
    torch.npu.set_device(dev)

    from harness import Case, check, report

    case = Case.load(args.case)
    meta = case.meta
    print(f"case {case.name}: layer {meta['layer']} seq {meta['seq_len']} "
          f"{meta['n_routed_experts']} experts top-{meta['top_k']}")

    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    get_context().set_server_args(
        ServerArgs(model_path=MODEL, device="npu", tp_size=args.tp,
                   dtype="bfloat16", moe_a2a_backend="none", trust_remote_code=True)
    )
    # RowParallelLinear.forward reaches for get_tp_group() unconditionally (the
    # symmetric-memory context), so a real -- if single-rank -- process group has to
    # exist. gloo, not hccl: a group of one performs no collective, and hccl init on a
    # bare process (no RANK/MASTER_ADDR) hangs. Nothing here communicates.
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, backend="gloo",
        distributed_init_method=f"tcp://127.0.0.1:{args.port}",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")

    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    cfg = build_config(meta)
    inter = cfg.moe_intermediate_size // args.tp
    print(f"a2a backend {get_moe_a2a_backend()}  ->  "
          f"{'DeepEP' if get_moe_a2a_backend().is_deepep() else 'AscendTP'} path; "
          f"intermediate_size_per_partition = {cfg.moe_intermediate_size}/{args.tp} "
          f"= {inter}")

    t0 = time.time()
    w13_all, w2_all, shared_w, gate_w, bias = load_experts_host(meta["layer"], cfg)
    # Slicing the 16 TP shards out of the full [E, 2I, H] tensor is a strided gather;
    # on a loaded host it is minutes, on the die it is milliseconds, and 14.5 GB of
    # bf16 experts fits in one 64 GB A3 die alongside everything else here.
    w13_all, w2_all = w13_all.to(dev), w2_all.to(dev)
    print(f"experts on device: w13 {tuple(w13_all.shape)} w2 {tuple(w2_all.shape)} "
          f"({time.time() - t0:.0f}s)", flush=True)

    ranks = ([int(r) for r in args.ranks.split(",")] if args.ranks
             else list(range(args.tp)))
    x_all = case.inputs["hidden_states"]
    tok_list = ([int(t) for t in args.tokens.split(",")] if args.tokens
                else [x_all.shape[0]])

    gate = build_router(cfg, gate_w, bias, dev)
    assert gate.weight.dtype == torch.bfloat16, (
        f"router weight is {gate.weight.dtype}; the defect-2 measurement assumes the "
        f"bf16 model dtype"
    )
    topk = build_topk(cfg, gate)
    layer, rc, runner, dispatcher = build_moe_runner(cfg, inter, meta["layer"])
    print(f"runner activation: {type(runner.activation).__name__} "
          f"(swiglu_limit={getattr(runner.activation, '_swiglu_limit', None)})   "
          f"dispatcher: {type(dispatcher).__name__}")

    exit_code = 0
    if not args.skip_check:
        # routing is replicated: every rank computes the same logits from the same
        # weights, so it is computed once and shared, exactly as TP does.
        routing = {}
        for n_tok in tok_list:
            n_tok = min(n_tok, x_all.shape[0])
            x = x_all[x_all.shape[0] - n_tok:].to(torch.bfloat16).to(dev)
            rl = gate(x)
            routing[n_tok] = (x, rl, topk(x, rl))
        acc = {n: [torch.zeros(n, cfg.hidden_size, dtype=torch.float32, device=dev),
                   torch.zeros(n, cfg.hidden_size, dtype=torch.float32, device=dev)]
               for n in routing}
        for r in ranks:
            t1 = time.time()
            w13, w2 = make_shard(w13_all, w2_all, r, inter, cfg.moe_intermediate_size)
            qi = attach_weights(layer, w13, w2)
            sh_mlp = build_shared(cfg, inter, shared_w, r, dev)
            for n_tok, (x, rl, tko) in routing.items():
                acc[n_tok][0] += run_routed(x, tko, qi, runner, dispatcher).float()
                acc[n_tok][1] += sh_mlp(x).float()
            del w13, w2, qi, sh_mlp
            layer.w13_weight = layer.w2_weight = None
            torch.npu.empty_cache()
            print(f"  rank {r:>2}/{args.tp} done ({time.time() - t1:.1f}s)", flush=True)

        for n_tok, (x, rl, tko) in routing.items():
            rows = slice(x_all.shape[0] - n_tok, x_all.shape[0])
            # DeepseekV2MoE.forward applies the routed scaling on the output because
            # should_fuse_routed_scaling_factor_in_topk is False for unquantized NPU.
            routed = acc[n_tok][0] * cfg.routed_scaling_factor
            sh = acc[n_tok][1]
            cand = {"routed_out": routed, "shared_out": sh, "moe_out": routed + sh,
                    "router_logits": rl.float()}
            sub = Case(case.name, {"hidden_states": x_all[rows]},
                       {k: v[rows] for k, v in case.ref_fp32.items() if k in cand},
                       {k: v[rows] for k, v in case.ref_bf16.items() if k in cand},
                       meta)
            res = check(sub, cand)

            ref_ids = meta["topk_ids_fp32"][rows].to(torch.int64)
            floor_ids = meta["topk_ids_bf16"][rows].to(torch.int64)
            npu_ids = tko.topk_ids.to(torch.int64).cpu()
            k = ref_ids.shape[1]

            def overlap(a, b, k=k):
                hit = (a.sort(-1).values.unsqueeze(-1)
                       == b.sort(-1).values.unsqueeze(-2)).any(-1)
                return hit.sum(-1).float().mean().item() / k

            ov, ovf = overlap(npu_ids, ref_ids), overlap(floor_ids, ref_ids)
            bad = int((~(npu_ids.sort(-1).values
                         == ref_ids.sort(-1).values).all(-1)).sum())
            extra = (f"tokens={n_tok}  ranks={len(ranks)}/{args.tp}  "
                     f"router bf16 -> top-{k} expert-set overlap vs fp32 ref "
                     f"{ov:.6f} ({bad}/{n_tok} tokens differ); "
                     f"bf16-reference floor {ovf:.6f}")
            exit_code |= report(f"{case.name}  T={n_tok}", res, extra)
            if ov < ovf:
                print("  [note] the NPU router picks a worse expert set than the bf16 "
                      "reference: the reference upcasts to fp32 (HF "
                      "Glm5NextTextTopkRouter), deepseek_v2.py MoEGate does not.")

    if args.verify_router_dtype:
        verify_router_dtype(cfg, gate, topk, meta, x_all, dev,
                            args.bench_iters, args.bench_warmup)

    if args.verify_deepep_clamp:
        w13, w2 = make_shard(w13_all, w2_all, 0, inter, cfg.moe_intermediate_size)
        verify_deepep_clamp(cfg, gate, topk, layer, dispatcher, w13, w2, x_all, dev)

    if args.bench:
        w13, w2 = make_shard(w13_all, w2_all, 0, inter, cfg.moe_intermediate_size)
        sh_mlp = build_shared(cfg, inter, shared_w, 0, dev)
        bench(cfg, gate, topk, layer, runner, dispatcher, sh_mlp,
              w13, w2, x_all, dev,
              [(16, "decode bs=16"), (8192, "prefill chunk 8192")],
              args.bench_iters, args.bench_warmup)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
