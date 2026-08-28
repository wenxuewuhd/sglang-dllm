#!/usr/bin/env python
"""Stage B: run the real sglang dense FFN on the NPU and score it against stage A.

    source $ROOT/env.sh
    PYTHONPATH=$REPO/python ASCEND_RT_VISIBLE_DEVICES=15 \
        $VENV/bin/python check_dense_ffn.py \
            --case $ROOT/goldens/dense_ffn_layer02_t8192.pt

The module under test is the production one: `DeepseekV2MLP`, which
`models/glm5_next.py` imports verbatim as `Glm5NextMLP` for the first
`first_k_dense_replace = 3` layers.  Nothing here reimplements the FFN -- only the
weights are loaded by hand, from the same checkpoint the server reads.

--------------------------------------------------------------------------------
Real deployment shapes, not toy shapes
--------------------------------------------------------------------------------

`$ROOT/run/launch_glm_bf16.sh` serves this model as **pure TP16** (no DP, no EP),
bf16, `page-size 64`, `context-length 32768`, `max-running-requests 16`.  So the shape
a single card's dense FFN actually sees is (verified by constructing the real layers,
not assumed):

    gate_up_proj.weight   [1536, 4096]    = MergedColumnParallelLinear(4096, [12288]*2)
                                            with output_partition_sizes [768, 768]
    down_proj.weight      [4096,  768]    = RowParallelLinear(12288, 4096),
                                            input_size_per_partition 768

i.e. per card the FFN is `[M, 4096] -> [M, 1536] -> [M, 768] -> [M, 4096]`, and the
16 partial outputs are summed by the all-reduce.  This script runs **all 16 rank
shards on one device** and sums them, which reproduces every per-card GEMM shape
exactly.  What it does *not* reproduce is the HCCL all-reduce itself; the summation
here is a local one, and the report separates the two so a reduction problem cannot be
mistaken for a GEMM problem.

M is swept over the values the deployment actually produces: decode batches of 1 and
16 (`--max-running-requests 16`), and a full chunked-prefill chunk of 8192 (the
`chunked_prefill_size` this config resolves to for a 64 GB device).  The dense FFN is a
strictly per-token map -- no token mixes with another -- so one reference of N tokens is
a valid reference for every M <= N, and `--batches` slices it.  M matters here because
**NPU bf16 matmul is not batch-shape invariant** and the kernel's tiling is chosen per
shape; it does not matter for the mathematics.

--------------------------------------------------------------------------------
What gets scored
--------------------------------------------------------------------------------

Per rank, and then once for the summed result:

    gate_up.rankNN   the merged GEMM output, BEFORE the clamp
    act.rankNN       after clamp + silu * up, i.e. what down_proj receives
    out.allreduce    the summed [M, 4096]

Splitting it that way is what makes the **clamped SwiGLU** falsifiable on its own.  GLM
needs `alpha=1.0, limit=10.0, bias=0.0, interleaved=False`; a kernel defaulted for
gpt-oss (`alpha=1.702, limit=7.0, bias=1.0, interleaved=True`) is wrong by ~109x and
raises nothing.  On this path sglang does not call a fused clipped-swiglu kernel at all
-- `DeepseekV2MLP.forward` takes the `_is_npu` branch, clamps the two halves explicitly
and calls `SiluAndMul.forward_npu` (`torch_npu.npu_swiglu`) -- so the trap does not
apply here, but the check still has to *prove* that rather than assume it.

One warning the check prints for you: on real layer-0/1/2 inputs the activations never
reach 10.0 (measured max |gate_up| is 2.17), so a scale=1 case cannot tell a correct
clamp from a missing one.  Generate a clamp-exercising case with
`reference_dense_ffn.py --scale-input 8` to actually gate the clamp.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch_npu  # noqa: F401  (registers the npu device and the torch_npu ops)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing  # noqa: E402
from harness import Case, Result, check, report  # noqa: E402

DEFAULT_MODEL = Path("/mnt/workspace/models/GLM-5.3-Flash-BF16")
DEV = "npu"

#: from $ROOT/run/launch_glm_bf16.sh
TP_SIZE = 16
#: decode batches (max-running-requests 16) and one full prefill chunk
DEFAULT_BATCHES = (1, 16, 8192)


def init_single_process_group(port: int) -> None:
    """A world-size-1 process group, only so `get_tp_group()` resolves.

    `RowParallelLinear.forward` reaches for the TP group unconditionally (to build a
    symmetric-memory context) even with `reduce_results=False`, so a module that never
    communicates still needs one to exist.  This group is world size 1 and is never
    used to communicate; the *sharding* comes from the explicit `tp_rank` / `tp_size`
    passed to each linear layer, which is what makes the per-card shapes real.
    """
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    torch.npu.set_device(0)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="hccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)


def load_shard(model_dir: Path, layer: int, rank: int, tp_size: int):
    """The gate/up/down slices rank `rank` owns, laid out as sglang expects them."""
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    handles: Dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(str(model_dir / shard), framework="pt")
        return handles[shard].get_tensor(name)

    p = f"model.language_model.layers.{layer}.mlp."
    gate = get(p + "gate_proj.weight")
    up = get(p + "up_proj.weight")
    down = get(p + "down_proj.weight")
    inter = gate.shape[0]
    per = inter // tp_size
    lo, hi = rank * per, (rank + 1) * per
    return {
        # MergedColumnParallelLinear concatenates the shards of its output_sizes in
        # order, so this rank's gate rows come first and its up rows second.
        "gate_up_proj.weight": torch.cat([gate[lo:hi], up[lo:hi]], dim=0),
        # RowParallelLinear splits the *input* dimension.
        "down_proj.weight": down[:, lo:hi],
    }


def build_mlp(cfg_meta: dict, rank: int, tp_size: int, weights: dict):
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP

    mlp = DeepseekV2MLP(
        hidden_size=cfg_meta["hidden_size"],
        intermediate_size=cfg_meta["intermediate_size"],
        hidden_act=cfg_meta["hidden_act"],
        tp_rank=rank,
        tp_size=tp_size,
        swiglu_limit=cfg_meta["swiglu_limit"],
        # The all-reduce is the thing this single-device harness cannot do; the caller
        # sums the 16 partials instead, and says so in the report.
        reduce_results=False,
    )
    sd = {k: v.to(torch.bfloat16) for k, v in weights.items()}
    missing, unexpected = mlp.load_state_dict(sd, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected weights for rank {rank}: {unexpected}")
    if missing:
        raise SystemExit(f"missing weights for rank {rank}: {missing}")
    return mlp.to(DEV).eval()


def run_rank(mlp, x: torch.Tensor):
    """One rank's forward, returning (gate_up before clamp, act, partial out)."""
    box = {}

    def gu_hook(_m, _a, out):
        box["gate_up"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

    def down_hook(_m, args, _kwargs):
        box["act"] = args[0].detach().clone()

    h1 = mlp.gate_up_proj.register_forward_hook(gu_hook)
    h2 = mlp.down_proj.register_forward_pre_hook(down_hook, with_kwargs=True)
    try:
        with torch.no_grad():
            out = mlp(x)
    finally:
        h1.remove()
        h2.remove()
    out = out[0] if isinstance(out, tuple) else out
    if "gate_up" not in box or "act" not in box:
        raise SystemExit(
            "the gate_up / down_proj hooks did not both fire -- DeepseekV2MLP took a "
            "fused fast path, so this check is not measuring what it claims to. "
            "Inspect DeepseekV2MLP.forward for the branch that was taken."
        )
    return box["gate_up"], box["act"], out


def time_one_rank(mlp, x: torch.Tensor, label: str, shape: str, iters: int):
    """Steady-state cost of ONE rank's dense FFN at this shape.

    Phased through the real module's own submodules rather than a copy of its
    arithmetic, so the breakdown cannot drift from what the model runs. The clamp and
    the activation are inside `mlp(...)` between the two projections, so they show up in
    the total but not as their own phase -- `DeepseekV2MLP.forward` has no seam there to
    mark without reimplementing it.
    """

    def one_call(t):
        with t.phase("gate_up_proj"):
            gate_up, _ = mlp.gate_up_proj(x)
        with t.phase("clamp+swiglu"):
            g, u = gate_up.chunk(2, dim=-1)
            lim = float(mlp.swiglu_limit)
            act = mlp.act_fn(
                torch.cat([g.clamp(max=lim), u.clamp(min=-lim, max=lim)], dim=-1)
            )
        with t.phase("down_proj"):
            out, _ = mlp.down_proj(act)
        return out

    def whole_module(t):
        return mlp(x)

    # The real forward is measured FIRST, so its `first_ms` is the only genuine
    # cold-start number here: whichever variant runs second finds every kernel already
    # compiled. `timing.render` flags the second one when it shows.
    t_whole = timing.measure(
        whole_module, label=f"{label} (real forward)", shape=shape, iters=iters
    )
    t_phased = timing.measure(
        one_call, label=f"{label} (phased, warm)", shape=shape, iters=iters
    )
    syncs = timing.count_syncs(whole_module)
    return [t_whole, t_phased], {t_whole.label: syncs}


def rank_slices(case: Case, rank: int, tp_size: int, m: int):
    """The (fp32, bf16) reference pair for one rank, cut out of the full reference."""
    pair = []
    for refs in (case.ref_fp32, case.ref_bf16):
        gu, act = refs["gate_up"], refs["act"]
        inter = act.shape[-1]
        per = inter // tp_size
        lo, hi = rank * per, (rank + 1) * per
        pair.append(
            {
                "gate_up": torch.cat(
                    [gu[:m, lo:hi], gu[:m, inter + lo : inter + hi]], dim=-1
                ),
                "act": act[:m, lo:hi],
            }
        )
    return pair


def _prefixed(results: List[Result], suffix: str) -> List[Result]:
    return [
        Result(f"{r.tensor}.{suffix}", r.err, r.floor, r.budget, r.note) for r in results
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--tp-size", type=int, default=TP_SIZE)
    ap.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCHES),
        help="token counts to run. Defaults to the deployment's own: decode 1 and 16, "
        "and one full 8192-token prefill chunk.",
    )
    ap.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=None,
        help="which TP ranks to run (default: all of them). Every rank holds a "
        "different slice of the intermediate dimension, so they are different "
        "numbers, not repeats.",
    )
    ap.add_argument(
        "--timing",
        action="store_true",
        help="also report steady-state per-rank timing (timing.py protocol): first "
        "call separately, steady-state p50, phase breakdown and host-sync count",
    )
    ap.add_argument("--timing-iters", type=int, default=30)
    ap.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="run only the timing. Needed for a real cold-start number: the accuracy "
        "pass runs the same shapes first and pays the compile cost, after which "
        "timing.py's first_ms is not a cold start any more.",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=29871,
        help="rendezvous port for the world-size-1 process group; change it if two "
        "checks run at once",
    )
    args = ap.parse_args()

    case = Case.load(args.case)
    meta = case.meta
    if meta.get("module") != "dense_ffn":
        raise SystemExit(f"{args.case} is a {meta.get('module')!r} case, not dense_ffn")
    layer = int(meta["layer"])
    n_ref = int(case.inputs["hidden_states"].shape[0])
    ranks = args.ranks if args.ranks is not None else list(range(args.tp_size))

    print(f"case {case.name}: {n_ref} reference tokens, layer {layer}")
    print(
        f"  hidden={meta['hidden_size']} intermediate={meta['intermediate_size']} "
        f"tp_size={args.tp_size} -> per-card [{meta['intermediate_size'] * 2 // args.tp_size}"
        f", {meta['hidden_size']}] gate_up, "
        f"[{meta['hidden_size']}, {meta['intermediate_size'] // args.tp_size}] down"
    )
    print(f"  swiglu_limit={meta['swiglu_limit']} rms_norm_eps={meta['rms_norm_eps']}")
    clip = meta.get("clip_fraction_fp32")
    if clip is not None and clip == 0.0:
        print(
            "  WARNING: nothing in this case reaches swiglu_limit, so it CANNOT "
            "distinguish a correct clamp from no clamp. Regenerate with "
            "reference_dense_ffn.py --scale-input 8 to gate the activation."
        )
    if meta.get("synthetic_input"):
        print(
            f"  NOTE: synthetic input (hidden state scaled by "
            f"{meta.get('input_scale')}); this case tests the activation, not the model."
        )

    init_single_process_group(args.port)
    if args.timing:
        timing.prime_device(DEV)

    print("\nloading TP shards ...", flush=True)
    mlps = {
        r: build_mlp(meta, r, args.tp_size, load_shard(args.model, layer, r, args.tp_size))
        for r in ranks
    }

    exit_code = 0
    x_full = case.inputs["hidden_states"].to(torch.bfloat16)
    for m in args.batches:
        if m > n_ref:
            print(
                f"\n=== M={m}: SKIPPED, the case only has {n_ref} tokens. Regenerate "
                f"with reference_dense_ffn.py --tokens {m}. ==="
            )
            exit_code = 1
            continue
        x = x_full[:m].to(DEV)
        results: List[Result] = []
        partials = []
        if args.skip_accuracy:
            print(f"\n=== dense FFN layer {layer}, M={m}: accuracy skipped ===")
            if args.timing:
                r0 = ranks[0]
                ts, syncs = time_one_rank(
                    mlps[r0],
                    x,
                    label=f"dense FFN layer {layer} rank{r0:02d} COLD",
                    shape=f"M={m}, tp={args.tp_size}",
                    iters=args.timing_iters,
                )
                print(timing.render(ts, syncs))
            continue

        for r in ranks:
            gu, act, out = run_rank(mlps[r], x)
            partials.append(out)
            ref32, ref16 = rank_slices(case, r, args.tp_size, m)
            sub = Case(f"{case.name}.rank{r:02d}", {}, ref32, ref16, {})
            results += _prefixed(check(sub, {"gate_up": gu, "act": act}), f"rank{r:02d}")

        # bf16 accumulation, the way an all-reduce over bf16 tensors would, and again in
        # fp32. If only the bf16 one fails, the GEMMs are fine and the reduction is the
        # lossy step -- a different bug with a different fix.
        summed_bf16 = partials[0].clone()
        for p in partials[1:]:
            summed_bf16 += p
        summed_fp32 = torch.stack([p.float() for p in partials]).sum(0)

        if len(ranks) == args.tp_size:
            out_case = Case(
                f"{case.name}.out",
                {},
                {"out": case.ref_fp32["out"][:m]},
                {"out": case.ref_bf16["out"][:m]},
                {},
            )
            results += _prefixed(check(out_case, {"out": summed_bf16}), "allreduce_bf16")
            results += _prefixed(check(out_case, {"out": summed_fp32}), "allreduce_fp32")
        else:
            # The summed output is only meaningful when every shard of the contraction
            # is present. Saying so beats reporting a guaranteed 60x failure that means
            # nothing except "you asked for a subset of the ranks".
            print(
                f"\n  (out.allreduce not scored: {len(ranks)}/{args.tp_size} ranks "
                f"requested, so the contraction over the intermediate dimension is "
                f"incomplete. Drop --ranks to score it.)"
            )

        worst = [r for r in results if not r.ok]
        code = report(
            f"dense FFN layer {layer}, M={m}, tp={args.tp_size} "
            f"({len(ranks)} rank shard(s) on one device)",
            # Printing 33 lines per M buries the answer; print every failure and only
            # the extreme of each passing family.
            worst + _summarise_ok(results),
            extra=f"per-card gate_up [{m}, {meta['intermediate_size'] * 2 // args.tp_size}]"
            f"  act [{m}, {meta['intermediate_size'] // args.tp_size}]"
            f"  out [{m}, {meta['hidden_size']}]",
        )
        exit_code |= code

        if args.timing:
            # One rank only: every rank runs the same shapes on the same device, so
            # timing all 16 would be 16 samples of one number dressed up as 16 numbers.
            r0 = ranks[0]
            ts, syncs = time_one_rank(
                mlps[r0],
                x,
                label=f"dense FFN layer {layer} rank{r0:02d}",
                shape=f"M={m}, tp={args.tp_size}, gate_up [{m}, "
                f"{meta['intermediate_size'] * 2 // args.tp_size}]",
                iters=args.timing_iters,
            )
            print(
                timing.render(
                    ts,
                    syncs,
                    extra_exclusions=(
                        "the dense FFN is a per-token map, so 'decode bs=16 at 32k "
                        "context' reduces to M=16 rows here -- context length does not "
                        "enter this module's shapes at all",
                    ),
                )
            )
    return exit_code


def _summarise_ok(results: List[Result]) -> List[Result]:
    """The worst passing result of each tensor family, so a clean run still shows the
    number it achieved instead of only a count."""
    best: Dict[str, Result] = {}
    for r in results:
        if not r.ok:
            continue
        family = r.tensor.split(".")[0]
        cur = best.get(family)
        ratio = r.err / r.budget if r.budget else float("inf")
        if cur is None or ratio > (cur.err / cur.budget if cur.budget else 0):
            best[family] = r
    return [
        Result(f"{r.tensor} (worst of family)", r.err, r.floor, r.budget, r.note)
        for r in best.values()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
