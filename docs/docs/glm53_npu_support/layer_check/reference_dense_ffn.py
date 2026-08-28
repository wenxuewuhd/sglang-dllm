#!/usr/bin/env python
"""Stage A for the dense FFN of GLM-5.3-Flash layers 0, 1 and 2.

`first_k_dense_replace = 3`, so the first three decoder layers have a plain
`Glm5NextTextMLP` (hidden 4096 -> intermediate 12288 -> 4096, no routing) while layers
3..44 are MoE.  That makes this the simplest module in the network and the right one to
prove the harness itself on.

Run with the *reference* venv (`$ROOT/.venv-ref`, transformers 5.16.1)::

    source $ROOT/env.sh
    OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 \
        $ROOT/.venv-ref/bin/python reference_dense_ffn.py \
            --layer 0 --tokens 128 --out $ROOT/goldens/dense_ffn_layer00.pt

Real weights, real input: the network actually runs on CPU up to the layer under test
(streamed layer by layer, see `trace_reference.LayerStreamer`) and a pre-hook on
`layer.mlp` grabs the tensor the MLP genuinely receives, then aborts the forward.  For
layers 0-2 that is at most three layers of prefix, so it is cheap.

--------------------------------------------------------------------------------
Three reference tensors, not one
--------------------------------------------------------------------------------

    gate_up   [S, 2*12288]   cat([gate_proj(x), up_proj(x)], -1), BEFORE the clamp
    act       [S, 12288]     silu(clamp(gate, max=L)) * clamp(up, -L, L)
    out       [S, 4096]      down_proj(act)

`out` alone would say "the FFN is wrong" and nothing more.  The split localises the
failure to one of three places, and in particular it isolates the **clamped SwiGLU**,
which is where this model's activation differs from a stock SwiGLU:

    gate: clamped ABOVE only    (min=None, max=swiglu_limit)
    up:   clamped on BOTH sides (min=-swiglu_limit, max=+swiglu_limit)
    then  silu(gate) * up       -- plain SiLU, no alpha, no bias, not interleaved

`swiglu_limit` is **10.0** for GLM-5.3-Flash (config.json), not the 7.0 that some
gpt-oss-derived kernels default to.  `rms_norm_eps` is 1e-5, not DeepSeek-V4's 1e-6.

The order of the two halves in `gate_up` is [gate | up], matching both HF's
`chunk(2, dim=-1)` and sglang's `MergedColumnParallelLinear(h, [inter] * 2)`.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Case  # noqa: E402
from trace_reference import DEFAULT_MODEL, build_streaming_model, make_input_ids  # noqa: E402


class _StopAtMLP(Exception):
    """Raised from the capture hook to abort the forward once we have the input."""


def capture_mlp_input(
    model_dir: Path, layer: int, input_ids: torch.Tensor, prefix_dtype: torch.dtype
):
    """Run the real network up to `layer.mlp` and return (its input, the live module).

    The forward is aborted from inside the hook, so `layer.mlp` never runs and -- more
    importantly -- the layer is still materialised when we return, so its real weights
    can be reused without a second pass over the checkpoint.
    """
    model, _captured, _routing, cfg, _streamer = build_streaming_model(
        model_dir, prefix_dtype, num_layers=layer + 1, verbose=True
    )
    if cfg.mlp_layer_types[layer] != "dense":
        raise SystemExit(
            f"layer {layer} has a {cfg.mlp_layer_types[layer]!r} mlp; this script is "
            f"for the dense ones (0..{cfg.first_k_dense_replace - 1})"
        )
    target = model.layers[layer].mlp
    box = {}

    def hook(_mod, args, kwargs):
        got = kwargs.get("x")
        if got is None:
            got = args[0]
        box["input"] = got.detach().to(torch.float32).clone()
        raise _StopAtMLP

    handle = target.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    except _StopAtMLP:
        pass
    finally:
        handle.remove()
    if "input" not in box:
        raise SystemExit("the capture hook never fired -- is layer.mlp really dense?")
    # deepcopy before the model (and its meta-restoring hooks) go away
    mlp = copy.deepcopy(target)
    del model
    return box["input"], mlp, cfg


def eval_mlp(state: dict, hidden: torch.Tensor, limit: float, dtype: torch.dtype):
    """Evaluate the dense FFN in one dtype, returning the three staged tensors.

    Written out rather than calling `Glm5NextTextMLP.forward` so that `gate_up` and
    `act` are observable; `out` is asserted against the real module's own output by the
    caller, so this staging cannot drift from HF's definition unnoticed.
    """
    x = hidden.to(dtype)
    w_gate = state["gate_proj.weight"].to(dtype)
    w_up = state["up_proj.weight"].to(dtype)
    w_down = state["down_proj.weight"].to(dtype)

    gate = torch.nn.functional.linear(x, w_gate)
    up = torch.nn.functional.linear(x, w_up)
    gate_c = gate.clamp(min=None, max=limit)
    up_c = up.clamp(min=-limit, max=limit)
    act = torch.nn.functional.silu(gate_c) * up_c
    out = torch.nn.functional.linear(act, w_down)
    return {
        "gate_up": torch.cat([gate, up], dim=-1).float(),
        "act": act.float(),
        "out": out.float(),
    }


def build_case(
    mlp, cfg, hidden: torch.Tensor, layer: int, meta_extra: dict, scale: float = 1.0
) -> Case:
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP

    limit = float(cfg.swiglu_limit)
    state = {k: v.detach().float() for k, v in mlp.state_dict().items()}
    hidden = hidden[0] if hidden.dim() == 3 else hidden  # drop the batch axis
    if scale != 1.0:
        hidden = hidden * scale

    tensors = {}
    for dtype, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        got = eval_mlp(state, hidden, limit, dtype)

        # Cross-check the staged arithmetic against HF's own module, in the same dtype.
        # Same operations in the same order, so this is exact; if it ever is not, the
        # staging above has drifted from the model definition and the case is invalid.
        ref = Glm5NextTextMLP(cfg)
        ref.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)
        ref = ref.to(dtype).eval()
        with torch.no_grad():
            hf_out = ref(hidden.to(dtype)).float()
        if not torch.equal(hf_out, got["out"]):
            raise SystemExit(
                f"{tag}: staged FFN does not reproduce Glm5NextTextMLP bit for bit "
                f"(max abs diff {(hf_out - got['out']).abs().max():.3e}). The staging "
                f"in eval_mlp() no longer matches the HF definition; fix it there."
            )
        del ref

        tensors[tag] = got
        frac = _clip_fraction(got["gate_up"], limit)
        print(
            f"  {tag}: gate_up absmax={got['gate_up'].abs().max():.4f} "
            f"clipped={int(frac * got['gate_up'].numel()):,}/{got['gate_up'].numel():,} "
            f"({frac:.3e})  out absmax={got['out'].abs().max():.4f}"
        )

    meta = {
        "module": "dense_ffn",
        "layer": layer,
        "tokens": int(hidden.shape[0]),
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "hidden_act": cfg.hidden_act,
        "swiglu_limit": limit,
        "rms_norm_eps": cfg.rms_norm_eps,
        "layout.gate_up": "[tokens, 2*intermediate]; gate half first, then up -- "
        "matches MergedColumnParallelLinear(hidden, [intermediate] * 2)",
        "clamp": "gate: (-inf, +limit]; up: [-limit, +limit]; then silu(gate) * up",
        "input_scale": scale,
        "synthetic_input": scale != 1.0,
        "clip_fraction_fp32": _clip_fraction(tensors["fp32"]["gate_up"], limit),
        "source": "transformers Glm5NextTextMLP (CPU)",
        **meta_extra,
    }
    return Case(
        name=f"dense_ffn.layer{layer:02d}",
        inputs={"hidden_states": hidden.float().contiguous()},
        ref_fp32=tensors["fp32"],
        ref_bf16=tensors["bf16"],
        meta=meta,
    )


def _clip_fraction(gate_up: torch.Tensor, limit: float) -> float:
    """How much of this case actually exercises the clamp.

    A case where nothing clips cannot distinguish a correct clamp from no clamp at all,
    so the number is printed at generation time rather than discovered later.
    """
    half = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :half], gate_up[..., half:]
    n = (gate > limit).sum() + (up.abs() > limit).sum()
    return float(n) / gate_up.numel()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--layer", type=int, default=0, help="one of the dense layers, 0-2")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--prefix-dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="dtype of the layers before the one under test (the serving dtype by "
        "default; it decides which input the case tests with, not the verdict -- both "
        "references see the same input, byte for byte)",
    )
    ap.add_argument(
        "--scale-input",
        type=float,
        default=1.0,
        help="multiply the captured hidden state by this before evaluating. On real "
        "inputs the dense layers never reach swiglu_limit=10 (measured: max |gate_up| "
        "is 2.17 at layer 2), so a scale=1 case cannot tell a correct clamp apart from "
        "no clamp at all. Use ~8 to make the clamp bind. The resulting case is marked "
        "synthetic_input=True in its meta -- it tests the activation, not the model.",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoConfig

    probe = AutoConfig.from_pretrained(str(args.model)).text_config
    input_ids, id_source = make_input_ids(
        args.model, args.tokens, args.seed, probe.vocab_size
    )
    print(
        f"running layers 0..{args.layer} on {input_ids.shape[1]} tokens "
        f"({args.prefix_dtype} prefix, {id_source} ids) ..."
    )
    hidden, mlp, cfg = capture_mlp_input(
        args.model, args.layer, input_ids, getattr(torch, args.prefix_dtype)
    )
    print(
        f"captured mlp input {tuple(hidden.shape)} absmax={hidden.abs().max():.4f} "
        f"rms={hidden.pow(2).mean().sqrt():.4f}"
    )

    case = build_case(
        mlp,
        cfg,
        hidden,
        args.layer,
        {
            "seed": args.seed,
            "prefix_dtype": args.prefix_dtype,
            "id_source": id_source,
            "input_ids": input_ids[0].tolist(),
        },
        scale=args.scale_input,
    )
    for name, ref32 in case.ref_fp32.items():
        ref16 = case.ref_bf16[name]
        denom = ref32.norm().clamp_min(1e-30)
        print(
            f"  floor {name:<12} {(ref32 - ref16).norm() / denom:.3e} "
            f"{tuple(ref32.shape)}"
        )
    case.save(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
