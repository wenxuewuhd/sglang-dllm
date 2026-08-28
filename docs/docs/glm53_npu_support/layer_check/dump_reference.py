#!/usr/bin/env python
"""Stage A: produce a `harness.Case` for one layer of GLM-5.3-Flash on CPU.

Run this with the *reference* venv (`$ROOT/.venv-ref`, transformers 5.16.1), never
with the sglang venv -- sglang pins transformers 5.12.1, which does not know
`glm5_next` at all.

    $ROOT/.venv-ref/bin/python dump_reference.py --module kda --layer 0 \
        --prefill 256 --decode 8 --out $ROOT/goldens/kda_layer00.pt

What it does, for any module:

1. Instantiate the real HF model **truncated to `layer + 1` decoder layers**, so
   only the weights up to the layer under test are read off disk, and run a real
   forward on real token ids.  A forward pre-hook on the module under test
   captures the tensor that module actually receives, then aborts the forward --
   the layer's own MLP never runs.
2. Re-evaluate that one module twice on the captured input: once with fp32
   weights, once with bf16 weights.  The distance between those two is the
   acceptance budget (`harness.check`), not a hand-picked threshold.

The prefix runs in bf16 (the serving dtype); only the module under test is
evaluated in both precisions.  The prefix dtype decides *which* input the case
tests with, not whether the comparison is fair -- both references see the same
input, byte for byte.

Adding a module means adding one `ModuleSpec` to `MODULES`: where to tap the
decoder layer, and how to turn (config, weights, input) into a Case.  Nothing
else here is KDA-specific.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Callable, Dict, NamedTuple, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Case  # noqa: E402

DEFAULT_MODEL = Path("/mnt/workspace/models/GLM-5.3-Flash-BF16")


# --------------------------------------------------------------- prefix run


class _StopAtLayer(Exception):
    """Raised from the capture hook to abort the forward once we have the input."""


def capture_module_input(
    model_dir: Path,
    layer: int,
    tap: Callable[[torch.nn.Module], torch.nn.Module],
    input_ids: torch.Tensor,
    prefix_dtype: torch.dtype,
):
    """Run the real model up to `layer` and return (captured input, the module).

    Returns the tensor the tapped submodule receives as its first positional /
    `hidden_states` argument, in fp32, plus the live submodule (so its real
    weights can be reused without a second pass over the checkpoint).
    """
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextModel

    config = AutoConfig.from_pretrained(str(model_dir))
    text_config = config.text_config
    if layer >= text_config.num_hidden_layers:
        raise SystemExit(
            f"layer {layer} out of range ({text_config.num_hidden_layers} layers)"
        )
    full_layer_types = list(text_config.layer_types)
    # Truncating the config is what keeps this cheap: from_pretrained then reads
    # only the shards holding layers 0..layer, not all 599 GB.
    text_config.num_hidden_layers = layer + 1
    model = Glm5NextModel.from_pretrained(
        str(model_dir), config=config, dtype=prefix_dtype
    )
    model.eval()

    decoder_layer = model.language_model.layers[layer]
    target = tap(decoder_layer)

    box = {}

    def hook(_mod, args, kwargs):
        got = kwargs.get("hidden_states")
        if got is None:
            got = args[0]
        box["input"] = got.detach().to(torch.float32)
        raise _StopAtLayer

    handle = target.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    except _StopAtLayer:
        pass
    finally:
        handle.remove()
    if "input" not in box:
        raise SystemExit("the capture hook never fired -- wrong tap for this layer?")

    text_config.layer_types = full_layer_types
    return box["input"], target, text_config, model


def make_input_ids(n: int, vocab_size: int, seed: int) -> torch.Tensor:
    """Deterministic token ids.

    A real prompt would be nicer but is not needed: what the layer under test
    sees is the *hidden state*, and any in-distribution id sequence produces one.
    Ids are kept away from the very top of the vocab (special tokens).
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size - 1024, (1, n), generator=g)


# ------------------------------------------------------------------- KDA


def build_kda_case(
    *,
    module,
    text_config,
    hidden: torch.Tensor,
    layer: int,
    prefill: int,
    decode: int,
    ref_chunk: int,
    meta_extra: dict,
) -> Case:
    """Prefill + decode through HF's `Glm5NextTextLinearAttention`, twice.

    The decode steps are teacher-forced on the same captured hidden states, so
    the NPU side can be handed exactly the same inputs.  Prefill and decode share
    one cache, which is the point: `state.*.after_prefill` and the decode outputs
    together say whether the conv / recurrent state survives the handoff.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextLinearAttention,
    )

    win = text_config.linear_attn_config["short_conv_kernel_size"] - 1
    total = prefill + decode
    if hidden.shape[1] < total:
        raise SystemExit(
            f"captured {hidden.shape[1]} tokens, need prefill+decode={total}"
        )
    hidden = hidden[:, :total]
    state = {k: v.detach().float() for k, v in module.state_dict().items()}

    tensors = {}
    for dtype, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        ref = Glm5NextTextLinearAttention(text_config, layer_idx=layer)
        ref.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)
        ref = ref.to(dtype).eval()

        cache = DynamicCache(config=text_config)
        with torch.no_grad():
            # Chunked prefill, same as the server: each chunk continues from the
            # cache the previous one left. Above ~8k tokens this is also the only
            # way the reference runs at all -- torch's CPU depthwise conv1d JIT
            # fails ("illegal immediate parameter") on a 32k-long window.
            step = ref_chunk or prefill
            outs_p = []
            for lo in range(0, prefill, step):
                hi = min(lo + step, prefill)
                o = ref(
                    hidden_states=hidden[:, lo:hi].to(dtype),
                    cache_params=cache,
                    attention_mask=torch.ones(1, hi - lo, dtype=torch.bool),
                )
                outs_p.append((o[0] if isinstance(o, tuple) else o).detach())
            out_p = torch.cat(outs_p, dim=1)
            del outs_p
            # HF keeps a `conv_kernel`-wide window (it appends before convolving);
            # sglang / the NPU pool keep `conv_kernel-1` -- the tokens the *next*
            # step still needs. Slice to that so stage B compares like with like.
            conv_p = cache.layers[layer].conv_states[0].detach()[..., -win:].clone()
            ssm_p = cache.layers[layer].recurrent_states[0].detach().clone()

            outs_d = []
            for t in range(prefill, total):
                o = ref(
                    hidden_states=hidden[:, t : t + 1].to(dtype),
                    cache_params=cache,
                    attention_mask=torch.ones(1, 1, dtype=torch.bool),
                )
                outs_d.append((o[0] if isinstance(o, tuple) else o).detach())
            conv_f = cache.layers[layer].conv_states[0].detach()[..., -win:].clone()
            ssm_f = cache.layers[layer].recurrent_states[0].detach().clone()

        out_d = (
            torch.cat(outs_d, dim=1)
            if outs_d
            else out_p.new_zeros(1, 0, out_p.shape[-1])
        )
        tensors[tag] = {
            "out.prefill": out_p[0].float(),
            "out.decode": out_d[0].float(),
            "state.conv.after_prefill": conv_p[0].float(),
            "state.ssm.after_prefill": ssm_p[0].float(),
            "state.conv.final": conv_f[0].float(),
            "state.ssm.final": ssm_f[0].float(),
        }
        print(
            f"  {tag}: prefill {tuple(out_p.shape)} absmax="
            f"{out_p.float().abs().max():.4f}  decode {tuple(out_d.shape)} "
            f"absmax={out_d.float().abs().max():.4f}"
        )

    meta = {
        "module": "kda",
        "layer": layer,
        "prefill": prefill,
        "decode": decode,
        "ref_prefill_chunk": ref_chunk or prefill,
        "hidden_size": int(hidden.shape[-1]),
        "num_heads": text_config.linear_attn_config["num_heads"],
        "head_dim": text_config.linear_attn_config["head_dim"],
        "conv_kernel": text_config.linear_attn_config["short_conv_kernel_size"],
        "gate_lower_bound": text_config.linear_attn_config.get("gate_lower_bound"),
        "rms_norm_eps": text_config.rms_norm_eps,
        # Layouts, so stage B does not have to guess which way round a state is.
        "layout.state.conv": (
            "[conv_dim, conv_kernel-1] (q|k|v concatenated); HF's own window is "
            "conv_kernel wide and was sliced to its last conv_kernel-1 columns"
        ),
        "layout.state.ssm": "[num_heads, head_k_dim, head_v_dim]",
        "source": "transformers Glm5NextTextLinearAttention (CPU)",
        **meta_extra,
    }
    return Case(
        name=f"kda.layer{layer:02d}",
        inputs={"hidden_states": hidden[0].float().contiguous()},
        ref_fp32=tensors["fp32"],
        ref_bf16=tensors["bf16"],
        meta=meta,
    )


# --------------------------------------------------------------- registry


class ModuleSpec(NamedTuple):
    #: which submodule of the decoder layer receives the tensor we want
    tap: Callable[[torch.nn.Module], torch.nn.Module]
    #: (config-declared) layer kinds this module is valid for
    layer_kinds: Tuple[str, ...]
    build: Callable[..., Case]


def _unsupported(name: str):
    def build(**_kwargs):  # noqa: ANN003
        raise SystemExit(
            f"--module {name}: no stage-A builder here yet. The DSA / MoE / dense-FFN "
            f"references live in reference_{name}.py; either call one from a "
            "ModuleSpec.build, or write one -- the capture / prefix machinery above "
            "is module-agnostic and already works."
        )

    return build


MODULES: Dict[str, ModuleSpec] = {
    "kda": ModuleSpec(
        tap=lambda layer: layer.self_attn,
        layer_kinds=("linear_attention",),
        build=build_kda_case,
    ),
    # Placeholders: the tap is right, only `build` is missing. Keeping them here
    # (rather than out of the dict) is what makes `--module dsa` fail with a
    # useful message instead of a KeyError.
    "dsa": ModuleSpec(
        tap=lambda layer: layer.self_attn,
        layer_kinds=("deepseek_sparse_attention",),
        build=_unsupported("dsa"),
    ),
    "moe": ModuleSpec(
        tap=lambda layer: layer.mlp,
        layer_kinds=("linear_attention", "deepseek_sparse_attention"),
        build=_unsupported("moe"),
    ),
    "ffn": ModuleSpec(
        tap=lambda layer: layer.mlp,
        layer_kinds=("linear_attention", "deepseek_sparse_attention"),
        build=_unsupported("ffn"),
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--module", default="kda", choices=sorted(MODULES))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prefill", type=int, default=256)
    ap.add_argument("--decode", type=int, default=8)
    ap.add_argument(
        "--ref-prefill-chunk",
        type=int,
        default=0,
        help="split the reference prefill into chunks of this many tokens "
        "(0 = one shot). Required above ~8k: torch's CPU depthwise conv1d "
        "cannot JIT a window that long. Stage B should use the same chunk.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--prefix-dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help="dtype of the layers *before* the one under test (the serving dtype "
        "by default; only decides which input the case tests with)",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = MODULES[args.module]
    prefix_dtype = getattr(torch, args.prefix_dtype)
    total = args.prefill + args.decode

    from transformers import AutoConfig

    probe = AutoConfig.from_pretrained(str(args.model)).text_config
    kind = probe.layer_types[args.layer]
    if kind not in spec.layer_kinds:
        raise SystemExit(
            f"layer {args.layer} is {kind!r}; --module {args.module} wants one of "
            f"{spec.layer_kinds}"
        )

    input_ids = make_input_ids(total, probe.vocab_size, args.seed)
    print(f"running layers 0..{args.layer} on {total} tokens ({args.prefix_dtype}) ...")
    hidden, module, text_config, model = capture_module_input(
        args.model, args.layer, spec.tap, input_ids, prefix_dtype
    )
    print(
        f"captured {tuple(hidden.shape)} absmax={hidden.abs().max():.4f} "
        f"rms={hidden.pow(2).mean().sqrt():.4f}"
    )
    # Keep the tapped module, drop the rest of the network before allocating the
    # fp32 copies.
    module = copy.deepcopy(module).float()
    del model

    case = spec.build(
        module=module,
        text_config=text_config,
        hidden=hidden,
        layer=args.layer,
        prefill=args.prefill,
        decode=args.decode,
        ref_chunk=args.ref_prefill_chunk,
        meta_extra={
            "seed": args.seed,
            "prefix_dtype": args.prefix_dtype,
            "input_ids_sha": int(input_ids.sum()),
        },
    )

    for name, ref32 in case.ref_fp32.items():
        ref16 = case.ref_bf16[name]
        denom = ref32.norm().clamp_min(1e-30)
        print(
            f"  floor {name:<28} {(ref32 - ref16).norm() / denom:.3e} "
            f"{tuple(ref32.shape)}"
        )
    case.save(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
