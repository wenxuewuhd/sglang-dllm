#!/usr/bin/env python
"""Stage A for mHC (manifold-constrained hyper-connections): CPU reference cases.

Run with the *reference* venv (``$ROOT/.venv-ref``, transformers 5.16.1); sglang
pins 5.12.1, which does not know ``glm5_next``.

    $ROOT/.venv-ref/bin/python reference_mhc.py --layers 1,20 --tokens 128 \
        --outdir $ROOT/goldens

mHC sits on *every* one of the 45 layers, twice per layer (an ``attn`` site and an
``ffn`` site), so what it has to be checked against is the input distribution it
actually sees at depth -- not a ``randn``. This script therefore does what
``dump_reference.py`` does for KDA: instantiates the real HF model truncated to
``max(layers)+1`` decoder layers, runs a real forward, and taps the module under
test with hooks.

Two tensors are captured per site, because mHC is two halves that are used at two
different points of the layer:

* ``hidden_streams`` ``[S, H, D]`` -- the ``hc_mult`` residual streams entering the
  site.  This drives the **pre** half (``pre`` / ``post`` / ``comb`` and the
  collapsed sublayer input).
* ``sublayer_out`` ``[S, D]`` -- what the attention (or MLP) actually returned for
  those streams.  This drives the **post** half
  ``out = post * x + combᵀ @ residual``.

Both are real activations from the real forward.

The references are the HF module evaluated twice on the same captured input, fp32
and bf16, exactly as ``../tools/golden_mhc.py`` does it (``load_state_dict`` with
the fp32-widened checkpoint values, then ``.to(dtype)``).  Note where the bf16
floor actually comes from here: ``fn`` is stored **bf16** in the checkpoint and HF
upcasts it (``self.fn.float()``), so it contributes nothing; ``base`` and ``scale``
are stored **fp32** and are *not* upcast, so rounding them is what separates the
two references for ``post``/``comb``.  ``collapsed`` and the post half additionally
round their bf16 outputs.

Sinkhorn is the part of mHC most likely to accumulate error (20 alternating
row/column normalisations), so every iterate is saved as well -- under ``inputs``,
not ``ref_*``, because the fused NPU kernel cannot expose its internal iterates and
``harness.check`` treats a missing reference tensor as a failure.  Stage B uses the
trace to replay the recurrence on device and to identify which iteration count the
kernel's output actually corresponds to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from golden_mhc import STAGES, load_weights  # noqa: E402
from harness import Case  # noqa: E402

DEFAULT_MODEL = Path("/mnt/workspace/models/GLM-5.3-Flash-BF16")

#: HF's decoder-layer attribute names for the two mHC sites, and the sublayer whose
#: output the post half consumes. Keys match ``golden_mhc.STAGES``.
SITES = {
    "attn": ("attn_hc", "self_attn"),
    "ffn": ("ffn_hc", "mlp"),
}


class _StopAtLayer(Exception):
    """Raised once the last requested tensor is in hand, to abort the forward."""


def make_input_ids(n: int, vocab_size: int, seed: int) -> torch.Tensor:
    """Deterministic ids, kept away from the special-token end of the vocab.

    Same convention as ``dump_reference.py``: what mHC sees is a hidden state, and
    any in-distribution id sequence produces one.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab_size - 1024, (1, n), generator=g)


def _first_tensor(out):
    while isinstance(out, (tuple, list)):
        out = out[0]
    return out


def capture_sites(
    model_dir: Path,
    sites: list[tuple[int, str]],
    input_ids: torch.Tensor,
    prefix_dtype: torch.dtype,
):
    """Run the real model up to the deepest requested site and tap all of them.

    Returns ``{(layer, stage): {"hidden_streams": ..., "sublayer_out": ...}}`` in
    fp32, plus the (restored) text config.
    """
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextModel

    config = AutoConfig.from_pretrained(str(model_dir))
    text_config = config.text_config
    n_layers = text_config.num_hidden_layers
    deepest = max(layer for layer, _ in sites)
    if deepest >= n_layers:
        raise SystemExit(f"layer {deepest} out of range ({n_layers} layers)")

    # Truncating the config is what keeps this affordable: from_pretrained then
    # reads only the shards holding layers 0..deepest, not all 599 GB.
    text_config.num_hidden_layers = deepest + 1
    model = Glm5NextModel.from_pretrained(
        str(model_dir), config=config, dtype=prefix_dtype
    )
    model.eval()
    text_config.num_hidden_layers = n_layers

    box: dict[tuple[int, str], dict[str, torch.Tensor]] = {(k): {} for k in sites}
    # The last capture in execution order; its hook aborts the forward.
    last = max(sites, key=lambda s: (s[0], 0 if s[1] == "attn" else 1))
    handles = []

    def pre_hook(key):
        def fn(_mod, args, kwargs):
            got = kwargs.get("hidden_streams")
            if got is None:
                got = args[0]
            box[key]["hidden_streams"] = got.detach().to(torch.float32)

        return fn

    def post_hook(key):
        def fn(_mod, _args, out):
            box[key]["sublayer_out"] = _first_tensor(out).detach().to(torch.float32)
            if key == last:
                raise _StopAtLayer

        return fn

    for layer, stage in sites:
        hc_name, sub_name = SITES[stage]
        decoder_layer = model.language_model.layers[layer]
        handles.append(
            getattr(decoder_layer, hc_name).register_forward_pre_hook(
                pre_hook((layer, stage)), with_kwargs=True
            )
        )
        handles.append(
            getattr(decoder_layer, sub_name).register_forward_hook(
                post_hook((layer, stage))
            )
        )

    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    except _StopAtLayer:
        pass
    finally:
        for h in handles:
            h.remove()

    for key, got in box.items():
        missing = {"hidden_streams", "sublayer_out"} - set(got)
        if missing:
            raise SystemExit(f"site {key}: hooks never produced {sorted(missing)}")
    del model
    return box, text_config


def sinkhorn_trace(block, streams, dtype):
    """Re-run the pre half step by step so every Sinkhorn iterate is observable.

    This is a transcription of ``Glm5NextTextHyperConnection.forward``; the caller
    asserts it reproduces the module's own outputs bit for bit, so a drift in
    transformers shows up as an assertion rather than as a silently wrong trace.
    """
    import torch.nn.functional as F

    hc = block.hc_mult
    eps = block.hc_eps
    flat = block.input_norm(streams.flatten(start_dim=2).float())
    pre_w, post_w, comb_w = F.linear(flat, block.fn.float()).split(
        [hc, hc, hc * hc], dim=-1
    )
    pre_b, post_b, comb_b = block.base.split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = block.scale.unbind(0)

    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)

    comb = torch.softmax(logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    trace = [comb]
    for _ in range(block.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        trace.append(comb)
    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
    return {
        "pre": pre,
        "post": post,
        "comb": comb,
        "collapsed": collapsed,
        "logits": logits,
        "trace": torch.stack([t[0].float() for t in trace]),
    }


def build_case(
    *,
    model_dir: Path,
    text_config,
    layer: int,
    stage: str,
    streams: torch.Tensor,
    sublayer_out: torch.Tensor,
    repeat: int,
    meta_extra: dict,
) -> Case:
    """Evaluate the HF mHC block twice on one captured site."""
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextHyperConnection,
    )

    distinct = int(streams.shape[1])
    if repeat > 1:
        streams = streams.repeat(1, repeat, 1, 1).contiguous()
        sublayer_out = sublayer_out.repeat(1, repeat, 1).contiguous()

    state = load_weights(model_dir, layer, stage)
    fp32_state = {k: v.to(torch.float32) for k, v in state.items()}

    refs: dict[str, dict[str, torch.Tensor]] = {}
    traces: dict[str, torch.Tensor] = {}
    logits: dict[str, torch.Tensor] = {}
    for dtype, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        block = Glm5NextTextHyperConnection(text_config)
        block.load_state_dict(fp32_state, strict=True)
        block = block.to(dtype).eval()

        s_d = streams.to(dtype)
        x_d = sublayer_out.to(dtype)
        with torch.no_grad():
            post, comb, collapsed = block(s_d)
            step = sinkhorn_trace(block, s_d, dtype)
            # HF applies the post half inside the decoder layer, in the layer's
            # dtype -- post/comb come back fp32 and are cast down there.
            out = post.to(dtype).unsqueeze(-1) * x_d.unsqueeze(-2) + torch.matmul(
                comb.to(dtype).transpose(-1, -2), s_d
            )

        # The step-by-step transcription must be the module, not a lookalike.
        assert torch.equal(step["post"], post), f"{tag}: post trace diverged"
        assert torch.equal(step["comb"], comb), f"{tag}: comb trace diverged"
        assert torch.equal(step["collapsed"], collapsed), f"{tag}: collapsed diverged"

        out32 = out[0].float().contiguous()
        refs[tag] = {
            "pre.post": post[0].float().contiguous(),
            "pre.comb": comb[0].float().contiguous(),
            "pre.collapsed": collapsed[0].float().contiguous(),
            "post.out": out32,
            # Same reference tensor, deliberately aliased (torch.save stores the
            # storage once). It is scored against a candidate that is fed the
            # *fp32 reference* post/comb instead of its own, so a post-half
            # failure cannot be blamed on the pre half, and vice versa.
            "post.out.isolated": out32,
        }
        traces[tag] = step["trace"]
        logits[tag] = step["logits"][0].float().contiguous()

        row = comb.float().sum(-1)
        col = comb.float().sum(-2)
        print(
            f"  {tag}: post{tuple(post.shape)} comb{tuple(comb.shape)} "
            f"collapsed{tuple(collapsed.shape)} out{tuple(out.shape)} | "
            f"comb row-sum {row.min():.6f}..{row.max():.6f} "
            f"col-sum {col.min():.6f}..{col.max():.6f}"
        )

    # Per-iteration fp32-vs-bf16 divergence: the Sinkhorn error trend of the
    # *reference itself*, which is the scale any device trend has to be read against.
    from reference.tolerance import rel_err  # noqa: E402

    per_iter = [
        rel_err(traces["bf16"][i], traces["fp32"][i]) for i in range(traces["fp32"].shape[0])
    ]
    print("  sinkhorn fp32-vs-bf16 per iteration:")
    print("   " + " ".join(f"{i}:{e:.2e}" for i, e in enumerate(per_iter)))

    meta = {
        "module": "mhc",
        "layer": layer,
        "stage": stage,
        "hc_mult": int(text_config.hc_mult),
        "hc_eps": float(text_config.hc_eps),
        "hc_sinkhorn_iters": int(text_config.hc_sinkhorn_iters),
        # 1e-5 for GLM-5.3-Flash. DeepSeek-V4 uses 1e-6; passing that here is a
        # silent accuracy bug, not an error, so it is recorded in the case.
        "rms_norm_eps": float(text_config.rms_norm_eps),
        "hidden_size": int(text_config.hidden_size),
        "post_mult_value": 2.0,
        "tokens": int(streams.shape[1]),
        "distinct_tokens": distinct,
        "repeat": repeat,
        "layer_type": text_config.layer_types[layer],
        "mlp_type": text_config.mlp_layer_types[layer],
        "sinkhorn_floor_per_iter": per_iter,
        "layout.hidden_streams": "[S, hc_mult, hidden]; flatten to [S, hc_mult*hidden]",
        "layout.pre.comb": "[S, hc_mult, hc_mult]; post half uses combᵀ @ residual",
        "source": "transformers Glm5NextTextHyperConnection (CPU)",
        **meta_extra,
    }
    inputs = {
        "hidden_streams": streams[0].float().contiguous(),
        "sublayer_out": sublayer_out[0].float().contiguous(),
        "weight.fn": state["fn"].to(torch.float32).contiguous(),
        "weight.base": state["base"].to(torch.float32).contiguous(),
        "weight.scale": state["scale"].to(torch.float32).contiguous(),
        "sinkhorn.logits.fp32": logits["fp32"],
        "sinkhorn.trace.fp32": traces["fp32"],
        "sinkhorn.trace.bf16": traces["bf16"],
    }
    return Case(
        name=f"mhc.{stage}.layer{layer:02d}.M{streams.shape[1]}",
        inputs=inputs,
        ref_fp32=refs["fp32"],
        ref_bf16=refs["bf16"],
        meta=meta,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument(
        "--layers",
        default="1,20",
        help="comma-separated decoder layers. Layer 0 is degenerate for mHC: the "
        "model expands one embedding into all hc_mult streams, so they are equal.",
    )
    ap.add_argument("--stages", default="attn,ffn")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument(
        "--repeats",
        default="1",
        help="comma-separated tile factors. One case per factor: the captured real "
        "token rows are tiled to M = tokens*factor before the references are "
        "computed, so the reference is evaluated at the deployed M. Needed because "
        "the deployed prefill chunk is 8192 tokens and a real 8192-token CPU "
        "forward through 40 layers is not affordable (DSA attention is O(S^2)); "
        "mHC is per-token, so tiling changes only M, never the arithmetic.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--prefix-dtype", default="bfloat16", choices=("bfloat16", "float32")
    )
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument(
        "--from-case",
        type=Path,
        default=None,
        help="re-derive from an existing case's captured tensors instead of running "
        "the model again. The prefix forward is the expensive part (a 41-layer CPU "
        "forward on a loaded machine can take over an hour); the reference itself is "
        "seconds. Use this with --repeats to reach a deployed M from a capture you "
        "already have. --layers/--stages/--tokens are read from the case.",
    )
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    repeats = [int(x) for x in args.repeats.split(",") if x.strip()]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = set(stages) - set(STAGES)
    if bad:
        raise SystemExit(f"unknown stage(s) {sorted(bad)}; known: {sorted(STAGES)}")
    if 0 in layers:
        print(
            "warning: layer 0's four streams are identical copies of the embedding; "
            "the case will not exercise stream mixing."
        )
    sites = [(layer, stage) for layer in layers for stage in stages]

    from transformers import AutoConfig

    if args.from_case is not None:
        src = Case.load(args.from_case)
        text_config = AutoConfig.from_pretrained(str(args.model)).text_config
        layer, stage = src.meta["layer"], src.meta["stage"]
        if src.meta.get("repeat", 1) != 1:
            raise SystemExit(
                f"{args.from_case} is itself a tiled case (repeat="
                f"{src.meta['repeat']}); tile from the untiled one"
            )
        box = {
            (layer, stage): {
                "hidden_streams": src.inputs["hidden_streams"].unsqueeze(0),
                "sublayer_out": src.inputs["sublayer_out"].unsqueeze(0),
            }
        }
        sites = [(layer, stage)]
        input_ids = torch.tensor([[src.meta.get("input_ids_sum", 0)]])
        print(
            f"re-deriving from {args.from_case}: layer {layer} / {stage}, "
            f"{src.inputs['hidden_streams'].shape[0]} captured token rows, no model load"
        )
    else:
        box = None

    if box is None:
        probe = AutoConfig.from_pretrained(str(args.model)).text_config
        input_ids = make_input_ids(args.tokens, probe.vocab_size, args.seed)
        deepest = max(layers)
        print(
            f"running layers 0..{deepest} on {args.tokens} tokens "
            f"({args.prefix_dtype}); tapping {len(sites)} site(s) ..."
        )
        box, text_config = capture_sites(
            args.model, sites, input_ids, getattr(torch, args.prefix_dtype)
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    for layer, stage in sites:
        got = box[(layer, stage)]
        streams, sublayer_out = got["hidden_streams"], got["sublayer_out"]
        print(
            f"\nlayer {layer} / {stage}: streams{tuple(streams.shape)} "
            f"absmax={streams.abs().max():.4f} rms={streams.pow(2).mean().sqrt():.4f} | "
            f"sublayer_out{tuple(sublayer_out.shape)} absmax={sublayer_out.abs().max():.4f}"
        )
        for repeat in repeats:
            m = args.tokens * repeat
            if repeat > 1:
                print(
                    f"  M={m}: {args.tokens} real token rows tiled {repeat}x. The "
                    f"reference is evaluated by HF at M={m}; only the input rows repeat."
                )
            case = build_case(
                model_dir=args.model,
                text_config=text_config,
                layer=layer,
                stage=stage,
                streams=streams,
                sublayer_out=sublayer_out,
                repeat=repeat,
                meta_extra={
                    "seed": args.seed,
                    "prefix_dtype": args.prefix_dtype,
                    "input_ids_sum": int(input_ids.sum()),
                },
            )
            for name, ref32 in case.ref_fp32.items():
                ref16 = case.ref_bf16[name]
                denom = ref32.norm().clamp_min(1e-30)
                print(
                    f"  floor {name:<20} {(ref32 - ref16).norm() / denom:.3e} "
                    f"{tuple(ref32.shape)}"
                )
            suffix = "" if repeat == 1 else f"_m{m}"
            out = args.outdir / f"mhc_{stage}_layer{layer:02d}{suffix}.pt"
            case.save(out)
            print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
