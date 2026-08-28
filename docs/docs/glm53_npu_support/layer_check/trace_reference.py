#!/usr/bin/env python
"""Stage A for the whole network: every layer's hidden state, on CPU, twice.

This is the tool that answers **"which layer did it start going wrong at?"**.  The
per-module checks (`dump_reference.py` + `check_<module>.py`) say whether one module is
right in isolation; this says where a full 45-layer run first leaves the reference.

Run it with the *reference* venv (`$ROOT/.venv-ref`, transformers 5.16.1).  sglang pins
transformers 5.12.1, which does not know `glm5_next` at all::

    source $ROOT/env.sh
    OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 \
        $ROOT/.venv-ref/bin/python trace_reference.py \
            --tokens 128 --out $ROOT/goldens/trace_128.pt

Then, once the whole network runs on device (P4.1), capture the same per-layer hidden
states there and hand both to `trace_compare.py`.

--------------------------------------------------------------------------------
Why this does not just call `from_pretrained`
--------------------------------------------------------------------------------

The checkpoint is 599 GB of bf16.  It *would* fit in this machine's 1.8 TB of RAM, but
the fp32 reference would be 1.2 TB, and holding either one is a needless risk when the
model is evaluated exactly once, strictly layer by layer.

So the model is built on the **meta device** (no storage at all) and each decoder layer
is materialised from the checkpoint by a forward pre-hook, then dropped back to meta by
the matching forward hook.  Peak resident memory is therefore *one* decoder layer -- a
MoE layer is 14.5 GB in bf16, 29 GB in fp32 -- instead of the whole network.

Everything else is the real `Glm5NextTextModel.forward`: the mHC stream expansion, the
mask construction, the `prev_topk_indices` chaining between DSA layers and the final
`hc_head` collapse are HF's own code, not a reimplementation.

--------------------------------------------------------------------------------
What a "hidden state" is here
--------------------------------------------------------------------------------

GLM-5.3-Flash uses manifold-constrained hyper-connections (mHC), so the tensor that
flows *between* decoder layers is **not** `[B, S, D]` -- it is `[B, S, hc_mult, D]`,
four parallel residual streams.  `hc_mult = 4`, so one layer's state at 128 tokens is
128 x 4 x 4096 fp32 = 8.4 MB, and a 45-layer trace is ~378 MB per precision.

The trace stores that four-stream tensor, batch dim squeezed out: `[S, hc_mult, D]`.
The NPU side must dump the same thing -- see `trace_compare.py` for the file format and
for where to hook.

--------------------------------------------------------------------------------
What this tool does NOT cover -- read this before trusting a clean trace
--------------------------------------------------------------------------------

The trace runs a **short prompt** (128 tokens by default) because a 45-layer CPU
forward is not cheap and this is a *locating* tool, not an accuracy evaluation.  That
choice buys tractability and costs shape coverage, and the cost is real:

* On this hardware **NPU bf16 matmul is not batch-shape invariant** -- the same input
  with M changed from 4096 to 4080 moves 5 of 4080 rows by one bf16 ulp (measured) --
  and operator tiling is chosen per shape.  A short-prompt trace exercises none of the
  tilings the real deployment uses (TP16, `page-size 64`, `context-length 32768`,
  `max-running-requests 16`, chunked prefill at 8192 tokens).
* Everything that only engages at length is invisible here: the KDA chunked-scan path,
  DSA pool selection (`index_topk = 2048` never binds at 128 tokens), and every
  paging/block-table code path.
* This project has already been bitten twice by exactly that gap -- FIA silently wrong
  by 200x when a parameter was dropped under the TND layout, and `npu_clipped_swiglu`
  wrong by 109x on its default arguments.  Neither raised an error.

So: **a clean trace is not a pass.**  It says "the per-layer arithmetic does not drift
at this shape", which narrows *where* to look.  Module-level checks on the real
deployment shapes (`check_<module>.py`) are what actually gate a module, and they are
not replaced by this.  Raise `--tokens` when you can afford it; the cost is recorded in
the report next to the default.

--------------------------------------------------------------------------------
fp32 vs bf16
--------------------------------------------------------------------------------

Both runs read the *same* bf16 checkpoint.  The fp32 run upcasts it (lossless); the
bf16 run uses it as stored.  So the two traces differ only in arithmetic precision, and
`harness.first_divergence` can use the distance between them as each layer's own noise
floor -- which is the whole point, because a fixed threshold is wrong for this model
(KDA layer-0 at seq=64 already has an fp32-vs-bf16 relative error of 1.06e-2).

One caveat the tool measures for you: MoE routing is a **discrete** function of the
hidden state, so from some layer on the fp32 and bf16 runs select *different experts*
for some tokens.  Past that layer the two references are no longer two precisions of the
same arithmetic and the "noise floor" is dominated by that discrete difference -- the
per-layer report prints the flip fraction so a suspiciously wide floor is explained
rather than trusted.

The bf16 run reproduces HF's `_keep_in_fp32_modules_strict` -- `conv1d`, `dt_bias`,
`A_log` and `e_score_correction_bias` stay fp32 even under `dtype=bfloat16`, because
that is what `from_pretrained(dtype=torch.bfloat16)` actually does and the reference has
to be the thing HF would give you.  `--no-keep-in-fp32` turns it off if you want to see
how much it is worth.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import save_trace  # noqa: E402

DEFAULT_MODEL = Path("/mnt/workspace/models/GLM-5.3-Flash-BF16")

#: HF's `_keep_in_fp32_modules_strict` for Glm5NextPreTrainedModel. Parameters whose
#: name contains one of these stay fp32 even when the model dtype is bf16.
KEEP_FP32 = ("conv1d", "dt_bias", "A_log", "e_score_correction_bias")

#: checkpoint-name -> module-name rewrites, from transformers'
#: `conversion_mapping.py` entry for "glm5_next". Kept explicit rather than imported
#: because that module is a private detail of the loader we are deliberately bypassing.
RENAMES = (
    (r"^hc_attn_(fn|base|scale)$", r"attn_hc.\1"),
    (r"^hc_ffn_(fn|base|scale)$", r"ffn_hc.\1"),
    (r"^self_attn\.(f_a_proj|f_b_proj|dt_bias|A_log)", r"self_attn.forget_gate.\1"),
)
_EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
_CONV_RE = re.compile(r"^self_attn\.([qkv])_conv1d\.weight$")


class LayerStreamer:
    """Reads one decoder layer's weights out of the sharded checkpoint on demand.

    Holds only `safe_open` handles, which mmap; nothing is resident between calls.
    """

    def __init__(self, model_dir: Path):
        self.dir = model_dir
        self.index: Dict[str, str] = json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self._handles: Dict[str, object] = {}

    def raw(self, name: str) -> torch.Tensor:
        shard = self.index[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.dir / shard), framework="pt")
        return self._handles[shard].get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.index

    @staticmethod
    def _cast(name: str, t: torch.Tensor, dtype: torch.dtype, keep_fp32: bool):
        if keep_fp32 and dtype != torch.float32 and any(k in name for k in KEEP_FP32):
            return t.to(torch.float32)
        return t.to(dtype)

    def _stack_experts(self, srcs: List[str], dtype: torch.dtype) -> torch.Tensor:
        """Materialise a `[E, ...]` expert tensor without ever holding a bf16 *and* an
        fp32 copy of the whole stack: allocate the destination in the target dtype and
        fill it one expert at a time.  At 288 experts the naive `stack().to(fp32)` costs
        an extra 9.7 GB of transient per MoE layer.
        """
        out = torch.empty((len(srcs), *self.raw(srcs[0]).shape), dtype=dtype)
        for i, s in enumerate(srcs):
            out[i] = self.raw(s).to(dtype)
        return out

    def layer_state(
        self, layer: int, dtype: torch.dtype, keep_fp32: bool = True
    ) -> Dict[str, torch.Tensor]:
        p = f"model.language_model.layers.{layer}."
        names = [k for k in self.index if k.startswith(p)]
        if not names:
            raise SystemExit(f"no weights for layer {layer} in the checkpoint index")
        sd: Dict[str, torch.Tensor] = {}
        experts: Dict[str, Dict[int, str]] = {"gate": {}, "up": {}, "down": {}}
        for full in names:
            s = full[len(p) :]
            m = _EXPERT_RE.match(s)
            if m:
                experts[m.group(2)][int(m.group(1))] = full
                continue
            if _CONV_RE.match(s):
                continue  # handled below, the three are fused into one depthwise conv
            for pat, rep in RENAMES:
                s2 = re.sub(pat, rep, s)
                if s2 != s:
                    s = s2
                    break
            sd[s] = self._cast(s, self.raw(full), dtype, keep_fp32)

        if self.has(p + "self_attn.q_conv1d.weight"):
            # HF runs ONE depthwise conv over cat([q, k, v]); keep that row order.
            sd["self_attn.conv1d.weight"] = self._cast(
                "conv1d",
                torch.cat(
                    [self.raw(f"{p}self_attn.{c}_conv1d.weight") for c in "qkv"], dim=0
                ),
                dtype,
                keep_fp32,
            )

        if experts["gate"]:
            n = len(experts["gate"])
            assert len(experts["up"]) == len(experts["down"]) == n, "ragged expert set"
            g0 = self.raw(experts["gate"][0])
            inter, hid = g0.shape
            # [E, 2*inter, hid], gate rows first -- `Glm5NextTextExperts._apply_gate`
            # does `chunk(2, dim=-1)` on the *output* of F.linear, so the first `inter`
            # output rows must be the gate half.
            gu = torch.empty((n, 2 * inter, hid), dtype=dtype)
            for e in range(n):
                gu[e, :inter] = self.raw(experts["gate"][e]).to(dtype)
                gu[e, inter:] = self.raw(experts["up"][e]).to(dtype)
            sd["mlp.experts.gate_up_proj"] = gu
            sd["mlp.experts.down_proj"] = self._stack_experts(
                [experts["down"][e] for e in range(n)], dtype
            )
        return sd


def build_streaming_model(
    model_dir: Path,
    dtype: torch.dtype,
    num_layers: int | None = None,
    keep_fp32: bool = True,
    verbose: bool = True,
):
    """A real `Glm5NextTextModel` whose decoder layers page in and out of RAM.

    Returns `(model, captured, config, streamer)`. `captured` is filled in layer order
    by the forward hooks during `model(...)`.
    """
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    cfg = AutoConfig.from_pretrained(str(model_dir)).text_config
    if num_layers is not None:
        cfg.num_hidden_layers = min(num_layers, cfg.num_hidden_layers)
    streamer = LayerStreamer(model_dir)

    with torch.device("meta"):
        model = Glm5NextTextModel(cfg)
    model.eval()

    # The embedding and the final norm are small and used at both ends of the forward,
    # so they stay resident. 154880 x 4096 is 1.3 GB bf16 / 2.5 GB fp32.
    model.embed_tokens.load_state_dict(
        {"weight": streamer.raw("model.language_model.embed_tokens.weight").to(dtype)},
        assign=True,
    )
    model.norm.load_state_dict(
        {"weight": streamer.raw("model.language_model.norm.weight").to(dtype)},
        assign=True,
    )

    captured: List[torch.Tensor] = []
    # Per MoE layer, the top-k expert ids each token selected. Tiny (tokens x 8 ints)
    # and it is the only thing that explains a noise floor that jumps rather than
    # grows: routing is a *discrete* function of the hidden state, so fp32 and bf16
    # can pick different experts and the two references then stop being two
    # evaluations of the same arithmetic.
    routing: Dict[int, torch.Tensor] = {}
    # state_dict() on a meta module returns meta tensors: exactly what we assign back
    # to free a layer.
    meta_sds = [dict(layer.state_dict()) for layer in model.layers]

    def router_hook(mod, _args, out):
        routing[mod._trace_idx] = out[2].detach().clone()  # (logits, weights, indices)

    def pre(mod, args, kwargs):
        mod._t0 = time.time()
        mod.load_state_dict(
            streamer.layer_state(mod._trace_idx, dtype, keep_fp32),
            assign=True,
            strict=True,
        )
        mod._t1 = time.time()

    def post(mod, args, kwargs, out):
        i = mod._trace_idx
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach().to(torch.float32).squeeze(0).clone())
        mod.load_state_dict(meta_sds[i], assign=True, strict=True)
        gc.collect()
        if verbose:
            now = time.time()
            print(
                f"  layer {i:>2} {cfg.layer_types[i][:6]}/{cfg.mlp_layer_types[i][:6]}"
                f"  load={mod._t1 - mod._t0:5.2f}s fwd={now - mod._t1:5.2f}s"
                f"  absmax={captured[-1].abs().max():.5f}"
                f"  rms={captured[-1].pow(2).mean().sqrt():.5f}",
                flush=True,
            )

    for i, layer in enumerate(model.layers):
        layer._trace_idx = i
        layer.register_forward_pre_hook(pre, with_kwargs=True)
        layer.register_forward_hook(post, with_kwargs=True)
        # Module objects are stable across the materialise/free cycle -- only their
        # parameters are swapped -- so a hook registered here survives it.
        if cfg.mlp_layer_types[i] == "sparse":
            layer.mlp.gate._trace_idx = i
            layer.mlp.gate.register_forward_hook(router_hook)

    return model, captured, routing, cfg, streamer


def make_input_ids(model_dir: Path, n: int, seed: int, vocab_size: int):
    """A short, in-distribution prompt.

    Prefers the real tokenizer on real text -- an id sequence the model has actually
    seen keeps the hidden-state magnitudes representative, which matters because the
    noise floor is measured *on this trace*.  Falls back to deterministic random ids
    (kept clear of the special-token block at the top of the vocab) if the tokenizer
    cannot be built, so the tool never silently depends on a tokenizer install.
    """
    text = (
        "The Ascend NPU port of GLM-5.3-Flash has 45 decoder layers: 34 of them use "
        "Kimi delta linear attention and 11 use DeepSeek sparse attention. The first "
        "three layers have a dense feed-forward network; the remaining forty-two are "
        "mixture-of-experts layers with 288 routed experts and one shared expert, "
        "eight of which are active per token. Numerical debugging of such a stack "
        "starts with a single question: which layer diverged first? Everything else "
        "follows from the answer, because a wrong layer poisons every layer after it "
        "and an error that grows smoothly is a different bug from one that appears "
        "all at once. This paragraph exists only to be tokenized into a prompt that "
        "is long enough to exercise the attention path and short enough to trace on "
        "a CPU in a couple of minutes."
    )
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(model_dir))
        ids = tok(text, return_tensors="pt").input_ids
        if ids.shape[1] >= n:
            return ids[:, :n], "tokenizer"
        # repeat rather than pad: padding would put the model off-distribution
        reps = -(-n // ids.shape[1])
        return ids.repeat(1, reps)[:, :n], "tokenizer(repeated)"
    except Exception as exc:  # noqa: BLE001
        print(f"  (tokenizer unavailable: {exc}; using deterministic random ids)")
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, vocab_size - 1024, (1, n), generator=g), "random"


def run_once(
    model_dir: Path,
    input_ids: torch.Tensor,
    dtype: torch.dtype,
    num_layers: int | None,
    keep_fp32: bool,
) -> tuple[List[torch.Tensor], Dict[int, torch.Tensor], torch.Tensor, float]:
    model, captured, routing, cfg, _ = build_streaming_model(
        model_dir, dtype, num_layers, keep_fp32
    )
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    elapsed = time.time() - t0
    final = out.last_hidden_state.detach().to(torch.float32).squeeze(0).clone()
    del model, out
    gc.collect()
    return captured, routing, final, elapsed


def routing_flips(
    r32: Dict[int, torch.Tensor], r16: Dict[int, torch.Tensor]
) -> Dict[int, float]:
    """Fraction of tokens whose selected expert *set* differs between the two runs.

    Compared as sets, not as vectors: `topk(..., sorted=False)` gives no order
    guarantee, so a positional comparison would report flips that are not flips.
    """
    out = {}
    for layer, a in sorted(r32.items()):
        b = r16.get(layer)
        if b is None or a.shape != b.shape:
            continue
        differs = (
            torch.sort(a, dim=-1).values != torch.sort(b, dim=-1).values
        ).any(dim=-1)
        frac = float(differs.float().mean())
        if frac > 0:
            out[layer] = frac
    return out


def peak_rss_gb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument(
        "--tokens",
        type=int,
        default=128,
        help="prompt length. This is a locating tool, not an accuracy eval -- 128 is "
        "plenty and 32k would cost hours of CPU for no extra information.",
    )
    ap.add_argument(
        "--layers",
        type=int,
        default=None,
        help="trace only the first N layers (smoke test; the full run is 45)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-keep-in-fp32",
        dest="keep_fp32",
        action="store_false",
        help="cast conv1d / dt_bias / A_log / e_score_correction_bias to bf16 too, "
        "instead of reproducing HF's _keep_in_fp32_modules_strict",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(args.model)).text_config
    n_layers = min(args.layers or cfg.num_hidden_layers, cfg.num_hidden_layers)
    input_ids, id_source = make_input_ids(
        args.model, args.tokens, args.seed, cfg.vocab_size
    )
    print(
        f"tracing {n_layers} layers on {input_ids.shape[1]} tokens "
        f"({id_source} ids), keep_in_fp32={args.keep_fp32}"
    )

    results = {}
    for tag, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        print(f"\n--- {tag} pass ---", flush=True)
        hidden, routing, final, elapsed = run_once(
            args.model, input_ids, dtype, n_layers, args.keep_fp32
        )
        print(
            f"  {tag}: {elapsed:.1f}s, {len(hidden)} layers, "
            f"peak RSS {peak_rss_gb():.1f} GB"
        )
        results[tag] = (hidden, routing, final, elapsed)

    h32, r32, f32, t32 = results["fp32"]
    h16, r16, f16, t16 = results["bf16"]

    # Print the floors now, so a trace that is useless (a layer where fp32 and bf16
    # already disagree by 100 %) is visible at generation time and not three days later
    # when somebody tries to compare an NPU run against it.
    print("\n--- per-layer fp32-vs-bf16 noise floor ---")
    from harness import first_divergence  # noqa: F401  (documents the pairing)
    from reference.tolerance import rel_err  # type: ignore  # noqa: E402

    flips = routing_flips(r32, r16)
    for i, (a, b) in enumerate(zip(h32, h16)):
        note = ""
        if i in flips:
            note = f"   routing flip on {flips[i] * 100:.1f}% of tokens"
        print(f"  layer {i:>2}: floor={rel_err(b, a):.3e}{note}")
    print(f"  final:    floor={rel_err(f16, f32):.3e}")
    if flips:
        worst = max(flips.values())
        print(
            f"\n  NOTE: the top-k expert set differs between the fp32 and bf16 runs "
            f"on up to {worst * 100:.1f}% of tokens. From the first layer "
            f"where that happens, the two references stop being two precisions of the "
            f"same arithmetic -- the floor is then dominated by a discrete routing "
            f"difference, not by rounding, and a wide floor there is NOT evidence that "
            f"a wide error is acceptable. Trust deep-layer verdicts only as 'this is "
            f"where it started', and confirm with the module check for that layer."
        )

    meta = {
        "model": str(args.model),
        "tokens": int(input_ids.shape[1]),
        "layers": n_layers,
        "hc_mult": cfg.hc_mult,
        "hidden_size": cfg.hidden_size,
        "id_source": id_source,
        "seed": args.seed,
        "keep_in_fp32": args.keep_fp32,
        # A tensor, not a list, and the layer kinds as one compact string each:
        # `harness.first_divergence` prints this whole dict as its header, so a meta
        # that carries 45-element lists makes its own report unreadable.
        "input_ids": input_ids[0].clone(),
        "layout.hidden": "[seq, hc_mult, hidden_size], the mHC multi-stream residual",
        "layer_kinds": "".join(
            "L" if k == "linear_attention" else "D" for k in cfg.layer_types[:n_layers]
        ),
        "mlp_kinds": "".join(
            "d" if k == "dense" else "s" for k in cfg.mlp_layer_types[:n_layers]
        ),
        # Summary only. The full per-layer profile is printed at generation time; a
        # 42-entry dict in here would swamp first_divergence's header, which prints meta.
        "routing_flip_first_layer": min(flips) if flips else None,
        "routing_flip_max": round(max(flips.values()), 4) if flips else 0.0,
        "seconds_fp32": round(t32, 1),
        "seconds_bf16": round(t16, 1),
        "final_hidden_rms_fp32": float(f32.pow(2).mean().sqrt()),
        "final_hidden_absmax_fp32": float(f32.abs().max()),
        "final_hidden_floor": float(rel_err(f16, f32)),
        "source": "transformers Glm5NextTextModel (CPU, streamed layer by layer)",
    }
    save_trace(args.out, h32, h16, meta)
    print(
        f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) "
        f"-- peak RSS {peak_rss_gb():.1f} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
