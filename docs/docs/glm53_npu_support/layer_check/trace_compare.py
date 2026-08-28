#!/usr/bin/env python
"""Stage B for the whole network: which layer diverged first?

Takes the CPU trace from `trace_reference.py` and a per-layer hidden-state dump from an
NPU run, and prints `harness.first_divergence`: every layer's relative error against the
fp32 reference, next to that layer's *own* measured fp32-vs-bf16 noise floor.

    source $ROOT/env.sh
    PYTHONPATH=$REPO/python $VENV/bin/python trace_compare.py \
        --trace $ROOT/goldens/trace_128.pt \
        --npu   $ROOT/goldens/npu_trace_128.pt

Every layer is printed, not just the first failure.  A run that drifts gradually and a
run that breaks at one layer look identical if you only report the first bad layer, and
they need completely different debugging.

--------------------------------------------------------------------------------
STATUS: the NPU capture side is a *format*, not a capture
--------------------------------------------------------------------------------

Producing the NPU dump needs the whole 45-layer network to run on device, which is P4.1
and is not done.  So this file defines and validates the interchange format and does the
comparison; `dump_npu_trace_hook` below is the capture helper, written but **not yet
exercised against a live model** -- see the repo report.  It is deliberately a ~20-line
hook rather than a script, because the run it has to attach to is a real server / offline
engine forward, not something this file should be launching.

--------------------------------------------------------------------------------
The file format
--------------------------------------------------------------------------------

A `torch.save`d dict::

    {
        "format": 1,                 # harness.CASE_FORMAT
        "hidden": [T_0, ..., T_44],  # one tensor per decoder layer, in layer order
        "meta":   {...},             # free-form; "input_ids" is checked if present
    }

Each `T_i` is that layer's **output** hidden state with the batch dimension removed:

    [seq, hc_mult, hidden_size]   ==  [128, 4, 4096]  for the default trace

That is the mHC four-stream residual GLM-5.3-Flash carries *between* decoder layers --
`Glm5NextTextModel` expands the embedding to `hc_mult` streams before layer 0 and only
collapses them in `hc_head` after layer 44.  Dumping the collapsed `[seq, hidden_size]`
instead would hide a bug that lives in one stream, so the trace stores all four.

dtype does not matter (it is upcast to fp32 before comparison); device does not matter
(it is moved to CPU).  Layer *order* and the token axis do matter, and both are checked.

--------------------------------------------------------------------------------
How to capture it on the NPU side
--------------------------------------------------------------------------------

The sglang model is `sglang.srt.models.glm5_next.Glm5NextForConditionalGeneration`; the
per-layer modules are `Glm5NextDecoderLayer` in `model.model.layers`.  Attach the hook
below to each of them and run **one prefill** of exactly the trace's `input_ids`, with
no radix-cache hit, no chunked prefill and no CUDA/NPU graph replay -- all three would
either split the sequence or skip the hooks:

    from trace_compare import dump_npu_trace_hook
    stop = dump_npu_trace_hook(model.model.layers, out_path, meta={"input_ids": ids})
    ...one forward...
    stop()          # writes the file

Two things to get right, both of which will otherwise produce a file that compares
cleanly against nothing:

1. **Batch of one, unchunked.**  sglang flattens the batch, so the hook sees
   `[num_tokens, ...]` with no batch axis.  With more than one request in flight the
   token axis is a concatenation of several sequences and the comparison is meaningless.

2. **The same token ids.**  `meta["input_ids"]` from the reference trace is the
   contract; pass them through and this script will refuse a mismatch rather than
   report a divergence at layer 0 that is really just a different prompt.

If the sglang layer turns out to carry the mHC streams in a different memory layout than
`[tokens, hc_mult, hidden]` (for example flattened to `[tokens, hc_mult * hidden]`),
reshape in the hook -- do **not** "fix" it by reshaping the reference, because the
reference is the thing being trusted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import CASE_FORMAT, first_divergence  # noqa: E402


def dump_npu_trace_hook(layers: Iterable[torch.nn.Module], out: Path, meta: dict):
    """Record every decoder layer's output; returns a `stop()` that writes the file.

    Kept tiny and dependency-free on purpose: it has to be pasteable into whatever
    harness is driving the device run.
    """
    out = Path(out)
    captured: List[torch.Tensor] = []
    handles = []

    def hook(_mod, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().to("cpu", torch.float32).clone())

    for layer in layers:
        handles.append(layer.register_forward_hook(hook))

    def stop() -> Path:
        for h in handles:
            h.remove()
        if not captured:
            raise SystemExit("no layer outputs captured -- did the forward run?")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": CASE_FORMAT, "hidden": captured, "meta": meta}, out)
        return out

    return stop


def load_npu_trace(path: Path) -> tuple[List[torch.Tensor], dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, list):  # tolerate a bare list of tensors
        return blob, {}
    got = blob.get("format")
    if got != CASE_FORMAT:
        raise SystemExit(
            f"{path}: npu trace format {got}, this harness speaks {CASE_FORMAT}"
        )
    return list(blob["hidden"]), dict(blob.get("meta", {}))


def _normalise(
    hidden: List[torch.Tensor], ref_shape: torch.Size
) -> List[torch.Tensor]:
    """Squeeze a leading batch axis and un-flatten `[tokens, hc_mult*hidden]`.

    Only shapes that are unambiguously the same tensor are accepted; anything else is
    an error, because silently reshaping a genuinely different layout is how a broken
    comparison passes.
    """
    n_elem = 1
    for d in ref_shape:
        n_elem *= d
    out = []
    for i, t in enumerate(hidden):
        if t.shape == ref_shape:
            out.append(t)
        elif t.dim() == len(ref_shape) + 1 and t.shape[0] == 1 and t.shape[1:] == ref_shape:
            out.append(t.squeeze(0))  # sglang keeps no batch axis, but a wrapper might
        elif t.dim() == 2 and t.shape[0] == ref_shape[0] and t.numel() == n_elem:
            out.append(t.reshape(ref_shape))  # [tokens, hc_mult*hidden] -> unflatten
        else:
            raise SystemExit(
                f"layer {i}: npu hidden state has shape {tuple(t.shape)}, reference "
                f"is {tuple(ref_shape)}. Fix the capture hook, not the reference -- "
                f"see the module docstring."
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--trace", type=Path, required=True, help="CPU trace from trace_reference.py"
    )
    ap.add_argument(
        "--npu", type=Path, required=True, help="per-layer dump from the NPU run"
    )
    ap.add_argument(
        "--allow-id-mismatch",
        action="store_true",
        help="compare even when the two runs used different token ids (almost always "
        "a mistake -- it makes layer 0 diverge for a reason that is not a port bug)",
    )
    args = ap.parse_args()

    ref = torch.load(args.trace, map_location="cpu", weights_only=False)
    hidden_npu, npu_meta = load_npu_trace(args.npu)
    ref_meta = ref.get("meta", {})

    ref_ids = ref_meta.get("input_ids")
    npu_ids = npu_meta.get("input_ids")
    if ref_ids is not None and npu_ids is not None:
        if list(ref_ids) != list(npu_ids):
            msg = (
                f"token ids differ: reference has {len(ref_ids)} ids, npu run has "
                f"{len(list(npu_ids))}, and they are not equal"
            )
            if not args.allow_id_mismatch:
                raise SystemExit(msg + " -- rerun the NPU side on meta['input_ids']")
            print(f"WARNING: {msg}")
    elif npu_ids is None:
        print(
            "WARNING: the npu trace carries no meta['input_ids']; nothing verifies "
            "that the two runs saw the same prompt."
        )

    hidden_npu = _normalise(hidden_npu, ref["hidden_fp32"][0].shape)
    # first_divergence prints the trace's meta verbatim as its header; keep a long
    # input_ids tensor from swamping the per-layer table.
    torch.set_printoptions(threshold=8, edgeitems=3)
    print(
        f"reference: {ref_meta.get('layers')} layers, "
        f"{ref_meta.get('tokens')} tokens, layout {ref_meta.get('layout.hidden')}"
    )
    first = first_divergence(args.trace, hidden_npu)
    if first is not None:
        kinds = ref_meta.get("layer_kinds") or ""
        mlps = ref_meta.get("mlp_kinds") or ""
        attn = {"L": "KDA linear attention", "D": "DSA sparse attention"}.get(
            kinds[first : first + 1], "?"
        )
        mlp = {"d": "dense FFN", "s": "MoE"}.get(mlps[first : first + 1], "?")
        print(
            f"\n  -> first divergence at layer {first}: {attn} + {mlp}. "
            f"Run that module's own check next -- this trace localises, it does not "
            f"diagnose, and its short prompt does not cover the real serving shapes."
        )
    return 1 if first is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
