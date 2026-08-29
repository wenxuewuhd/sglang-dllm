#!/usr/bin/env python
"""Compare a running SGLang server's token logprobs against a CPU reference.

Accuracy benchmarks are a blunt instrument for a port: sampling noise hides small
numerical drift, and a score tells you something broke without telling you where.
Teacher-forced logprobs over a fixed prompt set have neither problem -- one forward
per prompt, no sampling, and the first token that diverges points at the layer.

Two steps, deliberately split because their costs differ by three orders of magnitude:

    reference   run the HF model once on CPU and save the logprobs   (slow, one-off)
    compare     hit the server and diff against that file            (seconds, every iteration)

The reference needs the HF venv, which is the only place ``glm5_next`` exists:

    $ROOT/.venv-ref/bin/python logit_check.py reference --out ref.json
    $VENV/bin/python           logit_check.py compare   --ref ref.json --port 30003

``--streaming`` swaps ``from_pretrained`` for the layer-at-a-time model that
``layer_check/trace_reference.py`` builds, plus ``lm_head`` on the end. That is not an
optimisation, it is the only way to get the **fp32** reference at all: a whole-model
fp32 ``from_pretrained`` reaches 1.26 TB at 26% loaded on this machine and starts
swapping (measured). Streaming holds one decoder layer, so peak RSS is tens of GB and
the fp32 run is affordable. Both dtypes read the same bf16 checkpoint -- fp32 upcasts
it losslessly -- so the two references differ only in arithmetic precision, which is
exactly what ``--against`` needs.

``--ref-source`` picks what "correct" means. ``hf-cpu`` is ground truth. ``server``
records a second server instead, which is how you check a fused operator against the
torch path it replaced: capture with the fallback env vars set, then compare with them
unset. That catches a wrong kernel in minutes without a CPU reference at all.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Short, varied, and fixed. Content diversity matters more than length: a prompt set
# that is all English prose will not exercise the routing that non-Latin text does.
PROMPTS = [
    "The capital of France is Paris, and the capital of Germany is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
    "In 1969, humanity first set foot on the Moon. The mission was called",
    "水的化学式是 H2O，它由氢和氧两种元素组成。常温下水的沸点是",
    "The derivative of x^3 with respect to x is 3x^2, and the integral of 2x is",
    "Q: If a train travels 60 km in 45 minutes, what is its speed in km/h?\nA:",
    "SELECT name, COUNT(*) FROM orders GROUP BY name HAVING COUNT(*) >",
    "Photosynthesis converts light energy into chemical energy, storing it in",
]


# Every prompt above is under `index_topk = 2048`, so the DSA indexer selects the whole
# pool and the sparse path never binds -- the short set cannot tell a correct kpool from
# a broken one. These are long enough that it does. One paragraph repeated is fine here:
# the point is sequence length, and a repeated prompt is still in distribution.
_LONG_BASE = (
    "The Ascend NPU port of GLM-5.3-Flash has 45 decoder layers: 34 of them use Kimi "
    "delta linear attention and 11 use DeepSeek sparse attention. The first three "
    "layers have a dense feed-forward network; the remaining forty-two are "
    "mixture-of-experts layers with 288 routed experts and one shared expert, eight of "
    "which are active per token. Numerical debugging of such a stack starts with a "
    "single question: which layer diverged first? Everything else follows from the "
    "answer, because a wrong layer poisons every layer after it, and an error that "
    "grows smoothly is a different bug from one that appears all at once. "
)

LONG_PROMPTS = [
    _LONG_BASE * 24 + "\n\nQ: How many of the 45 layers use sparse attention?\nA:",
    _LONG_BASE * 24 + "\n\nSummarised in one sentence, the paragraph above says that",
]

PROMPT_SETS = {"short": PROMPTS, "long": LONG_PROMPTS}

#: Mirrors layer_check/tolerance.py, which cannot be imported here without pulling in
#: torch; keep the name, the default and the env var in step with it.
SLACK = float(os.environ.get("GLM53_TOL_SLACK", "2.0"))


def token_ids(model_dir: str, text: str) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return tok(text, add_special_tokens=True)["input_ids"]


def run_reference_hf_cpu(model_dir: str, dtype: str, prompts: list[str]) -> list[dict]:
    """One teacher-forced forward per prompt; returns logprob of each realised token."""
    import torch

    # Not AutoModelForCausalLM: GLM-5.3-Flash's architecture is
    # Glm5NextForConditionalGeneration (it has a vision tower), which
    # transformers 5.16.1 defines but does not register with that auto class, so
    # AutoModelForCausalLM raises "Unrecognized configuration class". The class
    # is also absent from the package's top-level exports; it has to come from
    # the module. Text-only inference still goes through it -- the language
    # model and the lm_head both live there.
    from transformers.models.glm5_next import Glm5NextForConditionalGeneration

    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    print(f"loading {model_dir} on CPU as {dtype} (this is the slow part)", flush=True)
    model = Glm5NextForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch_dtype, device_map="cpu", trust_remote_code=True
    ).eval()

    out = []
    for i, prompt in enumerate(prompts):
        ids = token_ids(model_dir, prompt)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        # Position t predicts token t+1, so the last position has nothing to score.
        realised = [logprobs[t, ids[t + 1]].item() for t in range(len(ids) - 1)]
        top1 = logprobs[:-1].argmax(-1).tolist()
        out.append({"prompt": prompt, "ids": ids, "logprobs": realised, "top1": top1})
        print(f"  [{i + 1}/{len(prompts)}] {len(ids)} tokens", flush=True)
    return out


def run_reference_hf_cpu_streaming(
    model_dir: str, dtype: str, prompts: list[str], keep_fp32: bool = True
) -> list[dict]:
    """The same measurement as ``run_reference_hf_cpu``, one decoder layer at a time.

    Reuses ``trace_reference.build_streaming_model`` rather than re-deriving the
    checkpoint-name rewrites, the fused depthwise conv and the stacked expert layout --
    those are fiddly and already tested by the trace tool. All this adds is ``lm_head``,
    which lives outside ``Glm5NextTextModel``, and the log-softmax.

    All prompts go through as ONE right-padded batch, and that is not a micro-
    optimisation. Materialising a layer costs a read and a dtype cast of the whole
    599 GB checkpoint per forward -- measured at 205 MB/s for bf16 and 77 MB/s for
    fp32 on this box -- so a prompt-at-a-time loop is 8 checkpoint passes per dtype,
    about 17 hours for fp32. Batched it is one pass.

    Right padding is exact here rather than approximate: every path in this model is
    causal (attention mask, the KDA recurrence, the depthwise conv) or per-token (MoE
    routing, the mHC sinkhorn over hc_mult streams), so a token appended after position
    t cannot reach position t. The padded tail is computed and discarded. The claim is
    also checked rather than assumed -- the bf16 batched run reproduces the unpadded
    ``from_pretrained`` reference.

    ``keep_fp32`` reproduces HF's ``_keep_in_fp32_modules_strict``: under
    ``dtype=bfloat16`` the conv1d / dt_bias / A_log / e_score_correction_bias parameters
    stay fp32, because that is what ``from_pretrained(dtype=torch.bfloat16)`` gives you
    and the bf16 reference has to be the thing HF would give you.
    """
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "layer_check"))
    from trace_reference import build_streaming_model  # noqa: E402

    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    print(f"building streaming {dtype} model from {model_dir}", flush=True)
    model, captured, _routing, cfg, streamer = build_streaming_model(
        Path(model_dir), torch_dtype, keep_fp32=keep_fp32, verbose=True
    )
    # tie_word_embeddings is False for this checkpoint and the two tensors really do
    # differ (checked), so lm_head has to come from its own entry.
    lm_head = streamer.raw("lm_head.weight").to(torch_dtype)

    id_lists = [token_ids(model_dir, prompt) for prompt in prompts]
    width = max(len(ids) for ids in id_lists)
    pad = cfg.pad_token_id if cfg.pad_token_id is not None else 0
    batch = torch.tensor([ids + [pad] * (width - len(ids)) for ids in id_lists])
    print(
        f"one pass, {len(id_lists)} prompts right-padded to {width} tokens", flush=True
    )

    t0 = time.time()
    with torch.no_grad():
        hidden = model(input_ids=batch).last_hidden_state
    # The per-layer hidden states the trace hooks collect are dead weight here.
    captured.clear()
    print(f"forward done in {time.time() - t0:.1f}s, projecting logits", flush=True)

    out = []
    for row, (prompt, ids) in enumerate(zip(prompts, id_lists)):
        logits = torch.nn.functional.linear(hidden[row, : len(ids)], lm_head).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        realised = [logprobs[t, ids[t + 1]].item() for t in range(len(ids) - 1)]
        out.append(
            {
                "prompt": prompt,
                "ids": ids,
                "logprobs": realised,
                "top1": logprobs[:-1].argmax(-1).tolist(),
            }
        )
    return out


def run_reference_server(
    model_dir: str, host: str, port: int, prompts: list[str], decode_tokens: int = 0
) -> list[dict]:
    """Same measurement, taken from a server instead of the HF model."""
    import requests

    out = []
    for prompt in prompts:
        ids = token_ids(model_dir, prompt)
        r = requests.post(
            f"http://{host}:{port}/generate",
            json={
                "input_ids": ids,
                "sampling_params": {
                    "max_new_tokens": max(decode_tokens, 1),
                    "temperature": 0,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=1800,
            # This box exports HTTP_PROXY=http://127.0.0.1:1056; requests honours it
            # even for 127.0.0.1 and the proxy answers 503. env.sh unsets it, but this
            # tool is run directly often enough that it should not depend on that.
            proxies={"http": None, "https": None},
        )
        r.raise_for_status()
        meta = r.json()["meta_info"]
        # [logprob, token_id, token_text]; the first entry has no predecessor.
        entries = [e for e in meta["input_token_logprobs"] if e[0] is not None]
        record = {
            "prompt": prompt,
            "ids": ids,
            "logprobs": [e[0] for e in entries],
            "top1": None,
        }
        if decode_tokens:
            # Greedy, so the id sequence is a function of the arithmetic alone: for
            # graph-vs-eager it must match exactly, and a single flipped id is a real
            # divergence rather than sampling noise. The logprobs alongside it say how
            # far off the run was at the point it flipped.
            record["out_ids"] = [e[1] for e in meta["output_token_logprobs"]]
            record["out_logprobs"] = [e[0] for e in meta["output_token_logprobs"]]
        out.append(record)
    return out


def _compare_decode(a: dict, b: dict) -> int:
    """Greedy continuations, when both sides recorded one. Returns 1 if they diverged.

    Prefill logprobs say nothing about the decode path -- the KDA recurrent update, the
    kpool decode cache write and the sparse-attention decode branch are all only
    reachable from here.
    """
    if "out_ids" not in a or "out_ids" not in b:
        return 0
    n = min(len(a["out_ids"]), len(b["out_ids"]))
    first = next((t for t in range(n) if a["out_ids"][t] != b["out_ids"][t]), None)
    dlp = max(
        (abs(a["out_logprobs"][t] - b["out_logprobs"][t]) for t in range(n)), default=0.0
    )
    if first is None:
        print(f"           decode: {n} tokens identical, max|dlp|={dlp:.3e}")
        return 0
    print(
        f"           decode: DIVERGES at generated token {first} "
        f"({a['out_ids'][first]} vs {b['out_ids'][first]}), max|dlp|={dlp:.3e}"
    )
    return 1


def compare(
    ref: list[dict],
    got: list[dict],
    floor: list[dict] | None = None,
    emit_floor: Path | None = None,
) -> int:
    """Diff two logprob captures. With ``floor``, also return a verdict.

    The verdict runs on ``mean|dlp|`` rather than ``max|dlp|``. Both sides of this
    comparison flip MoE experts on some tokens, and a flip moves one token's logprob a
    long way; ``max`` therefore reports whichever run happened to flip the worst token,
    which is noise. The mean is the stable statistic.
    """
    print(f"{'prompt':<10}{'tokens':>8}{'max|dlp|':>12}{'mean|dlp|':>12}{'dNLL':>12}", end="")
    print(f"{'floor':>12}{'x floor':>10}" if floor else "")
    worst = 0.0
    worst_where = None
    decode_bad = 0
    over_floor = 0
    stats = []
    for i, (a, b) in enumerate(zip(ref, got)):
        if a["ids"] != b["ids"]:
            print(f"#{i}: tokenisation differs -- {len(a['ids'])} vs {len(b['ids'])} ids")
            return 1
        n = min(len(a["logprobs"]), len(b["logprobs"]))
        diffs = [abs(a["logprobs"][t] - b["logprobs"][t]) for t in range(n)]
        if not diffs:
            print(f"#{i}: no comparable positions")
            continue
        mx = max(diffs)
        mean = sum(diffs) / n
        stats.append({"prompt": a["prompt"], "n": n, "max": mx, "mean": mean})
        if mx > worst:
            worst, worst_where = mx, (i, diffs.index(mx))
        nll_a = -sum(a["logprobs"][:n]) / n
        nll_b = -sum(b["logprobs"][:n]) / n
        print(
            f"#{i:<9}{n:>8}{mx:>12.3e}{mean:>12.3e}{nll_b - nll_a:>+12.3e}", end=""
        )
        if floor is None:
            print()
        else:
            f = floor[i]["mean"]
            budget = f * SLACK
            ratio = mean / budget if budget else float("inf")
            print(f"{f:>12.3e}{ratio:>9.2f}x{'' if ratio <= 1.0 else '  OVER'}")
            over_floor += ratio > 1.0
        decode_bad += _compare_decode(a, b)

    print(f"\nworst |dlogprob| = {worst:.3e}", end="")
    if worst_where is not None:
        print(f"  at prompt {worst_where[0]} token {worst_where[1]}")
    else:
        print()

    if emit_floor is not None:
        emit_floor.write_text(json.dumps(stats))
        print(f"wrote floor to {emit_floor}")
    if floor is not None:
        print(
            f"-> {len(stats) - over_floor}/{len(stats)} prompts within the measured "
            f"floor x slack {SLACK}"
        )
        return 1 if (over_floor or decode_bad) else 0
    # No fixed threshold. An earlier version of this file guessed "<1e-2 is bf16
    # rounding", and that guess is wrong for this model by more than an order of
    # magnitude: MoE routing flips between an fp32 and a bf16 evaluation from the
    # first MoE layer onward -- 12.5% of tokens at layer 3, 63.3% by layer 41 --
    # so the floor is a discrete routing difference, not rounding, and it widens
    # with depth.
    #
    # Measure the floor instead, the way ACCEPTANCE.md does it:
    #
    #     reference --streaming --dtype fp32 --out ref32.json
    #     reference --streaming --dtype bf16 --out ref16.json
    #     compare --ref ref32.json --against ref16.json --emit-floor floor.json
    #     compare --ref ref32.json --port 30003 --floor floor.json
    print("no fixed threshold -- measure one with --against --emit-floor, then --floor")
    return 1 if decode_bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reference", "compare"])
    ap.add_argument("--model", default="/mnt/workspace/models/GLM-5.3-Flash-BF16")
    ap.add_argument("--ref", type=Path, help="reference file to write (reference) or read (compare)")
    ap.add_argument("--out", type=Path, help="alias for --ref when capturing")
    ap.add_argument("--ref-source", choices=["hf-cpu", "server"], default="hf-cpu")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument(
        "--streaming",
        action="store_true",
        help="build the HF reference one decoder layer at a time instead of calling "
        "from_pretrained. Required for --dtype fp32: the whole-model fp32 load needs "
        "1.2 TB and swaps on this machine.",
    )
    ap.add_argument(
        "--prompt-set",
        choices=sorted(PROMPT_SETS),
        default="short",
        help="short: 8 varied prompts under 26 tokens. long: prompts past "
        "index_topk=2048, the only ones that make the DSA sparse selection bind.",
    )
    ap.add_argument(
        "--decode-tokens",
        type=int,
        default=0,
        help="also greedily generate this many tokens per prompt and record them "
        "(--ref-source server only). Covers the decode path, which prefill logprobs "
        "do not reach at all.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30003)
    ap.add_argument(
        "--emit-floor",
        type=Path,
        help="write the per-prompt distances of this comparison to a file, for use as "
        "--floor later. Only meaningful together with --against.",
    )
    ap.add_argument(
        "--floor",
        type=Path,
        help="a file written by --emit-floor. Turns the printed numbers into a "
        "pass/fail verdict at SLACK times the measured floor.",
    )
    ap.add_argument(
        "--against",
        type=Path,
        help="compare --ref against this second reference file instead of a live "
        "server. Two references of different dtype give the noise floor.",
    )
    args = ap.parse_args()

    if args.mode == "reference":
        path = args.out or args.ref
        if path is None:
            raise SystemExit("reference mode needs --out")
        prompts = PROMPT_SETS[args.prompt_set]
        if args.ref_source == "server":
            data = run_reference_server(
                args.model, args.host, args.port, prompts, args.decode_tokens
            )
        elif args.streaming:
            data = run_reference_hf_cpu_streaming(args.model, args.dtype, prompts)
        else:
            data = run_reference_hf_cpu(args.model, args.dtype, prompts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "source": args.ref_source,
                    "dtype": args.dtype,
                    "streaming": bool(args.streaming),
                    "prompt_set": args.prompt_set,
                    "decode_tokens": args.decode_tokens,
                    "data": data,
                }
            )
        )
        print(f"wrote {path}")
        return 0

    if args.ref is None:
        raise SystemExit("compare mode needs --ref")
    saved = json.loads(args.ref.read_text())
    print(f"reference: {saved['source']} ({saved['dtype']}) {saved.get('prompt_set', 'short')}")
    floor = json.loads(args.floor.read_text()) if args.floor else None
    if args.against is not None:
        other = json.loads(args.against.read_text())
        print(f"against:   {other['source']} ({other['dtype']}) {other.get('prompt_set', 'short')}")
        return compare(saved["data"], other["data"], floor, args.emit_floor)
    # The prompts have to come from the reference, not from --prompt-set: comparing a
    # server run against a reference built from a different prompt set silently diffs
    # unrelated numbers, and the id check below would be the only thing that caught it.
    return compare(
        saved["data"],
        run_reference_server(
            args.model,
            args.host,
            args.port,
            [d["prompt"] for d in saved["data"]],
            saved.get("decode_tokens", 0),
        ),
        floor,
        args.emit_floor,
    )


if __name__ == "__main__":
    raise SystemExit(main())
