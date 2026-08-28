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

``--ref-source`` picks what "correct" means. ``hf-cpu`` is ground truth. ``server``
records a second server instead, which is how you check a fused operator against the
torch path it replaced: capture with the fallback env vars set, then compare with them
unset. That catches a wrong kernel in minutes without a CPU reference at all.
"""

from __future__ import annotations

import argparse
import json
import math
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


def token_ids(model_dir: str, text: str) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return tok(text, add_special_tokens=True)["input_ids"]


def run_reference_hf_cpu(model_dir: str, dtype: str) -> list[dict]:
    """One teacher-forced forward per prompt; returns logprob of each realised token."""
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    print(f"loading {model_dir} on CPU as {dtype} (this is the slow part)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch_dtype, device_map="cpu", trust_remote_code=True
    ).eval()

    out = []
    for i, prompt in enumerate(PROMPTS):
        ids = token_ids(model_dir, prompt)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        # Position t predicts token t+1, so the last position has nothing to score.
        realised = [logprobs[t, ids[t + 1]].item() for t in range(len(ids) - 1)]
        top1 = logprobs[:-1].argmax(-1).tolist()
        out.append({"prompt": prompt, "ids": ids, "logprobs": realised, "top1": top1})
        print(f"  [{i + 1}/{len(PROMPTS)}] {len(ids)} tokens", flush=True)
    return out


def run_reference_server(model_dir: str, host: str, port: int) -> list[dict]:
    """Same measurement, taken from a server instead of the HF model."""
    import requests

    out = []
    for prompt in PROMPTS:
        ids = token_ids(model_dir, prompt)
        r = requests.post(
            f"http://{host}:{port}/generate",
            json={
                "input_ids": ids,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "return_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=1800,
        )
        r.raise_for_status()
        meta = r.json()["meta_info"]
        # [logprob, token_id, token_text]; the first entry has no predecessor.
        entries = [e for e in meta["input_token_logprobs"] if e[0] is not None]
        out.append(
            {
                "prompt": prompt,
                "ids": ids,
                "logprobs": [e[0] for e in entries],
                "top1": None,
            }
        )
    return out


def compare(ref: list[dict], got: list[dict]) -> int:
    print(f"{'prompt':<10}{'tokens':>8}{'max|dlp|':>12}{'mean|dlp|':>12}{'dNLL':>12}")
    worst = 0.0
    worst_where = None
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
        if mx > worst:
            worst, worst_where = mx, (i, diffs.index(mx))
        nll_a = -sum(a["logprobs"][:n]) / n
        nll_b = -sum(b["logprobs"][:n]) / n
        print(f"#{i:<9}{n:>8}{mx:>12.3e}{sum(diffs) / n:>12.3e}{nll_b - nll_a:>+12.3e}")

    print(f"\nworst |dlogprob| = {worst:.3e}", end="")
    if worst_where is not None:
        print(f"  at prompt {worst_where[0]} token {worst_where[1]}")
    else:
        print()
    # A bf16 forward through 45 layers moves logprobs by ~1e-2; an order past that is
    # a different function, not a different rounding.
    print("guide: <1e-2 is bf16 rounding, >1e-1 means the two paths disagree")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reference", "compare"])
    ap.add_argument("--model", default="/mnt/workspace/models/GLM-5.3-Flash-BF16")
    ap.add_argument("--ref", type=Path, help="reference file to write (reference) or read (compare)")
    ap.add_argument("--out", type=Path, help="alias for --ref when capturing")
    ap.add_argument("--ref-source", choices=["hf-cpu", "server"], default="hf-cpu")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30003)
    args = ap.parse_args()

    if args.mode == "reference":
        path = args.out or args.ref
        if path is None:
            raise SystemExit("reference mode needs --out")
        data = (
            run_reference_server(args.model, args.host, args.port)
            if args.ref_source == "server"
            else run_reference_hf_cpu(args.model, args.dtype)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"source": args.ref_source, "dtype": args.dtype, "data": data}))
        print(f"wrote {path}")
        return 0

    if args.ref is None:
        raise SystemExit("compare mode needs --ref")
    saved = json.loads(args.ref.read_text())
    print(f"reference: {saved['source']} ({saved['dtype']})")
    return compare(saved["data"], run_reference_server(args.model, args.host, args.port))


if __name__ == "__main__":
    raise SystemExit(main())
