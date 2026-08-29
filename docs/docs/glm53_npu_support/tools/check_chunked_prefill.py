#!/usr/bin/env python
"""Does a prompt longer than --chunked-prefill-size still come out right?

Nothing else in this directory reaches this path. The long prompt set in
logit_check.py is ~3256 tokens against a chunked_prefill_size of 8192, so a single
sequence is never split; batch-level chunking (several sequences sharing an 8192-token
budget) is a different mechanism from splitting one sequence across forward passes.

Splitting one sequence is the case that can go quietly wrong here, because GLM carries
state across the split that a stateless transformer does not:

  * 34 KDA layers carry a conv state and an SSM state. If either is not handed from
    chunk N to chunk N+1, everything before the boundary is silently forgotten.
  * The DSA kpool is written incrementally, and PLAN P3.4 lists "unaligned chunk
    starts" as unverified.

Two independent probes, because they fail differently:

  needle   a distinctive fact planted BEFORE the boundary, asked about at the end.
           Broken state carry-over loses it outright -- a semantic, unmissable failure.
  logprob  the same prompt scored with and without chunking. Judge the magnitude, not
           equality: chunking changes the GEMM M dimension, and bf16 matmul on this
           hardware is not batch-shape invariant (measured), so a correct
           implementation still moves by the shape floor, about 2e-2 mean.

    # with the server on its normal chunked_prefill_size
    $VENV/bin/python check_chunked_prefill.py --out chunked.json
    # then restart with --chunked-prefill-size 32768 and
    $VENV/bin/python check_chunked_prefill.py --out unchunked.json --against chunked.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

# The dev box exports HTTP_PROXY and requests honours it even for 127.0.0.1.
NO_PROXY = {"http": None, "https": None}

NEEDLE_EARLY = (
    "Remember this: the calibration constant for the Antares detector is 4173."
)
NEEDLE_LATE = (
    "Remember this too: the calibration constant for the Borealis detector is 8891."
)

FILLER = (
    "The Ascend NPU port of this model separates two kinds of attention across its "
    "decoder stack, and the memory behaviour of each one differs enough that a single "
    "description of the layer stack would be misleading. Linear attention keeps a "
    "recurrent state whose size does not grow with the sequence, while sparse "
    "attention keeps a pool of keys and selects among them. Numerical work on such a "
    "stack has to say which of the two it is talking about at every step. "
)


def build_prompt(tok, target_tokens: int, boundary: int) -> str:
    """Filler with one needle well before the chunk boundary and one well after."""
    unit = len(tok(FILLER)["input_ids"])
    before = FILLER * max(1, (boundary // 2) // unit)          # needle lands early
    after = FILLER * max(1, (boundary // 2) // unit)
    tail = FILLER * max(1, ((target_tokens - boundary) // 2) // unit)
    return (
        f"{NEEDLE_EARLY}\n\n{before}\n\n{after}\n\n{NEEDLE_LATE}\n\n{tail}\n\n"
        "Question: what is the calibration constant for the Antares detector?\n"
        "Answer with the number only."
    )


def ask(host, port, model, tok, prompt, n_new):
    ids = tok(prompt, add_special_tokens=True)["input_ids"]
    r = requests.post(
        f"http://{host}:{port}/generate",
        json={
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": n_new, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        timeout=1800,
        proxies=NO_PROXY,
    )
    r.raise_for_status()
    meta = r.json()["meta_info"]
    return {
        "n_prompt": len(ids),
        "logprobs": [e[0] for e in meta["input_token_logprobs"] if e[0] is not None],
        "out_ids": [e[1] for e in meta["output_token_logprobs"]],
        "text": r.json()["text"],
    }


def background_load(host, port, model, tok, stop, results):
    """Keep other requests decoding while the long prompt is being chunked.

    This is the case that a single-request test cannot reach. Between chunk N and
    chunk N+1 the long request is stashed out of the running batch while everything
    else keeps decoding, so its KDA mamba slot has to survive other requests' writes.
    A single-request test never has anything else in the batch.
    """
    import threading

    prompt = "Count from one to twenty in words, one per line.\n1. one\n"
    ids = tok(prompt, add_special_tokens=True)["input_ids"]

    def one():
        while not stop.is_set():
            try:
                r = requests.post(
                    f"http://{host}:{port}/generate",
                    json={"input_ids": ids,
                          "sampling_params": {"max_new_tokens": 64, "temperature": 0}},
                    timeout=600, proxies=NO_PROXY)
                results.append(r.json()["text"])
            except Exception:
                pass

    threads = [threading.Thread(target=one, daemon=True) for _ in range(8)]
    for t in threads:
        t.start()
    return threads


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30003)
    ap.add_argument("--model", default="/mnt/workspace/models/GLM-5.3-Flash-BF16")
    ap.add_argument("--tokens", type=int, default=12000)
    ap.add_argument("--boundary", type=int, default=8192,
                    help="the server's --chunked-prefill-size")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--against", type=Path)
    ap.add_argument("--concurrent", action="store_true",
                    help="keep 8 other requests decoding through the chunked prefill")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt = build_prompt(tok, args.tokens, args.boundary)

    bg_results, threads, stop = [], [], None
    if args.concurrent:
        import threading
        import time

        stop = threading.Event()
        threads = background_load(args.host, args.port, args.model, tok, stop, bg_results)
        time.sleep(5)  # let the background batch fill before the long prefill starts

    got = ask(args.host, args.port, args.model, tok, prompt, 24)

    if stop is not None:
        stop.set()
        bad = [t for t in bg_results if "two" not in t]
        print(f"background: {len(bg_results)} completions during the chunked prefill, "
              f"{len(bad)} degraded")

    print(f"prompt {got['n_prompt']} tokens, boundary at {args.boundary} "
          f"-> {'SPLIT' if got['n_prompt'] > args.boundary else 'NOT split (raise --tokens)'}")
    print(f"needle answer: {got['text']!r}")
    hit = "4173" in got["text"]
    print(f"early needle (planted before the boundary): {'FOUND' if hit else 'LOST'}")

    if args.out:
        args.out.write_text(json.dumps(got))
        print(f"wrote {args.out}")

    rc = 0 if hit else 1
    if args.against:
        other = json.loads(args.against.read_text())
        if other["n_prompt"] != got["n_prompt"]:
            print(f"prompts differ ({other['n_prompt']} vs {got['n_prompt']}), not comparable")
            return 1
        n = min(len(got["logprobs"]), len(other["logprobs"]))
        d = [abs(got["logprobs"][t] - other["logprobs"][t]) for t in range(n)]
        print(f"\nvs {args.against.name}: {n} positions, "
              f"max|dlp|={max(d):.3e}  mean|dlp|={sum(d) / n:.3e}")
        print(f"generated ids identical: {got['out_ids'] == other['out_ids']}")
        # The shape floor measured on this model is ~2e-2 mean; an order of magnitude
        # above that is not rounding, it is state that did not cross the boundary.
        print("mean|dlp| at ~2e-2 is the shape floor (chunking changes the GEMM M "
              "dimension); an order of magnitude above it means broken carry-over")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
