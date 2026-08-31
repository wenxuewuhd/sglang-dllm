#!/usr/bin/env python
"""Does GLM-5.3-Flash still work when the context is actually long?

The model's card says 1048576 tokens and its DSA/kpool machinery exists for exactly
that, but every accuracy run in this project stopped at 32768, and the longest prompt
used to judge numerics was 3256 tokens. This tool is the missing measurement.

Three probes, because they fail differently and none of them subsumes the others:

  needle   distinctive facts planted at several DEPTHS of one long prompt, all asked
           at the end. This is the only probe that says "the sparse selection chose
           the right region". It is also the only one that survives a config change:
           depth recall does not depend on TP width or quantisation, so a result here
           on INT8 TP8 still means something on BF16 TP16.

  prefix   the SAME first N tokens scored twice -- once as a request N tokens long,
           once as the head of a request that is many times longer. Every path in this
           model is causal, and with N a multiple of --chunked-prefill-size both runs
           split into the same chunks with the same GEMM shapes. So the expectation is
           not "within a floor", it is BIT-IDENTICAL. Anything else means the long
           tail reached backwards, which recall cannot see: a needle test passes just
           fine while the logits drift.

  timing   prefill wall time and decode ms/token against context length. The decode
           curve is the falsifiable one: index_topk caps how much KV a decode step
           reads and the KDA state is O(1), so ms/token should be roughly FLAT in
           context length. If it climbs, something is scanning the whole sequence.

Usage:

    $VENV/bin/python check_long_context.py --port 30023 --tokens 131072 \
        --out ctx131072.json
    $VENV/bin/python check_long_context.py --port 30023 --tokens 131072 \
        --prefix-check --prefix-len 32768

⚠ The needle prompt is built from token ids, not text, so --tokens is exact. Pieces
tokenised separately and concatenated do not equal the tokenisation of the joined
text; that is fine here (the model consumes ids) and it is the only way to place a
needle at a known depth.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

# The dev box exports HTTP_PROXY and requests honours it even for 127.0.0.1.
NO_PROXY = {"http": None, "https": None}

FILLER = (
    "The Ascend NPU port of this model separates two kinds of attention across its "
    "decoder stack, and the memory behaviour of each one differs enough that a single "
    "description of the layer stack would be misleading. Linear attention keeps a "
    "recurrent state whose size does not grow with the sequence, while sparse "
    "attention keeps a pool of keys and selects among them. Numerical work on such a "
    "stack has to say which of the two it is talking about at every step. "
)

#: (name, secret). Names are unrelated words so a partial recall is unambiguous about
#: WHICH depth was lost; secrets are four digits with no shared prefix so a near miss
#: is visible as a near miss rather than rounding to the same string.
NEEDLES = [
    ("Antares", "4173"),
    ("Borealis", "8891"),
    ("Cygnus", "2607"),
    ("Draco", "9435"),
    ("Eridanus", "5182"),
]

#: Where each needle goes, as a fraction of the prompt. 0.0 is the very front (the
#: region a sliding-window bug drops first) and 0.99 is inside the tail that kpool
#: always selects, which makes it the control: if 0.99 fails, the failure is not
#: about selection at all.
DEPTHS = [0.0, 0.25, 0.5, 0.75, 0.99]


def build_ids(tok, target_tokens: int):
    """A prompt of exactly ``target_tokens`` ids with one needle at each depth."""
    filler = tok(FILLER, add_special_tokens=False)["input_ids"]
    question = tok(
        "\n\nQuestion: state the calibration constant for each detector named above, "
        "one per line, in the form NAME=NUMBER. Answer with those five lines only.\n",
        add_special_tokens=False,
    )["input_ids"]
    needle_ids = [
        tok(
            f"\nRemember this: the calibration constant for the {name} detector "
            f"is {secret}.\n",
            add_special_tokens=False,
        )["input_ids"]
        for name, secret in NEEDLES
    ]

    body_len = target_tokens - len(question)
    if body_len <= sum(len(n) for n in needle_ids):
        raise SystemExit(f"--tokens {target_tokens} is too small for the needles")

    # Lay the filler down first, then overwrite at each depth. Overwriting rather than
    # inserting keeps the total exact and keeps every needle at the depth it names.
    body = (filler * (body_len // len(filler) + 1))[:body_len]
    placed = []
    for (name, secret), nid, depth in zip(NEEDLES, needle_ids, DEPTHS):
        at = min(int(body_len * depth), body_len - len(nid))
        body[at : at + len(nid)] = nid
        placed.append({"name": name, "secret": secret, "depth": depth, "token_pos": at})

    ids = body + question
    assert len(ids) == target_tokens, (len(ids), target_tokens)
    return ids, placed


def generate(host, port, ids, n_new, want_logprob=False, timeout=7200):
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": n_new, "temperature": 0},
    }
    if want_logprob:
        payload["return_logprob"] = True
        payload["logprob_start_len"] = 0
    t0 = time.time()
    r = requests.post(
        f"http://{host}:{port}/generate", json=payload, timeout=timeout, proxies=NO_PROXY
    )
    elapsed = time.time() - t0
    r.raise_for_status()
    body = r.json()
    meta = body["meta_info"]
    out = {
        "n_prompt": len(ids),
        "n_new": n_new,
        "elapsed_s": elapsed,
        "text": body["text"],
        "out_ids": [e[1] for e in meta.get("output_token_logprobs", [])] or None,
    }
    if want_logprob:
        out["logprobs"] = [
            e[0] for e in meta["input_token_logprobs"] if e[0] is not None
        ]
    return out


def generate_streaming(host, port, ids, n_new, timeout=7200):
    """One request, both timings: TTFT is the prefill, the rest is the decode.

    ⚠ Do NOT time the decode as (wall time of a max_new=N run) minus (wall time of a
    max_new=1 run). At 32k that works; at 128k the two prefills are ~30 s each and
    their run-to-run spread (~1 s, 3%) is larger than 64 tokens of decode (~1.8 s),
    so the subtraction returned a NEGATIVE ms/token -- measured, 2026-08-30. The
    error grows with context precisely where the decode curve matters most. One
    streamed request has no subtraction of two noisy numbers in it, and it prefills
    once instead of twice, which at 1M is the difference between 4 minutes and 8.
    """
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": n_new, "temperature": 0},
        "stream": True,
    }
    t0 = time.time()
    ttft = None
    n_chunks = 0
    n_out = 0
    text = ""
    with requests.post(
        f"http://{host}:{port}/generate",
        json=payload,
        timeout=timeout,
        proxies=NO_PROXY,
        stream=True,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            chunk = line[len("data:") :].strip()
            if chunk == "[DONE]":
                break
            if ttft is None:
                ttft = time.time() - t0
            n_chunks += 1
            try:
                body = json.loads(chunk)
                text = body["text"]
                # Chunks are not tokens. Under speculative decoding one chunk
                # carries a whole accepted run, so dividing the decode phase by
                # the chunk count reports ms per VERIFY STEP and calls it ms per
                # token -- measured 2026-08-30: 44.6 against a true 22.2, which
                # makes a 1.24x speedup read as a 1.6x slowdown.
                n_out = body["meta_info"]["completion_tokens"]
            except ValueError:
                pass
            except KeyError:
                # A rejected request streams one chunk with an "error" object and
                # no "text". Silently reporting zero tokens turns that into a
                # plausible-looking measurement, which is worse than a crash.
                raise SystemExit(f"server returned no text: {chunk[:400]}")
    total = time.time() - t0
    # The first token arrives with the first chunk, so the decode phase spans the
    # rest of them.
    decoded_after_first = max(n_out - 1, 1)
    return {
        "n_prompt": len(ids),
        "n_new": n_new,
        "ttft_s": ttft,
        "total_s": total,
        "n_chunks": n_chunks,
        "completion_tokens": n_out,
        "decode_ms_per_token": (total - ttft) / decoded_after_first * 1000,
        "text": text,
    }


def run_needle(args, tok):
    ids, placed = build_ids(tok, args.tokens)
    print(f"prompt: {len(ids)} tokens, needles at "
          f"{[p['token_pos'] for p in placed]}", flush=True)

    got = generate_streaming(args.host, args.port, ids, args.max_new)
    decode_ms = got["decode_ms_per_token"]
    print(f"  TTFT {got['ttft_s']:.1f} s ({len(ids) / got['ttft_s']:.0f} tok/s prefill)"
          f"   total {got['total_s']:.1f} s, {got['completion_tokens']} tokens in "
          f"{got['n_chunks']} chunks"
          f"   -> decode {decode_ms:.1f} ms/token", flush=True)

    text = got["text"]
    print(f"\nanswer: {text!r}\n")
    hits = []
    for p in placed:
        hit = p["secret"] in text
        hits.append(hit)
        print(f"  depth {p['depth']:>5} (token {p['token_pos']:>8})  "
              f"{p['name']:<9} {p['secret']}  {'FOUND' if hit else 'LOST'}")
    n_hit = sum(hits)
    print(f"\nrecall {n_hit}/{len(hits)}")

    result = {
        "tokens": args.tokens,
        "placed": placed,
        "hits": hits,
        "recall": n_hit / len(hits),
        "ttft_s": got["ttft_s"],
        "prefill_tok_s": len(ids) / got["ttft_s"],
        "total_s": got["total_s"],
        "decode_ms_per_token": decode_ms,
        "text": text,
    }
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
        print(f"wrote {args.out}")
    return 0 if n_hit == len(hits) else 1


def run_prefix(args, tok):
    """Score the same prefix alone and as the head of a much longer request.

    ⚠ --prefix-len must be a multiple of the server's --chunked-prefill-size, or the
    short run's last chunk is a different width from the long run's and the GEMM shape
    floor reappears -- at which point 'bit-identical' is the wrong expectation and the
    test says nothing.
    """
    ids, _ = build_ids(tok, args.tokens)
    n = args.prefix_len
    if n >= len(ids):
        raise SystemExit("--prefix-len must be shorter than --tokens")

    short = generate(args.host, args.port, ids[:n], 1, want_logprob=True)
    print(f"short: {short['n_prompt']} tokens, {len(short['logprobs'])} logprobs, "
          f"{short['elapsed_s']:.1f} s", flush=True)
    long_ = generate(args.host, args.port, ids, 1, want_logprob=True)
    print(f"long : {long_['n_prompt']} tokens, {len(long_['logprobs'])} logprobs, "
          f"{long_['elapsed_s']:.1f} s", flush=True)

    m = min(len(short["logprobs"]), len(long_["logprobs"]), n)
    d = [abs(short["logprobs"][t] - long_["logprobs"][t]) for t in range(m)]
    mx, mean = max(d), sum(d) / m
    first = next((t for t, v in enumerate(d) if v > 0), None)
    print(f"\ncommon prefix: {m} positions")
    print(f"  max|dlp|  = {mx:.3e}")
    print(f"  mean|dlp| = {mean:.3e}")
    print(f"  first differing position: {first}")
    print(
        "expectation is 0.000e+00 exactly: the model is causal and both runs split "
        "into the same chunks. A nonzero value means the tail reached backwards."
        if args.prefix_len % args.chunk == 0
        else f"⚠ --prefix-len {n} is not a multiple of --chunk {args.chunk}; the last "
        "chunk widths differ, so a small nonzero value is the shape floor, not a bug."
    )
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "prefix_len": n,
                    "tokens": args.tokens,
                    "positions": m,
                    "max_dlp": mx,
                    "mean_dlp": mean,
                    "first_diff": first,
                },
                indent=1,
            )
        )
        print(f"wrote {args.out}")
    return 0 if mx == 0.0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30023)
    ap.add_argument("--model", default="/mnt/workspace/models/GLM-5.3-Flash-BF16",
                    help="tokenizer source; the W8A8 dir has the same tokenizer")
    ap.add_argument("--tokens", type=int, default=None,
                    help="exact prompt length in tokens; use --fill-ctx instead to "
                         "derive it from the server's context length")
    ap.add_argument("--fill-ctx", type=int, default=None,
                    help="the server's --context-length. The prompt is sized to fill "
                         "it minus room for the answer: sglang rejects an input longer "
                         "than context_len - 6 (measured), and the generated tokens "
                         "have to fit under context_len as well, so the margin is "
                         "max_new + 64.")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--prefix-check", action="store_true")
    ap.add_argument("--prefix-len", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=8192,
                    help="the server's --chunked-prefill-size, for the warning above")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if (args.tokens is None) == (args.fill_ctx is None):
        raise SystemExit("pass exactly one of --tokens / --fill-ctx")
    if args.fill_ctx is not None:
        args.tokens = args.fill_ctx - args.max_new - 64
        print(f"--fill-ctx {args.fill_ctx} -> prompt {args.tokens} tokens "
              f"(leaving {args.max_new + 64} for the answer and sglang's own margin)")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.prefix_check:
        return run_prefix(args, tok)
    return run_needle(args, tok)


if __name__ == "__main__":
    raise SystemExit(main())
