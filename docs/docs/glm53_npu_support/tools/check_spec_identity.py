#!/usr/bin/env python
"""Compare what a speculative server writes against a non-speculative one.

⛔ **The obvious criterion does not hold on this backend, and this tool exists
partly to record that.** Greedy speculative decoding is exact in real arithmetic
-- a draft token is accepted only when it equals the target's argmax -- so the
output "must" be identical. It is not, and speculation is not the reason:
decoding the same prompt at batch width 1 and at batch width 8 with no
speculation at all also diverges, one prompt at token 16 and another at token 0
(measured 2026-08-30). Greedy output here is not batch-invariant, and turning MTP
on changes the target's forward from one token per request to N.

So a DIFFER result from this tool is not evidence of a verify bug. What it is
good for:

* **an IDENTICAL result still means something** -- it is a strong pass, just not
  a criterion you can demand;
* **locating** the first divergent token when you already have other reason to
  suspect a bug;
* comparing two builds at the SAME batch shape, where bit-identity is legal.

For "is the verify arithmetic correct", use GSM8K (statistical, tolerates the
nondeterminism) and the accept length, which needs no reference at all: a verify
step computing wrong logits stops agreeing with the draft, so accept length
collapses toward 1.0 and cannot be faked upward.

What it covers that a short smoke test does not: with a prompt longer than
``index_topk`` the kpool sparse path is really exercised, and every rejected
draft exercises the KDA conv/SSM rollback -- the snapshot path that only
speculative decoding can reach.

⚠ What it does NOT cover: state that is written and never read back. A rollback
reads the snapshot, so a corrupted snapshot shows up here; a buffer written
wrongly and never consumed does not. See REGRESSION.md's mixed-state section.

    # against a server with --speculative-algorithm NEXTN
    $VENV/bin/python check_spec_identity.py --port 30023 --out spec.json
    # restart the same server without the speculative flags, then
    $VENV/bin/python check_spec_identity.py --port 30023 --against spec.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

NO_PROXY = {"http": None, "https": None}

#: Short prompts first -- they are the ones a bring-up already passes, kept so a
#: regression that only affects them is still visible. The long ones are the
#: point: below `index_topk` the indexer selects everything and the sparse path
#: never binds.
SHORT_PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
    "水的化学式是 H2O，它由氢和氧两种元素组成。常温下水的沸点是",
    "Q: If a train travels 60 km in 45 minutes, what is its speed in km/h?\nA:",
]

_FILLER = (
    "The Ascend NPU port of this model separates two kinds of attention across its "
    "decoder stack. Linear attention keeps a recurrent state whose size does not "
    "grow with the sequence, while sparse attention keeps a pool of keys and "
    "selects among them. Numerical work on such a stack has to say which of the "
    "two it is talking about at every step. "
)


def build_prompts(tok, long_tokens: int) -> list[dict]:
    """Short prompts as text, long ones as exact-length id sequences."""
    cases = [{"name": f"short{i}", "text": p} for i, p in enumerate(SHORT_PROMPTS)]
    filler = tok(_FILLER, add_special_tokens=False)["input_ids"]
    tail = tok(
        "\n\nSummarise the paragraph above in one sentence.\n",
        add_special_tokens=False,
    )["input_ids"]
    body = (filler * (long_tokens // len(filler) + 1))[: long_tokens - len(tail)]
    cases.append({"name": f"long{long_tokens}", "ids": body + tail})
    return cases


def generate(host, port, case, n_new, timeout=3600):
    payload = {"sampling_params": {"max_new_tokens": n_new, "temperature": 0}}
    payload["input_ids" if "ids" in case else "text"] = case.get("ids") or case["text"]
    r = requests.post(
        f"http://{host}:{port}/generate", json=payload, timeout=timeout, proxies=NO_PROXY
    )
    r.raise_for_status()
    body = r.json()
    meta = body["meta_info"]
    return {
        "name": case["name"],
        "text": body["text"],
        # output_ids is the exact object under test; text can collapse two
        # different id sequences onto one string.
        "out_ids": body.get("output_ids"),
        "completion_tokens": meta["completion_tokens"],
        "spec_accept_length": meta.get("spec_accept_length"),
        "spec_verify_ct": meta.get("spec_verify_ct"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30023)
    ap.add_argument("--model", default="/mnt/workspace/models/GLM-5.3-Flash-BF16")
    ap.add_argument("--long-tokens", type=int, default=8192)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--against", type=Path)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    got = [
        generate(args.host, args.port, c, args.max_new)
        for c in build_prompts(tok, args.long_tokens)
    ]
    for g in got:
        spec = (
            f"  accept_length={g['spec_accept_length']:.3f} "
            f"verify_ct={g['spec_verify_ct']}"
            if g["spec_accept_length"] is not None
            else "  (no speculation)"
        )
        print(f"{g['name']:>12}: {g['completion_tokens']:>4} tokens{spec}")

    if args.out:
        args.out.write_text(json.dumps(got, ensure_ascii=False))
        print(f"wrote {args.out}")

    if not args.against:
        return 0

    other = json.loads(args.against.read_text())
    by_name = {o["name"]: o for o in other}
    bad = []
    print(f"\nvs {args.against.name}:")
    for g in got:
        o = by_name.get(g["name"])
        if o is None:
            print(f"  {g['name']:>12}: MISSING in reference")
            bad.append(g["name"])
            continue
        same_ids = g["out_ids"] == o["out_ids"]
        same_text = g["text"] == o["text"]
        # ids are the criterion; text is reported because that is what a human
        # reads when it fails.
        ok = same_ids if g["out_ids"] and o["out_ids"] else same_text
        first = None
        if not same_ids and g["out_ids"] and o["out_ids"]:
            for i, (a, b) in enumerate(zip(g["out_ids"], o["out_ids"])):
                if a != b:
                    first = i
                    break
            if first is None:
                first = min(len(g["out_ids"]), len(o["out_ids"]))
        print(
            f"  {g['name']:>12}: ids {'IDENTICAL' if same_ids else 'DIFFER'}"
            f"  text {'IDENTICAL' if same_text else 'DIFFER'}"
            + (f"  first differing token: {first}" if first is not None else "")
        )
        if not ok:
            bad.append(g["name"])
    print(
        f"\n{len(got) - len(bad)}/{len(got)} identical"
        + (f"  -- FAILED: {bad}" if bad else "")
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
