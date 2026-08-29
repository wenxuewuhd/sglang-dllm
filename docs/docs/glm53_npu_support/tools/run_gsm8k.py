#!/usr/bin/env python
"""GSM8K against a running server, in the protocol P4.2's 97.50% was measured in.

That number comes from the GPU cookbook with thinking ON -- ``sgl-eval run gsm8k
--thinking``, temperature 1.0, top_p 0.95, max_tokens 32768. It is NOT comparable to
sglang's own ``benchmark/gsm8k/bench_sglang.py``, which is 5-shot, greedy and
short-output. Use this one when the question is "did we hit the exit criterion", and
that one when the question is "is the server roughly sane".

Thinking needs no flag here: this checkpoint's chat_template.jinja ends every
generation prompt with ``<|assistant|><think>``, so /v1/chat/completions is already in
thinking mode and there is no toggle to get wrong.

Sampling is stochastic at temperature 1.0, so a single run has real variance -- the
cookbook's own three GPQA rounds spanned 3.5 points. Treat one run as one sample.

    $VENV/bin/python run_gsm8k.py --concurrency 128 --out gsm8k.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DATA = Path("/mnt/workspace/y00359136/work/glm53_dev/env/eval/gsm8k/test.jsonl")
# The dev box exports HTTP_PROXY=http://127.0.0.1:1056 and requests honours it even
# for 127.0.0.1, where the proxy answers 503.
NO_PROXY = {"http": None, "https": None}

INSTRUCTION = (
    "Solve the problem. Put your final numeric answer inside \\boxed{}."
)


def gold_answer(line: dict) -> float | None:
    # GSM8K's reference answer is the text after the '####' marker.
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", line["answer"])
    return float(m.group(1).replace(",", "")) if m else None


def predicted_answer(text: str) -> float | None:
    """The model's final number: from \\boxed{} if it used one, else the last number.

    Both halves need care. \\boxed{} content is not always a bare number -- measured on
    a full run, the model writes \\boxed{70\\%} and \\boxed{25 \\text{ hours}} -- so pull the
    number back out of it rather than handing the whole string to float(). And search
    only the segment after </think>: the reasoning trace is full of intermediate
    numbers, and taking the last one from the whole text would sometimes score the
    model's scratch work instead of its answer.

    An earlier version returned None when the boxed content would not parse, with no
    fallback. That scored 9 of 1319 correct answers as wrong -- an extraction artifact
    that looked exactly like a model error.
    """
    answer = text.split("</think>")[-1]
    boxed = re.findall(r"\\boxed\{([^}]*)\}", answer)
    for source in (boxed[-1] if boxed else "", answer):
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", source)
        if nums:
            try:
                return float(nums[-1].replace(",", ""))
            except ValueError:
                continue
    return None


#: Tokens the chat template and INSTRUCTION add on top of the bare question. Measured:
#: a question that tokenizes to 64 on its own is 90 prompt tokens at the server, so the
#: real overhead is ~26. 512 is deliberately loose -- it costs completion budget nobody
#: uses (the longest GSM8K question is 185 tokens and no thinking trace here comes near
#: 30k) and it removes a whole class of off-by-a-few 400s.
#:
#: This is measured rather than taken from `apply_chat_template`: this checkpoint keeps
#: its template in chat_template.jinja, which transformers 5.12.1 in the sglang venv
#: does not apply -- it renders empty and reports 2 tokens for any input.
PROMPT_OVERHEAD = 512


def completion_budget(args, tok, question: str) -> int:
    """What is actually left for the completion after the prompt.

    The cookbook ran max_tokens=32768, but on a longer context. Here --context-length
    is 32768, so asking for 32768 completion tokens on top of any prompt at all is a
    400: "You requested a total of 32848 tokens".
    """
    n_prompt = len(tok(question)["input_ids"]) + PROMPT_OVERHEAD
    return min(args.max_tokens, args.context_length - n_prompt)


def ask(args, tok, question: str) -> dict:
    r = requests.post(
        f"http://{args.host}:{args.port}/v1/chat/completions",
        json={
            "model": args.model,
            "messages": [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": completion_budget(args, tok, question),
        },
        timeout=args.timeout,
        proxies=NO_PROXY,
    )
    r.raise_for_status()
    choice = r.json()["choices"][0]
    return {
        "text": choice["message"]["content"] or "",
        "finish_reason": choice["finish_reason"],
        "completion_tokens": r.json()["usage"]["completion_tokens"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30003)
    ap.add_argument("--model", default="/mnt/workspace/models/GLM-5.3-Flash-BF16")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--limit", type=int, default=0, help="0 means all 1319")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--context-length", type=int, default=32768,
                    help="must match the server's --context-length; the completion "
                         "budget is this minus the prompt")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    lines = [json.loads(x) for x in args.data.read_text().splitlines() if x.strip()]
    if args.limit:
        lines = lines[: args.limit]
    print(f"{len(lines)} questions, concurrency {args.concurrency}, "
          f"temp {args.temperature} top_p {args.top_p} max_tokens {args.max_tokens}",
          flush=True)

    done = [0]
    t0 = time.time()

    def one(line):
        res = ask(args, tok, line["question"])
        done[0] += 1
        if done[0] % 50 == 0:
            el = time.time() - t0
            print(f"  {done[0]}/{len(lines)}  {el:.0f}s  "
                  f"({done[0] / el:.2f} q/s)", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(one, lines))
    elapsed = time.time() - t0

    correct = truncated = 0
    gen_tokens = 0
    records = []
    for line, res in zip(lines, results):
        gold = gold_answer(line)
        pred = predicted_answer(res["text"])
        ok = gold is not None and pred is not None and abs(pred - gold) < 1e-4
        correct += ok
        truncated += res["finish_reason"] == "length"
        gen_tokens += res["completion_tokens"]
        records.append({"question": line["question"], "gold": gold, "pred": pred,
                        "ok": ok, "finish_reason": res["finish_reason"],
                        "completion_tokens": res["completion_tokens"],
                        # Keeping the text is what makes a scoring fix re-scorable
                        # offline; without it the only way to re-score is a 25-minute
                        # re-run, and at temperature 1.0 that is a different sample.
                        "text": res["text"]})

    n = len(lines)
    print(f"\naccuracy        {correct}/{n} = {100 * correct / n:.2f}%")
    # A truncated generation is not a wrong answer, it is a missing one; reporting it
    # separately keeps a max_tokens that is too small from looking like a model error.
    print(f"stop rate       {100 * (n - truncated) / n:.2f}%  ({truncated} hit max_tokens)")
    print(f"wall            {elapsed:.0f}s  ({n / elapsed:.2f} q/s)")
    print(f"completion      {gen_tokens} tokens, mean {gen_tokens / n:.0f}/question, "
          f"{gen_tokens / elapsed:.0f} tok/s aggregate")
    if args.out:
        args.out.write_text(json.dumps(
            {"n": n, "correct": correct, "truncated": truncated,
             "elapsed_s": elapsed, "completion_tokens": gen_tokens,
             "concurrency": args.concurrency, "temperature": args.temperature,
             "top_p": args.top_p, "max_tokens": args.max_tokens,
             "records": records}))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
