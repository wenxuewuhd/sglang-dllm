#!/usr/bin/env python
"""MMLU scored on a single answer token, against the running server.

The point is not the MMLU number. The point is that the answer *is* the argmax
of one distribution, so a change that moves the distribution shows up directly
instead of being absorbed by 250 tokens of reasoning the way GSM8K absorbs it.

Three statistics come out, in increasing order of resolution:

  accuracy      the familiar one, and the least sensitive
  flip rate     fraction of questions where the predicted letter changes between
                two runs -- decisions changed, with no absorption at all
  margin        logprob(best) - logprob(runner-up), per question. Continuous and
                paired, so the shift between two runs has real error bars even
                when accuracy does not move.

Prompts are raw completions, not chat -- no template, no thinking, one token out.
``--concurrency`` must match between the runs being compared: concurrent prefill
grouping is not reproducible across runs (PLAN §4), and while argmax is far less
exposed to that than a logprob value is, matching it keeps the two runs' batch
shape distributions the same.

!! DISABLED BY DEFAULT -- THIS TOOL CRASHES THE SERVER ON THE ASCEND STACK. !!

The ``margin`` statistic is the whole reason this file asks for logprobs, and
``{"return_logprob": true, "top_logprobs_num": N}`` is exactly the request that
JITs an uncompiled Triton kernel here; ``bishengir-compile`` takes a SIGSEGV and
the server dies with it. Not a bug in this file and not ours to fix, so the tool
refuses to run rather than leaving the knowledge in a document nobody reads.

Use ``/var/tmp/glm53/acc/mmlu_safe.py`` instead: pure greedy single-token, no
logprob fields at all, so it gives accuracy and flip rate. It cannot give margin
-- margin needs the logprobs that crash the server, and there is no cheap
substitute. Set ``MMLU_ALLOW_LOGPROB=1`` to run this anyway, on a server you are
willing to lose.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# The corpus is not vendored -- it is too big and it is shared between checkouts --
# so its location comes from the environment. This used to fall back to two absolute
# paths under two different people's home directories, which made the tool silently
# unrunnable for anyone else and silently wrong for anyone who had a stale copy at
# one of them. glm53_env.sh exports GLM53_EVAL_DIR; --data overrides it.
_EVAL_DIR = os.environ.get("GLM53_EVAL_DIR") or ""
DEFAULT_DATA = Path(_EVAL_DIR) / "mmlu" / "test.parquet" if _EVAL_DIR else None
LETTERS = "ABCD"

#: Opt-in for the request shape that kills the server. Named for what it enables,
#: not for a severity word, so `grep MMLU_ALLOW_LOGPROB` finds both the gate and
#: every place someone chose to defeat it.
ALLOW_ENV = "MMLU_ALLOW_LOGPROB"

REFUSAL = f"""\
run_mmlu_1tok.py is disabled: on this Ascend stack it kills the server.

  It sends {{"return_logprob": true, "top_logprobs_num": N}}. That combination JITs
  an uncompiled Triton kernel, bishengir-compile takes a SIGSEGV, and the server
  process dies. This is not a fault in this script and fixing the Triton path is
  out of scope; the request shape is simply unusable here.

  Use instead:  /var/tmp/glm53/acc/mmlu_safe.py
                pure greedy single-token, no logprob fields -> cannot trigger it.
                Gives accuracy and flip rate. Does NOT give margin.

  There is no safe way to get `margin` on this stack: margin is defined as
  logprob(top1) - logprob(top2), and obtaining those logprobs is the crash.

  If you genuinely accept losing the server (e.g. a scratch instance that is not
  shared, and no one else's evaluation is in flight), re-run with:

      {ALLOW_ENV}=1 $VENV/bin/python run_mmlu_1tok.py ...
"""


def check_allowed() -> None:
    """Refuse to send the server-killing request unless explicitly opted in."""
    if os.environ.get(ALLOW_ENV) == "1":
        print(
            f"{ALLOW_ENV}=1 set: sending return_logprob requests. This is the "
            "combination that SIGSEGVs bishengir-compile and takes the server "
            "down with it.",
            file=sys.stderr,
            flush=True,
        )
        return
    print(REFUSAL, file=sys.stderr)
    raise SystemExit(2)


PROMPT = """The following is a multiple choice question about {subject}.

{question}
A. {a}
B. {b}
C. {c}
D. {d}
Answer:"""


def build(row: dict) -> str:
    ch = list(row["choices"])
    return PROMPT.format(
        subject=row["subject"].replace("_", " "),
        question=row["question"].strip(),
        a=ch[0],
        b=ch[1],
        c=ch[2],
        d=ch[3],
    )


def letter_ids(model_dir: str) -> dict[int, str]:
    """Map the token ids that spell an answer letter back to the letter.

    Both the space-prefixed and bare forms are single tokens here, and which one
    the model emits depends on how the prompt ends, so accept either.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    out = {}
    for letter in LETTERS:
        for form in (f" {letter}", letter):
            ids = tok.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                out.setdefault(ids[0], letter)
    return out


def ask(
    session: requests.Session,
    host: str,
    port: int,
    prompt: str,
    topk: int,
    timeout: float,
    ids_to_letter: dict[int, str],
) -> dict:
    r = session.post(
        f"http://{host}:{port}/generate",
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "top_logprobs_num": topk,
        },
        timeout=timeout,
        proxies={"http": None, "https": None},
    )
    r.raise_for_status()
    meta = r.json()["meta_info"]
    # output_top_logprobs is one list per generated position; we asked for one.
    top = meta["output_top_logprobs"][0]
    # [logprob, token_id, token_text]. The text is None unless the server is asked
    # to decode it, so match on the id; the list is already sorted by logprob, so
    # the first hit for a letter is that letter's best form.
    lp = {}
    for logprob, tid, _text in top:
        letter = ids_to_letter.get(tid)
        if letter is not None and letter not in lp:
            lp[letter] = logprob
    return lp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30051)
    ap.add_argument("--model", default="/mnt/workspace/models/GLM-5.3-Flash-W8A8")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--limit", type=int, default=0, help="0 means all 14042")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.data is None:
        ap.error(
            "no corpus: set GLM53_EVAL_DIR to a directory holding mmlu/test.parquet, "
            "or pass --data explicitly. (Sourcing glm53_env.sh exports it.)"
        )
    if not args.data.is_file():
        ap.error(f"no corpus at {args.data}")

    # After argparse so --help still works, but before the tokenizer, the dataset
    # read, and anything that touches the network.
    check_allowed()

    import pyarrow.parquet as pq

    ids_to_letter = letter_ids(args.model)
    rows = pq.read_table(str(args.data)).to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} questions, concurrency {args.concurrency}, top-{args.topk}")

    session = requests.Session()
    session.mount(
        "http://", requests.adapters.HTTPAdapter(pool_maxsize=args.concurrency)
    )

    t0 = time.time()
    done = [0]

    def one(idx_row):
        i, row = idx_row
        lp = ask(
            session, args.host, args.port, build(row), args.topk, args.timeout,
            ids_to_letter,
        )
        done[0] += 1
        if done[0] % 1000 == 0:
            print(f"  {done[0]}/{len(rows)}  {time.time() - t0:.0f}s")
        ranked = sorted(lp.items(), key=lambda kv: -kv[1])
        return {
            "i": i,
            "subject": row["subject"],
            "gold": LETTERS[int(row["answer"])],
            "pred": ranked[0][0] if ranked else None,
            "margin": (ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else None,
            "logprobs": lp,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        recs = list(ex.map(one, enumerate(rows)))
    recs.sort(key=lambda r: r["i"])

    n = len(recs)
    scored = [r for r in recs if r["pred"] is not None]
    correct = sum(r["pred"] == r["gold"] for r in scored)
    with_margin = [r for r in scored if r["margin"] is not None]
    mean_margin = sum(r["margin"] for r in with_margin) / max(len(with_margin), 1)

    print(f"\naccuracy        {correct}/{n} = {100 * correct / n:.2f}%")
    print(f"letter found    {len(scored)}/{n} = {100 * len(scored) / n:.2f}%")
    print(f"mean margin     {mean_margin:.4f} nats  (top1 - top2, over {len(with_margin)})")
    print(f"wall            {time.time() - t0:.0f}s")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "n": n,
                    "correct": correct,
                    "scored": len(scored),
                    "mean_margin": mean_margin,
                    "concurrency": args.concurrency,
                    "topk": args.topk,
                    "records": recs,
                }
            )
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
