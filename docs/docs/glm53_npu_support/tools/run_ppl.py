#!/usr/bin/env python
"""Held-out perplexity against the running server, in long windows.

Why this exists: GSM8K absorbs distributional error -- a wrong token gets
recovered over 250 tokens of reasoning -- so it cannot resolve a change that
moves the token distribution by a few multiples of the deployment's own floor.
Teacher-forced NLL absorbs nothing, and it is the only criterion here that
gets one number per token instead of one per question.

Two deliberate choices:

  * Windows default to 4096 tokens, above ``index_topk=2048``. Below that the DSA
    indexer selects everything and the sparse path is never taken, which is the
    trap the handoff doc flags for the smoke test -- it applies just as much to
    the 11-24 token prompts the logprob check uses.
  * Requests go out serially. The prefill grouping of concurrent requests is not
    reproducible across runs (PLAN §4), and this is a logprob measurement, so the
    one thing it must not have is batch-shape noise.

Per-window NLL is written out, not just the total, so two runs can be compared
as a paired sample rather than as two scalars.
"""

import argparse
import json
import os
import math
import time
from pathlib import Path

import requests

# The dataset lived in one person's home directory, which is not a dependency this
# tool should carry. GLM53_EVAL_DIR overrides; the original path remains the last
# fallback so an existing setup keeps working.
DEFAULT_DATA = Path(
    os.environ.get("GLM53_EVAL_DIR")
    or "/mnt/workspace/y00359136/work/glm53_dev/env/eval"
) / "wikitext" / "test.parquet"
if not DEFAULT_DATA.is_file():
    DEFAULT_DATA = Path("/mnt/workspace/l84414662/glm53/env/eval/wikitext/test.parquet")


def load_text(path: Path) -> str:
    import pyarrow.parquet as pq

    rows = pq.read_table(str(path)).to_pylist()
    return "".join(r["text"] for r in rows)


def window_nll(host: str, port: int, ids: list[int], timeout: float) -> tuple[float, int]:
    """Return (sum of -logprob, number of scored positions) for one window."""
    r = requests.post(
        f"http://{host}:{port}/generate",
        json={
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        timeout=timeout,
        # The box exports HTTP_PROXY and requests honours it even for 127.0.0.1,
        # where the proxy answers 503.
        proxies={"http": None, "https": None},
    )
    r.raise_for_status()
    entries = r.json()["meta_info"]["input_token_logprobs"]
    # The first position has no predecessor and carries a null logprob.
    lps = [e[0] for e in entries if e[0] is not None]
    return -sum(lps), len(lps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30051)
    ap.add_argument("--model", default="/mnt/workspace/models/GLM-5.3-Flash-W8A8")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument(
        "--window",
        type=int,
        default=4096,
        help="tokens per window; keep above index_topk=2048 so the sparse path runs",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 means every window")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok.encode(load_text(args.data), add_special_tokens=False)

    n_win = len(ids) // args.window
    if args.limit:
        n_win = min(n_win, args.limit)
    print(f"{len(ids)} tokens -> {n_win} windows of {args.window}")

    windows = []
    total_nll = 0.0
    total_tok = 0
    t0 = time.time()
    for w in range(n_win):
        chunk = ids[w * args.window : (w + 1) * args.window]
        nll, n = window_nll(args.host, args.port, chunk, args.timeout)
        windows.append({"i": w, "nll_sum": nll, "n": n, "ppl": math.exp(nll / n)})
        total_nll += nll
        total_tok += n
        if (w + 1) % 10 == 0 or w + 1 == n_win:
            print(
                f"  {w + 1}/{n_win}  {time.time() - t0:.0f}s  "
                f"running ppl {math.exp(total_nll / total_tok):.4f}"
            )

    ppl = math.exp(total_nll / total_tok)
    print(f"\nperplexity      {ppl:.4f}")
    print(f"mean NLL        {total_nll / total_tok:.6f} nats/token")
    print(f"scored          {total_tok} tokens in {n_win} windows of {args.window}")
    print(f"wall            {time.time() - t0:.0f}s")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "ppl": ppl,
                    "mean_nll": total_nll / total_tok,
                    "total_nll": total_nll,
                    "tokens": total_tok,
                    "window": args.window,
                    "n_windows": n_win,
                    "windows": windows,
                }
            )
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
