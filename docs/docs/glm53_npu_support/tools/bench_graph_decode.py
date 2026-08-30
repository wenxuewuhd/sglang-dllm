"""Decode throughput under NPU Graph, with prefill separated out.

Times ``max_new_tokens=1`` for the prefill, then ``DEC + 1``, and differences them.
A single end-to-end number will not do: at 3256 input tokens and concurrency 16 the
prefill is 52k tokens, which swamps the decode the graph is there to speed up.

Greedy with ``ignore_eos`` so every request generates exactly ``DEC`` tokens and the
only variable is the batch the scheduler forms.

Reads its prompts from the recorded references so the input lengths are the same ones
the correctness checks use. Run it against a server that is otherwise idle -- this box
is shared, and numbers taken while someone else's training job is up are worthless.

    $VENV/bin/python bench_graph_decode.py
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--concurrency", default="1,2,4,8,16",
                help="comma-separated; must not exceed --max-running-requests")
ap.add_argument("--decode-tokens", type=int, default=128)
ap.add_argument("--port", type=int, default=30003)
args = ap.parse_args()
CONCURRENCY = [int(x) for x in args.concurrency.split(",")]

NP = {"http": None, "https": None}
G = "/mnt/workspace/y00359136/work/glm53_dev/env/goldens/logits"
pools = {
    "short (13 tok)": [d["ids"] for d in json.load(open(f"{G}/ref_server_eager_short_d100.json"))["data"]],
    "long (3256 tok)": [d["ids"] for d in json.load(open(f"{G}/ref_server_eager_long_d100.json"))["data"]],
}
DEC = args.decode_tokens

def run(ids, n_new):
    r = requests.post(f"http://127.0.0.1:{args.port}/generate",
        json={"input_ids": ids, "sampling_params": {"max_new_tokens": n_new, "temperature": 0, "ignore_eos": True}},
        timeout=1800, proxies=NP)
    r.raise_for_status()

def warmup(pool):
    """Burn the first call before anything is timed.

    The prefill column is measured by timing max_new_tokens=1 and the decode column
    by differencing it out, so a cold first call inflates the subtrahend and makes
    decode look *faster*. The bias is one-directional, which is the dangerous kind:
    measured on this stack a cold first prefill took 1.88 s against a warm 0.39 s,
    and 1.5 s of that landed entirely in the term being subtracted. A single
    discarded call removes it.
    """
    run(pool[0], 1)


def wall(pool, n, n_new):
    reqs = [pool[i % len(pool)] for i in range(n)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(lambda x: run(x, n_new), reqs))
    return time.time() - t0

for name, pool in pools.items():
    print(f"\n--- {name}, {DEC} decode tokens", flush=True)
    print(f"{'conc':>6}{'prefill s':>11}{'decode s':>10}{'ms/token':>10}{'decode tok/s total':>20}",
          flush=True)
    warmup(pool)
    for n in CONCURRENCY:
        t_pre = wall(pool, n, 1)
        t_all = wall(pool, n, DEC + 1)
        t_dec = max(t_all - t_pre, 1e-9)
        print(f"{n:>6}{t_pre:>11.2f}{t_dec:>10.2f}{1000 * t_dec / DEC:>10.1f}"
              f"{n * DEC / t_dec:>20.1f}", flush=True)
