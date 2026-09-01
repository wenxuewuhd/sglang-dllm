"""Fire N concurrent greedy requests so raw_bs lands off a captured bucket.

The sequential checks all run raw_bs=1, which is itself a bucket, so they never pad.
Padding is where the graph-only failure mode lives: every padded row claims
req_pool_index 0 and mamba slot -1, and 34 KDA layers scatter by those indices
(see PLAN.md P6.6a).

Read the result with the padding column, not the match count. Measured 2026-08-29:
raw_bs 8 and 16 pad zero rows and still diverge from the bs=1 baseline as much as
raw_bs 13 with three padding rows -- so what moves the numbers is batch *width*
(bf16 GEMM is not batch-shape invariant, and MoE routing turns a ulp into a discrete
flip), not padding. A padding bug would look different: garbage text, not a
coherent continuation that branches at a near-tied token.

    $VENV/bin/python check_graph_padding.py
"""
import json
from concurrent.futures import ThreadPoolExecutor
import requests

PORT = 30003
G = "${GLM53_ROOT}/env/goldens/logits/ref_server_eager_short_d100.json"
ref = json.load(open(G))["data"]
NOPROXY = {"http": None, "https": None}

def one(rec, n_new):
    r = requests.post(
        f"http://127.0.0.1:{PORT}/generate",
        json={"input_ids": rec["ids"],
              "sampling_params": {"max_new_tokens": n_new, "temperature": 0},
              "return_logprob": True, "logprob_start_len": 0},
        timeout=1800, proxies=NOPROXY)
    r.raise_for_status()
    m = r.json()["meta_info"]
    return [e[1] for e in m["output_token_logprobs"]]

for n in (3, 5, 7, 8, 13, 16):
    subset = [ref[i % len(ref)] for i in range(n)]
    with ThreadPoolExecutor(max_workers=n) as ex:
        got = list(ex.map(lambda rec: one(rec, 40), subset))
    same = sum(g == subset[i]["out_ids"][:40] for i, g in enumerate(got))
    # A padded row that corrupts a real request's KDA state produces garbage, not a
    # one-token drift, so report the first mismatch position rather than a count.
    firsts = [next((t for t in range(40) if g[t] != subset[i]["out_ids"][t]), None)
              for i, g in enumerate(got)]
    firsts = [f for f in firsts if f is not None]
    bucket = next(b for b in (1, 2, 4, 8, 12, 16) if b >= n)
    print(f"raw_bs={n:>2} -> bucket {bucket:>2} ({bucket - n} padding rows): "
          f"{same}/{n} match the bs=1 baseline exactly"
          + (f", earliest divergence at generated token {min(firsts)}" if firsts else ""))
