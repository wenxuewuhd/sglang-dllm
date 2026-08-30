#!/usr/bin/env python
"""Kernel-level profile of a *running* sglang server's decode steps.

`layer_check/kernel_profile.py` drives one module directly; this drives the whole
network through the service, which is the only way to see the real per-step kernel
mix (graph replay, scheduler overhead, all 45 layers in their real order).

⚠ Service-level profiling killed the TP16 server once (16 schedulers SIGSEGV, and
the data was unreadable).  That run had profile_by_stage + record_shapes + 128
concurrent requests.  This driver defaults to the opposite of all three, and to a
small step count.  **Save any measurement you care about before running it.**

    $VENV/bin/python profile_server_decode.py --port 30013 --concurrency 1 \
        --steps 20 --out /var/tmp/glm53/prof/bs1

The load generator holds `--concurrency` requests in decode for the whole window
(ignore_eos, greedy), so every profiled step is a decode step of that width --
no prefill mixed in, provided --prefill-settle is long enough.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import requests

NOPROXY = {"http": None, "https": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30013)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=13)
    ap.add_argument(
        "--decode-tokens",
        type=int,
        default=600,
        help="must outlast the profile window at the measured ms/token",
    )
    ap.add_argument(
        "--prefill-settle",
        type=float,
        default=6.0,
        help="seconds to wait after issuing before starting the profiler, so the "
        "window contains no prefill batch",
    )
    ap.add_argument("--record-shapes", action="store_true")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    os.makedirs(args.out, exist_ok=True)

    ids = [100 + i for i in range(args.prompt_tokens)]

    def gen() -> None:
        try:
            requests.post(
                base + "/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": {
                        "max_new_tokens": args.decode_tokens,
                        "temperature": 0,
                        "ignore_eos": True,
                    },
                },
                timeout=1800,
                proxies=NOPROXY,
            )
        except Exception as e:  # the server may die under the profiler; say so
            print(f"  load thread: {type(e).__name__}: {e}", file=sys.stderr)

    threads = [threading.Thread(target=gen, daemon=True) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    print(f"issued {args.concurrency} request(s); settling {args.prefill_settle}s")
    time.sleep(args.prefill_settle)

    req = {
        "output_dir": args.out,
        "num_steps": args.steps,
        "activities": ["CPU", "GPU"],
        "profile_by_stage": False,
        "with_stack": False,
        "record_shapes": bool(args.record_shapes),
    }
    print("POST /start_profile", json.dumps(req))
    r = requests.post(base + "/start_profile", json=req, timeout=120, proxies=NOPROXY)
    print(" ->", r.status_code, r.text[:300])
    r.raise_for_status()

    # num_steps stops the profiler on its own; wait for the export to land.
    deadline = time.time() + 600
    seen = set()
    while time.time() < deadline:
        time.sleep(5)
        hits = []
        for root, _, files in os.walk(args.out):
            if "kernel_details.csv" in files:
                hits.append(os.path.join(root, "kernel_details.csv"))
        if hits and set(hits) == seen:
            break  # stopped growing
        seen = set(hits)
        try:
            requests.get(base + "/health", timeout=10, proxies=NOPROXY)
        except Exception:
            print("  server is not answering /health any more", file=sys.stderr)
    print(f"kernel_details.csv: {sorted(seen)}")

    if seen:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(os.path.dirname(here), "layer_check"))
        import kernel_profile as kp

        print(
            kp.summarize(
                args.out,
                label=args.label or f"decode bs={args.concurrency}",
                steps=args.steps,
                top=45,
            )
        )


if __name__ == "__main__":
    main()
