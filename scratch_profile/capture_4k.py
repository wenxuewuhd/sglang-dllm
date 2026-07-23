"""Capture steady-state 910B forwards at 4K-input / 1.5K-output, bs~72.
Feeder sends fixed 4096-token input_ids + max_new_tokens=1536 to hold ~72
concurrent long-context requests, waits for steady state, then arms the Level2
profiler for `steps` forwards. Kernel breakdown -> analyze.py."""

import argparse
import concurrent.futures as cf
import json
import threading
import time
import urllib.request


def post(url, payload, timeout=1200):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# deterministic varied 4096-token prompt (content irrelevant for kernel cost)
INPUT_IDS = [((i * 131 + 17) % 60000) + 1 for i in range(4096)]


def gen(host):
    try:
        post(
            f"{host}/generate",
            {
                "input_ids": INPUT_IDS,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1536},
            },
        )
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31600)
    ap.add_argument("--conc", type=int, default=72)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=float, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    host = f"http://127.0.0.1:{args.port}"
    post(f"{host}/flush_cache", {})
    stop = threading.Event()
    ex = cf.ThreadPoolExecutor(max_workers=args.conc * 3)

    def feed():
        while not stop.is_set():
            for _ in range(args.conc):
                ex.submit(gen, host)
            time.sleep(20)  # long reqs (1536 out) stay in flight a while

    threading.Thread(target=feed, daemon=True).start()
    print(
        f"[warmup] {args.warmup}s to steady-state bs~{args.conc} @4K ctx...", flush=True
    )
    time.sleep(args.warmup)
    print(f"[arm] {args.steps} forwards -> {args.out}", flush=True)
    post(
        f"{host}/start_profile",
        {
            "output_dir": args.out,
            "num_steps": args.steps,
            "activities": ["CPU", "GPU"],
            "record_shapes": False,
            "with_stack": False,
            "profile_by_stage": False,
        },
    )
    time.sleep(30)
    stop.set()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
