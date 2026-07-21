"""Capture several consecutive FDFO denoise rounds at steady state and split the
wall time into device-busy vs host-gap (scheduler / init / denoise between
forwards).

dLLM FDFO has no separate decode phase: every scheduler round is a DLLM_EXTEND
batch mixing new-prompt prefill blocks and ongoing denoise blocks (the server
logs them all as "Prefill batch", #running-req: 0). Request admission ramps up
gradually, so the batch only reaches steady-state fullness after ~60-90s. This
feeder keeps submitting requests to hold the batch full, waits for warmup, then
arms the profiler for `steps` rounds -- which lands on steady mixed denoise
batches, not the initial prompt-prefill wave.
"""

import argparse
import concurrent.futures as cf
import json
import threading
import time
import urllib.request

TOPICS = [
    "the history of container shipping",
    "photosynthesis at the molecular level",
    "CPU branch predictor design",
    "renewable energy storage economics",
    "the evolution of writing systems",
    "how vaccines train immunity",
    "deep sea exploration engineering",
    "error-correcting codes",
]


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def gen(host, i, max_new):
    prompt = (
        f"Write a long detailed multi-section essay about "
        f"{TOPICS[i % len(TOPICS)]} (variant {i}). Cover history, "
        f"technical details, future outlook."
    )
    try:
        post(
            f"{host}/generate",
            {
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new},
            },
        )
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31400)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=70)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    host = f"http://127.0.0.1:{args.port}"

    post(f"{host}/flush_cache", {})

    # Feeder: keep a full pipeline of requests in flight so the batch stays at
    # steady-state fullness through the capture, not draining after one wave.
    stop = threading.Event()
    ex = cf.ThreadPoolExecutor(max_workers=args.bs * 2)
    counter = [0]

    def feed():
        while not stop.is_set():
            # top up to ~1.5x bs of outstanding requests
            for _ in range(args.bs):
                ex.submit(gen, host, counter[0], 1024)
                counter[0] += 1
            time.sleep(8)

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()

    print(f"[warmup] {args.warmup}s to reach steady-state batch...", flush=True)
    time.sleep(args.warmup)

    print(f"[arm] {args.steps} FDFO rounds -> {args.out}", flush=True)
    post(
        f"{host}/start_profile",
        {
            "output_dir": args.out,
            "num_steps": args.steps,
            "activities": ["CPU", "GPU"],
            "record_shapes": True,
            "with_stack": False,
            "profile_by_stage": False,
        },
    )
    time.sleep(25)  # let the capture + trace export complete

    stop.set()
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
