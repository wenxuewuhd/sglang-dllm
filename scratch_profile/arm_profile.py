"""Arm a running server's profiler with stack collection OFF.

Use this INSTEAD of `bench_serving --profile`. bench_serving omits with_stack
from the /start_profile body, so the server falls back to its default of
with_stack=True (profiler_manager.py: `with_stack if with_stack is not None else
True`). That unwinds a Python stack for every profiled op, which inflates host
time, bloats the trace, and makes any host/gap/overlap reading unusable --
kernel durations stay valid, everything host-side does not.

Usage: start the server, start the dataset bench WITHOUT --profile, then run
this once the batch is at steady state:

    python arm_profile.py --port 31600 --wait 150 --steps 12 --out <dir>

It POSTs start_profile with with_stack=False / record_shapes=False and lets the
server auto-stop after `--steps` forwards.
"""

import argparse
import json
import urllib.request


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31600)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--wait",
        type=float,
        default=150,
        help="seconds to wait before arming, so the batch reaches steady state",
    )
    ap.add_argument("--steps", type=int, default=12, help="forwards to capture")
    ap.add_argument("--out", required=True, help="output directory for the trace")
    args = ap.parse_args()

    import time

    print(f"[wait] {args.wait}s for steady state...", flush=True)
    time.sleep(args.wait)
    body = {
        "output_dir": args.out,
        "num_steps": args.steps,
        "activities": ["CPU", "GPU"],
        "with_stack": False,
        "record_shapes": False,
        "profile_by_stage": False,
    }
    print(f"[arm] {body}", flush=True)
    print(post(f"http://{args.host}:{args.port}/start_profile", body), flush=True)
    print(
        f"[done] auto-stops after {args.steps} forwards; trace -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
