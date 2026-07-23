"""Reproduce the dLLM FDFO mass-abort KV leak: fire concurrent long
generations, then exit abruptly so every connection drops mid-flight."""

import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 31501
N = 128


def gen(i):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate",
        data=json.dumps(
            {
                "text": f"Write a very long detailed essay about topic number {i}, covering history, present, and future.",
                "sampling_params": {"temperature": 0, "max_new_tokens": 1024},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=600).read()
    except Exception:
        pass


ex = cf.ThreadPoolExecutor(max_workers=N)
for i in range(N):
    ex.submit(gen, i)
print("submitted, running 25s...", flush=True)
time.sleep(25)
print("hard exit -> all connections drop", flush=True)
os._exit(1)
