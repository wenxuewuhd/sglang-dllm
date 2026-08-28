"""Shared graph-capture verification helper.

The only interesting question is not "did capture succeed" but "does the replay
still compute a *function* of its inputs".  So every check runs the same three
steps:

    A  eager(inA)   -> refA      ; capture on inA ; replay -> gA   ; gA == refA?
    B  overwrite the *same device buffers* with inB, in place
       eager(inB)   -> refB      ; replay        -> gB   ; gB == refB?

Step B is the one that matters.  If any host-side Python read a device value at
capture time and baked the answer in, gB still equals refA and nothing raises.
"""
from __future__ import annotations
import time, torch


def rel(a, b):
    a = a.detach().float().cpu(); b = b.detach().float().cpu()
    return (a - b).norm().item() / max(b.norm().item(), 1e-12)


class Cap:
    def __init__(self, name):
        self.name = name
        self.pool = torch.npu.graph_pool_handle()
        self.stream = torch.npu.Stream()
        self.graph = None

    def capture(self, fn, warmup=3):
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g, pool=self.pool, stream=self.stream,
                             auto_dispatch_capture=True):
            out = fn()
        torch.npu.synchronize()
        self.graph = g
        return out

    def replay(self):
        self.graph.replay()
        torch.npu.synchronize()


def snap(d):
    """Deep-copy a dict of device tensors to cpu float32 for comparison."""
    return {k: v.detach().float().cpu().clone() for k, v in d.items()
            if torch.is_tensor(v)}


def compare(tag, got, ref, tol=0.0):
    bad = []
    for k in ref:
        if k not in got:
            bad.append(f"{k}: MISSING"); continue
        e = rel(got[k], ref[k])
        eq = torch.equal(got[k], ref[k])
        status = "bitwise" if eq else (f"rel={e:.3e}" if e <= tol else f"rel={e:.3e} MISMATCH")
        if not eq and e > tol:
            bad.append(f"{k}: {status}")
        print(f"    {tag:<22} {k:<24} {status}")
    return bad


def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]
