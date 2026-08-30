#!/usr/bin/env python
"""Check the per-forward kpool metadata cache against recomputing it.

Why this is a unit check and not an end-to-end one: on this backend a greedy
generation is not reproducible at any decode batch width above 1.  Measured --
same server process, same width, same prompts, repeated: outputs differ (see
REPORT 7b.15 and 7b.16).  The decode batch's *composition* is the scheduler's
to decide and a client cannot pin it, so there is no baseline to diff against
above batch 1, by any method.  The cache therefore has to be checked where it
is deterministic: drive it directly and compare against the uncached path.

What it checks, at several batch widths:

  hit      11 DSA layers in one forward get the same values the uncached path
           derives, and get them from the cache (identity, not just equality).
  values   new values written into the SAME buffers must miss (_version).
  rewrite  the same values rewritten in place must also miss -- conservative,
           because _version bumps on the write regardless of what was written.
  wrap     a new forward reusing identical buffers with NO write at all must
           still miss, caught by the layer ordering.  This is the case the
           fingerprint alone cannot see, and a stale hit here is silent:
           right shapes, wrong pool lengths.

Run: ASCEND_RT_VISIBLE_DEVICES=0 python check_step_metadata.py
"""
from __future__ import annotations

import sys, types

import torch
import torch_npu  # noqa: F401

import sglang.srt.runtime_context as rc
from sglang.srt.hardware_backend.npu.attention.kpool_indexer_npu import (
    KPoolNPUIndexerMixin,
)

#: `_step_metadata` asks the runtime context whether speculative decoding is on,
#: and a standalone harness has published no config. Stub it, and test both
#: answers: None means the cache is live, a number means it must be bypassed
#: (479e43e258 turns it off under spec, because a draft forward's DSA layer id is
#: above every target layer id and the ordering check cannot tell them apart).
_SPEC = [None]
rc.max_speculative_num_draft_tokens = lambda: _SPEC[0]

DEV = "npu:0"
KPOOL = 64
PAGE = 64
DSA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]


def broken_step_metadata(o, forward_batch, block_tables, batch, layer_id):
    """A plausible-looking cache with both invalidations removed.

    Keyed on `data_ptr` alone and with no layer-ordering check -- exactly the
    version someone would write who had not thought about in-place writes or
    about buffers being reused across forwards. The negative control asserts the
    battery above catches it. A check that cannot fail is not a check (7b.1).
    """
    seq_lens = forward_batch.seq_lens
    key = (seq_lens.data_ptr(), block_tables.data_ptr(), batch)
    cached = o._kpool_step_meta.get("entry")
    if cached is not None and cached[0] == key:
        return cached[1]
    saved = o._kpool_step_meta
    o._kpool_step_meta = {}
    try:
        meta = KPoolNPUIndexerMixin._step_metadata(
            o, forward_batch, block_tables, batch, layer_id
        )
    finally:
        o._kpool_step_meta = saved
    o._kpool_step_meta["entry"] = (key, meta)
    return meta


def make_self():
    o = types.SimpleNamespace()
    o._kpool_step_meta = {}
    o.index_kpool = KPOOL
    o._step_metadata = types.MethodType(
        KPoolNPUIndexerMixin._step_metadata.__get__(o, types.SimpleNamespace), o
    ).__func__.__get__(o)
    return o


def fb(seq_lens, batch):
    return types.SimpleNamespace(seq_lens=seq_lens, batch_size=batch)


def uncached(o, forward_batch, block_tables, batch):
    """The same derivation with the cache emptied -- the reference."""
    saved, o._kpool_step_meta = o._kpool_step_meta, {}
    try:
        return o._step_metadata(forward_batch, block_tables, batch, DSA_LAYERS[0])
    finally:
        o._kpool_step_meta = saved


def eq(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def run(batch: int, broken: bool = False) -> list[str]:
    fails = []
    o = make_self()
    if broken:
        o._step_metadata = types.MethodType(broken_step_metadata, o)
    seq_lens = torch.randint(200, 4000, (batch,), device=DEV, dtype=torch.int32)
    n_pages = 80
    block_tables = torch.randint(
        0, 4096, (batch, n_pages), device=DEV, dtype=torch.int32
    )
    f = fb(seq_lens, batch)

    ref = uncached(o, f, block_tables, batch)

    # hit: every DSA layer of one forward
    o._kpool_step_meta = {}
    first = o._step_metadata(f, block_tables, batch, DSA_LAYERS[0])
    if not eq(first, ref):
        fails.append(f"b={batch} first call != uncached")
    for lid in DSA_LAYERS[1:]:
        got = o._step_metadata(f, block_tables, batch, lid)
        if any(x is not y for x, y in zip(got, first)):
            fails.append(f"b={batch} layer {lid} recomputed instead of hitting")
        if not eq(got, ref):
            fails.append(f"b={batch} layer {lid} != uncached")

    # values: new values in the SAME buffer must miss
    seq_lens.copy_(torch.randint(200, 4000, (batch,), device=DEV, dtype=torch.int32))
    ref2 = uncached(o, f, block_tables, batch)
    got = o._step_metadata(f, block_tables, batch, DSA_LAYERS[0])
    if not eq(got, ref2):
        fails.append(f"b={batch} STALE after new values in the same buffer")

    # rewrite: same values rewritten in place -- must also miss (conservative)
    same = seq_lens.clone()
    seq_lens.copy_(same)
    got = o._step_metadata(f, block_tables, batch, DSA_LAYERS[0])
    if not eq(got, uncached(o, f, block_tables, batch)):
        fails.append(f"b={batch} wrong after an identical in-place rewrite")

    # wrap: a fresh forward, identical buffers, NO write -- only layer order can see it
    for lid in DSA_LAYERS[1:]:
        o._step_metadata(f, block_tables, batch, lid)
    got = o._step_metadata(f, block_tables, batch, DSA_LAYERS[0])
    if not eq(got, uncached(o, f, block_tables, batch)):
        fails.append(f"b={batch} STALE on a new forward with untouched buffers")

    # spec: with speculative decoding on the cache must be bypassed entirely,
    # so two calls in one forward must NOT share objects.
    _SPEC[0] = 2
    try:
        o._kpool_step_meta = {}
        a = o._step_metadata(f, block_tables, batch, DSA_LAYERS[0])
        b = o._step_metadata(f, block_tables, batch, DSA_LAYERS[1])
        if any(x is y for x, y in zip(a, b)):
            fails.append(f"b={batch} cache still live under speculative decoding")
        if not eq(b, uncached(o, f, block_tables, batch)):
            fails.append(f"b={batch} spec-path values wrong")
    finally:
        _SPEC[0] = None
    return fails


def main() -> int:
    allf = []
    for batch in (1, 2, 4, 13, 16):
        f = run(batch)
        print(f"  batch {batch:2d}: {'OK' if not f else 'FAIL'}")
        for x in f:
            print(f"      {x}")
        allf += f

    print("\n  negative control -- the same battery against a cache with both")
    print("  invalidations removed; it MUST fail, or the battery proves nothing:")
    caught = {b: len(run(b, broken=True)) for b in (1, 2, 4, 13, 16)}
    for b, n in caught.items():
        print(f"    batch {b:2d}: {n} failure(s) raised" + ("" if n else "   <-- NOT CAUGHT"))
    blind = [b for b, n in caught.items() if not n]
    if blind:
        allf.append(f"negative control passed at batch {blind} -- the battery is blind there")

    print(f"\nRESULT: {'PASS' if not allf else f'FAIL ({len(allf)})'}")
    return 1 if allf else 0


if __name__ == "__main__":
    sys.exit(main())
