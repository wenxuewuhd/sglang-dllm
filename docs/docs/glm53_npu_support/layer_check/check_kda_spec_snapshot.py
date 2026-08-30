#!/usr/bin/env python
"""Drive the two UNVERIFIED speculative KDA paths directly, with ragged batches.

`ascend_kda_backend.py` has two sites that only speculative decoding reaches,
both marked UNVERIFIED because this deployment runs no MTP:

  :513  the verify conv, handed a channel-major *copy* of the window-major pool
  :671  the snapshot scatter, handed a transposed *view* of the channel-major
        scratch, no copy -- the claim being that the kernel indexes by stride

Reading the source says both are consistent with how memory_pool.py allocates
these tensors.  That is not verification: three of the four operator
constraints this project has hit are silent, and "looks right in the source" is
exactly how they survive.  So drive them.

Ragged is the point.  The long-context line's MTE fault appears at batch 16 on
a ragged GSM8K batch but only at 32 on a uniform one, so every request here
gets a different sequence length, a different cache slot and a different step,
none of them an arange.

Every case ships a negative control: the same assertion run against a
deliberately wrong call.  A check that cannot fail proves nothing (REPORT 7b.1).

Run: ASCEND_RT_VISIBLE_DEVICES=0 python check_kda_spec_snapshot.py
"""
from __future__ import annotations

import sys

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"
DT = torch.bfloat16
# GLM-5.3-Flash KDA, from config.json + the allocator log (conv_state 0.08 GB):
# 34 KDA layers, 16+1 slots, K=4 so window=K-1=3, channels = 3 * 64 * 128.
WINDOW = 3
CHANNELS = 3 * 64 * 128
LAYERS = 4          # a slice is enough; the kernel strides over layers
DRAFT = 4
BATCHES = (2, 3, 8, 13, 16, 32)
# Slots and scratch rows are per-request, so they must outnumber the batch;
# step indices are not -- several requests legitimately land on the same step.
SLOTS = max(BATCHES) + 5
SCRATCH = max(BATCHES) + 3


def ragged(batch: int, hi: int, seed: int, distinct: bool = True) -> torch.Tensor:
    """Irregular and non-monotonic -- never an arange, never equal spacing."""
    g = torch.Generator().manual_seed(seed)
    if distinct:
        assert hi >= batch, (hi, batch)
        return torch.randperm(hi, generator=g)[:batch].to(torch.int32)
    return torch.randint(0, hi, (batch,), generator=g).to(torch.int32)


def ref_scatter(dst, src, dst_idx, src_idx, step_idx):
    """What the scatter must produce, written in plain torch."""
    out = dst.clone()
    for r in range(dst_idx.numel()):
        d, s, t = int(dst_idx[r]), int(src_idx[r]), int(step_idx[r])
        if d < 0 or s < 0 or t < 0:
            continue
        out[:, d] = src[:, s, t]
    return out


def case_scatter(batch: int) -> list[str]:
    from sgl_kernel_npu.mamba.speculative_state_scatter import (
        speculative_state_scatter_npu,
    )

    fails = []
    # Persistent pool: window-major, as _init_npu_conv_state builds it for KDA.
    dst = torch.randn(LAYERS, SLOTS, WINDOW, CHANNELS, dtype=DT, device=DEV)
    # Scratch: channel-major, as memory_pool.py allocates it (dense_conv_shapes
    # flips the KDA (window, channels) source tuple).
    scratch = torch.randn(LAYERS, SCRATCH, DRAFT, CHANNELS, WINDOW, dtype=DT, device=DEV)
    # ...and the backend hands the kernel a transposed VIEW of it, no copy.
    src_view = scratch.transpose(-1, -2)
    assert not src_view.is_contiguous(), "the view under test must be strided"

    dst_idx = ragged(batch, SLOTS, 11).to(DEV)
    src_idx = ragged(batch, SCRATCH, 22).to(DEV)
    step_idx = ragged(batch, DRAFT, 33, distinct=False).to(DEV)
    # One masked-out request, which the kernel must skip rather than write.
    if batch >= 3:
        dst_idx[1] = -1

    want = ref_scatter(dst, src_view, dst_idx, src_idx, step_idx)
    got = dst.clone()
    speculative_state_scatter_npu(got, src_view, dst_idx, src_idx, step_idx)
    if not torch.equal(got, want):
        n = (got != want).sum().item()
        fails.append(f"b={batch} scatter through a strided view is wrong ({n} elems)")

    # Negative control: feed the kernel a CONTIGUOUS copy of the wrong layout
    # (channel-major, i.e. forgetting the transpose). It must NOT match.
    bad = dst.clone()
    wrong = scratch[..., :WINDOW].reshape(LAYERS, SCRATCH, DRAFT, WINDOW, CHANNELS)
    speculative_state_scatter_npu(bad, wrong, dst_idx, src_idx, step_idx)
    if torch.equal(bad, want):
        fails.append(f"b={batch} NEGATIVE CONTROL PASSED -- the check is blind")
    return fails


def case_masked_all(batch: int) -> list[str]:
    """All requests masked out: the pool must come back untouched."""
    from sgl_kernel_npu.mamba.speculative_state_scatter import (
        speculative_state_scatter_npu,
    )

    fails = []
    dst = torch.randn(LAYERS, SLOTS, WINDOW, CHANNELS, dtype=DT, device=DEV)
    scratch = torch.randn(LAYERS, SCRATCH, DRAFT, CHANNELS, WINDOW, dtype=DT, device=DEV)
    keep = dst.clone()
    neg = torch.full((batch,), -1, dtype=torch.int32, device=DEV)
    speculative_state_scatter_npu(dst, scratch.transpose(-1, -2), neg, neg, neg)
    if not torch.equal(dst, keep):
        fails.append(f"b={batch} masked-out requests still wrote to the pool")
    return fails


def case_verify_conv(batch: int, C: int = 512) -> list[str]:
    """Drive :513's verify conv with ragged slots.

    No reimplementation of the vendor convolution: instead two properties that
    an index or slot error breaks and a correct implementation cannot.

    permutation  reorder the requests and their slot indices together; the
                 outputs must come back in the same new order.  A kernel that
                 reads the wrong slot cannot survive this, and it needs no
                 reference implementation to state.
    read-only    `update_persistent_state=False` is the entire reason :513 may
                 hand the operator a transposed *copy* of the pool -- if it
                 wrote back, the write would land in the copy and vanish.  So
                 assert the tensor it was given is byte-identical afterwards.
    """
    from sgl_kernel_npu.mamba.causal_conv1d_verify import (
        causal_conv1d_linear_verify_npu,
    )

    fails = []
    W, T = WINDOW, DRAFT
    pool_wm = torch.randn(SLOTS, W, C, dtype=DT, device=DEV)   # window-major pool
    weight = torch.randn(C, W + 1, dtype=DT, device=DEV)
    bias = torch.randn(C, dtype=DT, device=DEV)
    x = torch.randn(batch, C, T, dtype=DT, device=DEV)
    slots = ragged(batch, SLOTS, 44).to(DEV)
    inter = ragged(batch, SCRATCH, 55).to(DEV)

    def run(x_, slots_, inter_):
        # exactly what ascend_kda_backend.py:511 does
        state = pool_wm.transpose(-1, -2).contiguous()
        before = state.clone()
        win = torch.zeros(SCRATCH, T, C, W, dtype=DT, device=DEV)
        out = causal_conv1d_linear_verify_npu(
            x_.contiguous(), state, weight, bias, slots_, win, inter_,
            activation="silu", update_persistent_state=False,
        )
        return out, torch.equal(state, before)

    out0, ro = run(x, slots, inter)
    if not ro:
        fails.append(f"b={batch} C={C} operator wrote back through conv_state "
                     f"despite update_persistent_state=False -- :513's copy is unsound")

    g = torch.Generator().manual_seed(77)
    perm = torch.randperm(batch, generator=g)
    out1, _ = run(x[perm], slots[perm.to(DEV)], inter[perm.to(DEV)])
    if not torch.equal(out1, out0[perm.to(DEV)]):
        n = (out1 != out0[perm.to(DEV)]).sum().item()
        fails.append(f"b={batch} C={C} not permutation-equivariant ({n} elems) -- slot indexing")

    # Negative control: permute the requests but NOT their slots. Unless the
    # kernel ignores slots entirely, this must differ.
    if batch >= 2:
        out2, _ = run(x[perm], slots, inter)
        if torch.equal(out2, out0[perm.to(DEV)]):
            fails.append(f"b={batch} C={C} NEGATIVE CONTROL PASSED -- slots are being ignored")
    return fails


def main() -> int:
    allf = []
    for batch in BATCHES:
        # 512 exercises the multi-tile path (block_c is capped at 256); 24576 is
        # this model's real channel count, 3 * 64 heads * 128.
        f = (case_scatter(batch) + case_masked_all(batch)
             + case_verify_conv(batch, 512) + case_verify_conv(batch, CHANNELS))
        print(f"  batch {batch:2d}: {'OK' if not f else 'FAIL'}")
        for x in f:
            print(f"      {x}")
        allf += f
    print(f"\nRESULT: {'PASS' if not allf else f'FAIL ({len(allf)})'}")
    return 1 if allf else 0


if __name__ == "__main__":
    sys.exit(main())
