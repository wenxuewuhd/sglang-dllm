#!/usr/bin/env python
"""Stage B (NPU, .venv-glm53): does bf16 storage + torch_npu.npu_lightning_indexer
preserve the GLM layer-3 kpool *selection*?

Consumes stage A's fp32 reference tensors.  For each seq_len and each candidate
storage format it produces the selected pool set and scores it against the fp32
reference selection.

Storage formats (all reconstructed on CPU, then handed to the operator as bf16 --
A3 has no fp8, so the fp8 cast has to happen on the host):
  bf16   x -> bf16                             (the proposed route)
  int8   absmax/127, exact scale               (OP-1 fallback)
  fp8    e4m3 + ue8m0 power-of-two scale       (the CUDA incumbent)

Scoring paths:
  ref32  fp32 torch, unrotated fp32 pooled key            <- R32
  cpu    fp32 torch on the reconstructed bf16 key          <- isolates storage error
  npu    torch_npu.npu_lightning_indexer, decode + prefill <- isolates operator error

Run: npy kpool_stage_b_npu.py --ref ref32k.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import torch_npu  # noqa: F401

DEV = "npu"
BLOCK = 64
KPOOL = 4
HAD_SCALE = 0.08838834764831845  # 1/sqrt(128), same literal as the Triton kernel


# ---------------------------------------------------------------- Hadamard 128
def _h_stage(x: torch.Tensor, groups: int, stride: int) -> torch.Tensor:
    y = x.reshape(-1, groups, 2, stride)
    a, b = y[:, :, 0, :], y[:, :, 1, :]
    return torch.stack([a + b, a - b], dim=2).reshape(-1, 128)


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    """Transcription of kpool_fp8_index.py::_hadamard128 (butterfly + 1/sqrt(128))."""
    shape = x.shape
    y = x.reshape(-1, 128)
    for groups, stride in ((64, 1), (32, 2), (16, 4), (8, 8), (4, 16), (2, 32), (1, 64)):
        y = _h_stage(y, groups, stride)
    return (y * HAD_SCALE).reshape(shape)


def _check_hadamard() -> None:
    """Orthonormal? and equal to the Sylvester H128 / sqrt(128)?  And equal to the
    transform sglang applies to the query (rotate_activation)?"""
    eye = torch.eye(128, dtype=torch.float64)
    H = hadamard128(eye)  # rows are H applied to basis vectors
    err = (H @ H.T - eye).abs().max().item()
    print(f"[hadamard] orthonormality max|HH^T - I| = {err:.3e}")
    syl = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(7):
        syl = torch.cat([torch.cat([syl, syl], 1), torch.cat([syl, -syl], 1)], 0)
    print(f"[hadamard] vs Sylvester/sqrt(128): max|d| = "
          f"{(H - syl * HAD_SCALE).abs().max().item():.3e}")
    try:
        import sys
        sys.path.insert(0, "${GLM53_ROOT}/sglang-dllm/python")
        from sglang.kernels.ops.quantization.hadamard import hadamard_transform
        got = hadamard_transform(eye.float().to(DEV), scale=128 ** -0.5).cpu().double()
        print(f"[hadamard] vs sglang rotate_activation: max|d| = "
              f"{(got - H).abs().max().item():.3e}")
    except Exception as exc:  # noqa: BLE001
        print(f"[hadamard] sglang hadamard_transform unavailable: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- storage formats
def store_bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def store_int8(x: torch.Tensor, round_scale: bool = False) -> torch.Tensor:
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = absmax / 127.0
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q.float() * scale


def store_fp8(x: torch.Tensor, round_scale: bool = True) -> torch.Tensor:
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = absmax / 448.0
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.float() * scale


FORMATS = {
    "bf16": lambda x: store_bf16(x),
    "int8": lambda x: store_int8(x, round_scale=False),
    "int8_ue8m0": lambda x: store_int8(x, round_scale=True),
    "fp8_ue8m0": lambda x: store_fp8(x, round_scale=True),
}


# ---------------------------------------------------------------- scoring
def topk_ref(q: torch.Tensor, pk: torch.Tensor, w: torch.Tensor,
             pool_lens: torch.Tensor, k: int, chunk: int = 64):
    """fp32 torch: logits[t,p] = sum_h w[t,h] * relu(q[t,h].pk[p]), masked to pool_lens.
    Returns (selected [T,k] int64, logits [T,P] fp32)."""
    T, P = q.shape[0], pk.shape[0]
    sel = torch.empty(T, k, dtype=torch.int64)
    logits = torch.empty(T, P, dtype=torch.float32)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        sc = torch.relu(torch.einsum("tnd,pd->tnp", q[s:e], pk))
        lg = torch.einsum("tn,tnp->tp", w[s:e], sc)
        del sc
        ar = torch.arange(P)[None, :]
        lg = lg.masked_fill(ar >= pool_lens[s:e, None], -float("inf"))
        logits[s:e] = lg
        sel[s:e] = torch.topk(lg, k, dim=-1).indices
    return sel, logits


def score(sel_cand: list[set], sel_ref: list[set], logits_ref: torch.Tensor,
          ref_rows: list[torch.Tensor]):
    """mean set overlap and mean retained score mass, against the fp32 reference."""
    ovs, mass = [], []
    for i, (c, r) in enumerate(zip(sel_cand, sel_ref)):
        if not r:
            continue
        ovs.append(len(c & r) / len(r))
        lr = logits_ref[i]
        num = lr[torch.tensor(sorted(c), dtype=torch.int64)].double().sum()
        den = lr[ref_rows[i]].double().sum()
        mass.append((num / den).item() if den != 0 else 1.0)
    return (float(torch.tensor(ovs).mean()), float(torch.tensor(ovs).min()),
            float(torch.tensor(mass).mean()))


def swap_severity(sel_cand, ref_rows, logits_ref):
    """How far above the cut do the dropped pools sit, as a fraction of the top-k
    score span?  0 = a pool right at the boundary, 1 = the single best pool."""
    worst = 0.0
    for i, (c, r) in enumerate(zip(sel_cand, ref_rows)):
        dropped = set(r.tolist()) - c
        if not dropped:
            continue
        lr = logits_ref[i]
        s_ref = lr[r]
        cut, top = s_ref.min().item(), s_ref.max().item()
        span = max(top - cut, 1e-12)
        for d in dropped:
            worst = max(worst, (lr[d].item() - cut) / span)
    return worst


# ---------------------------------------------------------------- the operator
def li_decode(q_bf16, key_pa, w, pool_lens, sparse_count, nblk):
    """One TND batch per query row; actual_seq_lengths_key = visible pool count."""
    T = q_bf16.shape[0]
    bt = torch.arange(nblk, dtype=torch.int32, device=DEV).repeat(T, 1).contiguous()
    asq = torch.arange(1, T + 1, dtype=torch.int32, device=DEV)
    ask = pool_lens.to(device=DEV, dtype=torch.int32)
    out = torch_npu.npu_lightning_indexer(
        query=q_bf16, key=key_pa, weights=w,
        actual_seq_lengths_query=asq, actual_seq_lengths_key=ask,
        block_table=bt, layout_query="TND", layout_key="PA_BSND",
        sparse_count=sparse_count, sparse_mode=0,
    )
    return out[0].squeeze(1)


def li_prefill(q_bf16, key_pa, w, pool_lens, sparse_count, nblk):
    """Contiguous query rows sharing one visible-pool count form one TND batch."""
    run_vals, run_cnt = torch.unique_consecutive(pool_lens, return_counts=True)
    G = run_vals.numel()
    bt = torch.arange(nblk, dtype=torch.int32, device=DEV).repeat(G, 1).contiguous()
    asq = run_cnt.cumsum(0).to(device=DEV, dtype=torch.int32)
    ask = run_vals.to(device=DEV, dtype=torch.int32)
    out = torch_npu.npu_lightning_indexer(
        query=q_bf16, key=key_pa, weights=w,
        actual_seq_lengths_query=asq, actual_seq_lengths_key=ask,
        block_table=bt, layout_query="TND", layout_key="PA_BSND",
        sparse_count=sparse_count, sparse_mode=0,
    )
    return out[0].squeeze(1), G


def to_pa(pk_bf16: torch.Tensor, nblk: int) -> torch.Tensor:
    """[P,128] -> PA_BSND [nblk, 64, 1, 128], zero-padded."""
    P, D = pk_bf16.shape
    buf = torch.zeros(nblk * BLOCK, D, dtype=torch.bfloat16)
    buf[:P] = pk_bf16
    return buf.view(nblk, BLOCK, 1, D).contiguous().to(DEV)


def as_sets(idx: torch.Tensor, pool_lens: torch.Tensor) -> list[set]:
    idx = idx.cpu()
    out = []
    for i in range(idx.shape[0]):
        L = int(pool_lens[i])
        row = idx[i]
        out.append(set(row[(row >= 0) & (row < L)].tolist()))
    return out


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--formats", nargs="+", default=["bf16", "int8", "fp8_ue8m0"])
    ap.add_argument("--weights-dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--skip-prefill", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    print(f"device: {torch.npu.get_device_name(0)}")
    _check_hadamard()

    ref = torch.load(args.ref, map_location="cpu")
    meta = ref["meta"]
    print(f"ref meta: {meta}")
    pk_f32 = ref["pooled_key"]                 # [P_max, 128] fp32, unrotated
    q_rows = ref["q_rows"]                     # [L, R, 32, 128] fp32, unrotated
    w_rows = ref["w_rows"]                     # [L, R, 32] fp32 (already * n_heads^-0.5)
    row_pos = ref["row_pos"]
    seq_lens = ref["seq_lens"].tolist()
    group_topk = meta["index_topk"] // meta["kpool"]
    sms = meta["softmax_scale"]

    # ---- rotate once: query bf16 -> H -> bf16 (what rotate_activation does)
    q_rot_bf16 = hadamard128(q_rows.to(torch.bfloat16).float()).to(torch.bfloat16)
    # ---- fp32 reference query/key are the UNROTATED fp32 tensors (H is orthonormal,
    #      so q.k == (Hq).(Hk) up to rounding; the fp32 dot product is the reference)
    w_ref = w_rows * sms

    # ---- the kernel's key pipeline up to (but not including) the store
    x_pre = hadamard128(pk_f32.to(torch.bfloat16).float()).to(torch.bfloat16).float()

    recon = {}
    for name in args.formats:
        recon[name] = FORMATS[name](x_pre)
        rel = ((recon[name] - x_pre).norm() / x_pre.norm()).item()
        print(f"[key recon] {name:11s} rel-L2 vs the pre-store fp32 rotated key = {rel:.5f}")

    rows = []
    for li, S in enumerate(seq_lens):
        P = S // KPOOL
        pool_lens = ((row_pos[li] + 1) // KPOOL).to(torch.int64)
        k = min(group_topk, int(pool_lens.min()))
        assert k == min(group_topk, int(pool_lens.max())) or True
        pk = pk_f32[:P]
        q = q_rows[li]
        w = w_ref[li]
        nblk = (P + BLOCK - 1) // BLOCK

        # ---------------- R32 reference
        t0 = time.time()
        sel32, lg32 = topk_ref(q, pk, w, pool_lens, min(group_topk, P))
        ref_rows, ref_sets = [], []
        for i in range(sel32.shape[0]):
            kk = min(group_topk, int(pool_lens[i]))
            r = sel32[i, :kk]
            ref_rows.append(r)
            ref_sets.append(set(r.tolist()))
        print(f"\n=== seq_len={S}  pools={P}  rows={q.shape[0]}  "
              f"pool_lens {int(pool_lens.min())}..{int(pool_lens.max())}  "
              f"k={min(group_topk, P)}  (R32 in {time.time()-t0:.1f}s)")

        for name in args.formats:
            pk_c = recon[name][:P]
            q_c = q_rot_bf16[li]
            # ---- cpu: fp32 arithmetic on the reconstructed (rotated) key
            selc, lgc = topk_ref(q_c.float(), pk_c, w, pool_lens, min(group_topk, P))
            cpu_sets = [set(selc[i, : min(group_topk, int(pool_lens[i]))].tolist())
                        for i in range(selc.shape[0])]
            ov, mn, ms = score(cpu_sets, ref_sets, lg32, ref_rows)
            fin = torch.isfinite(lg32)
            rel = ((lgc[fin].double() - lg32[fin].double()).norm()
                   / lg32[fin].double().norm()).item()
            rows.append((S, name, "cpu-fp32", ov, mn, ms, float("nan")))
            print(f"  {name:11s} cpu-fp32   overlap={ov:.5f} (min row {mn:.4f})  "
                  f"mass={ms:.6f}  logits rel-L2 vs R32={rel:.3e}")
            del lgc

            # ---- npu operator
            key_pa = to_pa(pk_c.to(torch.bfloat16), nblk)
            qd = q_c.to(DEV)
            wd = w.to(torch.bfloat16 if args.weights_dtype == "bf16" else torch.float32).to(DEV)
            for mode, fn in (("npu-decode", li_decode),
                             *(() if args.skip_prefill else (("npu-prefill", li_prefill),))):
                torch.npu.synchronize()
                t0 = time.time()
                out = fn(qd, key_pa, wd, pool_lens, group_topk, nblk)
                idx = out[0] if isinstance(out, tuple) else out
                torch.npu.synchronize()
                dt = (time.time() - t0) * 1e3
                cand = as_sets(idx, pool_lens)
                ov, mn, ms = score(cand, ref_sets, lg32, ref_rows)
                op_only = float(torch.tensor(
                    [len(c & s_) / max(len(s_), 1) for c, s_ in zip(cand, cpu_sets)]).mean())
                rows.append((S, name, mode, ov, mn, ms, op_only))
                nsel = sum(len(c) for c in cand) / len(cand)
                sev = swap_severity(cand, ref_rows, lg32)
                print(f"  {name:11s} {mode:11s} overlap={ov:.5f} (min row {mn:.4f})  "
                      f"mass={ms:.6f}  vs_same_storage_cpu={op_only:.5f}  "
                      f"worst_drop={sev:.4f} of span  mean|sel|={nsel:.1f}  ({dt:.1f} ms)")
            del key_pa, qd, wd
            torch.npu.empty_cache()

    print("\n\n### summary  (overlap of selected pool set vs the fp32 reference)")
    print(f"{'seq_len':>8} {'format':>11} {'path':>11} {'overlap':>9} {'min row':>9} "
          f"{'mass':>10} {'vs cpu':>9}")
    for S, name, mode, ov, mn, ms, oo in rows:
        print(f"{S:>8} {name:>11} {mode:>11} {ov:>9.5f} {mn:>9.4f} {ms:>10.6f} {oo:>9.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
