#!/usr/bin/env python
"""P3.4 end-to-end: run IndexerKPool.forward_npu on real layer-3 weights.

Real classes wherever they exist -- IndexerKPool and NPUDSATokenToKVPool are the
production ones.  Only the attention backend and the ForwardBatch are faked, and
only in the fields forward_npu reads.

Inputs come from stage A2 (ref32k_v2.pt): the fp32 hidden state x and q_resid that
layer 3 actually sees, plus the fp32 reference pooled key / query / gate.

Run: npy kpool_e2e_npu.py --ref ref32k_v2.pt
"""
from __future__ import annotations

import argparse
import time
import types
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

MODEL = "/mnt/workspace/models/GLM-5.3-Flash-BF16"
PREFIX = "model.language_model.layers.{l}.self_attn.indexer."
DEV = "npu"


# ------------------------------------------------------------------ fakes
class FakeMeta:
    def __init__(self):
        self.block_tables = None


class FakeBackend:
    """Only the three attributes forward_npu reaches through."""

    def __init__(self, pool):
        self.token_to_kv_pool = pool
        self.req_to_token_pool = None
        self.forward_metadata = FakeMeta()


def make_fb(mode, batch_size, seq_lens, extend_lens, req_pool_indices, out_cache_loc):
    fb = types.SimpleNamespace()
    fb.forward_mode = mode
    fb.batch_size = batch_size
    fb.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    fb.seq_lens = fb.seq_lens_cpu.to(DEV, torch.int32)
    fb.extend_seq_lens_cpu = (
        None if extend_lens is None else torch.tensor(extend_lens, dtype=torch.int64)
    )
    fb.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int32, device=DEV)
    fb.out_cache_loc = torch.tensor(out_cache_loc, dtype=torch.int64, device=DEV)
    fb.positions = None
    return fb


# ------------------------------------------------------------------ build
def load_indexer(cfg, layer):
    from safetensors import safe_open
    import json

    from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool

    idx = IndexerKPool(
        hidden_size=cfg.hidden_size,
        index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        index_topk=cfg.index_topk,
        q_lora_rank=cfg.q_lora_rank,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_theta=None,
        layer_id=layer,
        scale_fmt=None,
        prefix="",
        quant_config=None,
        alt_stream=None,
        skip_rope=True,
        config=cfg,
    )
    d = Path(MODEL)
    wmap = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def get(name):
        shard = wmap[name]
        if shard not in handles:
            handles[shard] = safe_open(str(d / shard), framework="pt")
        return handles[shard].get_tensor(name)

    p = PREFIX.format(l=layer)
    # dtypes as the real loader leaves them: linear weights in the model dtype
    # (bf16), the fp32-declared params (ape, weights_proj, k_norm) in fp32.
    sd = {
        "wq_b.weight": get(p + "wq_b.weight").to(torch.bfloat16),
        "wk.weight": get(p + "wk.weight").to(torch.bfloat16),
        "k_norm.weight": get(p + "k_norm.weight").float(),
        "k_norm.bias": get(p + "k_norm.bias").float(),
        "weights_proj.weight": get(p + "weights_proj.weight").float(),
        "index_kpool_compress_ape": get(p + "index_kpool_compress_ape").float(),
        "index_kpool_compress_gate": get(p + "index_kpool_compress_gate").to(
            torch.bfloat16
        ),
    }
    missing, unexpected = idx.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    return idx.to(DEV).eval()


def make_pool(size, layer, max_running_requests=8):
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUDSATokenToKVPool

    return NPUDSATokenToKVPool(
        size=size,
        page_size=64,
        kv_lora_rank=512,
        dtype=torch.bfloat16,
        qk_rope_head_dim=0,
        layer_num=1,
        device=DEV,
        index_head_dim=128,
        enable_memory_saver=False,
        kv_cache_dim=512,
        start_layer=layer,
        end_layer=layer,
        index_buf_size=size,
        index_kpool=4,
        index_kpool_compress=True,
        tail_extra_slots=0,
        max_running_requests=max_running_requests,
    )


def page_map(n_pages, seed=0, shuffle=True):
    """Physical page id per logical page, as req_to_token would produce it.

    A permutation rather than identity so any page-arithmetic error in the
    write/read paths shows up as garbage instead of accidentally matching.
    Page 0 is left unused: out_cache_loc == 0 is the decode-invalid sentinel.
    """
    if not shuffle:
        return torch.arange(1, n_pages + 1, dtype=torch.int32)
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_pages, generator=g).to(torch.int32) + 1


# ------------------------------------------------------------------ reference
def topk_ref(q, pk, w, pool_lens, k, chunk=32):
    """fp32 CPU: logits[t,p] = sum_h w[t,h]*relu(q[t,h].pk[p]), masked to pool_lens."""
    T, P = q.shape[0], pk.shape[0]
    sel = torch.full((T, k), -1, dtype=torch.int64)
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


def ref_sets(sel, pool_lens, group_topk, seq_lens, kpool):
    """(pool sets, token sets) for the fp32 reference rows."""
    pools, toks = [], []
    for i in range(sel.shape[0]):
        kk = min(group_topk, int(pool_lens[i]))
        ps = set(sel[i, :kk].tolist())
        pools.append(ps)
        S = int(seq_lens[i])
        t = {p * kpool + j for p in ps for j in range(kpool)}
        t |= set(range(int(pool_lens[i]) * kpool, S))
        toks.append(t)
    return pools, toks


# ------------------------------------------------------------------ contract
def check_contract(out, seq_lens, pool_lens, group_topk, kpool, topk, tag):
    """-1 padding, valid values a prefix, dtype/width, expected cardinality."""
    errs = []
    if out.dtype != torch.int32:
        errs.append(f"dtype {out.dtype} != int32")
    if out.shape[1] != topk + kpool - 1:
        errs.append(f"width {out.shape[1]} != {topk + kpool - 1}")
    o = out.cpu()
    valid = o >= 0
    n_valid = valid.sum(1)
    ar = torch.arange(o.shape[1])[None, :]
    prefix = ar < n_valid[:, None]
    bad_prefix = (valid != prefix).any(1)
    if bad_prefix.any():
        r = int(bad_prefix.nonzero()[0])
        row = o[r]
        errs.append(
            f"valid values not a prefix in {int(bad_prefix.sum())} rows; "
            f"first row {r}: n_valid={int(n_valid[r])} "
            f"first_neg={int((row < 0).nonzero()[0]) if (row<0).any() else -1}"
        )
    exp = torch.minimum(pool_lens, torch.full_like(pool_lens, group_topk)) * kpool + (
        seq_lens % kpool
    )
    if not torch.equal(n_valid, exp.cpu()):
        d = (n_valid != exp.cpu()).nonzero().flatten()[:5].tolist()
        errs.append(
            f"cardinality mismatch on {int((n_valid != exp.cpu()).sum())} rows, "
            f"e.g. rows {d}: got {n_valid[d].tolist()} want {exp.cpu()[d].tolist()}"
        )
    for r in range(0, o.shape[0], max(1, o.shape[0] // 8)):
        row = o[r][: int(n_valid[r])]
        if row.numel() != row.unique().numel():
            errs.append(f"duplicate token ids in row {r}")
        if row.numel() and int(row.max()) >= int(seq_lens[r]):
            errs.append(f"row {r}: token {int(row.max())} >= seq_len {int(seq_lens[r])}")
    print(f"[contract {tag}] " + ("OK" if not errs else "; ".join(errs)))
    return not errs


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--stages", default="extend,decode,consistency")
    ap.add_argument("--decode-seqs", default="2048,8192")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.npu.set_device(0)
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    get_context().set_server_args(ServerArgs(model_path=MODEL, device="npu"))

    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        set_forward_context,
    )

    import json

    cfg = types.SimpleNamespace(
        **json.loads(Path(MODEL, "config.json").read_text())["text_config"]
    )
    KPOOL, TOPK = cfg.index_kpool, cfg.index_topk
    GTOPK = TOPK // KPOOL
    HD, NH = cfg.index_head_dim, cfg.index_n_heads

    ref = torch.load(args.ref, map_location="cpu")
    meta = ref["meta"]
    print(f"ref meta: {meta}")
    x_f32 = ref["x_f32"]
    q_resid_f32 = ref["q_resid_f32"]
    S_total = x_f32.shape[0]
    seq_lens_ref = ref["seq_lens"].tolist()
    row_pos = ref["row_pos"]
    R = row_pos.shape[1]

    idx = load_indexer(cfg, args.layer)
    print("indexer loaded")

    x_dev = x_f32.to(torch.bfloat16).to(DEV)
    q_dev = q_resid_f32.to(torch.bfloat16).to(DEV)

    stages = set(args.stages.split(","))
    results = {}

    # ---------------------------------------------------------- extend
    npages_tot = (S_total + 63) // 64
    pmap = page_map(npages_tot, seed=1)
    pool = make_pool(S_total, args.layer)
    backend = FakeBackend(pool)
    set_forward_context(ForwardContext(attn_backend=backend))

    want = {}  # seq_len -> (chunk_id, local row slice)
    for S in seq_lens_ref:
        c = (S - 1) // args.chunk
        lo = S - R - c * args.chunk
        want[S] = (c, lo, lo + R)

    ext_rows = {}
    if "extend" in stages:
        nchunks = S_total // args.chunk
        for c in range(nchunks):
            first, qlen = c * args.chunk, args.chunk
            seq_len = first + qlen
            ncol = (seq_len + 63) // 64
            backend.forward_metadata.block_tables = (
                pmap[:ncol].to(DEV).unsqueeze(0).contiguous()
            )
            fb = make_fb(
                ForwardMode.EXTEND,
                1,
                [seq_len],
                [qlen],
                [0],
                [1],
            )
            pos = torch.arange(first, seq_len, device=DEV, dtype=torch.int64)
            torch.npu.synchronize()
            t0 = time.time()
            out = idx.forward(
                x_dev[first:seq_len],
                q_dev[first:seq_len],
                pos,
                fb,
                args.layer,
            )
            torch.npu.synchronize()
            dt = (time.time() - t0) * 1e3
            print(
                f"[extend] chunk {c}: pos [{first},{seq_len}) out={tuple(out.shape)} "
                f"{out.dtype} {dt:.1f} ms"
            )
            for S, (cc, lo, hi) in want.items():
                if cc == c:
                    ext_rows[S] = out[lo:hi].cpu()
            del out
            torch.npu.empty_cache()
        results["extend_rows"] = ext_rows

    # ---------------------------------------------------------- scoring
    pk_f32 = ref["pooled_key"]
    q_rows = ref["q_rows"]
    w_rows = ref["w_rows"] * meta["softmax_scale"]

    def score(rows_by_S, tag):
        for li, S in enumerate(seq_lens_ref):
            if S not in rows_by_S:
                continue
            out = rows_by_S[S]
            pos = row_pos[li]
            slen = pos + 1
            plen = slen // KPOOL
            k = min(GTOPK, int(plen.max()))
            t0 = time.time()
            sel, lg = topk_ref(q_rows[li], pk_f32[: int(plen.max())], w_rows[li], plen, k)
            rp, rt = ref_sets(sel, plen, GTOPK, slen, KPOOL)
            ok = check_contract(out, slen, plen, GTOPK, KPOOL, TOPK, f"{tag} S={S}")
            povs, tovs, mass = [], [], []
            for i in range(out.shape[0]):
                row = out[i]
                cand_t = set(row[row >= 0].tolist())
                cand_p = {t // KPOOL for t in cand_t if t < int(plen[i]) * KPOOL}
                povs.append(len(cand_p & rp[i]) / max(len(rp[i]), 1))
                tovs.append(len(cand_t & rt[i]) / max(len(rt[i]), 1))
                l = lg[i]
                sel_idx = torch.tensor(sorted(cand_p), dtype=torch.int64)
                den = l[sel[i, : min(GTOPK, int(plen[i]))]].double().sum()
                num = l[sel_idx].double().sum() if sel_idx.numel() else torch.tensor(0.0)
                mass.append(float(num / den) if den != 0 else 1.0)
            pov = torch.tensor(povs)
            tov = torch.tensor(tovs)
            print(
                f"[{tag}] S={S:>6} pools={int(plen.max())} rows={out.shape[0]} "
                f"pool_overlap={pov.mean():.5f} (min {pov.min():.4f})  "
                f"token_overlap={tov.mean():.5f}  "
                f"score_mass={torch.tensor(mass).mean():.6f}  "
                f"contract={'ok' if ok else 'FAIL'}  ({time.time()-t0:.1f}s ref)"
            )

    if ext_rows:
        score(ext_rows, "extend")

    # ---------------------------------------------------------- decode
    for S_dec in [int(v) for v in args.decode_seqs.split(",")] if "decode" in stages else []:
        pre = S_dec - R
        assert pre % KPOOL == 0
        p2 = make_pool(S_dec + 64, args.layer)
        b2 = FakeBackend(p2)
        set_forward_context(ForwardContext(attn_backend=b2))
        pm2 = page_map((S_dec + 63) // 64, seed=1)
        ncol = (pre + 63) // 64
        b2.forward_metadata.block_tables = pm2[:ncol].to(DEV).unsqueeze(0).contiguous()
        fb = make_fb(ForwardMode.EXTEND, 1, [pre], [pre], [0], [1])
        idx.forward(
            x_dev[:pre], q_dev[:pre], torch.arange(pre, device=DEV), fb, args.layer
        )
        torch.npu.synchronize()
        print(f"[decode] primed with extend [0,{pre})")

        rows = []
        t0 = time.time()
        for p in range(pre, S_dec):
            seq_len = p + 1
            ncol = (seq_len + 63) // 64
            b2.forward_metadata.block_tables = (
                pm2[:ncol].to(DEV).unsqueeze(0).contiguous()
            )
            slot = int(pm2[p // 64]) * 64 + p % 64
            fb = make_fb(ForwardMode.DECODE, 1, [seq_len], None, [0], [slot])
            out = idx.forward(
                x_dev[p : p + 1],
                q_dev[p : p + 1],
                torch.tensor([p], device=DEV, dtype=torch.int64),
                fb,
                args.layer,
            )
            rows.append(out.cpu())
        torch.npu.synchronize()
        print(f"[decode] {S_dec - pre} steps in {time.time()-t0:.1f}s")
        score({S_dec: torch.cat(rows)}, "decode")
        del p2, rows
        torch.npu.empty_cache()

    # ---------------------------------------------------------- consistency
    if "consistency" in stages:
        N = 16
        TOT = 4096
        pmc = page_map((TOT + 63) // 64, seed=1)

        def run(split):
            p = make_pool(TOT, args.layer)
            b = FakeBackend(p)
            set_forward_context(ForwardContext(attn_backend=b))
            ncol = (split + 63) // 64
            b.forward_metadata.block_tables = (
                pmc[:ncol].to(DEV).unsqueeze(0).contiguous()
            )
            fb = make_fb(ForwardMode.EXTEND, 1, [split], [split], [0], [1])
            o = idx.forward(
                x_dev[:split],
                q_dev[:split],
                torch.arange(split, device=DEV),
                fb,
                args.layer,
            )
            outs = [o[-N:].cpu()] if split == TOT else []
            for t in range(split, TOT):
                seq_len = t + 1
                ncol = (seq_len + 63) // 64
                b.forward_metadata.block_tables = (
                    pmc[:ncol].to(DEV).unsqueeze(0).contiguous()
                )
                slot = int(pmc[t // 64]) * 64 + t % 64
                fb = make_fb(ForwardMode.DECODE, 1, [seq_len], None, [0], [slot])
                outs.append(
                    idx.forward(
                        x_dev[t : t + 1],
                        q_dev[t : t + 1],
                        torch.tensor([t], device=DEV, dtype=torch.int64),
                        fb,
                        args.layer,
                    ).cpu()
                )
            buf = p.get_index_k_with_scale_buffer(args.layer).clone().cpu()
            return buf, torch.cat(outs)

        buf_a, out_a = run(TOT)
        buf_b, out_b = run(TOT - N)
        npools = TOT // KPOOL
        locs = []
        for pid in range(npools):
            page = int(pmc[(pid // 64) * KPOOL])
            locs.append(page * 64 + pid % 64)
        locs = torch.tensor(locs)
        fa = buf_a.view(-1, HD)[locs]
        fb_ = buf_b.view(-1, HD)[locs]
        dec_lo = (TOT - N) // KPOOL   # pools path B wrote through the decode path
        for tag, lo, hi in (
            ("decode-written pools", dec_lo, npools),
            ("extend-written pools", 0, dec_lo),
        ):
            a, b = fa[lo:hi], fb_[lo:hi]
            nd = int((a != b).any(1).sum())
            md = (a.float() - b.float()).abs().max().item()
            print(
                f"[consistency] {tag} [{lo},{hi}): "
                f"{'IDENTICAL' if nd == 0 else f'{nd}/{hi-lo} differ, max|d|={md:.3e}'}"
            )
            if nd:
                bad = (a != b).any(1).nonzero().flatten()[:6].tolist()
                print(f"    e.g. pools {[lo+i for i in bad]}")
        nsame = 0
        for i in range(N):
            sa = set(out_a[i][out_a[i] >= 0].tolist())
            sb = set(out_b[i][out_b[i] >= 0].tolist())
            if sa == sb:
                nsame += 1
            else:
                print(
                    f"  row {i}: |a|={len(sa)} |b|={len(sb)} "
                    f"|a&b|={len(sa & sb)} overlap={len(sa & sb)/max(len(sa),1):.5f}"
                )
        print(
            f"[consistency] last-{N} selection rows extend vs decode: "
            f"{nsame}/{N} identical as sets; "
            f"elementwise {'equal' if torch.equal(out_a, out_b) else 'differs (order only)'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
