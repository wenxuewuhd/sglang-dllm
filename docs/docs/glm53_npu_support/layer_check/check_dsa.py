#!/usr/bin/env python
"""One GLM-5.3-Flash DSA layer (layer 3) at the REAL deployment shape.

Shape comes from $ROOT/run/launch_glm_bf16.sh: --tp-size 16, --page-size 64,
--context-length 32768, --max-running-requests 16, bf16, no cuda graph, no
overlap schedule, no radix cache.  Per die that means:
  MLA          4 local heads (64/16), qk_nope 256, v 256, kv_lora 512, rope 0
  indexer      all 32 heads on every rank (ReplicatedLinear, never TP-divided)
  o_proj       row-parallel, reduce_results=False -> this rank's PARTIAL sum,
               which is what the reference computes too
Decode runs a real batch of 16 ragged requests out to 32768 context.

Numbers reported per case:
  (a) NPU layer output vs the fp32 reference with the *fp32* selection
  (b) NPU layer output vs the fp32 reference with the *NPU's own* selection
  sel  fp32 attention on the NPU selection vs fp32 attention on the fp32
       selection -- the cost of the selection difference alone, no arithmetic
       noise, which is the whole gap between (a) and (b)
  floor  ACCEPTANCE two-reference floor: same pure-torch reference run on
         bf16-rounded inputs vs on fp32 inputs
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import fixture as F
import ref as RR

SP = "/tmp/claude-1000/-mnt-workspace-y00359136-work-glm53-dev-sglang-dllm/d927eb2c-8461-4d46-9b1e-b27511958e37/scratchpad"
SLACK = 2.0
KPOOL, GTOPK = 4, 512


def rows_from_topk(t):
    return [t[i][t[i] >= 0].long() for i in range(t.shape[0])]


class Fp32Select:
    """fp32 reference selection for any query row -- the pipeline stage A
    validated against Glm5NextTextIndexer.forward (overlap 1.000000 @ 4096)."""

    def __init__(self, d, sh, layer, cfg):
        p = f"model.language_model.layers.{layer}.self_attn.indexer."
        self.wq_b = sh.get(p + "wq_b.weight").float()
        self.q_resid = d["q_resid_f32"]
        self.w_all = d["w_all_f32"] * (cfg.index_head_dim ** -0.5)
        self.pk = d["pooled_key"]
        self.nh, self.hd = cfg.index_n_heads, cfg.index_head_dim

    def rows(self, positions):
        out = []
        for t in positions:
            t = int(t)
            plen = (t + 1) // KPOOL
            q = (self.q_resid[t] @ self.wq_b.T).view(self.nh, self.hd)
            sc = torch.relu(q @ self.pk[:plen].T)
            lg = (self.w_all[t][:, None] * sc).sum(0)
            k = min(GTOPK, plen)
            sel = torch.topk(lg, k).indices
            tok = (sel[:, None] * KPOOL + torch.arange(KPOOL)).flatten()
            tail = torch.arange(plen * KPOOL, t + 1)
            out.append(torch.cat([tok, tail]).sort().values.long())
        return out


def score(tag, lref, npu_out, npu_rows, ref_rows, qno32, qno16, kv32, kv16):
    r32_a = lref.attend(qno32, kv32, ref_rows, torch.float32)
    r16_a = lref.attend(qno16, kv16, ref_rows, torch.bfloat16)
    r32_b = lref.attend(qno32, kv32, npu_rows, torch.float32)
    r16_b = lref.attend(qno16, kv16, npu_rows, torch.bfloat16)
    ea, fa = RR.rel(npu_out, r32_a), RR.rel(r16_a, r32_a)
    eb, fb = RR.rel(npu_out, r32_b), RR.rel(r16_b, r32_b)
    sel = RR.rel(r32_b, r32_a)
    ov = sum(len(set(a.tolist()) & set(b.tolist())) / max(len(a), 1)
             for a, b in zip(ref_rows, npu_rows)) / len(ref_rows)
    print(f"  {tag}")
    print(f"    selection overlap vs fp32 ref = {ov:.5f}   "
          f"selection-only cost = {sel:.3e}")
    print(f"    (a) end-to-end            rel={ea:.3e}  floor={fa:.3e}  "
          f"ratio={ea/max(fa,1e-12):5.2f}  cos={RR.cos(npu_out, r32_a):.7f}")
    print(f"    (b) controlled (NPU idx)  rel={eb:.3e}  floor={fb:.3e}  "
          f"ratio={eb/max(fb,1e-12):5.2f}  cos={RR.cos(npu_out, r32_b):.7f}  "
          f"{'PASS' if eb <= fb * SLACK else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--tp", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--ext-seqs", type=int, nargs="+", default=[4096, 32768])
    ap.add_argument("--ext-rows", type=int, default=256)
    ap.add_argument("--stages", default="extend,batch_extend,decode,contract")
    args = ap.parse_args()
    stages = set(args.stages.split(","))

    sa, mc = F.boot()
    print(f"server args: page_size={sa.page_size} "
          f"chunked_prefill_size={sa.chunked_prefill_size} "
          f"max_running_requests={sa.max_running_requests}")
    sh = F.Shards()
    cfg = mc.hf_text_config
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        set_forward_context,
    )

    with F.tp_override(args.tp, 0):
        lref = RR.LayerRef(sh, args.layer, cfg, tp_size=args.tp, tp_rank=0)
        m = F.make_attn(mc, args.layer, sh, tp_size=args.tp, tp_rank=0)
        print(f"TP{args.tp} rank0: local_heads={m.num_local_heads} "
              f"indexer_heads={m.indexer.n_heads} q_b={tuple(m.q_b_proj.weight.shape)} "
              f"kv_b={tuple(m.kv_b_proj.weight.shape)} o={tuple(m.o_proj.weight.shape)} "
              f"w_kc={tuple(m.w_kc.shape)} w_vc={tuple(m.w_vc.shape)} "
              f"scaling={m.scaling}")

        d = torch.load(f"{SP}/ref32k_v2.pt", map_location="cpu")
        da = torch.load(f"{SP}/dsa/ref_attn.pt", map_location="cpu")
        x_all = d["x_f32"]
        kv32_all = da["kv_a_f32_32768"]
        x_dev = x_all.to(torch.bfloat16).to(F.DEV)
        fsel = Fp32Select(d, sh, args.layer, cfg)

        def ctx(size, nreq=4, max_ctx=None):
            kv, req = F.make_pools(mc, args.layer, size,
                                   max_running_requests=max(nreq, 16),
                                   max_ctx=max_ctx or size, nreq=nreq + 1)
            be = F.make_backend(mc, kv, req, sa)
            set_forward_context(ForwardContext(attn_backend=be))
            return kv, req, be

        # ------------------------------------------------ single-request extend
        if "extend" in stages:
            for S in args.ext_seqs:
                size = ((S + 63) // 64) * 64
                kv, req, be = ctx(size, 1, max_ctx=size)
                pm = F.page_map(size // 64, seed=1)
                slots = F.fill_req_to_token(req, 0, pm, S)
                first, out, tk = 0, None, None
                t0 = time.time()
                while first < S:
                    n = min(args.chunk, S - first)
                    sl = first + n
                    fb = F.make_fb(ForwardMode.EXTEND, [sl], [n], [0],
                                   slots[first:sl].tolist())
                    be.init_forward_metadata(fb)
                    o, k, _, _ = F.run_layer(
                        m, torch.arange(first, sl, device=F.DEV, dtype=torch.int64),
                        x_dev[first:sl], fb)
                    torch.npu.synchronize()
                    if sl == S:
                        out, tk = o[-args.ext_rows:].float().cpu(), k[-args.ext_rows:].cpu()
                    del o, k
                    first = sl
                dt = time.time() - t0
                buf = kv.get_key_buffer(args.layer).view(-1, 512)[slots.to(F.DEV)].float().cpu()
                print(f"[extend TP{args.tp} S={S}] chunk={args.chunk} {dt:.1f}s  "
                      f"kv-write rel={RR.rel(buf, kv32_all[:S]):.3e} "
                      f"(bf16 store floor {RR.rel(kv32_all[:S].to(torch.bfloat16).float(), kv32_all[:S]):.3e})")
                del buf
                R = args.ext_rows
                pos = list(range(S - R, S))
                qno32 = lref.q_absorbed(x_all[S - R:S], torch.float32)
                qno16 = lref.q_absorbed(x_all[S - R:S], torch.bfloat16)
                kv16 = lref.kv_latent(x_all[:S], torch.bfloat16)
                score(f"extend S={S}, last {R} rows", lref, out,
                      rows_from_topk(tk), fsel.rows(pos), qno32, qno16,
                      kv32_all[:S], kv16)
                del kv, req, be, qno32, qno16, kv16
                torch.npu.empty_cache()

        # ------------------------------------------------ ragged batch extend
        if "batch_extend" in stages:
            lens = [8192, 5000, 3333]
            pages = [(L + 63) // 64 for L in lens]
            size = (sum(pages) + 1) * 64
            kv, req, be = ctx(size, len(lens), max_ctx=32768 + 64)
            pm = F.page_map(sum(pages) + 1, seed=3)
            off, slots = 1, []
            for i, L in enumerate(lens):
                sub = pm[off:off + pages[i]]
                slots.append(F.fill_req_to_token(req, i, sub, L))
                off += pages[i]
            fb = F.make_fb(ForwardMode.EXTEND, lens, lens, list(range(len(lens))),
                           torch.cat(slots).tolist())
            be.init_forward_metadata(fb)
            xin = torch.cat([x_dev[:L] for L in lens])
            posin = torch.cat([torch.arange(L, device=F.DEV, dtype=torch.int64)
                               for L in lens])
            o, tk, _, _ = F.run_layer(m, posin, xin, fb)
            torch.npu.synchronize()
            o = o.float().cpu(); tk = tk.cpu()
            print(f"[batch extend] {len(lens)} ragged reqs {lens} -> {tuple(o.shape)}")
            base = 0
            for i, L in enumerate(lens):
                sel = list(range(base + L - 32, base + L))
                pos = list(range(L - 32, L))
                qno32 = lref.q_absorbed(x_all[L - 32:L], torch.float32)
                qno16 = lref.q_absorbed(x_all[L - 32:L], torch.bfloat16)
                kv16 = lref.kv_latent(x_all[:L], torch.bfloat16)
                score(f"req {i} (len {L}), last 32 rows", lref, o[sel],
                      rows_from_topk(tk[sel]), fsel.rows(pos), qno32, qno16,
                      kv32_all[:L], kv16)
                base += L
                del qno32, qno16, kv16
            del kv, req, be, o, tk
            torch.npu.empty_cache()

        # ------------------------------------------------ batch-16 decode
        if "decode" in stages:
            lens = [32768, 32767, 30011, 28672, 24576, 20481, 16384, 12289,
                    9999, 8192, 6145, 4096, 3077, 2048, 1025, 512]
            pages = [(L + 63) // 64 for L in lens]
            size = (sum(pages) + 1) * 64
            print(f"[decode batch {len(lens)}] pool {size} slots "
                  f"({size*512*2/2**30:.2f} GiB kv), lens {lens}")
            kv, req, be = ctx(size, len(lens), max_ctx=32768 + 64)
            pm = F.page_map(sum(pages) + 1, seed=5)
            off, slots = 1, []
            for i, L in enumerate(lens):
                sub = pm[off:off + pages[i]]
                slots.append(F.fill_req_to_token(req, i, sub, L))
                off += pages[i]
            t0 = time.time()
            for i, L in enumerate(lens):           # chunked prefill, one req at a time
                first = 0
                while first < L - 1:
                    n = min(args.chunk, L - 1 - first)
                    sl = first + n
                    fb = F.make_fb(ForwardMode.EXTEND, [sl], [n], [i],
                                   slots[i][first:sl].tolist())
                    be.init_forward_metadata(fb)
                    F.run_layer(m, torch.arange(first, sl, device=F.DEV,
                                                dtype=torch.int64),
                                x_dev[first:sl], fb)
                    first = sl
            torch.npu.synchronize()
            print(f"    prefill of {sum(lens)-len(lens)} tokens in {time.time()-t0:.1f}s")
            fb = F.make_fb(ForwardMode.DECODE, lens, None, list(range(len(lens))),
                           [int(slots[i][lens[i] - 1]) for i in range(len(lens))])
            be.init_forward_metadata(fb)
            pos = torch.tensor([L - 1 for L in lens], device=F.DEV, dtype=torch.int64)
            xin = torch.cat([x_dev[L - 1:L] for L in lens])
            t0 = time.time()
            o, tk, _, _ = F.run_layer(m, pos, xin, fb)
            torch.npu.synchronize()
            print(f"    decode step bs={len(lens)} in {(time.time()-t0)*1e3:.1f} ms "
                  f"-> {tuple(o.shape)}")
            o = o.float().cpu(); tk = tk.cpu()
            npu_rows = rows_from_topk(tk)
            ref_rows = fsel.rows([L - 1 for L in lens])
            qno32 = torch.cat([lref.q_absorbed(x_all[L - 1:L], torch.float32)
                               for L in lens])
            qno16 = torch.cat([lref.q_absorbed(x_all[L - 1:L], torch.bfloat16)
                               for L in lens])
            # per-request, because each row attends over its own prefix
            for i, L in enumerate(lens):
                kv16 = lref.kv_latent(x_all[:L], torch.bfloat16)
                score(f"decode req {i} len={L}", lref, o[i:i+1], npu_rows[i:i+1],
                      ref_rows[i:i+1], qno32[i:i+1], qno16[i:i+1],
                      kv32_all[:L], kv16)
                del kv16
            del kv, req, be
            torch.npu.empty_cache()

        # ------------------------------------------------ prefix contract
        if "contract" in stages:
            from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
                forward_dsa_core_npu, forward_dsa_prepare_npu)
            S = 32768
            size = ((S + 63) // 64) * 64
            kv, req, be = ctx(size, 1, max_ctx=size)
            pm = F.page_map(size // 64, seed=1)
            slots = F.fill_req_to_token(req, 0, pm, S)
            first = 0
            while first < S - 1:
                n = min(args.chunk, S - 1 - first)
                sl = first + n
                fb = F.make_fb(ForwardMode.EXTEND, [sl], [n], [0],
                               slots[first:sl].tolist())
                be.init_forward_metadata(fb)
                F.run_layer(m, torch.arange(first, sl, device=F.DEV,
                                            dtype=torch.int64), x_dev[first:sl], fb)
                first = sl
            torch.npu.synchronize()
            t = S - 1
            fb = F.make_fb(ForwardMode.DECODE, [S], None, [0], [int(slots[t])])
            be.init_forward_metadata(fb)
            pack = forward_dsa_prepare_npu(
                m, torch.tensor([t], device=F.DEV, dtype=torch.int64),
                x_dev[t:t+1], fb, None, None)
            q_pe, k_pe, qno, kn, tk, fb2, za, pos2 = pack
            good, _ = forward_dsa_core_npu(m, q_pe, k_pe, qno, kn, tk, fb2, za, pos2)
            good = good.float().cpu()
            row = tk[0].cpu()
            nval = int((row >= 0).sum())
            print(f"[contract] real shape, decode, seq={S}, {nval} valid indices")
            for name, mod in (
                ("-1 rotated to slot 0 (not a prefix)",
                 lambda r: torch.cat([torch.tensor([-1], dtype=r.dtype), r[:nval],
                                      r[nval + 1:]])),
                ("-1 injected at slot 1024 (even)",
                 lambda r: torch.cat([r[:1024], torch.tensor([-1], dtype=r.dtype),
                                      r[1024:-1]])),
                ("-1 injected at slot 1025 (odd)",
                 lambda r: torch.cat([r[:1025], torch.tensor([-1], dtype=r.dtype),
                                      r[1025:-1]])),
            ):
                bad = tk.clone()
                bad[0] = mod(row).to(bad.device)
                b, _ = forward_dsa_core_npu(m, q_pe, k_pe, qno, kn, bad, fb2, za, pos2)
                b = b.float().cpu()
                print(f"    {name:<38} max|out|={b.abs().max():.4e}  "
                      f"rel vs unperturbed={RR.rel(b, good):.3e}  "
                      f"{'SILENT ZERO' if b.abs().max() < 1e-20 else ('identical' if torch.equal(b, good) else 'SILENTLY WRONG')}")
            del kv, req, be
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
