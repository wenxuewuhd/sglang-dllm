"""DSA decode under NPU-graph capture, at the shipped shape, scored against the
fp32 reference -- not just against itself.

Shape is check_dsa.py's decode stage verbatim (TP16 rank 0, page 64, ctx 32768,
16 ragged requests, max-running-requests 16), so the eager numbers this compares
against are the ones PLAN P3.5 already accepted.

Three separate questions:
  cap   does torch.npu.graph capture the decode path at all?
  bake  after capture, does a replay still track its *device* inputs?  Run with
        new hidden states, and again with new seq_lens, and demand the replay
        agree with eager on the new inputs.  A host value frozen into the graph
        shows up here and nowhere else.
  gold  is the tensor coming out of the replayed graph still the right tensor?
        Scored with check_dsa's own two-reference method.
"""
import sys, argparse, time, torch
import os as _os
from pathlib import Path as _Path
LC = str(_Path(__file__).resolve().parent.parent)          # .../layer_check
G = str(_Path(__file__).resolve().parent)                  # .../layer_check/graph_capture
# The DSA fp32 references are multi-hundred-MB dumps that do not go in the repo.
# Point SCRATCH at wherever dump_reference.py / reference_dsa.py wrote them.
SP = _os.environ.get("SCRATCH", "/tmp/glm53_scratch")

sys.path.insert(0, LC); sys.path.insert(0, G)
import gcap
ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--tp", type=int, default=16)
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--lens", default="")
ap.add_argument("--skip-gold", action="store_true")
a = ap.parse_args()
torch.set_grad_enabled(False)
import torch_npu, custom_ops  # noqa
import tp_fixture as F
import reference_mla_math as RR
import types

# Verbatim from check_dsa.py: the fp32 reference selection and the topk->rows
# decode.  Copied rather than imported because check_dsa.py's own imports are
# named for a different file layout.
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



lens = ([int(x) for x in a.lens.split(",")] if a.lens else
        [32767, 32766, 30011, 28672, 24576, 20481, 16384, 12289,
         9999, 8192, 6145, 4096, 3077, 2048, 1025, 512])
BS = len(lens)
sa, mc = F.boot()
sh = F.Shards(); cfg = mc.hf_text_config
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, set_forward_context
from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
    forward_dsa_core_npu, forward_dsa_prepare_npu)

with F.tp_override(a.tp, 0):
    lref = RR.LayerRef(sh, a.layer, cfg, tp_size=a.tp, tp_rank=0)
    m = F.make_attn(mc, a.layer, sh, tp_size=a.tp, tp_rank=0)
    print(f"TP{a.tp} rank0 layer{a.layer}: local_heads={m.num_local_heads} "
          f"indexer_heads={m.indexer.n_heads}  bs={BS}  lens={lens}", flush=True)
    d = torch.load(f"{SP}/ref32k_v2.pt", map_location="cpu")
    da = torch.load(f"{SP}/dsa/ref_attn.pt", map_location="cpu")
    x_all = d["x_f32"]; kv32_all = da["kv_a_f32_32768"]
    x_dev = x_all.to(torch.bfloat16).to(F.DEV)
    fsel = Fp32Select(d, sh, a.layer, cfg)

    pages = [(L + 64) // 64 for L in lens]
    size = (sum(pages) + 1) * 64
    kv, req = F.make_pools(mc, a.layer, size, max_running_requests=max(BS, 16),
                           max_ctx=32768 + 64, nreq=BS + 1)
    be = F.make_backend(mc, kv, req, sa)
    set_forward_context(ForwardContext(attn_backend=be))
    pm = F.page_map(sum(pages) + 1, seed=5)
    off, slots = 1, []
    for i, L in enumerate(lens):
        slots.append(F.fill_req_to_token(req, i, pm[off:off + pages[i]], L + 1)); off += pages[i]

    t0 = time.time()
    for i, L in enumerate(lens):
        first = 0
        while first < L - 1:
            n = min(a.chunk, L - 1 - first); sl = first + n
            fbp = F.make_fb(ForwardMode.EXTEND, [sl], [n], [i], slots[i][first:sl].tolist())
            be.init_forward_metadata(fbp)
            F.run_layer(m, torch.arange(first, sl, device=F.DEV, dtype=torch.int64),
                        x_dev[first:sl], fbp)
            first = sl
    torch.npu.synchronize()
    print(f"  prefill {sum(lens)-BS} tokens in {time.time()-t0:.0f}s", flush=True)

    # ---------------- static decode buffers, as the runner owns them ----------
    st_seq_lens = torch.tensor(lens, device=F.DEV, dtype=torch.int32)
    st_seq_lens_cpu = torch.tensor(lens, dtype=torch.int32)
    st_out_cache = torch.tensor([int(slots[i][lens[i] - 1]) for i in range(BS)],
                                device=F.DEV, dtype=torch.int64)
    st_pos = torch.tensor([L - 1 for L in lens], device=F.DEV, dtype=torch.int64)
    st_x = torch.cat([x_dev[L - 1:L] for L in lens]).contiguous()
    st_req_idx = torch.arange(BS, device=F.DEV, dtype=torch.int32)

    fb = types.SimpleNamespace()
    fb.forward_mode = ForwardMode.DECODE
    fb.batch_size = BS
    fb.seq_lens = st_seq_lens; fb.seq_lens_cpu = st_seq_lens_cpu
    fb.extend_seq_lens = fb.extend_seq_lens_cpu = None
    fb.extend_prefix_lens = fb.extend_prefix_lens_cpu = None
    fb.req_pool_indices = st_req_idx
    fb.out_cache_loc = st_out_cache
    fb.positions = st_pos
    fb.spec_info = None; fb.spec_algorithm = None
    fb.attn_cp_metadata = None; fb.token_to_kv_pool = None

    be.init_cuda_graph_state(max_bs=BS, max_num_tokens=BS)
    def prep(in_capture): be.init_forward_metadata_out_graph(fb, in_capture=in_capture)
    def step():
        be.init_forward_metadata_in_graph(fb)
        q_pe, k_pe, qno, kn, tk, fb2, za, pos2 = forward_dsa_prepare_npu(
            m, st_pos, st_x, fb, None, None)
        out, _ = forward_dsa_core_npu(m, q_pe, k_pe, qno, kn, tk, fb2, za, pos2)
        return {"out": out, "topk": tk.float()}

    prep(True)
    refA = gcap.snap(step())
    cap = gcap.Cap("dsa")
    try:
        gout = cap.capture(step)
    except Exception as e:
        print(f"  CAPTURE FAILED: {type(e).__name__}: {str(e)[:1200]}")
        import traceback; traceback.print_exc(); raise SystemExit(1)
    print("  capture OK", flush=True)

    def scenario(tag, mutate):
        mutate(); prep(False)
        ref = gcap.snap(step())
        prep(False); cap.replay()
        return gcap.compare(f"replay({tag})", gcap.snap(gout), ref), ref

    badA, rA = scenario("A same input", lambda: None)

    # ---------------- golden: score the REPLAYED output ----------------------
    rc_gold = 0
    if not a.skip_gold:
        prep(False); cap.replay()
        o = gout["out"].float().cpu(); tk = gout["topk"].to(torch.int32).cpu()
        npu_rows = rows_from_topk(tk)
        ref_rows = fsel.rows([L - 1 for L in lens])
        qno32 = torch.cat([lref.q_absorbed(x_all[L - 1:L], torch.float32) for L in lens])
        qno16 = torch.cat([lref.q_absorbed(x_all[L - 1:L], torch.bfloat16) for L in lens])
        print("\n=== DSA graph-replay vs fp32 reference (two-reference method) ===")
        print("  candidate = the `out` tensor read back out of the replayed NPUGraph")
        SLACK = 2.0
        for i, L in enumerate(lens):
            kv16 = lref.kv_latent(x_all[:L], torch.bfloat16)
            k32 = kv32_all[:L]
            rr, nr = ref_rows[i:i+1], npu_rows[i:i+1]
            r32_a = lref.attend(qno32[i:i+1], k32, rr, torch.float32)
            r16_a = lref.attend(qno16[i:i+1], kv16, rr, torch.bfloat16)
            r32_b = lref.attend(qno32[i:i+1], k32, nr, torch.float32)
            r16_b = lref.attend(qno16[i:i+1], kv16, nr, torch.bfloat16)
            ea, fa = RR.rel(o[i:i+1], r32_a), RR.rel(r16_a, r32_a)
            eb, fb_ = RR.rel(o[i:i+1], r32_b), RR.rel(r16_b, r32_b)
            ov = len(set(rr[0].tolist()) & set(nr[0].tolist())) / max(len(rr[0]), 1)
            ok = eb <= fb_ * SLACK
            rc_gold |= 0 if ok else 1
            print(f"  req {i:>2} len={L:<6} overlap={ov:.5f}  "
                  f"(a) e2e {ea:.3e}/{fa:.3e}={ea/max(fa,1e-12):5.2f}x  "
                  f"(b) ctrl {eb:.3e}/{fb_:.3e}={eb/max(fb_,1e-12):5.2f}x  "
                  f"{'ok' if ok else 'FAIL'}", flush=True)
            del kv16
        print(f"  -> {'all requests within budget' if rc_gold==0 else 'SOME REQUESTS OUT OF BUDGET'}")

    # ---------------- bake checks -------------------------------------------
    g2 = torch.Generator().manual_seed(31)
    def mutB():
        st_x.copy_((torch.randn(st_x.shape, generator=g2) * 0.4).to(torch.bfloat16).to(F.DEV))
    badB, rB = scenario("B new x", mutB)
    def mutC():
        st_x.copy_((torch.randn(st_x.shape, generator=g2) * 0.4).to(torch.bfloat16).to(F.DEV))
        new = [L + 1 for L in lens]
        st_seq_lens.copy_(torch.tensor(new, dtype=torch.int32))
        st_seq_lens_cpu.copy_(torch.tensor(new, dtype=torch.int32))
        st_pos.copy_(torch.tensor(lens, dtype=torch.int64))
        st_out_cache.copy_(torch.tensor([int(slots[i][lens[i]]) for i in range(BS)],
                                        dtype=torch.int64))
    badC, rC = scenario("C new x+seqlen", mutC)
    print(f"    (A->B rel={gcap.rel(rB['out'], rA['out']):.3e}  "
          f"B->C rel={gcap.rel(rC['out'], rB['out']):.3e}  "
          f"topk A->C changed: {not torch.equal(rA['topk'], rC['topk'])})")

    prep(False)
    te = gcap.bench(step); prep(False)
    tg = gcap.bench(cap.replay)
    th = gcap.bench(lambda: (prep(False), cap.replay()))
    print(f"  eager {te:.3f} ms  replay {tg:.3f} ms  replay+prep {th:.3f} ms  "
          f"[CONTENDED MACHINE -- reference only, not a conclusion]")
    print("VERDICT:", "PASS" if not (badA or badB or badC or rc_gold)
          else f"FAIL bake A={badA} B={badB} C={badC} gold_rc={rc_gold}")
