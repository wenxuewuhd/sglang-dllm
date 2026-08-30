"""The padded-batch test: the failure mode that ONLY exists under graph capture.

A captured graph has a fixed batch width.  When fewer requests are running, the
runner pads: req_pool_indices -> 0, seq_lens -> 0, out_cache_loc -> 0
(cuda_graph_buffer_registry.PaddingPolicy.ZERO / FILL_SENTINEL).  Padding rows
therefore *name request 0*.  If any writer scatters by req_pool_index without
excluding them, request 0's kpool tail ring and index cache get clobbered by
garbage -- silently, with plausible-looking output.

So: capture at bs=16, replay with only N real rows, and check request 0 against
an unpadded eager run of the same N requests.
"""
import sys, argparse, time, torch, types
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
ap.add_argument("--lens", default="4096,3077,2048,1025,512,384,320,256,192,4032,3968,3904,3840,3776,3712,3648")
ap.add_argument("--real", type=int, default=3)
ap.add_argument("--chunk", type=int, default=4096)
ap.add_argument("--skip-eager-ref", action="store_true")
a = ap.parse_args()
torch.set_grad_enabled(False)
import torch_npu, custom_ops  # noqa
import tp_fixture as F
sa, mc = F.boot()
sh = F.Shards(); cfg = mc.hf_text_config
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, set_forward_context
from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
    forward_dsa_core_npu, forward_dsa_prepare_npu)

lens = [int(x) for x in a.lens.split(",")]
BS, R = len(lens), a.real
with F.tp_override(a.tp, 0):
    m = F.make_attn(mc, a.layer, sh, tp_size=a.tp, tp_rank=0)
    pages = [(L + 64) // 64 for L in lens]
    size = (sum(pages) + 1) * 64
    kv, req = F.make_pools(mc, a.layer, size, max_running_requests=BS,
                           max_ctx=max(lens) + 128, nreq=BS + 1)
    be = F.make_backend(mc, kv, req, sa)
    set_forward_context(ForwardContext(attn_backend=be))
    pm = F.page_map(sum(pages) + 1, seed=5)
    off, slots = 1, []
    for i, L in enumerate(lens):
        slots.append(F.fill_req_to_token(req, i, pm[off:off + pages[i]], L + 1)); off += pages[i]
    gcpu = torch.Generator().manual_seed(3)
    x_pool = (torch.randn(a.chunk, cfg.hidden_size, generator=gcpu) * 0.3).to(torch.bfloat16).to(F.DEV)
    for i, L in enumerate(lens):
        first = 0
        while first < L - 1:
            n = min(a.chunk, L - 1 - first); sl = first + n
            fbp = F.make_fb(ForwardMode.EXTEND, [sl], [n], [i], slots[i][first:sl].tolist())
            be.init_forward_metadata(fbp)
            F.run_layer(m, torch.arange(first, sl, device=F.DEV, dtype=torch.int64), x_pool[:n], fbp)
            first = sl
    torch.npu.synchronize()
    print(f"TP{a.tp} layer{a.layer} bs={BS} real={R}  prefill done", flush=True)

    # ---- pool state snapshot / restore so every run starts identically
    ikc = kv.index_key_cache
    pool_bufs = list(kv.kv_buffer) + list(kv._compress_tail_k) + list(kv._compress_tail_score) \
                + list(ikc.buffer if isinstance(ikc.buffer, (list, tuple)) else [ikc.buffer])
    saved = [b.clone() for b in pool_bufs]
    def restore():
        for b, s in zip(pool_bufs, saved): b.copy_(s)

    def mk_fb(bs, seq_lens_l, req_idx_l, ocl_l, pos_l):
        fb = types.SimpleNamespace()
        fb.forward_mode = ForwardMode.DECODE; fb.batch_size = bs
        fb.seq_lens = torch.tensor(seq_lens_l, device=F.DEV, dtype=torch.int32)
        fb.seq_lens_cpu = torch.tensor(seq_lens_l, dtype=torch.int32)
        fb.extend_seq_lens = fb.extend_seq_lens_cpu = None
        fb.extend_prefix_lens = fb.extend_prefix_lens_cpu = None
        fb.req_pool_indices = torch.tensor(req_idx_l, device=F.DEV, dtype=torch.int32)
        fb.out_cache_loc = torch.tensor(ocl_l, device=F.DEV, dtype=torch.int64)
        fb.positions = torch.tensor(pos_l, device=F.DEV, dtype=torch.int64)
        fb.spec_info = None; fb.spec_algorithm = None
        fb.attn_cp_metadata = None; fb.token_to_kv_pool = None
        return fb

    full_seq  = list(lens)
    full_req  = list(range(BS))
    full_ocl  = [int(slots[i][lens[i] - 1]) for i in range(BS)]
    full_pos  = [L - 1 for L in lens]
    # padded view: real rows first R, rest padded exactly as the registry does
    pad_seq = full_seq[:R] + [0] * (BS - R)
    pad_req = full_req[:R] + [0] * (BS - R)
    pad_ocl = full_ocl[:R] + [0] * (BS - R)
    pad_pos = full_pos[:R] + [0] * (BS - R)

    x16 = x_pool[:BS].clone().contiguous()
    def run(fb, x, bs):
        be.init_cuda_graph_state(max_bs=bs, max_num_tokens=bs)
        be.init_forward_metadata_out_graph(fb, in_capture=True)
        be.init_forward_metadata_in_graph(fb)
        p = forward_dsa_prepare_npu(m, fb.positions, x, fb, None, None)
        out, _ = forward_dsa_core_npu(m, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
        return out, p[4]

    def tail_of_req0():
        return kv._compress_tail_k[0][0].float().cpu().clone()

    # (0) unpadded EAGER-metadata reference (the path check_dsa.py validated)
    def run_eager(fb, x):
        be.init_forward_metadata(fb)
        p = forward_dsa_prepare_npu(m, fb.positions, x, fb, None, None)
        out, _ = forward_dsa_core_npu(m, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
        return out, p[4]
    restore()
    if a.skip_eager_ref:
        o_e = torch.zeros(1); tk_e = None
    else:
      o_e, tk_e = run_eager(mk_fb(R, full_seq[:R], full_req[:R], full_ocl[:R], full_pos[:R]), x16[:R])
      o_e = o_e.float().cpu().clone(); tk_e = tk_e.cpu().clone()
      print(f"  [ref] eager-metadata bs={R} max|out|={o_e.abs().max():.4e}")

    # (1) unpadded, graph-metadata path, bs=R
    restore()
    o_ref, tk_ref = run(mk_fb(R, full_seq[:R], full_req[:R], full_ocl[:R], full_pos[:R]), x16[:R], R)
    o_ref = o_ref.float().cpu().clone(); tk_ref = tk_ref.cpu().clone(); tail_ref = tail_of_req0()

    for it in range(3):
        restore()
        oX, tkX = run(mk_fb(R, full_seq[:R], full_req[:R], full_ocl[:R], full_pos[:R]), x16[:R], R)
        print(f"    [repeat {it}] unpadded graph-md max|out|={oX.abs().max().item():.4e} "
              f"rel vs first={gcap.rel(oX.float().cpu(), o_ref):.3e}")
    # (2) padded eager, bs=16 with 13 padding rows
    restore()
    o_pe, tk_pe = run(mk_fb(BS, pad_seq, pad_req, pad_ocl, pad_pos), x16, BS)
    o_pe = o_pe.float().cpu().clone(); tk_pe = tk_pe.cpu().clone(); tail_pe = tail_of_req0()

    # (3) padded, captured at bs=16 with all-real rows then replayed padded
    restore()
    st_seq = torch.tensor(full_seq, device=F.DEV, dtype=torch.int32)
    st_seq_cpu = torch.tensor(full_seq, dtype=torch.int32)
    st_req = torch.tensor(full_req, device=F.DEV, dtype=torch.int32)
    st_ocl = torch.tensor(full_ocl, device=F.DEV, dtype=torch.int64)
    st_pos = torch.tensor(full_pos, device=F.DEV, dtype=torch.int64)
    st_x = x16.clone()
    fbS = types.SimpleNamespace(forward_mode=ForwardMode.DECODE, batch_size=BS,
        seq_lens=st_seq, seq_lens_cpu=st_seq_cpu, extend_seq_lens=None,
        extend_seq_lens_cpu=None, extend_prefix_lens=None, extend_prefix_lens_cpu=None,
        req_pool_indices=st_req, out_cache_loc=st_ocl, positions=st_pos,
        spec_info=None, spec_algorithm=None, attn_cp_metadata=None, token_to_kv_pool=None)
    be.init_cuda_graph_state(max_bs=BS, max_num_tokens=BS)
    be.init_forward_metadata_out_graph(fbS, in_capture=True)
    def step():
        be.init_forward_metadata_in_graph(fbS)
        p = forward_dsa_prepare_npu(m, st_pos, st_x, fbS, None, None)
        out, _ = forward_dsa_core_npu(m, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])
        return {"out": out, "topk": p[4].float()}
    cap = gcap.Cap("dsa_pad")
    gout = cap.capture(step)
    print("  capture OK (bs=16, all rows real)", flush=True)
    # now replay with the padded batch
    restore()
    st_seq.copy_(torch.tensor(pad_seq, dtype=torch.int32)); st_seq_cpu.copy_(torch.tensor(pad_seq, dtype=torch.int32))
    st_req.copy_(torch.tensor(pad_req, dtype=torch.int32))
    st_ocl.copy_(torch.tensor(pad_ocl, dtype=torch.int64))
    st_pos.copy_(torch.tensor(pad_pos, dtype=torch.int64))
    be.init_forward_metadata_out_graph(fbS, in_capture=False)
    cap.replay()
    o_pg = gout["out"].float().cpu().clone(); tk_pg = gout["topk"].cpu().clone(); tail_pg = tail_of_req0()

    print(f"\n  magnitudes: unpadded max|out|={o_ref.abs().max():.4e}  "
          f"padded-eager max|out[:R]|={o_pe[:R].abs().max():.4e} (pad rows max={o_pe[R:].abs().max():.4e})  "
          f"padded-replay max|out[:R]|={o_pg[:R].abs().max():.4e}")
    print(f"  rows 0..{R-1} (real requests), three ways:")
    for nm, o, tk, tl in (("padded eager  vs unpadded eager", o_pe, tk_pe, tail_pe),
                          ("padded REPLAY vs unpadded eager", o_pg, tk_pg, tail_pg)):
        eo = gcap.rel(o[:R], o_ref)
        same_tk = torch.equal(tk[:R], tk_ref)
        etl = gcap.rel(tl, tail_ref)
        print(f"    {nm:<34} out rel={eo:.3e}  topk {'IDENTICAL' if same_tk else 'DIFFERS'}  "
              f"req0 tail-ring rel={etl:.3e} {'ok' if etl < 1e-6 else '<-- CLOBBERED'}")
    eo = gcap.rel(o_pg[:R], o_pe[:R])
    if not a.skip_eager_ref:
        print(f"    unpadded graph-md vs unpadded eager-md out rel={gcap.rel(o_ref, o_e):.3e}  "
              f"topk {'IDENTICAL' if torch.equal(tk_ref, tk_e) else 'DIFFERS'}")
    print(f"    padded REPLAY vs padded eager      out rel={eo:.3e}  "
          f"topk {'IDENTICAL' if torch.equal(tk_pg[:R], tk_pe[:R]) else 'DIFFERS'}")
