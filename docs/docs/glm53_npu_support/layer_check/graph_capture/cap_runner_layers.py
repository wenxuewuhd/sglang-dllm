"""Two whole GLM-5.3-Flash decoder layers in ONE graph, captured by the REAL
NPUGraphRunner.

The rest of graph_capture/ answers "can this module be captured".  This answers
the two questions that only appear once the runner is in the loop:

  1. Does the deployment's own capture/replay plumbing work for GLM?
     `NPUGraphRunner` -> `NPUCudaGraphBackend` -> the attention backends'
     `init_forward_metadata_out_graph` / `_replay_metadata` /
     `_apply_cuda_graph_metadata` split.  Two of the three open hazards in
     SHARED_CHANGES.md live in exactly that code and are unreachable from a
     bare `torch.npu.graph(...)` capture.
  2. Does a *whole layer* -- and two of them, of different attention families --
     survive being one graph?  Separately capturable does not imply jointly
     capturable: the mHC four-stream residual crosses attention and comes back,
     a cube-heavy attention and a vector-heavy MoE share one graph, DSA and KDA
     touch the same pools in one replay, and layer 3's output is layer 4's
     input.

Three questions, the same three as ../README.md:

  cap   the runner captures every bs bucket into one shared memory pool.
  bake  replay after rewriting the runner's own device buffers in place --
        *including* seq_lens -- must agree with eager BIT FOR BIT.  A host
        value frozen at capture shows up here and nowhere else.
  gold  the tensor the graph hands back is scored against the CPU trace
        golden with harness.py's two-reference rule.  Only meaningful at
        --tp 1 (see --golden).

Run:
    source $ROOT/env.sh
    D=$REPO/docs/docs/glm53_npu_support/layer_check/graph_capture
    ASCEND_RT_VISIBLE_DEVICES=14 PYTHONPATH=$REPO/python:$PYTHONPATH \
        $VENV/bin/python $D/cap_runner_layers.py --real-weights
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

LC = str(Path(__file__).resolve().parent.parent)
G = str(Path(__file__).resolve().parent)
sys.path.insert(0, LC)
sys.path.insert(0, G)

ap = argparse.ArgumentParser()
ap.add_argument("--tp", type=int, default=16,
                help="TP width to emulate; 16 = the shipped shape (rank 0's "
                     "partial), 1 = the whole thing (comparable to the trace)")
ap.add_argument("--layers", default="3,4", help="two adjacent layer ids")
ap.add_argument("--bs", default="1,2,4,8,16", help="capture bs buckets")
ap.add_argument("--ctx", type=int, default=4096)
ap.add_argument("--prefill", type=int, default=127)
ap.add_argument("--nreq", type=int, default=17)
ap.add_argument("--port", type=int, default=29761)
ap.add_argument("--real-weights", action="store_true",
                help="load the checkpoint's own weights (slow); otherwise noise")
ap.add_argument("--golden", default="",
                help="path to trace_128.pt; scores the graph output against it "
                     "(only sound with --tp 1)")
a = ap.parse_args()

torch.set_grad_enabled(False)
import torch_npu, custom_ops  # noqa: E402,F401

torch.npu.set_device(0)
import runner_fixture as F  # noqa: E402

DEV = "npu"
PAGE = 64
LIDS = [int(x) for x in a.layers.split(",")]
BSL = [int(x) for x in a.bs.split(",")]


def hr(t=""):
    print(f"\n{'='*76}\n{t}" if t else "=" * 76, flush=True)


# ---------------------------------------------------------------- snapshot
def collect_state(named):
    """Every device tensor reachable one/two levels down from these pool
    objects, with a name path.  Both a KDA decode (conv/ssm) and a DSA decode
    (latent KV, index cache, compress-tail ring) mutate their pools, so 'eager
    then replay' would otherwise compare step N with step N+1."""
    out = []
    seen = set()

    def add(name, t):
        if isinstance(t, torch.Tensor) and t.device.type == "npu" and t.numel():
            if id(t) not in seen:
                seen.add(id(t))
                out.append((name, t))

    def walk(name, o, depth):
        d = getattr(o, "__dict__", None)
        if d is None or depth < 0:
            return
        for k, v in list(d.items()):
            if k.startswith("__"):
                continue
            if isinstance(v, torch.Tensor):
                add(f"{name}.{k}", v)
            elif isinstance(v, (list, tuple)):
                for i, x in enumerate(v):
                    if isinstance(x, torch.Tensor):
                        add(f"{name}.{k}[{i}]", x)
                    elif depth > 0:
                        walk(f"{name}.{k}[{i}]", x, depth - 1)
            elif hasattr(v, "__dict__") and depth > 0:
                walk(f"{name}.{k}", v, depth - 1)

    for nm, o in named:
        walk(nm, o, 2)
    return out


def diff_axes(t, e, max_show=8):
    """Which indices along each of the first two axes actually moved."""
    d = (t != e)
    parts = []
    for ax in range(min(2, t.dim())):
        red = list(range(t.dim()))
        red.remove(ax)
        idx = torch.nonzero(d.any(dim=tuple(red)) if red else d).flatten().tolist()
        parts.append(f"dim{ax}={idx[:max_show]}{'...' if len(idx) > max_show else ''}"
                     f"/{t.shape[ax]}")
    return " ".join(parts)


class Snap:
    def __init__(self, tensors):
        self.t = tensors
        self.saved = [x.clone() for x in tensors]

    def restore(self):
        for x, s in zip(self.t, self.saved):
            x.copy_(s)

    def refresh(self):
        for x, s in zip(self.t, self.saved):
            s.copy_(x)


def bitwise(a_, b_):
    if a_.dtype != b_.dtype or a_.shape != b_.shape:
        return False, f"shape/dtype {a_.shape}{a_.dtype} vs {b_.shape}{b_.dtype}"
    same = torch.equal(a_, b_)
    if same:
        return True, "bit-identical"
    d = (a_.float() - b_.float()).abs()
    rel = (d.max() / (b_.float().abs().max() + 1e-12)).item()
    return False, f"{int((a_ != b_).sum())} / {a_.numel()} differ, max rel {rel:.3e}"


# ---------------------------------------------------------------- build
hr("build")
sa, mc = F.boot(port=a.port, capture_bs=BSL, ctx=a.ctx,
                max_running=a.nreq - 1, page=PAGE)
cfg = mc.hf_text_config
full_layers = [l for l in LIDS if not cfg.is_kda_layer(l)]
kda_layers = [l for l in LIDS if cfg.is_kda_layer(l)]
print(f"  layers {LIDS}: full-attn(DSA) {full_layers}, linear(KDA) {kda_layers}")
assert full_layers and kda_layers, "the point is to have one of each in one graph"

npages = a.ctx // PAGE
kv, req = F.build_pools(mc, full_attn_layers=full_layers, kda_layers=kda_layers,
                        kv_pages=a.nreq * npages + 2, page=PAGE,
                        max_ctx=a.ctx, num_reqs=a.nreq, tp=a.tp)
TRACE = None
if a.golden:
    TRACE = torch.load(a.golden, map_location="cpu", weights_only=False)
    assert TRACE["meta"]["hc_mult"] * TRACE["meta"]["hidden_size"] == \
        cfg.hc_mult * cfg.hidden_size
    a.prefill = TRACE["meta"]["tokens"] - 1
    print(f"  golden trace: {TRACE['meta']['tokens']} tokens, "
          f"{TRACE['meta']['layers']} layers, source={TRACE['meta']['source']}")
model = F.TwoLayerGlm(cfg, LIDS, n_table_rows=max(256, a.prefill + 64), tp=a.tp,
                      stash_rows=(max(BSL) if a.golden else 0))
if a.real_weights:
    t0 = time.time()
    F.load_real_weights(model, mc, a.tp)
    print(f"  checkpoint weights loaded in {time.time()-t0:.0f}s")
else:
    F.random_init(model)
    print("  RANDOM weights (plumbing only; --real-weights for the real thing)")
g = torch.Generator(device="cpu").manual_seed(7)
if TRACE is not None:
    # The layer BEFORE ours: its output is our first layer's input.  Stored as
    # [seq, hc_mult, hidden]; sglang carries [seq, hc_mult*hidden] and
    # hc_expand is `x.repeat(1, n)`, i.e. stream-major -- so flatten(1) is the
    # right reinterpretation.
    h_in = TRACE["hidden_fp32"][LIDS[0] - 1].flatten(1)
    GOLD_ROWS = h_in.shape[0]
    model.h_table[:GOLD_ROWS].copy_(h_in.to(torch.bfloat16).to(DEV))
    GOLD_TABLE = model.h_table.clone()
else:
    GOLD_ROWS = 0
    GOLD_TABLE = None
    model.h_table.copy_(
        (torch.randn(model.h_table.shape, generator=g) * 0.5).to(torch.bfloat16))

mr = F.build_model_runner(mc, sa, model, kv, req, max_bs=max(BSL), page=PAGE)
be = F.build_backend(mr, full_layers)
F.patch_shared_path_gaps()
print(f"  backend {type(be).__name__} / {type(be.full_attn_backend).__name__}"
      f" + {type(be.linear_attn_backend).__name__}")

# Request slot 0 and mamba slot 0 are the reserved padding dummies
# (ReqToTokenPool / MambaSlotAllocator), so real requests start at 1 --
# otherwise the padding question below is untestable.
REQS = list(range(1, a.nreq))
for r in REQS:
    t = torch.arange(a.ctx, dtype=torch.int64)
    pages = (r - 1) * npages + t // PAGE + 1
    req.req_to_token[r, : a.ctx] = (
        (pages * PAGE + (t % PAGE)).to(torch.int32).to(DEV))
req.req_index_to_mamba_index_mapping[
    torch.tensor(REQS, device=DEV)
] = torch.arange(1, len(REQS) + 1, dtype=torch.int32, device=DEV)


# ---------------------------------------------------------------- batches
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardBatch, ForwardMode, CaptureHiddenMode,
)
from sglang.srt.model_executor.forward_context import (  # noqa: E402
    ForwardContext, forward_context,
)


def slots(r, lo, hi):
    return req.req_to_token[r, lo:hi].to(torch.int64)


def make_fb(mode, reqs, seq_lens, extend_lens=None, token_ids=None):
    bs = len(reqs)
    rp = torch.tensor(reqs, dtype=torch.int64, device=DEV)
    sl_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    sl = sl_cpu.to(DEV)
    if extend_lens is None:                       # decode: one token per req
        locs = torch.cat([slots(r, s - 1, s) for r, s in zip(reqs, seq_lens)])
        pos = torch.tensor([s - 1 for s in seq_lens], dtype=torch.int64, device=DEV)
        ntok = bs
    else:
        locs = torch.cat([slots(r, s - e, s)
                          for r, s, e in zip(reqs, seq_lens, extend_lens)])
        pos = torch.cat([torch.arange(s - e, s, dtype=torch.int64, device=DEV)
                         for s, e in zip(seq_lens, extend_lens)])
        ntok = sum(extend_lens)
    ids = (token_ids if token_ids is not None
           else torch.arange(ntok, dtype=torch.int64, device=DEV))
    fb = ForwardBatch(
        forward_mode=mode,
        batch_size=bs,
        input_ids=ids,
        req_pool_indices=rp,
        seq_lens=sl,
        seq_lens_cpu=sl_cpu,
        seq_lens_sum=int(sl_cpu.sum()),
        out_cache_loc=locs,
        positions=pos,
        return_logprob=False,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        global_forward_mode=mode,
    )
    if extend_lens is not None:
        fb.extend_seq_lens = torch.tensor(extend_lens, dtype=torch.int32, device=DEV)
        fb.extend_seq_lens_cpu = list(extend_lens)
        pre = [s - e for s, e in zip(seq_lens, extend_lens)]
        fb.extend_prefix_lens = torch.tensor(pre, dtype=torch.int32, device=DEV)
        fb.extend_prefix_lens_cpu = pre
        fb.extend_num_tokens = ntok
        starts = [0]
        for e in extend_lens[:-1]:
            starts.append(starts[-1] + e)
        fb.extend_start_loc = torch.tensor(starts, dtype=torch.int32, device=DEV)
        fb.extend_start_loc_cpu = starts
    fb.token_to_kv_pool = kv
    fb.req_to_token_pool = req
    return fb


def eager(fb):
    """One eager forward through the same two layers, planning through the
    backend's eager entry point."""
    with forward_context(ForwardContext(attn_backend=be)):
        be.init_forward_metadata(fb)
        out = model.forward(fb.input_ids, fb.positions, fb)
    torch.npu.synchronize()
    return out.hidden_states.clone()


# ---------------------------------------------------------------- prefill
hr(f"prefill {a.prefill} tokens x {len(REQS)} requests (fills DSA KV + index "
   f"cache, and the KDA conv/ssm state)")
tok = torch.arange(a.prefill, dtype=torch.int64, device=DEV)
# Batched in groups that fit the shipped 8192-token prefill chunk, so no single
# extend is wider than the deployment ever runs.
per_group = max(1, 8192 // a.prefill)
for i in range(0, len(REQS), per_group):
    grp = REQS[i:i + per_group]
    fbP = make_fb(ForwardMode.EXTEND, grp, [a.prefill] * len(grp),
                  extend_lens=[a.prefill] * len(grp),
                  token_ids=tok.repeat(len(grp)))
    eager(fbP)
print(f"  prefill done ({per_group} request(s) per extend)", flush=True)

def reserved_slots(kv, req, page):
    """Where a PADDED graph row is *supposed* to write, per pool, and why.

    A captured graph has a fixed width.  When fewer requests are running the
    runner zeroes the tail rows (`seq_lens -> fill`, `req_pool_indices -> 0`,
    `out_cache_loc -> 0`), so every padding row names request 0 and token slot
    0.  Each pool keeps somewhere harmless for those writes; a padded replay
    that touches anything ELSE has silently corrupted a live request.

    Returns {tensor-name-suffix: (axis, {allowed indices})}.  Everything is read
    off the pool objects rather than hardcoded, so a pool that changes its
    reserved slot moves this rule with it."""
    rules = {}
    # out_cache_loc padding policy is ZERO (build_decode_registry), and token
    # slot 0 is never handed to a real request.
    rules["kv.kv_buffer"] = (0, {0})
    rules["kv.k_buffer"] = (0, {0})
    rules["kv.v_buffer"] = (0, {0})
    # NPUDSATokenToKVPool.scratch_loc -- an index-cache slot no block table can
    # name (memory_pool_npu.py: `(buffer[0].shape[0] - 1) * page_size`).
    sl = getattr(kv, "scratch_loc", None)
    if sl is not None:
        rules["kv.index_key_cache.buffer"] = (0, {sl // page})
    # the spare compress-tail ring row NPUDSATokenToKVPool adds for exactly this
    tsr = getattr(kv, "_tail_scratch_row", None)
    if tsr is not None:
        rules["kv.index_key_cache.pool._compress_tail_k"] = (0, {tsr})
        rules["kv.index_key_cache.pool._compress_tail_score"] = (0, {tsr})
        rules["kv._compress_tail_k"] = (0, {tsr})
        rules["kv._compress_tail_score"] = (0, {tsr})
    # MambaSlotAllocator keeps mamba slot 0 free (mem_cache/allocator/mamba.py:
    # free_slots = arange(1, size+1)); state tensors are [layer, slot, ...].
    for key in ("conv", "temporal", "intermediate_ssm"):
        rules[f"req.mamba_pool.mamba_cache.{key}"] = (1, {0})
    return rules


def allowed_diff(name, t, e, rules):
    """True when this tensor's diff is confined to its reserved slots."""
    base = name.split("[")[0]
    rule = rules.get(base)
    if rule is None:
        return False, "no reserved slot declared for this tensor"
    axis, allowed = rule
    if t.dim() <= axis:
        return False, f"tensor has no axis {axis}"
    d = (t != e)
    red = [i for i in range(t.dim()) if i != axis]
    moved = set(torch.nonzero(d.any(dim=tuple(red)) if red else d)
                .flatten().tolist())
    extra = sorted(moved - allowed)
    return (not extra), (f"moved dim{axis}={sorted(moved)} allowed={sorted(allowed)}"
                         + (f" EXTRA={extra[:8]}" if extra else ""))


named_state = collect_state([("kv", kv), ("req", req),
                             ("full", be.full_attn_backend),
                             ("lin", be.linear_attn_backend)])
state = [t for _n, t in named_state]
state_names = [n for n, _t in named_state]
RESERVED = reserved_slots(kv, req, PAGE)
print(f"  tracking {len(state)} mutable device tensors for save/restore")
snap = Snap(state)

# ---------------------------------------------------------------- Q1 cap
hr("Q1 cap -- NPUGraphRunner captures every bs bucket")
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (  # noqa: E402
    NPUGraphRunner,
)

snap.restore()
t0 = time.time()
try:
    runner = NPUGraphRunner(mr)
except Exception as e:                                        # noqa: BLE001
    import traceback
    traceback.print_exc()
    print(f"  CAPTURE FAILED: {type(e).__name__}: {str(e)[:600]}")
    raise SystemExit(1)
cap_s = time.time() - t0
keys = sorted(k.size for k in runner.backend._graphs)
print(f"  captured bs buckets {keys} in {cap_s:.1f}s "
      f"(CONTENDED MACHINE -- timing is not a result)")
print(f"  one shared memory pool handle: {runner.backend._pool!r}")
print(f"  runner.seq_len_fill_value = {runner.seq_len_fill_value}")
# Capture ran real forwards through the pools; re-baseline.
snap.restore()

# ---------------------------------------------------------------- hazards
hr("hazard probes (SHARED_CHANGES.md 'three open issues')")
lin = be.linear_attn_backend
full = be.full_attn_backend
print(f"  [1] padding fill value: runner fills padded seq_lens with "
      f"{runner.seq_len_fill_value} (top-level backend "
      f"{type(be).__name__}.get_cuda_graph_seq_len_fill_value()); the linear "
      f"half {type(lin).__name__} caches "
      f"{lin.get_cuda_graph_seq_len_fill_value()}")
print(f"      -> mismatch: "
      f"{runner.seq_len_fill_value != lin.get_cuda_graph_seq_len_fill_value()}")
print(f"      _replay_metadata resolved to "
      f"{type(lin)._replay_metadata.__qualname__}")
md = full.graph_metadata.get(max(keys))
print(f"  [2] AscendAttnBackend.graph_metadata[{max(keys)}].seq_lens_cpu_list "
      f"after capture = {md.seq_lens_cpu_list}")
print(f"      seq_lens_cpu_int = {md.seq_lens_cpu_int} "
      f"actual_seq_lengths_kv = {md.actual_seq_lengths_kv}")

# ---------------------------------------------------------------- Q2 bake
hr("Q2 bake -- replay must follow its DEVICE inputs, bit for bit")


def one_case(name, reqs, seq_lens, ids, expect_pad):
    bs = len(reqs)
    fb_e = make_fb(ForwardMode.DECODE, reqs, seq_lens, token_ids=ids)
    snap.restore()
    ref = eager(fb_e)
    st_e = [x.clone() for x in state]

    fb_g = make_fb(ForwardMode.DECODE, reqs, seq_lens, token_ids=ids)
    snap.restore()
    with forward_context(ForwardContext(attn_backend=be)):
        got = runner.execute(fb_g).hidden_states.clone()
    torch.npu.synchronize()
    ok, why = bitwise(got, ref)
    pad = runner.bs - runner.raw_bs
    tag = f"bs {bs} -> bucket {runner.bs} (pad {pad})"
    print(f"  {name:<34} {tag:<28} out: {'OK  ' if ok else 'FAIL'} {why}")
    # every mutated pool tensor must land in the same place too
    bad, benign = [], []
    for nm, t, e in zip(state_names, state, st_e):
        if torch.equal(t, e):
            continue
        if pad:
            ok_r, why_r = allowed_diff(nm, t, e, RESERVED)
            (benign if ok_r else bad).append((nm, tuple(t.shape), why_r))
        else:
            bad.append((nm, tuple(t.shape), diff_axes(t, e)))
    if not bad and not benign:
        print(f"  {'':<34} {'':<28} pool: OK   all {len(state)} tensors match")
    elif not bad:
        print(f"  {'':<34} {'':<28} pool: OK   real rows match; "
              f"{len(benign)} tensor(s) moved ONLY in their reserved slot:")
        for nm, sh, why_r in benign:
            print(f"  {'':<34} {'':<28}       {nm} {sh}: {why_r}")
    else:
        print(f"  {'':<34} {'':<28} pool: FAIL {len(bad)} tensor(s) moved "
              f"outside any reserved slot:")
        for nm, sh, why_r in bad:
            print(f"  {'':<34} {'':<28}       {nm} {sh}: {why_r}")
    assert pad == expect_pad or expect_pad is None, (pad, expect_pad)
    rec = dict(reqs=list(reqs), seq_lens=list(seq_lens), bucket=runner.bs,
               pad=pad, fb=fb_g)
    LAST.update(rec)
    if pad:
        PADDED.update(rec)
    return ok and not bad


results = {}
LAST = {}
PADDED = {}
n = len(REQS)
base_len = a.prefill + 1
ids0 = torch.arange(a.prefill, a.prefill + max(BSL), dtype=torch.int64, device=DEV)

# A: exactly a captured bucket, no padding
bsA = min(max(k for k in keys if k <= n), n)
results["A same inputs"] = one_case(
    "A  captured bucket, no padding", REQS[:bsA], [base_len] * bsA,
    ids0[:bsA], expect_pad=0)

# B: new hidden rows only -- same seq_lens
model.h_table.copy_(
    (torch.randn(model.h_table.shape, generator=g) * 0.5).to(torch.bfloat16))
results["B new hidden"] = one_case(
    "B  new hidden rows", REQS[:bsA], [base_len] * bsA,
    ids0[:bsA], expect_pad=0)

# C: new seq_lens TOO -- the only case that can catch a baked host length
lensC = [base_len - 1 - (i % 7) * 3 for i in range(bsA)]
results["C new seq_lens"] = one_case(
    "C  new hidden AND new seq_lens", REQS[:bsA], lensC, ids0[:bsA],
    expect_pad=0)

# D: token ids permuted -- the runner's own input_ids buffer must drive the gather
idsD = ids0[:bsA].flip(0).contiguous()
results["D permuted input_ids"] = one_case(
    "D  permuted input_ids", REQS[:bsA], lensC, idsD, expect_pad=0)

# E: a raw_bs that is NOT a bucket -> the runner pads the tail
raw = max(1, bsA - 3)
if raw < bsA:
    results["E padded replay"] = one_case(
        "E  padded replay", REQS[:raw], lensC[:raw], ids0[:raw],
        expect_pad=None)

# F: a different bucket entirely -- proves the buckets share one pool safely
for k in keys:
    if k == bsA or k > n:
        continue
    results[f"F bucket {k}"] = one_case(
        f"F  bucket {k} from the shared pool", REQS[:k],
        [base_len - 2 * (i % 5) for i in range(k)], ids0[:k], expect_pad=0)

hr("hazard verdict -- what the runner path actually does")

# [1] the fill-value mismatch: is the fallback reachable, and what would it say?
P = PADDED or LAST
bsL, padL = P["bucket"], P["pad"]
realL = P["seq_lens"]
print(f"  [1] last replay: bucket {bsL}, {padL} padding row(s)")
print(f"      the runner ALWAYS supplies num_padding "
      f"(decode_cuda_graph_runner build_replay_fb_view: num_padding=bs-raw_bs),")
print(f"      so MambaAttnBackendBase._replay_metadata's "
      f"`if num_padding is None` fallback is dead on this path.")
probe = torch.full((bsL,), runner.seq_len_fill_value, dtype=torch.int64)
probe[: bsL - padL] = torch.tensor(realL, dtype=torch.int64)
would = int(torch.count_nonzero(
    probe == lin.get_cuda_graph_seq_len_fill_value()))
truth = int(torch.count_nonzero(probe == runner.seq_len_fill_value))
print(f"      if it were reached with the runner's own padded seq_lens "
      f"{probe.tolist()}:")
print(f"        fallback would count num_padding = {would} "
      f"(compares against the LINEAR half's cached fill value "
      f"{lin.get_cuda_graph_seq_len_fill_value()})")
print(f"        truth is {padL} "
      f"(the runner filled with {runner.seq_len_fill_value})")
print(f"      -> latent: {'YES, would treat padding rows as real' if would != truth else 'no'}")

# ... and what that actually costs.  Re-plan the SAME padded replay through the
# public backend entry point with num_padding withheld (what any caller that
# does not set the field gets), then replay the captured graph and see where the
# KDA state landed.
if PADDED:
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (  # noqa
        build_replay_fb_view,
    )

    fbP2 = make_fb(ForwardMode.DECODE, P["reqs"], P["seq_lens"],
                   token_ids=ids0[: len(P["reqs"])])
    snap.restore()
    with forward_context(ForwardContext(attn_backend=be)):
        ref_pad = runner.execute(fbP2).hidden_states.clone()
    st_ok = [x.clone() for x in state]

    fbP3 = make_fb(ForwardMode.DECODE, P["reqs"], P["seq_lens"],
                   token_ids=ids0[: len(P["reqs"])])
    snap.restore()
    with forward_context(ForwardContext(attn_backend=be)):
        runner.load_batch(fbP3)                      # fills the static buffers
        view = build_replay_fb_view(
            forward_batch=fbP3, buffers=runner.buffers, bs=runner.bs,
            raw_bs=runner.raw_bs, num_tokens=runner.bs,
            seq_len_fill_value=runner.seq_len_fill_value,
            capture_forward_mode=ForwardMode.DECODE, is_encoder_decoder=False,
        )
        view.num_padding = None                      # the caller that forgets
        be.init_forward_metadata_out_graph(view)
        runner.backend.replay(runner._replay_graph_key, view)
    torch.npu.synchronize()
    hurt, contained = [], []
    for nm, t, e in zip(state_names, state, st_ok):
        if torch.equal(t, e):
            continue
        ok_r, why_r = allowed_diff(nm, t, e, RESERVED)
        (contained if ok_r else hurt).append((nm, why_r))
    print(f"      consequence of actually taking that branch "
          f"(same padded batch, num_padding withheld):")
    for nm, why_r in contained + hurt:
        print(f"        {nm}: {why_r}")
    if hurt:
        print(f"        ^ LIVE REQUEST STATE CORRUPTED: those writes are not "
              f"confined to a reserved slot.")
    elif contained:
        print(f"        ^ the padded rows were treated as REAL (mamba index 0 "
              f"instead of the -1 PAD_SLOT_ID sentinel), so they wrote state "
              f"the correct replay does not write -- but only into the slot "
              f"MambaSlotAllocator reserves for exactly that.  The blast "
              f"radius is contained BY THE POOL LAYOUT, not by the metadata "
              f"code: a pool that handed slot 0 to a live request would lose "
              f"that request's state.")
    else:
        print(f"        no pool tensor moved differently -- harmless for this "
              f"batch.")
    # the replayed output is discarded above; the padded rows never reach it
    print(f"      output rows [:raw_bs] vs the correct padded replay: "
          f"{bitwise(runner.backend._outputs[runner._replay_graph_key].hidden_states[: runner.raw_bs].clone(), ref_pad[: runner.raw_bs])[1]}")
    snap.restore()

# [2] the baked host seq_lens_cpu_list
bsC, padC = LAST["bucket"], LAST["pad"]
mdL = full.graph_metadata[bsC]
padded_true = LAST["seq_lens"] + [runner.seq_len_fill_value] * padC
print(f"  [2] after a replay whose real seq_lens were {padded_true}:")
print(f"        graph_metadata[{bsC}].seq_lens_cpu_list = {mdL.seq_lens_cpu_list}"
      f"   <- set once at capture, never refreshed")
print(f"        graph_metadata[{bsC}].seq_lens (device)  = "
      f"{mdL.seq_lens.tolist()}   <- refreshed in place")
print(f"        seq_lens_cpu_int = {mdL.seq_lens_cpu_int} "
      f"(None in graph mode: only init_forward_metadata sets it)")
print(f"      -> the host list IS stale. Case C above changed seq_lens and the "
      f"replay still matched eager bit for bit, so GLM's DSA decode does not "
      f"read it (forward_sparse short-circuits before forward_decode_graph).")

# [3] the MoE group_list host materialization
from sglang.srt.layers.moe.utils import get_moe_a2a_backend  # noqa: E402

_a2a = get_moe_a2a_backend()
_sparse = [l for l in LIDS if model.model.layers[str(l)].is_layer_sparse]
_experts = (model.model.layers[str(_sparse[0])].mlp.experts
            if _sparse else None)
print(f"  [3] moe a2a backend = {_a2a}, deepep = {_a2a.is_deepep()}; "
      f"experts quant_method = "
      f"{type(_experts.quant_method).__name__ if _experts is not None else None}")
print(f"      moe_runner_backend = "
      f"{getattr(_experts, 'moe_runner_backend', None)}")
print(f"      -> the ascend moe_runner's deepep branch (group_list built from a "
      f"host list) is {'LIVE' if _a2a.is_deepep() else 'not reached'}; the "
      f"shipped GLM recipe is --moe-a2a-backend none.")

hr("verdict")
for k, v in results.items():
    print(f"  {k:<28} {'PASS' if v else 'FAIL'}")
bad = [k for k, v in results.items() if not v]
print(f"\n  -> {len(results)-len(bad)}/{len(results)} pass")

# ---------------------------------------------------------------- gold
if a.golden:
    hr("Q3 gold -- the tensor the GRAPH returns, scored against the CPU trace")
    from tolerance import ABS_MIN, SLACK, noise_floor, rel_err  # noqa

    if a.tp != 1:
        print(f"  !! --tp {a.tp}: the modules compute rank 0's PARTIAL "
              f"contribution, which is not what the trace holds. The scores "
              f"below are meaningless; rerun with --tp 1.")
    pos = GOLD_ROWS - 1
    r0 = REQS[0]
    model.h_table.copy_(GOLD_TABLE)
    snap.restore()
    fbG = make_fb(ForwardMode.DECODE, [r0], [GOLD_ROWS],
                  token_ids=torch.tensor([pos], dtype=torch.int64, device=DEV))
    with forward_context(ForwardContext(attn_backend=be)):
        outG = runner.execute(fbG).hidden_states.clone()
    torch.npu.synchronize()
    got = {LIDS[-1]: outG[0]}
    for lid in LIDS[:-1]:
        got[lid] = model.layer_out[lid][0].clone()

    print(f"  decode of token {pos} with {pos} tokens of context, bs=1, "
          f"replayed from the captured graph")
    rows = []
    for lid in LIDS:
        ref32 = TRACE["hidden_fp32"][lid][pos].flatten().float()
        ref16 = TRACE["hidden_bf16"][lid][pos].flatten().float()
        err = rel_err(got[lid].to("cpu", torch.float32), ref32)
        floor = noise_floor(ref32, ref16)
        bud = max(floor * SLACK, ABS_MIN)
        ok_g = err <= bud
        rows.append(ok_g)
        print(f"  [{'ok  ' if ok_g else 'FAIL'}] layer {lid} output        "
              f"err={err:.3e}  floor={floor:.3e}  budget={bud:.3e}  "
              f"({err / bud:.2f}x budget)")
    print(f"  -> {sum(rows)}/{len(rows)} within budget "
          f"(slack {SLACK}, abs floor {ABS_MIN})")
    if a.tp == 1 and not all(rows):
        bad.append("gold")

del runner
torch.npu.empty_cache()
print("\ndone", flush=True)
raise SystemExit(1 if bad else 0)
