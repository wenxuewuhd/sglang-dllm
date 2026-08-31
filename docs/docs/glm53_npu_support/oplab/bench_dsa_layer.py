#!/usr/bin/env python
"""One GLM-5.3-Flash DSA layer, decode, bs=1, in an NPU graph -- no server.

WHAT THIS IS
------------
The 11 DSA (DeepSeek sparse attention) layers of GLM-5.3-Flash INT8 cost
**5.395 ms of the 31.274 ms decode step** on one Atlas A3 die at bs=1 with the
NPU graph on.  That is 891 kernel launches, **81 per layer, 468 us per layer**.
This script rebuilds that layer -- INT8 NZ projections, the kpool indexer, the
lightning indexer, the sparse flash attention and both MLA absorbs -- with the
deployed shapes and a full-size 1.25 M-token KV pool, and times it inside a
captured NPU graph.  No model, no checkpoint, no server.

Full reference kernel list: `../int8_singlecard/data/kernel_attribution_cfgI.txt`,
section `--- DSA ---`.  The head of it:

     us/step   n  us/call  operator                input shapes
    ---------------------------------------------------------------------------
       693.6  11    63.1   QuantBatchMatmulV3      "1,16384;128,1024,16,32;4096;1"
       357.4  11    32.5   SparseFlashAttention    "1,64,512;19479,64,1,512;..."
       303.9  11    27.6   QuantBatchMatmulV3      "1,1536;512,96,16,32;16384;1"
       300.0  11    27.3   BatchMatMulV2           "64,1,256;64,512,256"
       265.7  22    12.1   MatMulV2                "1,4096;128,4096"
       220.1  11    20.0   MatMulV2                "1,1536;4096,1536"
       218.2  11    19.8   batch_matmul_transpose_0
       206.8  11    18.8   QuantBatchMatmulV3      "1,4096;64,256,16,32;2048;1"
       204.9  11    18.6   LightningIndexer        "1,32,128;19480,64,1,128;..."
       ... ~46 more groups, mostly index and bookkeeping
    ---------------------------------------------------------------------------
      5143    891          = 5.395 ms/step, 468 us/layer

THE SEQUENCE (deepseek_v2_attention_mla_npu.py forward_dsa_prepare/core_npu)

  A. latents          x[1,4096] -> DynamicQuant -> QuantBatchMatmulV3 -> [1,2048]
                      split 1536 q_lora | 512 kv_lora
                      RmsNorm(1536), RmsNorm(512)
                      q_b_proj  1536 -> 64x256      QuantBatchMatmulV3
                      bmm q_nope x w_kc             BatchMatMulV2 "64,1,256;64,512,256"
                        = the W^UK absorb, q into the 512-wide latent space
  B. indexer          wq_b   1536 -> 32x128         MatMulV2 "1,1536;4096,1536"
                      wk     4096 -> 128            MatMulV2 "1,4096;128,4096"
                      k_norm LayerNormV3(128)
                      gate   4096 -> 128            MatMulV2 "1,4096;128,4096"
                      weights_proj 4096 -> 32 fp32  MatMulV2 "1,4096;32,4096"
                      Hadamard-128 in fp32          MatMulV2 "32,128;128,128" / "1,128;128,128"
                      compress 4 tokens -> 1 pool   ReduceMax/ReduceSum "1,4,128;1"
                      ring + cache writes           ScatterNdUpdate x3
                      npu_lightning_indexer         LightningIndexer -> 512 pool ids
                      expand 512 pools -> 2048      Add "1,512,1;4", BroadcastTo
                      + 3 tail columns              -> [1,2051] int32
  C. attention        kv_buffer[loc] = k_nope       IndexPutV2 "1246656,1,512;..."
                      npu_sparse_flash_attention    SparseFlashAttention
  D. out              batch_matmul_transpose (W^UV) batch_matmul_transpose_0
                      o_proj 16384 -> 4096          QuantBatchMatmulV3

SEQUENCE LENGTH -- THIS IS THE POINT OF THE SCRIPT
--------------------------------------------------
DSA and KDA scale completely differently, and making that difference visible is
the whole value of this pair of benchmarks.

**Nothing about the sparse attention itself grows with n.**  `index_topk = 2048`
caps how many tokens `npu_sparse_flash_attention` reads, forever.  What grows is
the **indexer**: it must score every closed pool, and with `index_kpool = 4` that
is **n/4 pools**, not n/64.  (The task brief said n/64; the config says
`index_kpool: 4` and REPORT.md 1 says "n/4 个 pool".  n/64 would be the page
count.)  So the expectation is `LightningIndexer` linear in n and everything
else flat.

The sweep does **not** change any tensor shape -- the KV and index pools are
allocated at their full deployed size (19479 / 19480 pages, 1.25 M tokens) exactly
as the server allocates them, and only the int32 length tensors that the graph
reads change between replays.  That is also how the served graph works.

HOW TO RUN
----------
    source <repo>/env/env.sh
    ASCEND_RT_VISIBLE_DEVICES=<die> python bench_dsa_layer.py

    --sections layer,sweep,refs,family    (default layer,sweep,refs)
    --seq-lens 1024,4096,32768,131072,1048576
    --reps 30 --warmup 10

Needs ~2.2 GB of HBM for the pools (`family` needs ~19 GB).

Needs `torch`, `torch_npu`, `sgl_kernel_npu` and four helpers from the sglang
tree -- two pure-torch kpool functions and one Triton kernel, all of them kernel
code rather than model code.  See the README section "what this imports".
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import itertools
import os
import shutil
import statistics
import sys
import time

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  registers the npu device and torch.ops.npu

try:
    import sgl_kernel_npu  # noqa: F401  registers torch.ops.npu.batch_matmul_transpose
except ImportError:
    sys.exit("sgl_kernel_npu is not importable; source the project env.sh first")

# `npu_format_cast` to FRACTAL_NZ is a **silent no-op** unless internal formats
# are enabled, and that switch does not exist until `transfer_to_npu` has been
# imported.  Without these three lines the INT8 weights stay ND, the profiler
# reports `"1,16384;16384,4096;4096;1"` instead of `"...;128,1024,16,32;..."`,
# and o_proj costs 103 us instead of 63 -- with nothing but a `[W...] Warning:
# Cannot create tensor with internal format` on stderr to say so.
# The price of the import is pitfall #1 in the README: it makes `tensor.is_cuda`
# return True for every tensor on this box, including ones made before the
# import.  Use `device.type`, never `is_cuda`.
from torch_npu.contrib import transfer_to_npu  # noqa: F401,E402

torch.cuda.is_available = lambda: False  # transfer_to_npu mocks this True
torch_npu.npu.config.allow_internal_format = True
torch_npu.npu.set_compile_mode(jit_compile=False)

try:
    from sglang.srt.hardware_backend.npu.attention.kpool_indexer_npu import (
        compress_pool_bf16,
        hadamard_transform_npu,
    )
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
        append_kpool_tail_to_topk,
        expand_pooled_groups_to_topk,
    )
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"cannot import the kpool helpers ({exc}).\n"
        "They are kernel code, not model code -- put the sglang tree on PYTHONPATH:\n"
        "  export PYTHONPATH=<worktree>/python:$PYTHONPATH"
    )

DEV = "npu:0"
BF = torch.bfloat16
#: ACL_FORMAT_FRACTAL_NZ -- the layout npu_quant_matmul wants for INT8 weights
FRACTAL_NZ = 29

# ---- GLM-5.3-Flash deployed values (config.json text_config, TP=1) ----------
HIDDEN = 4096
N_HEADS = 64  # num_attention_heads
Q_LORA = 1536  # q_lora_rank
KV_LORA = 512  # kv_lora_rank
QK_HEAD = 256  # qk_head_dim == qk_nope_head_dim (qk_rope_head_dim == 0, NoPE)
V_HEAD = 256  # v_head_dim
IDX_HEADS = 32  # index_n_heads
IDX_DIM = 128  # index_head_dim
IDX_TOPK = 2048  # index_topk
IDX_KPOOL = 4  # index_kpool   <- 4, not 64.  64 is the page size.
PAGE = 64  # --page-size
RMS_EPS = 1e-5
DSA_LAYERS = 11

QB_OUT = N_HEADS * QK_HEAD  # 16384
QKV_A_OUT = Q_LORA + KV_LORA  # 2048
TOPK_COLS = IDX_TOPK + IDX_KPOOL - 1  # 2051
GROUP_TOPK = IDX_TOPK // IDX_KPOOL  # 512
SLOTS_PER_PAGE = PAGE // IDX_KPOOL  # 16
#: The only rope width npu_sparse_flash_attention accepts on a NoPE model.
NOPE_ROPE_WIDTH = 64

#: Served pool geometry (mem-fraction 0.80 on one 64 GB die).  The kv pool and
#: the index pool deliberately differ by one page.
KV_PAGES = 19479
IDX_PAGES = 19480
KV_TOKENS = KV_PAGES * PAGE  # 1,246,656
IDX_TOKENS = IDX_PAGES * PAGE  # 1,246,720
REQ_POOL_ROWS = 19  # tail ring rows; the profile's 76 = 19 x 4

BW_GBPS = 1250.0
L2_MIB = 168.0

#: config I, per call, device us, keyed by (Type, input shapes).  `None` shapes
#: match any shape for that Type.
REF = {
    ("QuantBatchMatmulV3", "1,16384;128,1024,16,32;4096;1"): 63.1,
    ("SparseFlashAttention", None): 32.5,
    ("QuantBatchMatmulV3", "1,1536;512,96,16,32;16384;1"): 27.6,
    ("BatchMatMulV2", "64,1,256;64,512,256"): 27.3,
    ("MatMulV2", "1,4096;128,4096"): 12.1,
    ("MatMulV2", "1,1536;4096,1536"): 20.0,
    ("batch_matmul_transpose_0", None): 19.8,
    ("QuantBatchMatmulV3", "1,4096;64,256,16,32;2048;1"): 18.8,
    ("LightningIndexer", None): 18.6,
    ("ReduceSum", "1,4,128;1"): 8.5,
    ("Add", "1,512,1;4"): 9.6,
    ("IndexPutV2", "1246656,1,512;1,1,512;1;1;1"): 8.2,
    ("MatMulV2", "1,4096;32,4096"): 8.0,
    ("BroadcastTo", "1,512,1;3"): 7.3,
    ("MatMulV2", "128,128;128,128"): 6.5,
    ("ScatterNdUpdate", "1246720,1,128;1,1;1,1,128"): 6.2,
    ("LayerNormV3", "1,128;128;128"): 5.9,
    ("RmsNorm", "1,1536;1536"): 5.5,
    ("ReduceMax", "1,4,128;1"): 5.4,
    ("DynamicQuant", "1,16384"): 4.6,
    ("RmsNorm", "1,1,512;512"): 4.0,
    ("DynamicQuant", "1,1536"): 2.6,
}
REF_LAYER_US = 490.4  # 5143.0 / 11
REF_STEP_MS = 5.395
REF_KERNELS_PER_LAYER = 81

PROF_ROOT = os.environ.get("OPLAB_PROF_DIR", "/var/tmp/glm53/oplab/dsa")


def nz_int8(k: int, n: int, gen):
    """An INT8 weight in the FRACTAL_NZ layout npu_quant_matmul consumes.

    The served loader does `weight.transpose(0,1).contiguous()` then
    `npu_format_cast`, which turns a `[n, k]` checkpoint tensor into `[k, n]`
    and then into NZ.  The profiler reports the NZ shape as `[n/32, k/16, 16, 32]`
    -- that is how the reference table's `128,1024,16,32` decodes to 16384->4096.
    Alignment rule for INT8 NZ: `k % 16 == 0 and n % 32 == 0`.
    """
    assert k % 16 == 0 and n % 32 == 0, f"INT8 NZ needs k%16==0, n%32==0; got {k},{n}"
    w = torch.randint(-8, 8, (k, n), generator=gen, dtype=torch.int8).to(DEV)
    out = torch_npu.npu_format_cast(w.contiguous(), FRACTAL_NZ)
    if torch_npu.get_npu_format(out) != FRACTAL_NZ:
        raise SystemExit(
            "npu_format_cast did not produce FRACTAL_NZ (it fails silently).\n"
            "Enable internal formats before allocating: import "
            "torch_npu.contrib.transfer_to_npu, then set "
            "torch_npu.npu.config.allow_internal_format = True."
        )
    return out


# ---------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------
class DSALayer:
    """One DSA layer's weights, caches and decode body.

    Every buffer is allocated once.  The sequence length lives only in the four
    int32 metadata tensors, which `set_seq_len` rewrites in place -- so one
    captured graph serves every length, exactly as the served graph does.
    """

    def __init__(self, seq_len=512, context_len=32768, share=None, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)

        def rnd(*shape, dtype=BF, scale=0.05):
            return (torch.randn(*shape, generator=g) * scale).to(dtype).to(DEV)

        # -- INT8 NZ projections (the only quantized ones on a DSA layer) -----
        self.w_qkv_a = nz_int8(HIDDEN, QKV_A_OUT, g)  # fused q_a + kv_a_with_mqa
        self.s_qkv_a = torch.rand(QKV_A_OUT, generator=g).float().to(DEV) * 0.01
        self.w_q_b = nz_int8(Q_LORA, QB_OUT, g)
        self.s_q_b = torch.rand(QB_OUT, generator=g).float().to(DEV) * 0.01
        self.w_o = nz_int8(QB_OUT, HIDDEN, g)
        self.s_o = torch.rand(HIDDEN, generator=g).float().to(DEV) * 0.01

        # -- bf16 projections.  The checkpoint's `ignore` list holds indexer.wq_b,
        #    .wk, .weights_proj, .index_kpool_compress_gate and kv_b_proj, which
        #    is why these are MatMulV2 and not QuantBatchMatmulV3.
        # w_kc is stored transposed on purpose, so its physical buffer really is
        # [64, 512, 256] and the bmm runs with transpose_b -- that is what makes
        # the profile read "64,1,256;64,512,256".
        self.w_kc = rnd(N_HEADS, KV_LORA, QK_HEAD)
        self.w_vc = rnd(N_HEADS, KV_LORA, V_HEAD)
        self.wq_b = rnd(IDX_HEADS * IDX_DIM, Q_LORA)  # [4096, 1536]
        self.wk = rnd(IDX_DIM, HIDDEN)  # [128, 4096]
        self.gate_w = rnd(IDX_DIM, HIDDEN)  # index_kpool_compress_gate
        self.wproj = (torch.randn(IDX_HEADS, HIDDEN, generator=g) * 0.05).float().to(DEV)
        self.q_a_norm_w = torch.ones(Q_LORA, device=DEV, dtype=BF)
        self.kv_a_norm_w = torch.ones(KV_LORA, device=DEV, dtype=BF)
        self.k_norm_w = torch.ones(IDX_DIM, device=DEV, dtype=torch.float32)
        self.k_norm_b = torch.zeros(IDX_DIM, device=DEV, dtype=torch.float32)
        # fp32, not bf16.  `compress_pool_bf16` does `ape.float()`, so a bf16 ape
        # would add a Cast "4,128" per layer -- and the served inventory has the
        # Add "1,4,128;4,128" but no such Cast, which pins the parameter's dtype.
        self.ape = (torch.randn(IDX_KPOOL, IDX_DIM, generator=g) * 0.1).float().to(DEV)
        self.scaling = QK_HEAD**-0.5
        self.idx_scale = IDX_HEADS**-0.5 * IDX_DIM**-0.5

        # -- caches.  Shared across layers in the `family` section: the KV pool
        #    is per layer in the server, but the zero ropes and metadata are not.
        s = share if share is not None else {}
        self.kv_buffer = torch.zeros(KV_TOKENS, 1, KV_LORA, device=DEV, dtype=BF)
        self.index_k = torch.zeros(IDX_PAGES, PAGE, 1, IDX_DIM, device=DEV, dtype=BF)
        self.tail_k = torch.zeros(REQ_POOL_ROWS, IDX_KPOOL, IDX_DIM, device=DEV, dtype=BF)
        self.tail_s = torch.zeros(REQ_POOL_ROWS, IDX_KPOOL, IDX_DIM, device=DEV, dtype=BF)

        # A real key rope is a paged cache holding nothing but zeros.  Allocate
        # it once and share it: an `expand` here is materialised on every call
        # (torch_npu makes inputs contiguous), which cost 0.70 ms/step in the
        # served run before it was found.
        if "q_rope" not in s:
            s["q_rope"] = torch.zeros(1, N_HEADS, NOPE_ROPE_WIDTH, device=DEV, dtype=BF)
            s["k_rope"] = torch.zeros(
                KV_PAGES, PAGE, 1, NOPE_ROPE_WIDTH, device=DEV, dtype=BF
            )
        self.q_rope, self.k_rope = s["q_rope"], s["k_rope"]

        # -- metadata.  These are the only tensors the sweep touches. ---------
        # The block table is `context_length / page_size` wide, NOT
        # `current_seq_len / page_size`: the server allocates it once from
        # --context-length.  Getting this wrong is invisible in the totals and
        # very visible in the inventory -- it changes the recorded input shape of
        # SparseFlashAttention, LightningIndexer AND Index at once, which is
        # three of the four biggest operators in the layer.
        self.context_len = context_len
        self.max_pages = -(-context_len // PAGE)
        self.block_tables = torch.zeros(1, self.max_pages, dtype=torch.int32, device=DEV)
        self.pooled_bt = torch.zeros(
            1, self.max_pages // IDX_KPOOL, dtype=torch.int32, device=DEV
        )
        # dtypes follow ForwardBatch: seq_lens int32, positions/out_cache_loc/
        # req_pool_indices int64.  `pos < seq_lens` therefore mixes int64 with
        # int32 and emits a Cast -- that Cast is real, it is in the served run.
        self.seq_lens = torch.zeros(1, dtype=torch.int32, device=DEV)
        self.pool_lens = torch.zeros(1, dtype=torch.int32, device=DEV)
        self.positions = torch.zeros(1, dtype=torch.int64, device=DEV)
        self.out_cache_loc = torch.zeros(1, dtype=torch.int64, device=DEV)
        self.cu_seqlens_q = torch.ones(1, dtype=torch.int32, device=DEV)
        self.req_idx = torch.zeros(1, dtype=torch.int64, device=DEV)
        # `_decode_arange`: cached, because rebuilding it per layer would put a
        # Range kernel in every DSA layer (the served tree hoisted these in
        # c69883df97).
        self.rows = torch.arange(1, device=DEV)
        self.pool_arange = torch.arange(IDX_KPOOL, device=DEV)
        self.scratch_loc = IDX_TOKENS - 1
        #: a python int, not a tensor -- `torch.where(cond, t, python_int)` is
        #: SelectV2 "1;1;" while `torch.where(cond, t, tensor)` is "1;1;1", and
        #: the served run has one of each.
        self.tail_scratch_row = REQ_POOL_ROWS - 1

        self.x = rnd(1, HIDDEN)
        self.set_seq_len(seq_len)

    # -- the only thing the sequence-length sweep changes ---------------------
    def set_seq_len(self, n: int):
        """Point the graph's metadata at a sequence of length `n`.

        Writes in place, so a captured graph picks the new values up on its next
        replay.  No shape moves: the pools stay at their full deployed size.
        """
        pages = -(-n // PAGE)
        if pages > self.max_pages:
            raise ValueError(
                f"n={n} needs {pages} pages but the block table holds "
                f"{self.max_pages} (context_len={self.context_len}).  Build the "
                "layer with a bigger context_len."
            )
        if pages > KV_PAGES:
            raise ValueError(f"n={n} needs {pages} pages, the KV pool holds {KV_PAGES}")
        bt = torch.zeros(1, self.max_pages, dtype=torch.int32)
        bt[0, :pages] = torch.arange(pages, dtype=torch.int32)
        self.block_tables.copy_(bt.to(DEV))
        self.pooled_bt.copy_(self.block_tables[..., ::IDX_KPOOL])
        self.seq_lens.fill_(n)
        self.pool_lens.fill_(n // IDX_KPOOL)
        self.positions.fill_(n - 1)
        self.out_cache_loc.fill_((n - 1) % KV_TOKENS)
        self.seq_len = n

    def body(self):
        """One decode step of one DSA layer.

        No `.item()`, no `int(tensor)`, no `.nonzero()`, no `bool(t.any())`.  The
        served code went out of its way to eliminate all four so that decode
        could be captured; a benchmark that reintroduced one would not capture.
        """
        # ---------------- A. latents -------------------------------------
        qa, qa_s = torch.ops.npu.npu_dynamic_quant(self.x)
        fused = torch.ops.npu.npu_quant_matmul(
            qa, self.w_qkv_a, self.s_qkv_a,
            pertoken_scale=qa_s.flatten(), bias=None, output_dtype=BF,
        )  # [1, 2048]
        q_lora, latent = fused.split([Q_LORA, KV_LORA], dim=-1)
        q_lora = torch_npu.npu_rms_norm(q_lora, self.q_a_norm_w, RMS_EPS)[0]
        k_nope = torch_npu.npu_rms_norm(
            latent.unsqueeze(1), self.kv_a_norm_w, RMS_EPS
        )[0]  # [1,1,512]

        qb, qb_s = torch.ops.npu.npu_dynamic_quant(q_lora)
        q = torch.ops.npu.npu_quant_matmul(
            qb, self.w_q_b, self.s_q_b,
            pertoken_scale=qb_s.flatten(), bias=None, output_dtype=BF,
        ).view(-1, N_HEADS, QK_HEAD)
        # W^UK absorb: q_nope [1,64,256] -> [1,64,512], the SFA query
        q_absorbed = torch.bmm(q.transpose(0, 1), self.w_kc.transpose(-1, -2)).transpose(0, 1)

        # ---------------- B. indexer -------------------------------------
        query = F.linear(q_lora, self.wq_b).view(-1, IDX_HEADS, IDX_DIM)  # [1,32,128]
        key = F.linear(self.x, self.wk)  # [1,128]
        # The round trip through fp32 is explicit and it is load bearing: it is
        # two Cast "1,128" per layer in the served inventory (5 total, and the
        # bench had 3 until this line matched).  Handing bf16 straight to
        # F.layer_norm with fp32 weights emits no Cast at all -- measured.
        key = F.layer_norm(
            key.float(), (IDX_DIM,), self.k_norm_w, self.k_norm_b, RMS_EPS
        ).to(BF)
        query = hadamard_transform_npu(query)
        gate_score = F.linear(self.x, self.gate_w)  # [1,128]

        # -- kpool_decode_update_index_cache, transcribed --------------------
        # Expression for expression from
        # memory_pool_npu.py::kpool_decode_update_index_cache.  The operand
        # *kinds* matter as much as the arithmetic: a python int and a 1-element
        # tensor produce different kernels (SelectV2 "1;1;" vs "1;1;1"), and a
        # scalar comparison and a tensor comparison produce different kernels
        # (Less "1;" vs "1;1").  The served inventory pins all of them.
        req, pos = self.req_idx, self.positions
        valid = (
            (req >= 0)
            & (req < REQ_POOL_ROWS)
            & (self.out_cache_loc != 0)
            & (pos >= 0)
            & (pos < self.seq_lens)
        )
        safe_req = req.clamp(0, REQ_POOL_ROWS - 1)
        safe_pos = pos.clamp(min=0)

        # `safe_pos % pool_size` is written three times in the source (closing,
        # start, and the ring scatter, whose tail_width also equals pool_size).
        # It is one value; computing it once is what the served graph ends up
        # doing, and recomputing it would add three FloorMod + three BroadcastTo
        # per layer that the served inventory does not have.
        pos_mod = safe_pos % IDX_KPOOL
        closing = (valid & (pos_mod == IDX_KPOOL - 1)).unsqueeze(1)

        start = safe_pos - pos_mod
        phys = (start.unsqueeze(1) + self.pool_arange.unsqueeze(0)) % IDX_KPOOL
        flat_k = self.tail_k.view(-1, IDX_DIM)
        flat_s = self.tail_s.view(-1, IDX_DIM)
        gather = (safe_req.unsqueeze(1) * IDX_KPOOL + phys).reshape(-1)
        slot_k = flat_k.index_select(0, gather).view(1, IDX_KPOOL, IDX_DIM)
        slot_s = flat_s.index_select(0, gather).view(1, IDX_KPOOL, IDX_DIM)
        slot_k[:, IDX_KPOOL - 1] = key
        slot_s[:, IDX_KPOOL - 1] = gate_score

        pool_id = safe_pos // IDX_KPOOL
        page_col = ((pool_id // SLOTS_PER_PAGE) * IDX_KPOOL).clamp(
            0, self.block_tables.shape[1] - 1
        )
        page = self.block_tables[self.rows, page_col].long()
        loc = torch.where(
            closing.squeeze(1),
            page * PAGE + pool_id % SLOTS_PER_PAGE,
            torch.full_like(page, self.scratch_loc),   # tensor -> SelectV2 "1;1;1"
        )
        torch_npu.npu_scatter_nd_update_(
            self.index_k.view(-1, 1, IDX_DIM),
            loc.reshape(-1, 1).long(),
            compress_pool_bf16(slot_k, slot_s, self.ape).reshape(-1, 1, IDX_DIM),
        )
        # python int, not a tensor -> SelectV2 "1;1;" (+ a Fill for the scalar)
        dest = torch.where(valid, safe_req, self.tail_scratch_row)
        scatter = (dest * IDX_KPOOL + pos_mod).reshape(-1, 1)
        torch_npu.npu_scatter_nd_update_(
            flat_k.view(-1, 1, IDX_DIM), scatter, key.reshape(-1, 1, IDX_DIM)
        )
        torch_npu.npu_scatter_nd_update_(
            flat_s.view(-1, 1, IDX_DIM), scatter, gate_score.reshape(-1, 1, IDX_DIM)
        )

        # -- score every closed pool.  THIS is the O(n) term. ----------------
        # _kpool_head_gate_npu multiplies by two scalars in sequence, not by one
        # pre-folded constant -- the served inventory has Mul "1,32;" twice.
        weights = (
            F.linear(self.x.float(), self.wproj)
            * IDX_HEADS**-0.5
            * IDX_DIM**-0.5
        ).contiguous()
        selected = torch_npu.npu_lightning_indexer(
            query=query.contiguous(),
            key=self.index_k,
            weights=weights,
            actual_seq_lengths_query=self.cu_seqlens_q,
            actual_seq_lengths_key=self.pool_lens,
            block_table=self.pooled_bt,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=GROUP_TOPK,
            sparse_mode=0,
        )[0].squeeze(1)  # [1, 512] int32 pool ids

        # -- 512 pools -> 2048 tokens, then 3 tail columns -> [1,2051] --------
        expanded = expand_pooled_groups_to_topk(
            selected.contiguous(),
            selected.ge(selected.new_zeros(())),
            topk=IDX_TOPK,
            pool_size=IDX_KPOOL,
        )
        topk_indices = append_kpool_tail_to_topk(
            expanded,
            seq_lens=self.seq_lens,
            pool_lens=self.pool_lens,
            pool_size=IDX_KPOOL,
        )

        # ---------------- C. sparse attention ----------------------------
        self.kv_buffer[self.out_cache_loc] = k_nope  # IndexPutV2
        k_pa = self.kv_buffer.view(KV_PAGES, PAGE, 1, KV_LORA)
        attn = torch_npu.npu_sparse_flash_attention(
            query=q_absorbed.contiguous(),
            key=k_pa,
            value=k_pa,
            query_rope=self.q_rope,
            key_rope=self.k_rope,
            sparse_indices=topk_indices.unsqueeze(-2),  # [1,1,2051]
            scale_value=self.scaling,
            actual_seq_lengths_query=self.cu_seqlens_q,
            actual_seq_lengths_kv=self.seq_lens,
            block_table=self.block_tables,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )[0].view(-1, N_HEADS, KV_LORA)

        # ---------------- D. W^UV absorb + o_proj ------------------------
        out = torch.empty(attn.shape[0], N_HEADS, V_HEAD, dtype=attn.dtype, device=DEV)
        torch.ops.npu.batch_matmul_transpose(attn.contiguous(), self.w_vc, out)
        oq, oq_s = torch.ops.npu.npu_dynamic_quant(out.reshape(-1, QB_OUT))
        return torch.ops.npu.npu_quant_matmul(
            oq, self.w_o, self.s_o,
            pertoken_scale=oq_s.flatten(), bias=None, output_dtype=BF,
        )


# ---------------------------------------------------------------------------
# measurement primitives  (identical to bench_kda_layer.py -- kept duplicated
# so each script stands alone)
# ---------------------------------------------------------------------------
def capture(fn, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    g = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(g):
        fn()
    torch.npu.synchronize()
    g.replay()
    torch.npu.synchronize()

    def replay(k=1):
        for _ in range(k):
            g.replay()

    return replay, g


MARKER_NUMEL = 1357

#: only kernels at least this long take part in the dispersion test; below it
#: the p90/p50 ratio measures the timer, not the die.
DISPERSION_FLOOR_US = 5.0


def profile(cases, nrep=30, tag="p"):
    """Profile each (label, callable) in one session, split by a marker kernel.

    Splitting the row list into equal blocks is wrong the moment two cases differ
    in launch count, and it fails silently by attributing one case's kernels to
    its neighbour.  DSA cases differ (the indexer's launch count is not constant
    across lengths), so the marker is not optional here.
    """
    for _, fn in cases:
        fn(3)
    torch.npu.synchronize()
    # With one case there is nothing to split, and the marker would show up in
    # the raw kernel_details.csv that regress_against_network.py reads -- as an
    # op group the network does not have.  Single-case profiles stay clean.
    use_marker = len(cases) > 1
    mk = torch.zeros(MARKER_NUMEL, device=DEV)

    def mark():
        if use_marker:
            mk.add_(1.0)

    mark()
    torch.npu.synchronize()

    out = os.path.join(PROF_ROOT, tag)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    exp = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
    )
    labels, walls = [], []
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=True,
        experimental_config=exp,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out),
    ) as p:
        for lab, fn in cases:
            mark()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            fn(nrep)
            torch.npu.synchronize()
            walls.append((lab, (time.perf_counter() - t0) * 1e6 / nrep))
            labels.append(lab)
        mark()
        torch.npu.synchronize()
        p.step()

    hits = sorted(glob.glob(os.path.join(out, "**", "kernel_details.csv"), recursive=True))
    if not hits:
        raise SystemExit(f"profiler wrote no kernel_details.csv under {out}")
    rows = list(csv.DictReader(open(hits[-1], newline="")))
    rows.sort(key=lambda r: float(r["Start Time(us)"].strip()))
    tag_m = str(MARKER_NUMEL)
    if not use_marker:
        return [(labels[0], rows)], walls
    cuts = [i for i, r in enumerate(rows) if tag_m in r["Input Shapes"]]
    if len(cuts) != len(cases) + 1:
        raise SystemExit(
            f"marker kernel appeared {len(cuts)} times, expected {len(cases)+1}. "
            "Case attribution would be wrong, so refusing to report numbers."
        )
    # Drop the markers themselves: they are measurement scaffolding, not part of
    # the layer, and leaving them in puts an `Add "1357;"` row in every inventory.
    blocks = [
        [r for r in rows[cuts[i] + 1 : cuts[i + 1]] if tag_m not in r["Input Shapes"]]
        for i in range(len(cases))
    ]
    return list(zip(labels, blocks)), walls


def agg_rows(rows, nrep):
    agg = collections.defaultdict(list)
    for r in rows:
        key = (r["Type"] or r["Name"], r["Input Shapes"].strip('" '))
        agg[key].append(float(r["Duration(us)"]))
    return agg


def total_us(agg, nrep):
    return sum(statistics.median(v) * len(v) / nrep for v in agg.values())


def tabulate(rows, nrep, layers=DSA_LAYERS, top=None):
    agg = agg_rows(rows, nrep)
    items = sorted(agg.items(), key=lambda kv: -statistics.median(kv[1]) * len(kv[1]))
    print(
        f"  {'operator':<26} {'shapes':<34} {'n/l':>4} "
        f"{'p50':>8} {'p90':>8} {'max':>8} {'ref':>7} {'p50/ref':>8}"
    )
    total = 0.0
    shown = 0
    tail = 0.0
    for (typ, shapes), v in items:
        v = sorted(v)
        n_per = len(v) / nrep
        p50 = statistics.median(v)
        p90 = v[min(len(v) - 1, int(len(v) * 0.9))]
        total += p50 * n_per
        ref = REF.get((typ, shapes)) or REF.get((typ, None))
        if top is not None and shown >= top:
            tail += p50 * n_per
            continue
        shown += 1
        rs = f"{ref:7.1f}" if ref else "      -"
        rr = f"{p50/ref:8.2f}x" if ref else "        "
        print(
            f"  {typ[:26]:<26} {shapes[:34]:<34} {n_per:4.1f} "
            f"{p50:8.2f} {p90:8.2f} {v[-1]:8.2f} {rs} {rr}"
        )
    if tail:
        print(f"  {'... %d smaller groups' % (len(items)-shown):<26} "
              f"{'':<34} {'':>4} {tail:8.2f}")
    nk = sum(len(v) for v in agg.values()) / nrep
    print(f"  {'-'*26} {'-'*34} {'-'*4} {'-'*8}")
    print(
        f"  {'per layer':<26} {'':<34} {nk:4.0f} {total:8.2f} us"
        f"   (config I: {REF_KERNELS_PER_LAYER} kernels, {REF_LAYER_US:.1f} us)"
    )
    print(
        f"  {'x %d layers' % layers:<26} {'':<34} {nk*layers:4.0f} "
        f"{total*layers/1000:8.3f} ms   target {REF_STEP_MS:.3f} ms "
        f"({100*(total*layers/1000/REF_STEP_MS-1):+.1f}%)"
    )
    return total, agg


def contention_warning(agg, refs):
    """Two different failures, two different messages.  They are not the same thing.

    * **dispersion** (p90/p50) over the reference shapes -> the die is SHARED.
      Interference is bursty, so it shows up in the tail first.  This is the
      test that actually detects a co-tenant.
    * **median** off against a fixed shape, with the tail clean -> the die is
      fine and something about the RUN does not match the reference: the
      sequence length, the context length, the pool size, a dtype.

    Getting this backwards costs real time.  An earlier version of this harness
    led with the median, and on a verifiably idle die (3136 MiB used, no other
    process) it printed "THIS DIE IS SHARED" because `--ref-seq` was 512 instead
    of the aligned 256 -- IndexPutV2's median was 1.62x while its dispersion was
    1.01x, i.e. perfectly clean.  The README's own rule ("interference lives in
    the tail; a median alone will miss it") applies in both directions.
    """
    hits = []
    for key, v in agg.items():
        ref = refs.get(key) or refs.get((key[0], None))
        if not ref or not v:
            continue
        v = sorted(v)
        p50 = statistics.median(v)
        p90 = v[min(len(v) - 1, int(len(v) * 0.9))]
        hits.append((key, p50 / ref, p90 / p50 if p50 else 1.0, p50))
    if not hits:
        return
    worst_ratio = max(h[1] for h in hits)
    # Dispersion is only meaningful above the timer's own granularity.  A 1.3 us
    # kernel with 0.4 us of jitter reads as p90/p50 = 1.30 on a verifiably idle
    # die -- measured, in the `family` section, where it false-fired.  Judge
    # sharing on kernels big enough for the ratio to mean something.
    big = [h for h in hits if h[3] >= DISPERSION_FLOOR_US]
    worst_spread = max((h[2] for h in big), default=1.0)

    def show(n=4):
        for key, r, sp, _p in sorted(hits, key=lambda h: -max(h[1], h[2]))[:n]:
            print(f"       {key[0][:24]:<24} {str(key[1])[:26]:<26} "
                  f"p50/ref {r:5.2f}x  p90/p50 {sp:5.2f}x")

    if worst_spread > 1.25:
        print(
            f"\n  !! THIS DIE IS SHARED -- do not report these as operator costs.\n"
            f"     worst p90/p50 dispersion : {worst_spread:.2f}x   (over kernels >= "
            f"{DISPERSION_FLOOR_US:.0f} us; clean die: <= 1.10x)\n"
            f"     worst median vs config I : {worst_ratio:.2f}x"
        )
        show()
        print(
            "     Run `npu-smi info`.  A busy die inflates everything by roughly 1.7x,\n"
            "     which lands inside the '20% is fine / 2x means wrong shape' gap."
        )
    elif worst_ratio > 1.25:
        print(
            f"\n  ?  MEDIAN OFF BUT TAIL CLEAN -- this is most likely a PARAMETER\n"
            f"     mismatch, not a busy die.\n"
            f"     worst median vs config I : {worst_ratio:.2f}x\n"
            f"     worst p90/p50 dispersion : {worst_spread:.2f}x   (clean -- nobody else\n"
            f"                                 is on this die)"
        )
        show()
        print(
            "     Check --ref-seq (config I is ~256), --context-len (32768), and that\n"
            "     the INT8 weights really are NZ.  Only the dispersion test detects a\n"
            "     co-tenant; a median on its own cannot tell the two apart."
        )


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------
def sec_layer(args):
    print("\n=== layer: one DSA layer, captured and replayed ===")
    lay = DSALayer(seq_len=args.ref_seq, context_len=args.context_len)
    replay, _g = capture(lay.body, warmup=args.warmup)
    blocks, walls = profile([("dsa", replay)], nrep=args.reps, tag="layer")
    rows = blocks[0][1]
    print(
        f"  bs=1, T=1, seq_len {args.ref_seq}, {N_HEADS} heads, kv_lora {KV_LORA}, "
        f"q_lora {Q_LORA}\n  indexer {IDX_HEADS}x{IDX_DIM}, topk {IDX_TOPK}, "
        f"kpool {IDX_KPOOL}, page {PAGE}, pool {KV_TOKENS} tokens"
    )
    print(f"  graph replay, {args.reps} reps after {args.warmup} warmups\n")
    total, agg = tabulate(rows, args.reps, top=args.top)
    print(f"  wall clock per replay (host, reference only): {walls[0][1]:.1f} us")
    contention_warning(agg, REF)


def sec_sweep(args):
    print("\n=== sweep: DSA against sequence length -- the whole point ===")
    print("  index_topk = 2048 caps the attention itself.  The indexer does not")
    print("  get a cap: it scores n/index_kpool = n/4 pools, so it is O(n).")
    print("  Nothing below changes a tensor shape; only the int32 length")
    print("  metadata the captured graph reads changes, which is how the served")
    print("  graph handles varying lengths too.\n")

    ns = [n for n in args.seq_lens if -(-n // PAGE) <= KV_PAGES]
    dropped = [n for n in args.seq_lens if n not in ns]
    if dropped:
        print(f"  ! skipping {dropped}: longer than the {KV_TOKENS}-token KV pool\n")
    # The block table has to hold the longest length in the sweep, so it is
    # wider here than in the `layer` section.  That is faithful -- a server with
    # a 1M context length allocates a 16384-entry block table whatever the
    # current sequence length is -- but it does change the recorded input shape
    # of SparseFlashAttention / LightningIndexer / Index, so the sweep's
    # inventory is NOT directly comparable to config I.  Use the `layer`
    # section for the inventory check.
    lay = DSALayer(seq_len=ns[0], context_len=max(max(ns), args.context_len))
    replay, _g = capture(lay.body, warmup=args.warmup)

    # One profiler session per length.  Rewriting the metadata has to happen
    # OUTSIDE the timed window -- `set_seq_len` launches its own kernels, and
    # they would land in whichever case happened to follow them.
    per_n, aggs = {}, {}
    for n in ns:
        lay.set_seq_len(n)
        torch.npu.synchronize()
        blocks, _ = profile([(f"n={n}", replay)], nrep=args.reps, tag=f"sweep{n}")
        a = agg_rows(blocks[0][1], args.reps)
        aggs[n] = a
        per_n[n] = total_us(a, args.reps)

    print(f"  {'seq len':>10} {'us/layer':>10} {'x11 (ms)':>10} {'vs 1st':>9} {'kernels':>8}")
    first = per_n[ns[0]]
    for n in ns:
        nk = sum(len(v) for v in aggs[n].values()) / args.reps
        print(
            f"  {n:>10} {per_n[n]:10.2f} {per_n[n]*DSA_LAYERS/1000:10.3f} "
            f"{100*(per_n[n]/first-1):+8.1f}% {nk:8.0f}"
        )

    # Which operators actually move?
    print("\n  per-operator, only the ones that move by more than 2 us:")
    keys = set().union(*[set(a) for a in aggs.values()])
    rowsout = []
    for k in keys:
        vals = [
            statistics.median(aggs[n][k]) * len(aggs[n][k]) / args.reps
            if k in aggs[n] else 0.0
            for n in ns
        ]
        if max(vals) - min(vals) > 2.0:
            rowsout.append((max(vals) - min(vals), k, vals))
    rowsout.sort(reverse=True)
    hdr = "  " + f"{'operator':<26} {'shapes':<22}" + "".join(f"{n:>11}" for n in ns)
    print(hdr)
    for _, k, vals in rowsout:
        print(f"  {k[0][:26]:<26} {k[1][:22]:<22}" + "".join(f"{v:11.2f}" for v in vals))
    if not rowsout:
        print("  (none -- report this, it would contradict the O(n) indexer)")

    # Fit only where the attention has already saturated.  Below ~4k the curve
    # is dominated by SparseFlashAttention still filling up to its topk cap, and
    # fitting a line through that mixes two different mechanisms into one slope.
    fit_ns = [n for n in ns if n >= 4096]
    if len(fit_ns) >= 3:
        import statistics as st

        xs = fit_ns
        ys = [per_n[n] * DSA_LAYERS / 1000 for n in xs]
        mx, my = st.mean(xs), st.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        slope = num / den if den else 0.0
        print(
            f"\n  least squares over n >= 4096 (attention saturated), "
            f"{len(xs)} points:\n"
            f"    DSA family = {my - slope*mx:.3f} + {slope:.3e} x n   ms/step"
        )
        print(
            "  Two mechanisms, and the sweep separates them:\n"
            "    * SparseFlashAttention grows only until it hits index_topk=2048,\n"
            "      then is flat forever.  That is the cap working.\n"
            "    * LightningIndexer is the O(n) term: it scores n/4 pools and\n"
            "      nothing caps that.\n"
            "  For scale, the multi-card TP8 line fitted the WHOLE decode step at\n"
            "  27.3 + 5.4e-6 x n ms/token over 32k..1M.  This slope is the TP1 DSA\n"
            "  family only -- the two are not the same quantity and the ratio\n"
            "  between them is not evidence of anything on its own."
        )


def sec_refs(args):
    print("\n=== refs: which DSA operators are already at their floor? ===")
    print("  Same four controls as the KDA bench: a stock kernel of the same")
    print("  shape, a pure read of the same bytes, a read+write, and a trivial")
    print("  kernel.  Plus the pool reads, because DSA's cost is pool-shaped.\n")
    lay = DSALayer(seq_len=args.ref_seq, context_len=args.context_len)
    tiny = torch.randn(1, device=DEV)
    x16 = torch.zeros(1, QB_OUT, device=DEV, dtype=BF)
    qi, qs = torch.ops.npu.npu_dynamic_quant(x16)
    qi15, qs15 = torch.ops.npu.npu_dynamic_quant(torch.zeros(1, Q_LORA, device=DEV, dtype=BF))
    qk = torch.zeros(N_HEADS, 1, QK_HEAD, device=DEV, dtype=BF)
    k_pa = lay.kv_buffer.view(KV_PAGES, PAGE, 1, KV_LORA)

    def loop(f):
        return lambda k=1: [f() for _ in range(k)] and None

    cases = [
        ("o_proj  INT8 16384->4096", loop(lambda: torch.ops.npu.npu_quant_matmul(
            qi, lay.w_o, lay.s_o, pertoken_scale=qs.flatten(), output_dtype=BF))),
        ("  same bytes, pure read (int8 w)", loop(lambda: lay.w_o.float().sum())),
        ("q_b_proj INT8 1536->16384", loop(lambda: torch.ops.npu.npu_quant_matmul(
            qi15, lay.w_q_b, lay.s_q_b, pertoken_scale=qs15.flatten(), output_dtype=BF))),
        ("w_kc bmm [64,1,256]x[64,512,256]", loop(lambda: torch.bmm(
            qk, lay.w_kc.transpose(-1, -2)))),
        ("  same bytes, pure read w_kc.sum()", loop(lambda: lay.w_kc.sum())),
        ("wq_b   bf16 1536->4096", loop(lambda: F.linear(
            torch.zeros(1, Q_LORA, device=DEV, dtype=BF), lay.wq_b))),
        ("  same bytes, pure read", loop(lambda: lay.wq_b.sum())),
        ("KV pool  read all 1.19 GiB", loop(lambda: lay.kv_buffer.sum())),
        ("KV pool  read 2048 tokens (topk)", loop(lambda: lay.kv_buffer[:IDX_TOPK].sum())),
        ("index pool read all 0.30 GiB", loop(lambda: lay.index_k.sum())),
        ("index pool read n/4 at 32k", loop(lambda: lay.index_k.view(-1, IDX_DIM)[:8192].sum())),
        ("zero k_rope read 0.15 GiB", loop(lambda: lay.k_rope.sum())),
        ("trivial  torch.add on 1 element", loop(lambda: torch.add(tiny, 1.0))),
    ]
    blocks, _ = profile(cases, nrep=args.reps, tag="refs")

    def mib(t):
        return t.numel() * t.element_size() / 2**20

    mibs = [
        mib(lay.w_o), mib(lay.w_o) * 5, mib(lay.w_q_b),
        mib(lay.w_kc), mib(lay.w_kc), mib(lay.wq_b), mib(lay.wq_b),
        mib(lay.kv_buffer), IDX_TOPK * KV_LORA * 2 / 2**20,
        mib(lay.index_k), 8192 * IDX_DIM * 2 / 2**20,
        mib(lay.k_rope), 0.0,
    ]
    print(f"  {'case':<38} {'MiB':>9} {'floor':>8} {'p50':>9} {'p90':>9} {'/floor':>8}")
    for (lab, rows), m in zip(blocks, mibs):
        d = sorted(float(r["Duration(us)"]) for r in rows)
        if not d:
            print(f"  {lab:<38} (no kernels recorded)")
            continue
        p50 = statistics.median(d)
        p90 = d[min(len(d) - 1, int(len(d) * 0.9))]
        floor = m * 2**20 / (BW_GBPS * 1e9) * 1e6 if m else 0.0
        fs = f"{floor:8.2f}" if floor else "       -"
        rr = f"{p50/floor:7.2f}x" if floor else "        "
        print(f"  {lab:<38} {m:9.2f} {fs} {p50:9.2f} {p90:9.2f} {rr}")
    print(
        f"\n  L2 here is ~{L2_MIB:.0f} MiB.  Anything smaller than that, measured by\n"
        "  hammering the same buffer, is L2-warm and will beat its own HBM floor;\n"
        "  the served step is not warm.  Treat any ratio below 1.0 as a warning."
    )


def sec_family(args):
    n = args.family_layers
    print(f"\n=== family: {n} distinct DSA layers chained in one graph ===")
    need = n * (KV_TOKENS * KV_LORA + IDX_TOKENS * IDX_DIM) * 2 / 2**30
    print(f"  allocating ~{need:.1f} GB of pools ({n} of the {DSA_LAYERS} layers)")
    share = {}
    try:
        lays = [
            DSALayer(seq_len=args.ref_seq, context_len=args.context_len,
                     share=share, seed=i)
            for i in range(n)
        ]
    except RuntimeError as exc:
        print(f"  ! could not allocate ({str(exc).splitlines()[0][:90]})")
        print("    lower --family-layers, or wait for the die to free up.")
        return

    def chain():
        h = lays[0].x
        for lay in lays:
            lay.x.copy_(h)
            h = lay.body()
        return h

    replay, _g = capture(chain, warmup=max(3, args.warmup // 3))
    nrep = max(10, args.reps // 3)
    blocks, _ = profile([("family", replay)], nrep=nrep, tag="family")
    agg = agg_rows(blocks[0][1], nrep)
    tot = total_us(agg, nrep)
    scaled = tot * DSA_LAYERS / n
    nk = sum(len(v) for v in agg.values()) / nrep
    print(
        f"  {n} layers = {tot/1000:.3f} ms, {nk:.0f} kernels;  scaled to "
        f"{DSA_LAYERS} layers = {scaled/1000:.3f} ms   target {REF_STEP_MS:.3f} ms "
        f"({100*(scaled/1000/REF_STEP_MS-1):+.1f}%)"
    )
    # No contention check here.  This section pools N *different* layers into one
    # distribution per operator, so its p90/p50 measures the spread between
    # layers -- which is real -- and not interference.  Measured on a verifiably
    # idle die (2877 MiB, 0% AI core): it reported 1.26x and "THIS DIE IS
    # SHARED".  The test is only valid where every sample is the same work,
    # which is the `layer` section.
    print("  (no die-sharing check here: see the `layer` section -- pooling "
          "distinct layers\n   makes the dispersion test meaningless)")


SECTIONS = {
    "layer": sec_layer,
    "sweep": sec_sweep,
    "refs": sec_refs,
    "family": sec_family,
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sections", default="layer,sweep,refs")
    ap.add_argument("--seq-lens", default="128,512,1024,4096,32768,131072,1048576")
    ap.add_argument("--context-len", type=int, default=32768,
                    help="served --context-length.  Sets the block table width "
                         "(context_len/64 entries), which is part of the recorded "
                         "input shape of SparseFlashAttention, LightningIndexer "
                         "and Index.  Config I ran 32768 -> a [1,512] table.")
    ap.add_argument("--ref-seq", type=int, default=256,
                    help="length the `layer` and `refs` sections use.  Config I "
                         "was profiled on a 13-token prompt (see "
                         "tools/profile_server_decode.py --prompt-tokens 13); its "
                         "SparseFlashAttention median of 32.08 us sits between "
                         "this bench's n=128 and n=256, so 256 is the aligned "
                         "point.  The reference DSA numbers are SHORT CONTEXT "
                         "numbers.")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--top", type=int, default=22,
                    help="how many operator rows to print before folding the tail")
    ap.add_argument("--family-layers", type=int, default=DSA_LAYERS)
    args = ap.parse_args()
    args.seq_lens = [int(v) for v in args.seq_lens.split(",")]
    want = list(SECTIONS) if args.sections == "all" else args.sections.split(",")
    bad = [s for s in want if s not in SECTIONS]
    if bad:
        raise SystemExit(f"unknown section(s) {bad}; pick from {list(SECTIONS)}")

    print(f"device        : {torch.npu.get_device_name(0)}")
    print(f"visible dies  : {os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '(all)')}")
    print(f"torch {torch.__version__}  torch_npu {torch_npu.__version__}")
    print(
        f"target        : {REF_LAYER_US:.1f} us/layer x {DSA_LAYERS} layers "
        f"= {REF_STEP_MS} ms/step, {REF_KERNELS_PER_LAYER} kernels/layer "
        "(config I, bs=1, graph on, one A3 die)"
    )
    print("mode          : NPU graph capture + replay; numbers are profiler device time")
    for s in want:
        SECTIONS[s](args)
    print(f"\nprofiler output under {PROF_ROOT} (delete when done)")


if __name__ == "__main__":
    main()
