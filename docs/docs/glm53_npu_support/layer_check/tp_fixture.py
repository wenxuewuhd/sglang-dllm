"""Shared fixture: build a real GLM-5.3-Flash layer-3 DSA attention module,
a real AscendAttnBackend, real pools -- on one NPU die, no server.

Only ModelRunner and ForwardBatch are faked, and only in the fields the real
code reads.
"""
from __future__ import annotations

import json
import os
import types
from pathlib import Path

import torch

MODEL = "/mnt/workspace/models/GLM-5.3-Flash-BF16"
CKPT = "model.language_model.layers.{l}."
DEV = "npu"


# ---------------------------------------------------------------- weights
class Shards:
    def __init__(self, model_dir=MODEL):
        from safetensors import safe_open

        self.dir = Path(model_dir)
        self.map = json.loads(
            (self.dir / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.h = {}
        self._safe_open = safe_open

    def get(self, name):
        shard = self.map[name]
        if shard not in self.h:
            self.h[shard] = self._safe_open(str(self.dir / shard), framework="pt")
        return self.h[shard].get_tensor(name)


# ---------------------------------------------------------------- env
def boot(tp_port=None):
    import random

    if tp_port is None:
        tp_port = random.randint(29600, 30600)
    torch.set_grad_enabled(False)
    torch.npu.set_device(0)
    from sglang.srt import runtime_context as rc
    from sglang.srt.server_args import ServerArgs

    # Mirrors $ROOT/run/launch_glm_bf16.sh, except tp_size: this is one process
    # on one die, so the process-level TP stays 1 and the *attention* TP shape is
    # imposed with tp_override(16) where the module and backend read it.
    sa = ServerArgs(
        model_path=MODEL,
        device="npu",
        tp_size=1,
        page_size=64,
        attention_backend="ascend",
        trust_remote_code=True,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        context_length=32768,
        max_running_requests=16,
        mem_fraction_static=0.85,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
    )
    rc.publish(sa, role="scheduler")
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(tp_port))
    init_distributed_environment(
        backend="hccl",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{tp_port}",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    from sglang.srt.configs.model_config import ModelConfig

    mc = ModelConfig.from_server_args(sa)
    return sa, mc


# ---------------------------------------------------------------- pools
def make_pools(mc, layer, size, max_running_requests=8, max_ctx=None, nreq=4):
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUDSATokenToKVPool
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

    kv = NPUDSATokenToKVPool(
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
    req = ReqToTokenPool(
        size=max(nreq, 4),
        max_context_len=max_ctx or (size + 64),
        device=DEV,
        enable_memory_saver=False,
    )
    return kv, req


def page_map(n_pages, seed=1):
    """Physical page id per logical page.  Shuffled so any page-arithmetic bug
    shows up as garbage instead of accidentally matching.  Page 0 is left
    unused (out_cache_loc == 0 is the decode-invalid sentinel)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_pages, generator=g).to(torch.int32) + 1


def fill_req_to_token(req_pool, req_idx, pmap, n_tokens):
    t = torch.arange(n_tokens, dtype=torch.int64)
    slots = pmap[t // 64].to(torch.int64) * 64 + (t % 64)
    req_pool.req_to_token[req_idx, :n_tokens] = slots.to(torch.int32).to(DEV)
    return slots


# ---------------------------------------------------------------- backend
def tp_override(attn_tp_size, attn_tp_rank=0):
    """Pretend to be rank `attn_tp_rank` of an `attn_tp_size`-way attention TP
    group inside one process.  DeepseekV2AttentionMLA and AscendAttnBackend both
    read get_parallel().attn_tp_{size,rank} to size their local head count and to
    hand tp_rank/tp_size to Column/RowParallelLinear, so overriding those two is
    what makes a single die build the shape it would build under --tp-size 16."""
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().override(
        attn_tp_size=attn_tp_size, attn_tp_rank=attn_tp_rank
    )


def make_backend(mc, kv_pool, req_pool, sa):
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )

    mr = types.SimpleNamespace()
    mr.device = DEV
    mr.page_size = 64
    mr.model_config = mc
    mr.dtype = mc.dtype
    mr.req_to_token_pool = req_pool
    mr.token_to_kv_pool = kv_pool
    mr.is_draft_worker = False
    mr.spec_algorithm = None
    mr.is_hybrid_swa = False
    mr.sliding_window_size = None
    mr.server_args = sa
    mr.ps = types.SimpleNamespace(attn_cp_size=1)
    return AscendAttnBackend(mr)


# ---------------------------------------------------------------- module
def make_attn(mc, layer, sh: Shards, tp_size=1, tp_rank=0):
    """Real DeepseekV2AttentionMLA, built the way Glm5NextDecoderLayer builds it,
    with layer `layer`'s real weights and the real post-load w_kc/w_vc absorb.

    With tp_size>1 the module is the rank-`tp_rank` shard: q_b_proj / kv_b_proj
    are column-parallel (head-sharded), o_proj is row-parallel (input-sharded),
    fused_qkv_a_proj_with_mqa and the whole indexer are replicated -- the indexer
    keeps all 32 heads on every rank because IndexerKPool uses ReplicatedLinear
    and never divides index_n_heads."""
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.model_loader.utils import set_default_torch_dtype

    cfg = mc.hf_text_config
    with set_default_torch_dtype(torch.bfloat16), tp_override(tp_size, tp_rank):
        m = DeepseekV2AttentionMLA(
            config=cfg,
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            rope_theta=cfg.rope_theta,
            rope_scaling=cfg.rope_scaling,
            max_position_embeddings=cfg.max_position_embeddings,
            quant_config=None,
            layer_id=layer,
            reduce_results=False,
            prefix="self_attn",
            alt_stream=None,
            is_nextn=False,
            skip_rope=True,
        )
    nh = cfg.num_attention_heads
    assert nh % tp_size == 0
    lh = nh // tp_size
    qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    kvo = cfg.qk_nope_head_dim + cfg.v_head_dim
    h0, h1 = tp_rank * lh, (tp_rank + 1) * lh
    p = CKPT.format(l=layer)
    sd = {}
    sd["fused_qkv_a_proj_with_mqa.weight"] = torch.cat(
        [
            sh.get(p + "self_attn.q_a_proj.weight"),
            sh.get(p + "self_attn.kv_a_proj_with_mqa.weight"),
        ],
        dim=0,
    ).to(torch.bfloat16)
    sd["q_a_layernorm.weight"] = sh.get(p + "self_attn.q_a_layernorm.weight").to(
        torch.bfloat16
    )
    sd["q_b_proj.weight"] = (
        sh.get(p + "self_attn.q_b_proj.weight")[h0 * qk : h1 * qk].to(torch.bfloat16)
    )
    sd["kv_a_layernorm.weight"] = sh.get(p + "self_attn.kv_a_layernorm.weight").to(
        torch.bfloat16
    )
    sd["kv_b_proj.weight"] = (
        sh.get(p + "self_attn.kv_b_proj.weight")[h0 * kvo : h1 * kvo].to(torch.bfloat16)
    )
    sd["o_proj.weight"] = (
        sh.get(p + "self_attn.o_proj.weight")[
            :, h0 * cfg.v_head_dim : h1 * cfg.v_head_dim
        ]
        .contiguous()
        .to(torch.bfloat16)
    )
    ip = p + "self_attn.indexer."
    sd["indexer.wq_b.weight"] = sh.get(ip + "wq_b.weight").to(torch.bfloat16)
    sd["indexer.wk.weight"] = sh.get(ip + "wk.weight").to(torch.bfloat16)
    sd["indexer.k_norm.weight"] = sh.get(ip + "k_norm.weight").float()
    sd["indexer.k_norm.bias"] = sh.get(ip + "k_norm.bias").float()
    sd["indexer.weights_proj.weight"] = sh.get(ip + "weights_proj.weight").float()
    sd["indexer.index_kpool_compress_ape"] = sh.get(
        ip + "index_kpool_compress_ape"
    ).float()
    sd["indexer.index_kpool_compress_gate"] = sh.get(
        ip + "index_kpool_compress_gate"
    ).to(torch.bfloat16)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    missing = [k for k in missing if not k.startswith("attn_m")]
    assert not missing, missing
    m = m.to(DEV).eval()

    # post_load_weights: absorb kv_b_proj into w_kc / w_vc, exactly as
    # deepseek_weight_loader.post_load_weights does for the bf16 / _is_npu case.
    w = m.kv_b_proj.weight
    w_kc, w_vc = w.unflatten(0, (-1, m.qk_nope_head_dim + m.v_head_dim)).split(
        [m.qk_nope_head_dim, m.v_head_dim], dim=1
    )
    m.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
    m.w_vc = w_vc.contiguous().transpose(1, 2).contiguous()
    return m


# ---------------------------------------------------------------- batches
def make_fb(mode, seq_lens, extend_lens, req_pool_indices, out_cache_loc):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    fb = types.SimpleNamespace()
    fb.forward_mode = mode
    fb.batch_size = len(seq_lens)
    fb.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    fb.seq_lens = fb.seq_lens_cpu.to(DEV, torch.int32)
    if extend_lens is None:
        fb.extend_seq_lens = None
        fb.extend_seq_lens_cpu = None
        fb.extend_prefix_lens = None
        fb.extend_prefix_lens_cpu = None
    else:
        fb.extend_seq_lens_cpu = list(extend_lens)
        fb.extend_seq_lens = torch.tensor(extend_lens, dtype=torch.int32, device=DEV)
        pre = [s - e for s, e in zip(seq_lens, extend_lens)]
        fb.extend_prefix_lens_cpu = pre
        fb.extend_prefix_lens = torch.tensor(pre, dtype=torch.int32, device=DEV)
    fb.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.int32, device=DEV)
    fb.out_cache_loc = torch.tensor(out_cache_loc, dtype=torch.int64, device=DEV)
    fb.positions = None
    fb.spec_info = None
    fb.spec_algorithm = None
    fb.attn_cp_metadata = None
    fb.token_to_kv_pool = None
    return fb


def run_layer(m, positions, x, fb):
    """The real chain: forward_dsa_prepare_npu -> indexer -> forward_dsa_core_npu."""
    from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
        forward_dsa_core_npu,
        forward_dsa_prepare_npu,
    )

    (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        topk_indices,
        fb2,
        za,
        pos2,
    ) = forward_dsa_prepare_npu(m, positions, x, fb, None, None)
    out, _ = forward_dsa_core_npu(
        m, q_pe, k_pe, q_nope_out, k_nope, topk_indices, fb2, za, pos2
    )
    return out, topk_indices, q_nope_out, k_nope
