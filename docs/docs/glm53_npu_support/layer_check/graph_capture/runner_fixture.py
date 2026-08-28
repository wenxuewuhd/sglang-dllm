"""Build the real NPUGraphRunner around a real two-decoder-layer GLM-5.3-Flash
stack, on one die, with no server.

The rest of graph_capture/ captures single modules with `torch.npu.graph(...)`
directly.  That skips the code the deployment actually runs: the runner's static
buffers, its padding policy, and the attention backends' capture/replay metadata
split (`init_forward_metadata_out_graph` / `_replay_metadata` /
`_apply_cuda_graph_metadata`).  This fixture puts those back.

What is real here
    - `Glm5NextDecoderLayer` for two adjacent layers, built by the model's own
      constructor and loaded with the checkpoint's own weights through the
      model's own `load_weights`.  One DSA layer + one KDA layer by default, so
      one graph carries both attention families, both MoE blocks, and the mHC
      four-stream residual crossing the layer boundary.
    - `AscendAttnBackend` + `AscendKDAAttnBackend` under
      `AscendKDAHybridLinearAttnBackend` -- the exact triple
      `attention_registry.py` builds for GLM on NPU.
    - `NPUDSATokenToKVPool` and `HybridReqToTokenPool` (mamba side included).
    - `NPUGraphRunner` itself: its capture loop, its bs buckets, its one shared
      memory pool, its `load_batch` / `execute`.

What is faked
    - `ModelRunner`: a `ModelRunner.__new__` shell with the ~25 attributes the
      runner reads.  Nothing else of the scheduler exists.
    - The embedding: `hidden = h_table[input_ids]`.  The table is ours; the
      `input_ids` that index it are the runner's own static buffer, so the
      hidden state still enters through a runner-managed device buffer.
    - Tensor parallelism: one process pretends to be rank `tp_rank` of a
      `tp`-way group by overriding the parallel context (the same trick
      `tp_fixture.py` uses).  With `tp=16` the shapes are the shipped ones and
      every collective is a no-op on a world-size-1 group, i.e. the module
      computes rank 0's *partial* contribution.  With `tp=1` the module computes
      the whole thing and its output is comparable to the CPU trace golden.
"""
from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Dict, List, Optional

import torch

MODEL = "/mnt/workspace/models/GLM-5.3-Flash-BF16"
DEV = "npu"


# ---------------------------------------------------------------- boot
def boot(*, port: int, ctx: int = 32768, max_running: int = 16, page: int = 64,
         capture_bs: Optional[List[int]] = None):
    """ServerArgs + distributed + ModelConfig, matching $ROOT/run/launch_glm_bf16.sh
    except tp_size (this is one process on one die; the TP *shape* is imposed by
    `parallel_override` where the modules read it)."""
    from sglang.srt import runtime_context as rc
    from sglang.srt.server_args import ServerArgs

    kw = dict(
        model_path=MODEL,
        device="npu",
        tp_size=1,
        page_size=page,
        attention_backend="ascend",
        trust_remote_code=True,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        context_length=ctx,
        max_running_requests=max_running,
        mem_fraction_static=0.6,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        moe_a2a_backend="none",
    )
    if capture_bs is not None:
        kw["cuda_graph_bs_decode"] = list(capture_bs)
    sa = ServerArgs(**kw)
    rc.publish(sa, role="scheduler")

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment(
        backend="hccl", world_size=1, rank=0, local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    from sglang.srt.configs.model_config import ModelConfig

    mc = ModelConfig.from_server_args(sa)
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    initialize_dp_attention(sa, mc)
    try:
        from sglang.srt.layers.moe.utils import initialize_moe_config

        initialize_moe_config()
    except Exception:
        pass
    # The loader's single model-instantiation point normally runs this gate
    # before any layer exists; every MoE layer then reads its answer through
    # `is_shared_experts_fusion_disabled()`.  Skipping it would build the MoE
    # with the shared expert fused into slot n_routed_experts while the weight
    # loader keeps the `mlp.shared_experts.*` names -- the exact divergence
    # `Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason`
    # warns about ("a divergence drops the shared-expert weights and runs the
    # fused slot uninitialized").
    from sglang.srt.layers.moe.utils import install_shared_experts_fusion_decision
    from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration

    install_shared_experts_fusion_decision(
        Glm5NextForConditionalGeneration, mc.hf_config, None
    )
    return sa, mc


def parallel_override(tp: int, tp_rank: int = 0):
    """Pretend to be rank `tp_rank` of a `tp`-way TP group inside one process.

    Covers the three axes the GLM layer reads: `attn_tp_*` (attention head
    shard), `moe_tp_*` (expert intermediate shard), and plain `tp_*` (the
    communicator's own bookkeeping).  The collectives still run on the real
    world-size-1 group, so they are identity -- which is what makes the result
    rank `tp_rank`'s partial contribution rather than the reduced sum."""
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().override(
        tp_size=tp, tp_rank=tp_rank,
        attn_tp_size=tp, attn_tp_rank=tp_rank,
        moe_tp_size=tp, moe_tp_rank=tp_rank,
        moe_ep_size=1, moe_ep_rank=0,
    )


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

    def iter_layers(self, layer_ids):
        """(name, tensor) for every checkpoint entry of these decoder layers,
        with the checkpoint's own naming (`model.language_model.layers.N....`)."""
        want = {f"model.language_model.layers.{l}." for l in layer_ids}
        for name in self.map:
            if any(name.startswith(w) for w in want):
                yield name, self.get(name)


# ---------------------------------------------------------------- model
class TwoLayerGlm(torch.nn.Module):
    """`model.layers.{a,b}` and nothing else, wired the way `Glm5NextModel`
    wires its layer loop.

    `forward` has the signature the graph runner calls
    (`input_ids, positions, forward_batch`) and returns a `LogitsProcessorOutput`
    whose `hidden_states` is the mHC four-stream residual leaving the last layer
    -- `[num_tokens, hc_mult * hidden]`, the exact tensor the CPU trace golden
    stores between layers."""

    def __init__(self, cfg, layer_ids: List[int], n_table_rows: int, tp: int,
                 tp_rank: int = 0, stash_rows: int = 0):
        super().__init__()
        from sglang.srt.model_loader.utils import set_default_torch_dtype
        from sglang.srt.models.glm5_next import Glm5NextDecoderLayer

        self.cfg = cfg
        self.layer_ids = list(layer_ids)
        self.hc_hidden = cfg.hc_mult * cfg.hidden_size
        with set_default_torch_dtype(torch.bfloat16), torch.device(DEV), \
                parallel_override(tp, tp_rank):
            layers = {}
            for lid in self.layer_ids:
                layers[lid] = Glm5NextDecoderLayer(
                    config=cfg, layer_id=lid, quant_config=None,
                    prefix=f"model.layers.{lid}", alt_stream=None,
                )
        # ModuleDict keyed by the *real* layer id so `named_parameters()` yields
        # `model.layers.3....`, which is what `load_weights` looks up.  Int
        # indexing is added because `post_load_weights` does
        # `self.model.layers[layer_id]`.
        class _IntKeyModuleDict(torch.nn.ModuleDict):
            def __getitem__(self, k):
                return super().__getitem__(str(k))

        self.model = torch.nn.Module()
        self.model.layers = _IntKeyModuleDict(
            {str(lid): layers[lid] for lid in self.layer_ids}
        )
        self.model.start_layer = min(self.layer_ids)
        self.model.end_layer = max(self.layer_ids) + 1
        # Stand-in embedding: a static table the runner's own `input_ids`
        # buffer indexes.  Rows are set by the caller (trace hidden states, or
        # random).
        self.h_table = torch.zeros(
            (n_table_rows, self.hc_hidden), dtype=torch.bfloat16, device=DEV
        )
        self.config = cfg
        # Optional per-layer output stash, for scoring the FIRST layer's output
        # against the trace golden as well as the last one.  Static buffers, so
        # the copy is recorded into the graph once and refreshed on every
        # replay.  Off by default: it adds a node the deployment does not have.
        self.stash_rows = stash_rows
        self.layer_out = (
            {lid: torch.zeros((stash_rows, self.hc_hidden), dtype=torch.bfloat16,
                              device=DEV) for lid in self.layer_ids}
            if stash_rows else {}
        )
        # The runner asks for these; a plain decode target has neither.
        self.capture_aux_hidden_states = False

    def named_layers(self):
        return [(lid, self.model.layers[str(lid)]) for lid in self.layer_ids]

    def forward(self, input_ids, positions, forward_batch, **kwargs):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.utils import BumpAllocator

        hidden_states = torch.nn.functional.embedding(input_ids, self.h_table)
        residual = None
        topk_indices = None
        zero_allocator = BumpAllocator(
            buffer_size=len(self.layer_ids) * 2,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for lid, layer in self.named_layers():
            hidden_states, residual, topk_indices = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                zero_allocator,
                None,
                prev_topk_indices=topk_indices,
                next_full_attention_layer_id=None,
            )
            if self.stash_rows:
                n = min(hidden_states.shape[0], self.stash_rows)
                self.layer_out[lid][:n].copy_(hidden_states[:n])
        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )


def load_real_weights(model: TwoLayerGlm, mc, tp: int, tp_rank: int = 0,
                      verbose: bool = True):
    """Load the checkpoint's own weights for these two layers, through the
    model's own `load_weights` (expert mapping, fused q/kv-a, fused qkvbfg,
    conv1d fuse, ...), then run the post-load absorb.

    The loader is `Glm5NextForConditionalGeneration.load_weights` bound to our
    stub, so name mapping and every `weight_loader` are the shipped ones."""
    from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration

    sh = Shards()
    model.quant_config = None
    model.mm_config = mc.hf_config
    model.encoder_only = False
    model.language_only = True
    model.fuse_qkv_a_proj = True
    first = model.model.layers[model.layer_ids[0]]
    model.num_fused_shared_experts = getattr(
        first.mlp, "num_fused_shared_experts", 0)
    # post_load_weights walks self.model.layers; ours is a ModuleDict of the two
    # real layers, which is exactly what it needs.
    with parallel_override(tp, tp_rank):
        Glm5NextForConditionalGeneration.load_weights(
            model, sh.iter_layers(model.layer_ids)
        )
    # A parameter the loader never touched keeps its `torch.empty` garbage and
    # would sail through every bitwise check while making the golden score
    # meaningless, so refuse to hand back a partially loaded model.
    dead = [n for n, q in model.named_parameters()
            if not torch.isfinite(q.float()).all() or q.abs().max().item() == 0.0]
    assert not dead, f"parameters not loaded (all-zero or non-finite): {dead}"
    if verbose:
        for lid, layer in model.named_layers():
            print(f"  layer {lid}: loaded, linear_attn={layer.is_linear_attn} "
                  f"sparse={layer.is_layer_sparse}")


# ---------------------------------------------------------------- pools
def build_pools(mc, *, full_attn_layers, kda_layers, kv_pages, page, max_ctx,
                num_reqs, tp):
    """The two pools the GLM decode path touches: the DSA latent/index KV pool
    (full-attention layers) and the hybrid req pool that also owns the KDA
    conv/ssm state (linear layers)."""
    from sglang.srt.configs.mamba_utils import (
        KimiLinearCacheParams, KimiLinearStateShape, Mamba2StateDType,
    )
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUDSATokenToKVPool
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    cfg = mc.hf_text_config
    la = cfg.linear_attn_config
    shape = KimiLinearStateShape.create(
        tp_world_size=tp,
        num_heads=la["num_heads"],
        head_dim=la["head_dim"],
        conv_kernel_size=la["short_conv_kernel_size"],
    )
    cache_params = KimiLinearCacheParams(
        shape=shape, layers=list(kda_layers),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
    )
    size = kv_pages * page
    kv = NPUDSATokenToKVPool(
        size=size,
        page_size=page,
        kv_lora_rank=cfg.kv_lora_rank,
        dtype=torch.bfloat16,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        layer_num=len(full_attn_layers),
        device=DEV,
        index_head_dim=cfg.index_head_dim,
        enable_memory_saver=False,
        kv_cache_dim=cfg.kv_lora_rank,
        start_layer=min(full_attn_layers),
        end_layer=max(full_attn_layers),
        index_buf_size=size,
        index_kpool=cfg.index_kpool,
        index_kpool_compress=cfg.index_kpool_compress,
        tail_extra_slots=0,
        max_running_requests=num_reqs,
    )
    req = HybridReqToTokenPool(
        size=num_reqs,
        mamba_size=num_reqs,
        mamba_spec_state_size=num_reqs,
        max_context_len=max_ctx,
        device=DEV,
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=list(kda_layers),
        enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=None,
        enable_overlap_schedule=False,
    )
    return kv, req


# ---------------------------------------------------------------- runner shell
def build_model_runner(mc, sa, model, kv_pool, req_pool, *, max_bs, page):
    """A `ModelRunner.__new__` shell carrying only what the graph runner and the
    attention backends read.  Anything they touch that is not here fails loudly
    with AttributeError rather than silently defaulting."""
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    mr = ModelRunner.__new__(ModelRunner)
    mr.device = DEV
    mr.gpu_id = 0
    mr.dtype = torch.bfloat16
    mr.kv_cache_dtype = torch.bfloat16
    mr.kv_cache_dtype_str = "auto"
    mr.page_size = page
    mr.model_config = mc
    mr.server_args = sa
    mr.model = model
    mr.is_draft_worker = False
    mr.is_generation = True
    mr.is_hybrid_swa = False
    mr.sliding_window_size = None
    mr.spec_algorithm = SpeculativeAlgorithm.from_string(None)
    mr.req_to_token_pool = req_pool
    mr.token_to_kv_pool = kv_pool
    mr.token_to_kv_pool_allocator = types.SimpleNamespace(page_size=page)
    mr.ps = ParallelState.trivial()
    mr.tp_group = get_parallel().tp_group
    mr.lora_manager = None
    mr.canary_manager = None
    mr.hisparse_coordinator = None
    mr.device_timer = None
    mr.capture_tail_hooks = []
    mr.shared_read_done_event = None
    mr.decode_attn_backend_group = None
    mr.ngram_embedding_manager = types.SimpleNamespace(enabled=False, table=None)
    mr.graph_shared_output = GraphSharedOutput(device=DEV, max_rows=max_bs)
    mr._kernel_warmed_up = True          # skip BaseRunner.warmup entirely
    mr.get_pp_proxy_topk_size = lambda: None
    mr.get_pp_proxy_residual_num_blocks = lambda: None
    mr.decode_num_tokens_per_req = lambda num_draft_tokens=None: 1
    from sglang.srt.layers.attention.linear.utils import resolve_linear_attn_backends

    mr.linear_attn_backends = resolve_linear_attn_backends()
    return mr


def build_backend(mr, full_attn_layers):
    """The exact triple `attention_registry.py` builds for GLM on NPU."""
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )
    from sglang.srt.hardware_backend.npu.attention.ascend_kda_backend import (
        AscendKDAAttnBackend, AscendKDAHybridLinearAttnBackend,
    )

    full = AscendAttnBackend(mr)
    linear = AscendKDAAttnBackend(mr)
    hybrid = AscendKDAHybridLinearAttnBackend(full, linear, list(full_attn_layers))
    mr.attn_backend = hybrid
    return hybrid


def absorb_kv_b(layer):
    """`post_load_weights`'s bf16/NPU branch for one MLA layer: fold kv_b_proj
    into w_kc / w_vc.  Only needed on the random-weight path; the real loader
    runs it for us."""
    m = layer.self_attn
    if not hasattr(m, "kv_b_proj"):
        return
    w = m.kv_b_proj.weight
    w_kc, w_vc = w.unflatten(0, (-1, m.qk_nope_head_dim + m.v_head_dim)).split(
        [m.qk_nope_head_dim, m.v_head_dim], dim=1
    )
    m.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
    m.w_vc = w_vc.contiguous().transpose(1, 2).contiguous()


def random_init(model, seed: int = 0):
    """Fill every parameter with small noise (norms with 1.0) so the plumbing
    can be exercised without paying the checkpoint read."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    for n, p in model.named_parameters():
        if p.dim() == 1 and ("norm" in n or "layernorm" in n):
            p.data.fill_(1.0)
        else:
            p.data.copy_((torch.randn(p.shape, generator=g) * 0.02).to(p.dtype))
    for _lid, layer in model.named_layers():
        absorb_kv_b(layer)


def patch_shared_path_gaps():
    """Work around a SHARED-path import that breaks any non-CUDA MoE decode.

    `DeepseekV2MoE.forward_normal` (which is GLM's MoE -- `Glm5NextMoE is
    DeepseekV2MoE`) ends every forward with
    `maybe_fuse_routed_scale_and_shared_add`, and that function's *first*
    statement is `from ...quantization.expert_pack import ExpertPackMoEMethod`,
    whose module header does `from sgl_kernel.quantization import
    ggml_moe_a8_vec`.  `sgl_kernel` is the CUDA extension; on NPU the import
    raises ModuleNotFoundError before any quant-method check runs.

    This is shared code (`layers/quantization/mxfp4_flashinfer_trtllm_moe.py`),
    so this repo's rules say report, do not edit -- see SHARED_CHANGES.md.  The
    substitute below is the `fused=False` branch verbatim, which is the branch
    NPU would take (none of the four MxFP4 / ExpertPack methods can be
    instantiated without CUDA kernels).
    """
    from sglang.srt.models import deepseek_v2

    def _npu_safe(experts, routed, shared, routed_scaling_factor):
        if shared is not None:
            routed += shared
        return routed

    deepseek_v2.maybe_fuse_routed_scale_and_shared_add = _npu_safe
    return _npu_safe
