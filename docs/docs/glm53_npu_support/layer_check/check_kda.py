#!/usr/bin/env python
"""Stage B: run one GLM-5.3-Flash KDA layer through the real Ascend backend.

    source $ROOT/env.sh
    PYTHONPATH=$REPO/python $VENV/bin/python check_kda.py \
        --case $ROOT/goldens/kda_layer00_s4096.pt --ranks all

The path under test is the production one: `AscendKDAAttnBackend.forward_extend`
and `.forward_decode`, driven through `RadixLinearAttention`, against a real
`HybridReqToTokenPool` / `MambaPool` allocated with the NPU KDA conv layout.  The
KDA math is never reimplemented here -- only the projections around it, the
ForwardBatch, and the tensor-parallel split.

Deployment shapes, not toy shapes.  `$ROOT/run/launch_glm_bf16.sh` serves with
`--tp-size 16 --page-size 64 --context-length 32768 --max-running-requests 16`,
so per card KDA has **4 heads**, not 64.  The 16 ranks are run one after another
on a single die and their `o_proj` partial sums added, which reproduces the
per-card shapes exactly and still lets the whole layer be scored against the
unsharded CPU reference.

Two independent things are checked:

* **numbers** -- prefill and decode output, plus the conv and recurrent state,
  scored by `harness.check` (two-reference budget, never a fixed threshold);
* **state continuity** -- the decode steps run against whatever state prefill
  left in the pool, and a decoy batch of other requests runs alongside the
  scored one, so a cross-request state leak shows up as a failure rather than as
  a plausible-looking number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Case, check, report  # noqa: E402

MODEL = Path("/mnt/workspace/models/GLM-5.3-Flash-BF16")
DEV = "npu"


# ---------------------------------------------------------------- weights


CKPT_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "b_proj.weight",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "g_a_proj.weight",
    "g_b_proj.weight",
    "o_norm.weight",
    "o_proj.weight",
    "A_log",
    "dt_bias",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
)


def load_layer_weights(model_dir: Path, layer: int) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"model.language_model.layers.{layer}.self_attn."
    handles: Dict[str, object] = {}
    out = {}
    for suffix in CKPT_SUFFIXES:
        name = prefix + suffix
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(str(model_dir / shard), framework="pt")
        out[suffix] = handles[shard].get_tensor(name)
    return out


class ShardedKDAWeights:
    """One tensor-parallel rank's slice of a KDA layer, on device.

    The split mirrors `Glm5NextLinearAttention`: q/k/v/beta/f_b/g_b/dt_bias/A_log
    and the three conv sub-blocks are head-sharded; f_a/g_a/o_norm are
    replicated; o_proj is row-parallel, so each rank produces a partial sum of
    the layer output that the caller adds up.
    """

    def __init__(self, full: Dict[str, torch.Tensor], *, rank: int, tp: int):
        heads = full["A_log"].numel()
        if heads % tp:
            raise SystemExit(f"{heads} heads do not split over tp={tp}")
        self.num_heads = heads // tp
        self.head_dim = full["dt_bias"].numel() // heads
        d = self.num_heads * self.head_dim
        hs = slice(rank * self.num_heads, (rank + 1) * self.num_heads)
        ds = slice(rank * d, (rank + 1) * d)

        bf = lambda t: t.to(DEV, torch.bfloat16).contiguous()  # noqa: E731
        f32 = lambda t: t.to(DEV, torch.float32).contiguous()  # noqa: E731

        self.wq = bf(full["q_proj.weight"][ds])
        self.wk = bf(full["k_proj.weight"][ds])
        self.wv = bf(full["v_proj.weight"][ds])
        self.wb = bf(full["b_proj.weight"][hs])
        self.wfa = bf(full["f_a_proj.weight"])
        self.wfb = bf(full["f_b_proj.weight"][ds])
        self.wga = bf(full["g_a_proj.weight"])
        self.wgb = bf(full["g_b_proj.weight"][ds])
        self.wo = bf(full["o_proj.weight"][:, ds])
        self.o_norm_weight = full["o_norm.weight"].to(DEV, torch.bfloat16).contiguous()
        # sglang keeps A_log / dt_bias / the conv weights in fp32 (params_dtype).
        self.A_log = f32(full["A_log"][hs]).view(1, 1, self.num_heads, 1)
        self.dt_bias = f32(full["dt_bias"][ds])
        self.conv_weights = f32(
            torch.cat(
                [
                    full["q_conv1d.weight"].squeeze(1)[ds],
                    full["k_conv1d.weight"].squeeze(1)[ds],
                    full["v_conv1d.weight"].squeeze(1)[ds],
                ],
                dim=0,
            )
        )
        self.q_dim = self.k_dim = self.v_dim = d

    def project(self, hidden: torch.Tensor):
        """hidden [T, hidden_size] bf16 -> (mixed_qkv, forget_gate, beta, o_gate)."""
        q = hidden @ self.wq.T
        k = hidden @ self.wk.T
        v = hidden @ self.wv.T
        mixed_qkv = torch.cat([q, k, v], dim=-1)
        beta = hidden @ self.wb.T
        forget_gate = (hidden @ self.wfa.T) @ self.wfb.T
        o_gate = (hidden @ self.wga.T) @ self.wgb.T
        return mixed_qkv, forget_gate, beta, o_gate


# ------------------------------------------------------------ mock runner


class MockGLMModelConfig:
    """The handful of `ModelConfig` fields the mamba/KDA backend reads."""

    def __init__(self, context_len: int, num_heads: int, head_dim: int):
        from sglang.srt.configs.model_config import AttentionArch

        self.attention_arch = AttentionArch.MHA
        self.context_len = context_len
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_heads
        self.head_dim = head_dim
        self.v_head_dim = head_dim
        self.swa_v_head_dim = head_dim
        self.is_encoder_decoder = False
        self.is_multimodal = False
        self.is_generation = True
        self.quantization = None
        self.is_hybrid_swa = False
        self.is_local_attention_model = False
        self.attention_chunk_size = None
        self.sliding_window_size = None
        self.hf_config = SimpleNamespace(architectures=["Glm5NextForCausalLM"])
        self.hf_config.get_text_config = lambda: self.hf_config
        self.hf_text_config = self.hf_config
        self.linear_attn_registry_result = None

    def get_max_num_attention_heads(self) -> int:
        return self.num_attention_heads

    def get_num_kv_heads(self, tp_size: int, dcp_size: int = 1) -> int:
        return self.num_key_value_heads


def build_backend(*, tp: int, batch: int, max_context_len: int, num_heads: int,
                  head_dim: int, conv_kernel: int, page_size: int):
    """A real `AscendKDAAttnBackend` on a real pool, with a stand-in runner."""
    from sglang.srt.configs.mamba_utils import (
        KimiLinearCacheParams,
        KimiLinearStateShape,
        Mamba2StateDType,
    )
    from sglang.srt.distributed.parallel_state_wrapper import ParallelState
    from sglang.srt.hardware_backend.npu.attention.ascend_kda_backend import (
        AscendKDAAttnBackend,
    )
    from sglang.srt.layers.attention.linear.utils import resolve_linear_attn_backends
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.runtime_context import get_context

    shape = KimiLinearStateShape.create(
        tp_world_size=tp,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_size=conv_kernel,
    )
    cache_params = KimiLinearCacheParams(
        shape=shape,
        layers=[0],
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
    )

    runner = ModelRunner.__new__(ModelRunner)
    runner.device = DEV
    runner.dtype = torch.bfloat16
    runner.kv_cache_dtype = torch.bfloat16
    runner.kv_cache_dtype_str = "auto"
    runner.gpu_id = 0
    runner.ps = ParallelState.trivial()
    runner.page_size = page_size
    runner.is_draft_worker = False
    runner.model_config = MockGLMModelConfig(max_context_len, num_heads, head_dim)
    runner.token_to_kv_pool = None
    runner.token_to_kv_pool_allocator = SimpleNamespace(page_size=page_size)
    runner.server_args = get_context().server_args
    runner.req_to_token_pool = HybridReqToTokenPool(
        size=batch,
        mamba_size=batch,
        mamba_spec_state_size=batch,
        max_context_len=max_context_len,
        device=DEV,
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=[0],
        enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=None,
        enable_overlap_schedule=False,
    )
    runner.linear_attn_backends = resolve_linear_attn_backends()
    backend = AscendKDAAttnBackend(runner)
    return runner, backend


def make_layer(w: ShardedKDAWeights, lower_bound: Optional[float]):
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

    layer = RadixLinearAttention(
        layer_id=0,
        num_q_heads=w.num_heads,
        num_k_heads=w.num_heads,
        num_v_heads=w.num_heads,
        head_q_dim=w.head_dim,
        head_k_dim=w.head_dim,
        head_v_dim=w.head_dim,
        conv_weights=w.conv_weights,
        bias=None,
        A_log=w.A_log,
        dt_bias=w.dt_bias,
    )
    layer.lower_bound = lower_bound
    return layer


# ------------------------------------------------------------ batch build


def make_forward_batch(*, mode, runner, seq_lens: List[int], input_lens: List[int],
                       max_context_len: int, page_size: int):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    bs = len(seq_lens)
    device = DEV
    req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)
    runner.req_to_token_pool.req_index_to_mamba_index_mapping[req_pool_indices] = (
        torch.arange(bs, dtype=torch.int32, device=device)
    )
    prefix_lens = [s - q for s, q in zip(seq_lens, input_lens)]
    positions, out_cache_locs = [], []
    for i, (prefix, qlen) in enumerate(zip(prefix_lens, input_lens)):
        for off in range(qlen):
            positions.append(prefix + off)
            out_cache_locs.append(page_size + i * max_context_len + prefix + off)

    batch = ForwardBatch(
        forward_mode=mode,
        batch_size=bs,
        input_ids=torch.zeros(len(positions), dtype=torch.int64, device=device),
        req_pool_indices=req_pool_indices,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32, device="cpu"),
        out_cache_loc=torch.tensor(out_cache_locs, dtype=torch.int64, device=device),
        seq_lens_sum=sum(seq_lens),
        positions=torch.tensor(positions, dtype=torch.int64, device=device),
    )
    if mode.is_extend():
        ext = torch.tensor(input_lens, dtype=torch.int32, device=device)
        batch.extend_prefix_lens = torch.tensor(
            prefix_lens, dtype=torch.int32, device=device
        )
        batch.extend_prefix_lens_cpu = list(prefix_lens)
        batch.extend_seq_lens = ext
        batch.extend_seq_lens_cpu = list(input_lens)
        batch.extend_start_loc = torch.zeros_like(ext)
        if bs > 1:
            batch.extend_start_loc[1:] = torch.cumsum(ext[:-1], dim=0)
        batch.extend_num_tokens = sum(input_lens)
    return batch


# --------------------------------------------------------------- the run


class RankRunner:
    """Drives one TP rank's KDA layer through prefill then decode."""

    def __init__(self, *, weights, backend, layer, o_norm, tp: int, batch: int,
                 golden_slot: int, page_size: int, max_context_len: int, runner):
        self.w = weights
        self.backend = backend
        self.layer = layer
        self.o_norm = o_norm
        self.tp = tp
        self.batch = batch
        self.golden_slot = golden_slot
        self.page_size = page_size
        self.max_context_len = max_context_len
        self.runner = runner
        self.timings: Dict[str, List[float]] = {}
        # Off while counting host<->device round trips, so the harness's own
        # segment barriers are not mistaken for the backend's.
        self.sync_between_segments = True

    def _tick(self, name: str, t0: float):
        if self.sync_between_segments:
            torch.npu.synchronize()
        self.timings.setdefault(name, []).append((time.perf_counter() - t0) * 1e3)

    def _core(self, forward_batch, hidden_rows: torch.Tensor, is_decode: bool):
        """The model-side wrapper around the backend, as `glm5_next.py` writes it."""
        t0 = time.perf_counter()
        mixed_qkv, forget_gate, beta, o_gate = self.w.project(hidden_rows)
        self._tick("proj", t0)

        if not is_decode:
            forget_gate = forget_gate.unsqueeze(0)
        beta = beta.unsqueeze(0)

        # What `AscendKDAHybridLinearAttnBackend` would route a KDA layer to; the
        # hybrid wrapper only picks between the linear and full-attention
        # sub-backends, and every layer here is linear.
        entry = (
            self.backend.forward_decode if is_decode else self.backend.forward_extend
        )
        t0 = time.perf_counter()
        core = entry(
            layer=self.layer,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=forget_gate,
            b=beta,
        )
        self._tick("kda_decode" if is_decode else "kda_extend", t0)

        t0 = time.perf_counter()
        gate = o_gate.unflatten(-1, (-1, self.w.head_dim))
        core = self.o_norm(core, gate)
        core = core.squeeze(0).flatten(-2)
        out = core @ self.w.wo.T
        self._tick("onorm_oproj", t0)
        return out

    DECOY_SHIFT = 977

    def _decoy_len(self, slot: int, qlen: int) -> int:
        """Ragged extend lengths, because production batches are ragged.

        The scored request keeps the full chunk; the others get progressively
        shorter ones so `query_start_loc` is not a uniform stride and the varlen
        chunk indexing is actually exercised.
        """
        if slot == self.golden_slot or qlen == 1:
            return qlen
        return max(1, qlen - (slot + 1) * 64 % max(qlen // 2, 1))

    def _pack(self, hidden: torch.Tensor, start: int, qlen: int):
        """Packed rows for the whole batch, plus (offset, len) of the scored one."""
        chunks, lens = [], []
        for slot in range(self.batch):
            n = self._decoy_len(slot, qlen)
            lens.append(n)
            if slot == self.golden_slot:
                chunks.append(hidden[start : start + n])
            else:
                # Different data for every other slot: if the backend mixed two
                # requests' states, the scored slot's output would move.
                shift = (slot + 1) * self.DECOY_SHIFT
                idx = (torch.arange(start, start + n) + shift) % hidden.shape[0]
                chunks.append(hidden[idx])
        offset = sum(lens[: self.golden_slot])
        return torch.cat(chunks, dim=0), lens, offset

    def prefill(self, hidden: torch.Tensor, prefill_len: int, chunk: int):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        outs = []
        pos = 0
        seen = [0] * self.batch
        while pos < prefill_len:
            qlen = min(chunk, prefill_len - pos)
            rows, lens, offset = self._pack(hidden, pos, qlen)
            fb = make_forward_batch(
                mode=ForwardMode.EXTEND,
                runner=self.runner,
                seq_lens=[s + n for s, n in zip(seen, lens)],
                input_lens=lens,
                max_context_len=self.max_context_len,
                page_size=self.page_size,
            )
            self.backend.init_forward_metadata(fb)
            out = self._core(fb, rows, is_decode=False)
            outs.append(out[offset : offset + qlen].float().cpu())
            seen = [s + n for s, n in zip(seen, lens)]
            pos += qlen
        self.decoy_seen = seen
        return torch.cat(outs, dim=0)

    def decode(self, hidden: torch.Tensor, prefill_len: int, steps: int):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        outs = []
        seen = getattr(self, "decoy_seen", [prefill_len] * self.batch)
        for i in range(steps):
            t = prefill_len + i
            rows, lens, offset = self._pack(hidden, t, 1)
            fb = make_forward_batch(
                mode=ForwardMode.DECODE,
                runner=self.runner,
                seq_lens=[s + 1 for s in seen],
                input_lens=[1] * self.batch,
                max_context_len=self.max_context_len,
                page_size=self.page_size,
            )
            self.backend.init_forward_metadata(fb)
            out = self._core(fb, rows, is_decode=True)
            outs.append(out[offset].float().cpu())
            seen = [s + 1 for s in seen]
        return torch.stack(outs, dim=0)

    def states(self):
        cache = self.runner.req_to_token_pool.mamba2_layer_cache(0)
        conv = cache.conv[0][self.golden_slot].float().cpu()
        ssm = cache.temporal[self.golden_slot].float().cpu()
        return conv, ssm

# ------------------------------------------------------------------ bench

#: Set for the duration of a `timing.measure` so the operator wrappers below can
#: open a phase on the timer that is currently recording.
_ACTIVE_TIMER = None

#: Every operator the Ascend KDA chain calls, as (module, attribute). Wrapping
#: them turns the chain into `timing.Timer` phases without a host sync.
_OP_SITES = (
    "causal_conv1d_fn_npu",
    "causal_conv1d_update_npu",
    "fused_kda_gate_npu",
    "l2norm_fwd",
    "chunk_local_cumsum",
    "chunk_kda_scaled_dot_kkt_fwd",
    "solve_tril_npu",
    "recompute_w_u_fwd_npu",
    "chunk_gated_delta_rule_fwd_h_npu",
    "chunk_gla_fwd_o_gk_npu",
    "prepare_chunk_indices",
)


def install_op_phases():
    import sglang.srt.hardware_backend.npu.attention.ascend_kda_backend as ak
    from sglang.srt.layers.attention.linear.kernels import kda_triton

    targets = [(ak, n) for n in _OP_SITES]
    targets.append((kda_triton, "fused_sigmoid_gating_delta_rule_update"))

    saved = []
    for owner, name in targets:
        orig = getattr(owner, name)
        saved.append((owner, name, orig))

        def make(orig=orig, name=name):
            def phased(*a, **kw):
                if _ACTIVE_TIMER is None:
                    return orig(*a, **kw)
                with _ACTIVE_TIMER.phase(name):
                    return orig(*a, **kw)

            return phased

        setattr(owner, name, make())
    return saved


def restore_op_phases(saved):
    for owner, name, orig in saved:
        setattr(owner, name, orig)


def run_bench(args, *, weights_full, meta) -> int:
    """Steady-state cost of one KDA layer at the deployment shape, one rank.

    Uses `timing.py` so the number is comparable with the other module checks in
    this directory, and never prints one without its exclusions.

    What is inside the timed region is one layer's own work: the projections, the
    Ascend backend call, and the output norm / projection.  Building the
    ForwardBatch and `init_forward_metadata` are outside it deliberately -- the
    runtime does those once per forward for all 34 KDA layers, so charging them
    to one layer would overstate it.
    """
    import os

    import timing
    from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    global _ACTIVE_TIMER

    num_heads = int(meta["num_heads"])
    head_dim = int(meta["head_dim"])
    ctx = args.bench_context
    bs = args.batch
    max_context_len = ((ctx + args.page_size) // args.page_size + 2) * args.page_size

    w = ShardedKDAWeights(weights_full, rank=0, tp=args.tp)
    runner, backend = build_backend(
        tp=args.tp,
        batch=bs,
        max_context_len=max_context_len,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel=int(meta["conv_kernel"]),
        page_size=args.page_size,
    )
    layer = make_layer(w, meta["gate_lower_bound"])
    o_norm = FusedRMSNormGated(
        head_dim, eps=float(meta["rms_norm_eps"]), activation="sigmoid"
    ).to(DEV)
    o_norm.weight.data.copy_(w.o_norm_weight)

    hidden_size = weights_full["q_proj.weight"].shape[1]
    torch.manual_seed(0)
    timing.prime_device(DEV)

    def build(mode, seq_lens, input_lens):
        rows = (
            torch.randn(
                sum(input_lens), hidden_size, device=DEV, dtype=torch.bfloat16
            )
            * 0.1
        )
        fb = make_forward_batch(
            mode=mode,
            runner=runner,
            seq_lens=seq_lens,
            input_lens=input_lens,
            max_context_len=max_context_len,
            page_size=args.page_size,
        )
        backend.init_forward_metadata(fb)
        is_decode = mode.is_decode()

        def call(t):
            with t.phase("qkv+gate proj"):
                mixed_qkv, forget_gate, beta, o_gate = w.project(rows)
            if not is_decode:
                forget_gate = forget_gate.unsqueeze(0)
            beta = beta.unsqueeze(0)
            entry = backend.forward_decode if is_decode else backend.forward_extend
            with t.phase("kda backend"):
                core = entry(
                    layer=layer, forward_batch=fb, mixed_qkv=mixed_qkv, a=forget_gate,
                    b=beta,
                )
            with t.phase("o_norm + o_proj"):
                core = o_norm(core, o_gate.unflatten(-1, (-1, head_dim)))
                return core.squeeze(0).flatten(-2) @ w.wo.T

        return call

    # Ragged decode near the context limit: what --max-running-requests 16 looks
    # like once the requests have drifted apart.
    dec_seqs = [ctx - 1 - (i * 137) % 4096 for i in range(bs)]
    cases = [
        (
            "KDA prefill chunk",
            f"tp{args.tp} rank, {num_heads // args.tp}x{head_dim}, "
            f"chunk={args.prefill_chunk} at prefix={ctx - args.prefill_chunk}",
            ForwardMode.EXTEND,
            [ctx],
            [args.prefill_chunk],
        ),
        (
            "KDA decode step",
            f"tp{args.tp} rank, {num_heads // args.tp}x{head_dim}, "
            f"bs={bs} ragged seq~{ctx}",
            ForwardMode.DECODE,
            dec_seqs,
            [1] * bs,
        ),
    ]

    load = os.getloadavg()
    print(
        f"host load average at start: {load[0]:.1f} / {load[1]:.1f} / {load[2]:.1f} "
        f"over {os.cpu_count()} cores -- these layers are launch-bound, so a busy "
        "host inflates every number below"
    )
    results, syncs = [], {}
    for label, shape, mode, seq_lens, input_lens in cases:
        call = build(mode, seq_lens, input_lens)
        def phased_call(t, call=call):
            global _ACTIVE_TIMER
            _ACTIVE_TIMER = t
            return call(t)

        saved = install_op_phases()
        try:
            r = timing.measure(
                phased_call,
                label=label,
                shape=shape,
                warmup=args.bench_warmup,
                iters=args.bench_iters,
                device=DEV,
            )
        finally:
            restore_op_phases(saved)
            _ACTIVE_TIMER = None
        # Counted without the phase wrappers, and outside the timed region.
        syncs[label] = timing.count_syncs(call, device=DEV)
        results.append(r)

    print(
        timing.render(
            results,
            syncs,
            extra_exclusions=(
                "ForwardBatch construction and init_forward_metadata: the runtime "
                "pays those once per forward for all 34 KDA layers, not per layer",
                "the operator phases are nested inside 'kda backend', so the "
                "sum-of-phases line double-counts them; each operator line is one "
                "call, and causal_conv1d_fn_npu / l2norm_fwd run more than once",
                f"host load average when measured: {load[0]:.1f} (1 min)",
            ),
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--tp", type=int, default=16, help="deployment is --tp-size 16")
    ap.add_argument(
        "--ranks",
        default="all",
        help="'all' (reconstruct the whole layer) or a comma list, e.g. '0'",
    )
    ap.add_argument("--batch", type=int, default=16, help="--max-running-requests 16")
    ap.add_argument("--golden-slot", type=int, default=0)
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument(
        "--bench",
        action="store_true",
        help="time the layer at the deployment shape instead of scoring it",
    )
    ap.add_argument("--bench-context", type=int, default=32768)
    ap.add_argument("--bench-warmup", type=int, default=5)
    ap.add_argument("--bench-iters", type=int, default=30)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.npu.set_device(args.device)

    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    get_context().set_server_args(
        ServerArgs(model_path=str(args.model), device="npu", tp_size=1)
    )

    case = Case.load(args.case)
    meta = case.meta
    print(f"case {case.name}: {meta}")
    prefill_len = int(meta["prefill"])
    decode_len = int(meta["decode"])
    num_heads = int(meta["num_heads"])
    head_dim = int(meta["head_dim"])
    conv_kernel = int(meta["conv_kernel"])
    lower_bound = meta["gate_lower_bound"]
    layer_id = int(meta["layer"])

    hidden = case.inputs["hidden_states"].to(DEV, torch.bfloat16).contiguous()
    total = prefill_len + decode_len
    ranks = (
        list(range(args.tp))
        if args.ranks == "all"
        else [int(x) for x in args.ranks.split(",")]
    )
    max_context_len = ((total + args.page_size) // args.page_size + 1) * args.page_size

    from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated

    print(f"loading layer {layer_id} weights ...")
    full = load_layer_weights(args.model, layer_id)

    if args.bench:
        return run_bench(args, weights_full=full, meta=meta)

    hidden_size = full["q_proj.weight"].shape[1]
    partial_prefill = torch.zeros(prefill_len, hidden_size, dtype=torch.float64)
    partial_decode = torch.zeros(decode_len, hidden_size, dtype=torch.float64)
    conv_parts, ssm_parts = [], []
    all_timings: Dict[str, List[float]] = {}

    for rank in ranks:
        w = ShardedKDAWeights(full, rank=rank, tp=args.tp)
        runner, backend = build_backend(
            tp=args.tp,
            batch=args.batch,
            max_context_len=max_context_len,
            num_heads=num_heads,
            head_dim=head_dim,
            conv_kernel=conv_kernel,
            page_size=args.page_size,
        )
        layer = make_layer(w, lower_bound)
        o_norm = FusedRMSNormGated(
            head_dim, eps=float(meta["rms_norm_eps"]), activation="sigmoid"
        ).to(DEV)
        o_norm.weight.data.copy_(w.o_norm_weight)

        rr = RankRunner(
            weights=w,
            backend=backend,
            layer=layer,
            o_norm=o_norm,
            tp=args.tp,
            batch=args.batch,
            golden_slot=args.golden_slot,
            page_size=args.page_size,
            max_context_len=max_context_len,
            runner=runner,
        )
        t0 = time.perf_counter()
        out_p = rr.prefill(hidden, prefill_len, args.prefill_chunk)
        conv, ssm = rr.states()
        out_d = rr.decode(hidden, prefill_len, decode_len)
        conv_f, ssm_f = rr.states()
        partial_prefill += out_p.double()
        partial_decode += out_d.double()
        conv_parts.append((conv, conv_f))
        ssm_parts.append((ssm, ssm_f))
        for k, v in rr.timings.items():
            all_timings.setdefault(k, []).extend(v)
        print(
            f"  rank {rank:>2}/{args.tp}: heads={w.num_heads} "
            f"conv={tuple(conv.shape)} ssm={tuple(ssm.shape)} "
            f"({time.perf_counter() - t0:.1f}s)"
        )
        del runner, backend, rr
        torch.npu.empty_cache()

    candidate = {
        "out.prefill": partial_prefill.float(),
        "out.decode": partial_decode.float(),
    }
    if len(ranks) == args.tp:
        candidate.update(reassemble_states(conv_parts, ssm_parts, num_heads, head_dim,
                                           args.tp))

    results = check(case, candidate)
    extra = (
        f"tp={args.tp} ranks={len(ranks)}/{args.tp} batch={args.batch} "
        f"golden_slot={args.golden_slot} prefill={prefill_len} "
        f"chunk={args.prefill_chunk} decode={decode_len}"
    )
    rc = report(f"{case.name} on AscendKDAAttnBackend", results, extra)
    print("\n  segment timings (ms, all calls, warm-up NOT excluded):")
    for k, v in all_timings.items():
        v = sorted(v)
        print(
            f"    {k:<14} n={len(v):<5} first={v[0]:.2f} p50={statistics.median(v):.2f} "
            f"max={v[-1]:.2f}"
        )
    return rc


def reassemble_states(conv_parts, ssm_parts, num_heads, head_dim, tp):
    """Per-rank state slices -> the unsharded layout the CPU reference stores.

    conv per rank is [q|k|v] over that rank's heads, so the full conv state is
    the three sub-blocks each concatenated over ranks -- not the ranks
    concatenated whole.  The NPU pool keeps [channels, window]; the reference
    stores the same, so no transpose.  The temporal state is [H, V, K] on NPU
    (`chunk_gated_delta_rule_fwd_h_npu` writes it transposed) against the
    reference's [H, K, V].
    """
    d = (num_heads // tp) * head_dim
    out = {}
    for tag, idx in (("after_prefill", 0), ("final", 1)):
        blocks = [[], [], []]
        for conv in (p[idx] for p in conv_parts):
            for b in range(3):
                blocks[b].append(conv[b * d : (b + 1) * d])
        out[f"state.conv.{tag}"] = torch.cat([torch.cat(b) for b in blocks], dim=0)
        out[f"state.ssm.{tag}"] = torch.cat(
            [p[idx].transpose(-1, -2) for p in ssm_parts], dim=0
        )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
