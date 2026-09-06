# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Set, Tuple

import torch

from sglang.srt.arg_groups.overrides import model_config_of
from sglang.srt.layers.moe.kt_expert_masks import (
    ensure_kt_layer_masks,
    get_layer_gpu_experts_mask,
    get_layer_logical_to_gpu_index,
)
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_schedule,
)
from sglang.srt.utils import get_compiler_backend, is_npu
from sglang.srt.utils.kt_accel import (
    kt_current_stream,
    kt_current_stream_handle,
    kt_device_synchronize,
    kt_new_event,
    kt_new_stream,
    kt_stream_context,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)
_is_npu = is_npu()
_npu_report_subscribed: Set[int] = set()

# KT_SIDE_STREAM=1 (opt-in): enqueue the CPU-MoE host callback on a dedicated
# side stream so the resident-expert GroupedMatmul keeps running on the compute
# stream instead of being stream-order blocked behind the callback round trip.
# Fork/join events express the dependency across the Ascend dispatch span:
#   compute: ..router.. fork ------ dispatch + GPU experts + combine ---- wait(join) -- merge
#   side:            wait(fork) D2H inputs + host callback          join
# Only the NPU-graph submit path forks; the eager path (``submit_forward``) is a
# kt-kernel async submit that is not stream ordered against the compute stream,
# so it has nothing to hide behind a second stream.
_KT_SIDE_STREAM = os.environ.get("KT_SIDE_STREAM", "") == "1"
_kt_side_stream = None


def _get_kt_side_stream(device: torch.device):
    """One process-wide side stream, created outside graph capture."""
    global _kt_side_stream
    if _kt_side_stream is None:
        _kt_side_stream = kt_new_stream(device)
    return _kt_side_stream


def _npu_use_graph_host_callback(device: torch.device) -> bool:
    if device.type != "npu":
        return False
    try:
        if torch.npu.is_current_stream_capturing():
            return True
    except Exception:
        pass
    try:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        return get_is_capture_mode()
    except Exception:
        return False


def _ensure_npu_subscribe_report(stream) -> None:
    key = int(stream.npu_stream)
    if key in _npu_report_subscribed:
        return
    import torch_npu

    try:
        torch_npu.npu._subscribe_report(stream)
    except RuntimeError as exc:
        # torch_npu >= 2.10 pre-subscribes capture streams inside NPUGraph, so a
        # second AclrtSubscribeReport on the same stream fails with error 107011.
        # The stream IS subscribed in that case, so only swallow that specific
        # error and let anything else propagate.
        if "107011" not in str(exc):
            raise
    _npu_report_subscribed.add(key)


@torch.no_grad()
def _kt_npu_graph_host_forward(args) -> None:
    wrapper, hidden_states, stream_handle = args
    wrapper.run_pinned_forward_sync(hidden_states, stream_handle)


def resolve_kt_weight_path_for_layer(weight_path: str, layer_idx: int) -> str:
    """Resolve a per-layer KT weight path without requiring a launcher patch.

    The native MXFP4 safetensors loader takes a checkpoint *directory* and needs
    no substitution; the per-layer GGUF loader takes a template. Both go through
    here so the launcher does not have to know which is in use.
    """
    if "{layer_idx}" in weight_path:
        return weight_path.format(layer_idx=layer_idx)
    if weight_path.count("{}") == 1:
        return weight_path.replace("{}", str(layer_idx), 1)
    if weight_path.count("{}") > 1:
        logger.warning(
            "KT weight path has multiple '{}' placeholders; using it literally"
        )
    return weight_path


@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        num_gpu_experts: Number of experts to run on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
        gpu_experts_mask: Per-logical-expert flag, True where the expert is
            resident on the accelerator (see ``kt_expert_masks``)
        logical_to_gpu_index: Resident weight slot per logical expert, -1 for
            the CPU-only experts
    """

    layer_idx: int
    num_gpu_experts: int
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    num_layers: Optional[int] = None
    gpu_experts_mask: Optional[torch.Tensor] = None
    logical_to_gpu_index: Optional[torch.Tensor] = None


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured, None otherwise
    """
    moe_args = get_exec().moe
    if moe_args.kt_weight_path is None:
        return None

    on_npu = server_args.device == "npu" or (server_args.device is None and _is_npu)
    if on_npu:
        if not moe_args.kt_num_gpu_experts or moe_args.kt_num_gpu_experts < 1:
            raise ValueError(
                "KT expert offload on NPU currently requires --kt-num-gpu-experts >= 1"
            )
        if server_args.tp_size != 1 or server_args.ep_size != 1:
            raise ValueError(
                "KT expert offload on NPU currently supports only "
                "--tensor-parallel-size 1 and --expert-parallel-size 1"
            )
        if moe_args.moe_a2a_backend != "none":
            raise ValueError(
                "KT expert offload on NPU requires --moe-a2a-backend none; "
                "the Ascend dispatcher hook does not support A2A dispatchers yet"
            )
        if moe_args.kt_method in ("AMXINT4", "AMXINT8", "RAWINT4"):
            # operators/amx/la/amx_kernels.hpp guards clean_c/run_tile/run_full_tile
            # with `#ifdef HAVE_AMX` and no `#else`, so on a host without Intel AMX
            # these methods load and run but never write the output matrix. Fail
            # loudly instead of serving silent garbage.
            raise ValueError(
                f"--kt-method {moe_args.kt_method} needs Intel AMX; on this host use "
                "--kt-method MXFP4 (native MXFP4 safetensors, AVX512-BF16 kernel)."
            )
        # Cross-repo contract: this variable is consumed by the companion
        # kt-kernel build, not by SGLang. SGLang owns the ACL report
        # subscription for the streams used by graph host callbacks, so
        # kt-kernel must not attach a second subscriber to the same stream --
        # a double AclrtSubscribeReport fails with ACL error 107011. A
        # kt-kernel that does not understand this variable will still try to
        # subscribe on its own; use the matching kt-kernel version.
        os.environ["KT_EXTERNAL_NPU_REPORT_SUBSCRIBER"] = "1"

    num_layers = getattr(
        model_config_of(server_args).hf_config, "num_hidden_layers", None
    )

    # Expert placement is a whole-model decision, so it is resolved once and
    # then sliced per layer. ``num_gpu_experts`` follows from the mask instead
    # of ``kt_num_gpu_experts`` so that a placement which cannot host the
    # requested count stays self-consistent.
    ensure_kt_layer_masks(server_args)
    gpu_experts_mask = get_layer_gpu_experts_mask(layer_idx)
    logical_to_gpu_index = get_layer_logical_to_gpu_index(layer_idx)
    num_gpu_experts = int(gpu_experts_mask.sum().item())
    if moe_args.kt_num_gpu_experts and num_gpu_experts == 0:
        raise ValueError(
            f"KT expert placement left layer {layer_idx} with no resident "
            f"experts while --kt-num-gpu-experts is {moe_args.kt_num_gpu_experts}; "
            "the layer is classified dense by first_k_dense_replace / moe_layer_freq."
        )

    return KTConfig(
        layer_idx=layer_idx,
        num_gpu_experts=num_gpu_experts,
        cpuinfer_threads=get_exec().moe.kt_cpuinfer,
        threadpool_count=get_exec().moe.kt_threadpool_count,
        weight_path=get_exec().moe.kt_weight_path,
        chunked_prefill_size=get_schedule().chunked_prefill_size,
        method=get_exec().moe.kt_method,
        max_deferred_experts_per_token=get_exec().moe.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
        gpu_experts_mask=gpu_experts_mask,
        logical_to_gpu_index=logical_to_gpu_index,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(
    topk_ids: torch.Tensor, logical_to_gpu_index: torch.Tensor
) -> torch.Tensor:
    """Rewrite routed logical expert ids into resident accelerator weight slots.

    ``logical_to_gpu_index`` already stores -1 for the offloaded experts, so the
    gather both remaps the resident experts and produces the -1 sentinel that
    makes the GPU MoE kernel skip the CPU ones. Unlike the previous in-place
    version this returns a new tensor and does not mutate the caller's routing.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] holding logical expert ids
        logical_to_gpu_index: Per-layer logical id -> resident slot table

    Returns:
        A new topk_ids tensor with CPU expert ids masked as -1
    """
    return logical_to_gpu_index[topk_ids].to(topk_ids.dtype)


def mask_cpu_expert_routing_npu(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    safe_id_table: torch.Tensor,
    weight_keep_table: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map CPU routes to a zero-weight valid expert for NPU grouped matmul.

    Ascend routing kernels do not accept the ``-1`` sentinel used by CUDA. A
    valid expert id with a zero weight is equivalent and keeps the NPU kernel
    inputs well formed. Resident experts are rewritten to their weight slot,
    which is the identity only for the prefix placement.

    Both tables are pre-baked per layer in ``process_weights_after_loading``
    and already live on the compute device, in ``topk_ids``' own dtype:

      ``safe_id_table[e]``     = resident slot of expert ``e``, else 0
      ``weight_keep_table[e]`` = 1 if ``e`` is resident, else 0

    so the whole remap is two gathers and one multiply. Deriving it from the
    mask at call time instead costs ~9 kernels per layer (two device casts,
    two gathers, a dtype cast, two ``zeros_like`` and two ``where``) to produce
    a ``[tokens, top_k]`` tensor -- ~344 launches per decode step, all of them
    executing for a few microseconds on six elements.
    """
    return (
        safe_id_table[topk_ids],
        topk_weights * weight_keep_table[topk_ids],
    )


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (0 to num_gpu_experts-1) using any quantization method
    - CPU experts (num_gpu_experts to total_experts-1) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoE(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, num_gpu_experts=4, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.gpu_experts_mask = kt_config.gpu_experts_mask
        self.logical_to_gpu_index = kt_config.logical_to_gpu_index
        self.num_gpu_experts = kt_config.num_gpu_experts
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_parallel().tp_rank

        # KT wrapper will be initialized in create_weights
        self.wrapper: Optional[KTMoEWrapper] = None

        # Ascend pre-dispatch / post-combine hook state. Set here rather than
        # lazily because ``__getattr__`` forwards unknown names to gpu_method,
        # which would turn a missing attribute into a confusing delegation.
        self._ascend_pending_hidden_states = None
        self._ascend_pending_graph = False
        # Side-stream fork/join state (KT_SIDE_STREAM=1 only; None keeps the
        # post-combine hook on the single-stream path).
        self._ascend_pending_join = None
        self._kt_side_events = []
        # Pre-baked routing tables, keyed by dtype; filled in
        # process_weights_after_loading.
        self._safe_id_tables = {}
        self._weight_keep_tables = {}

        # Store parameters needed for KT initialization
        self._layer_params = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU experts: 0 to num_gpu_experts-1
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        # 2. Pin the placement tables to this layer's expert count. An
        # expert-parallel shard makes the layer's local expert count differ from
        # the global placement table; the prefix placement is identical either
        # way, so it is rebuilt at the local width and every consumer below
        # (the CPU kernel, routing, the checkpoint loader) reads one table.
        if self.gpu_experts_mask is None or self.gpu_experts_mask.numel() != num_experts:
            self.gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool)
            self.gpu_experts_mask[: self.num_gpu_experts] = True
            self.logical_to_gpu_index = torch.where(
                self.gpu_experts_mask,
                torch.arange(num_experts, dtype=torch.long),
                torch.full((num_experts,), -1, dtype=torch.long),
            )

        # 3. Initialize KT wrapper for CPU experts
        # CPU experts: the logical experts whose gpu_experts_mask entry is False
        if self.tp_rank == 0:
            self.wrapper = KTMoEWrapper(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=self.gpu_experts_mask,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                numa_nodes=None,
                weight_path=resolve_kt_weight_path_for_layer(
                    self.kt_config.weight_path, self.kt_config.layer_idx
                ),
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
                method=self.kt_config.method,
                max_deferred_experts_per_token=layer_max_deferred,
                swiglu_limit=layer.moe_runner_config.swiglu_limit or 0.0,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        # 2. Move the placement tables to the accelerator once, at this
        # deterministic pre-capture point. NPU has no eager warmup forward, so
        # the first forward already runs under graph capture, where the
        # host-to-device copy of a per-forward ``.to(device)`` is rejected by ACL
        # (error 107030). With the tables resident, ``.to(same_device)`` is a
        # no-op. The all-CPU layer never routes to the accelerator and may have
        # no tables at all.
        if self.num_gpu_experts > 0 and self.gpu_experts_mask is not None:
            device = layer.w13_weight.device
            self.gpu_experts_mask = self.gpu_experts_mask.to(device)
            if self.logical_to_gpu_index is not None:
                self.logical_to_gpu_index = self.logical_to_gpu_index.to(device)
            # Pre-bake the routing remap into two lookup tables so the hot path
            # is two gathers and a multiply (see mask_cpu_expert_routing_npu).
            # topk_ids' integer dtype is not known here, so bake one table per
            # plausible dtype; the pick at call time is a dict lookup, not an op.
            slots = torch.where(
                self.gpu_experts_mask,
                self.logical_to_gpu_index,
                torch.zeros_like(self.logical_to_gpu_index),
            )
            self._safe_id_tables = {
                dt: slots.to(dt) for dt in (torch.int32, torch.int64)
            }
            keep = self.gpu_experts_mask.to(torch.float32)
            self._weight_keep_tables = {
                dt: keep.to(dt) for dt in (torch.float32, torch.bfloat16, torch.float16)
            }

        # 3. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            kt_device_synchronize(layer.w13_weight.device)

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            physical_to_logical_map_cpu = (
                get_global_expert_location_metadata()
                .physical_to_logical_map_cpu[self.kt_config.layer_idx]
                .contiguous()
            )
            self.wrapper.load_weights(physical_to_logical_map_cpu)

            # Subscribe before NPU graph capture. Subscribing lazily from the
            # first captured forward performs an ACL control operation that is
            # illegal during capture.
            if layer.w13_weight.device.type == "npu":
                # Best effort only: ``_subscribe_report`` is a private torch_npu
                # API, and the lazy path still subscribes outside capture, so a
                # missing API must not break model loading.
                try:
                    _ensure_npu_subscribe_report(
                        kt_current_stream(layer.w13_weight.device)
                    )
                    if _KT_SIDE_STREAM:
                        # The side-stream variant launches the host callback on
                        # the side stream, so it needs the same pre-capture
                        # subscription; creating the stream here also keeps
                        # stream creation out of the capture forward.
                        _ensure_npu_subscribe_report(
                            _get_kt_side_stream(layer.w13_weight.device)
                        )
                except Exception as exc:
                    logger.warning(
                        "[KT] pre-capture ACL report subscribe failed (non-fatal): %s",
                        exc,
                    )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config
        if self.override_num_local_experts:
            # AscendTPDispatcher sizes its expert_tokens buffer from
            # ``num_experts`` while the runner uses ``num_local_experts``.
            # Both must describe the resident NPU expert set.
            moe_runner_config.num_experts = self.num_gpu_experts
            moe_runner_config.num_local_experts = self.num_gpu_experts
        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, moe_runner_config)

    def attach_dispatcher(self, dispatcher) -> None:
        """Attach KT at the correct side of the Ascend TP dispatcher.

        Ascend dispatch expands and quantizes tokens before
        ``quant_method.apply``, and its dispatch output carries no
        ``topk_output`` at all. CPU MoE must therefore consume the original
        token rows and be merged after finalize routing, so the wrapper-only
        submit/sync placement used on CUDA is invalid here.
        """
        if dispatcher.__class__.__name__ != "AscendTPDispatcher":
            return
        dispatcher.register_pre_dispatch_hook(self._ascend_pre_dispatch)
        dispatcher.register_post_combine_hook(self._ascend_post_combine)

    def _submit_raw(self, hidden_states: torch.Tensor, topk_output) -> None:
        if self.tp_rank != 0 or self.wrapper is None:
            return
        topk_weights, topk_ids, _ = topk_output
        self.wrapper.submit_forward(
            hidden_states,
            topk_ids,
            topk_weights,
            kt_current_stream_handle(hidden_states.device),
        )

    def _submit_raw_npu_graph(self, hidden_states: torch.Tensor, topk_output) -> None:
        import torch_npu

        assert self.wrapper is not None
        topk_weights, topk_ids, _ = topk_output
        stream = kt_current_stream(hidden_states.device)
        _ensure_npu_subscribe_report(stream)
        stream_handle = kt_current_stream_handle(hidden_states.device)
        self.wrapper.copy_inputs_to_cpu_buffers(hidden_states, topk_ids, topk_weights)
        torch_npu.npu._launch_host_func(
            stream,
            _kt_npu_graph_host_forward,
            (self.wrapper, hidden_states, stream_handle),
        )

    def _submit_raw_npu_graph_forked(
        self, hidden_states: torch.Tensor, topk_output
    ) -> None:
        """Fork the captured CPU-MoE submit onto the side stream.

        The matching join runs in ``_ascend_post_combine``; everything the
        compute stream enqueues in between (Ascend dispatch, resident-expert
        GroupedMatmul, combine) overlaps the host callback.
        """
        device = hidden_states.device
        compute_stream = kt_current_stream(device)
        side_stream = _get_kt_side_stream(device)
        # A captured graph keeps record/wait nodes that reference these event
        # objects, so a fresh pair per capture is kept alive for the process.
        # Only capture-mode forwards reach this path, so the list is bounded by
        # (captured shapes x warmups) and does not grow during serving.
        fork_event = kt_new_event(device)
        join_event = kt_new_event(device)
        self._kt_side_events.append((fork_event, join_event))

        fork_event.record(compute_stream)
        side_stream.wait_event(fork_event)
        with kt_stream_context(side_stream, device):
            # ``kt_current_stream`` resolves to the side stream inside the
            # block, so the D2H input copies and ``_launch_host_func`` both
            # enqueue there.
            self._submit_raw_npu_graph(hidden_states, topk_output)
        self._ascend_pending_join = (join_event, side_stream, compute_stream)

    def _routing_tables(self, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        """Pick the pre-baked tables matching the routing tensors' dtypes."""
        ids = self._safe_id_tables.get(topk_ids.dtype)
        w = self._weight_keep_tables.get(topk_weights.dtype)
        if ids is None:
            ids = self.logical_to_gpu_index.clamp(min=0).to(topk_ids.dtype)
            self._safe_id_tables[topk_ids.dtype] = ids
        if w is None:
            w = self.gpu_experts_mask.to(topk_weights.dtype)
            self._weight_keep_tables[topk_weights.dtype] = w
        return ids, w

    def _ascend_pre_dispatch(self, dispatcher, hidden_states, topk_output):
        del dispatcher
        # A layer that forked under capture and then runs an eager forward
        # (prefill, or a batch size with no captured graph) takes the
        # non-forked branch below and would otherwise leave the previous
        # capture's join event here, for this forward's post-combine to wait
        # on. Harmless today -- the event is long since recorded, so the wait
        # is a no-op -- but it is a stale cross-stream dependency, so clear it.
        self._ascend_pending_join = None
        use_graph = (
            self.tp_rank == 0
            and self.wrapper is not None
            and _npu_use_graph_host_callback(hidden_states.device)
        )
        self._ascend_pending_hidden_states = hidden_states
        self._ascend_pending_graph = use_graph
        if use_graph and _KT_SIDE_STREAM:
            self._submit_raw_npu_graph_forked(hidden_states, topk_output)
        elif use_graph:
            self._submit_raw_npu_graph(hidden_states, topk_output)
        else:
            self._submit_raw(hidden_states, topk_output)

        id_tbl, w_tbl = self._routing_tables(
            topk_output.topk_ids, topk_output.topk_weights
        )
        safe_ids, safe_weights = mask_cpu_expert_routing_npu(
            topk_output.topk_ids, topk_output.topk_weights, id_tbl, w_tbl
        )
        return hidden_states, topk_output._replace(
            topk_ids=safe_ids,
            topk_weights=safe_weights,
        )

    def _ascend_post_combine(self, dispatcher, hidden_states):
        del dispatcher
        original = self._ascend_pending_hidden_states
        pending_join = self._ascend_pending_join
        self._ascend_pending_join = None
        if pending_join is not None:
            join_event, side_stream, compute_stream = pending_join
            # The compute stream resumes only once the side stream's host
            # callback has returned, so the pinned CPU output is complete before
            # ``sync`` enqueues its H2D copy.
            join_event.record(side_stream)
            compute_stream.wait_event(join_event)
        cpu_output = self.sync(
            original,
            cpu_already_synced=self._ascend_pending_graph,
        )
        self._ascend_pending_hidden_states = None
        self._ascend_pending_graph = False
        return hidden_states + cpu_output

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        assert self.moe_runner_config.activation == "silu", (
            "Only SiLU activation is supported."
        )

        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task to CPU (non-blocking)
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights, kt_current_stream_handle(x.device)
        )

    def sync(self, x: torch.Tensor, *, cpu_already_synced: bool = False) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        if cpu_already_synced:
            # The graph host callback already ran the CPU MoE to completion;
            # only the device-side copy of the pinned output is left.
            return self.wrapper.copy_forward_output_to_device(x)
        # Wait for CPU computation and retrieve results
        return self.wrapper.sync_forward(x, kt_current_stream_handle(x.device))

    def _submit_cpu_npu_graph(
        self,
        dispatch_output: "StandardDispatchOutput",
        x: torch.Tensor,
    ) -> None:
        """Capture CPU MoE as an Ascend graph host callback."""
        import torch_npu

        assert self.wrapper is not None
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        stream = kt_current_stream(x.device)
        _ensure_npu_subscribe_report(stream)
        stream_handle = kt_current_stream_handle(x.device)
        self.wrapper.copy_inputs_to_cpu_buffers(x, topk_ids, topk_weights)
        torch_npu.npu._launch_host_func(
            stream,
            _kt_npu_graph_host_forward,
            (self.wrapper, x, stream_handle),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        # The AscendTP path submits CPU work in a pre-dispatch hook and merges
        # it in a post-combine hook (see ``attach_dispatcher``). Here only the
        # resident NPU experts run, on already-expanded dispatcher output.
        if dispatch_output.format.is_ascend_tp():
            return self.gpu_method.apply(layer, dispatch_output)

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        use_npu_graph = (
            self.tp_rank == 0
            and self.wrapper is not None
            and _npu_use_graph_host_callback(x.device)
        )

        # Step 1: Submit CPU expert computation (non-blocking, or captured host callback)
        if use_npu_graph:
            self._submit_cpu_npu_graph(dispatch_output, x)
        elif self.tp_rank == 0:
            self.submit(layer, dispatch_output)

        # Step 2/3: Run the resident accelerator experts. The all-CPU case must
        # not call grouped matmul with an empty weight tensor.
        if self.num_gpu_experts > 0:
            topk_ids = topk_output.topk_ids
            if x.device.type == "npu":
                id_tbl, w_tbl = self._routing_tables(
                    topk_ids, topk_output.topk_weights
                )
                masked_topk_ids, masked_topk_weights = mask_cpu_expert_routing_npu(
                    topk_ids, topk_output.topk_weights, id_tbl, w_tbl
                )
                masked_topk_output = topk_output._replace(
                    topk_ids=masked_topk_ids,
                    topk_weights=masked_topk_weights,
                )
            else:
                masked_topk_output = topk_output._replace(
                    topk_ids=mask_cpu_expert_ids(topk_ids, self.logical_to_gpu_index)
                )
            masked_dispatch_output = dispatch_output._replace(
                topk_output=masked_topk_output
            )
            output = self.gpu_method.apply(layer, masked_dispatch_output).hidden_states
        else:
            output = torch.zeros_like(x)

        # Step 4: Synchronize CPU results and merge with accelerator results.
        if self.tp_rank == 0:
            cpu_output = self.sync(x, cpu_already_synced=use_npu_graph)
            output = output + cpu_output

        return StandardCombineInput(hidden_states=output)

    def map_logical_expert_id_for_gpu_load(self, logical_expert_id: int) -> int:
        """Map a checkpoint expert id to its accelerator weight slot.

        Returns -1 when the expert is offloaded to the CPU and must not be
        loaded into the accelerator weight tensors at all.
        """
        if self.logical_to_gpu_index is None:
            if logical_expert_id < self.num_gpu_experts:
                return logical_expert_id
            return -1
        return int(self.logical_to_gpu_index[logical_expert_id].item())

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)
