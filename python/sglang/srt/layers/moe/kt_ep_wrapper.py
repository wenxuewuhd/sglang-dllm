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

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.runtime_context import (
    get_parallel,
    get_schedule,
)
from sglang.srt.utils import get_compiler_backend
from sglang.srt.utils.kt_accel import (
    kt_current_stream,
    kt_current_stream_handle,
    kt_device_synchronize,
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
_npu_report_subscribed: Set[int] = set()


def _npu_use_graph_host_callback(device: torch.device) -> bool:
    if device.type != "npu":
        return False
    try:
        if torch.npu.is_current_stream_capturing():
            return True
    except Exception:
        pass
    try:
        from sglang.srt.model_executor.runner import get_is_capture_mode

        return get_is_capture_mode()
    except Exception:
        return False


def _ensure_npu_subscribe_report(stream) -> None:
    key = int(stream.npu_stream)
    if key in _npu_report_subscribed:
        return
    import torch_npu

    torch_npu.npu._subscribe_report(stream)
    _npu_report_subscribed.add(key)


@torch.no_grad()
def _kt_npu_graph_host_forward(args) -> None:
    wrapper, hidden_states, stream_handle = args
    wrapper.run_pinned_forward_sync(hidden_states, stream_handle)


def resolve_kt_weight_path_for_layer(weight_path: str, layer_idx: int) -> str:
    """Resolve a per-layer KT weight path without requiring a launcher patch."""
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
    if server_args.kt_weight_path is None:
        return None

    if server_args.device == "npu":
        if not server_args.kt_num_gpu_experts or server_args.kt_num_gpu_experts < 1:
            raise ValueError(
                "KT expert offload on NPU currently requires "
                "--kt-num-gpu-experts >= 1"
            )
        if server_args.tp_size != 1 or server_args.ep_size != 1:
            raise ValueError(
                "KT expert offload on NPU currently supports only "
                "--tensor-parallel-size 1 and --expert-parallel-size 1"
            )
        if server_args.moe_a2a_backend != "none":
            raise ValueError(
                "KT expert offload on NPU requires --moe-a2a-backend none; "
                "the Ascend dispatcher hook does not support A2A dispatchers yet"
            )
        # torch_npu owns the ACL report subscriber used by graph host callbacks.
        # Tell kt-kernel not to attach a second subscriber to the same stream.
        os.environ["KT_EXTERNAL_NPU_REPORT_SUBSCRIBER"] = "1"

    # Try to get num_layers from model config
    num_layers = None
    try:
        hf_config = server_args.get_model_config().hf_config
        num_layers = getattr(hf_config, "num_hidden_layers", None)
    except Exception:
        # If we can't get the config, num_layers will be None
        pass

    return KTConfig(
        layer_idx=layer_idx,
        num_gpu_experts=server_args.kt_num_gpu_experts,
        cpuinfer_threads=server_args.kt_cpuinfer,
        threadpool_count=server_args.kt_threadpool_count,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=get_schedule().chunked_prefill_size,
        method=server_args.kt_method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:
    """Mask CPU expert IDs by setting them to -1.

    This function masks expert IDs that should be computed on CPU (IDs >= num_gpu_experts)
    so they won't be computed on GPU. The masked IDs are set to -1, which causes the
    GPU MoE kernel to skip those experts.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing expert IDs
        num_gpu_experts: Number of experts that should run on GPU (experts 0 to num_gpu_experts-1)

    Returns:
        Modified topk_ids tensor with CPU expert IDs masked as -1
    """
    topk_ids[topk_ids >= num_gpu_experts] = -1
    return topk_ids


def mask_cpu_expert_routing_npu(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_gpu_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map CPU routes to a zero-weight valid expert for NPU grouped matmul.

    Ascend routing kernels do not accept the ``-1`` sentinel used by CUDA.  A
    valid expert id with a zero weight is equivalent and keeps the NPU kernel
    inputs well formed.
    """
    is_gpu = topk_ids < num_gpu_experts
    safe_ids = torch.where(is_gpu, topk_ids, torch.zeros_like(topk_ids))
    safe_weights = torch.where(is_gpu, topk_weights, torch.zeros_like(topk_weights))
    return safe_ids, safe_weights


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
        self.num_gpu_experts = kt_config.num_gpu_experts
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_parallel().tp_rank

        # KT wrapper will be initialized in create_weights
        self.wrapper: Optional[KTMoEWrapper] = None

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

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts: num_gpu_experts to num_experts-1
        if self.tp_rank == 0:
            gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool)
            gpu_experts_mask[: self.num_gpu_experts] = True
            self.wrapper = KTMoEWrapper(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=gpu_experts_mask,
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

        # 2. Load CPU weights using KT wrapper
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

            # Subscribe before NPU graph capture.  Subscribing lazily from the
            # first captured forward performs an ACL control operation that is
            # illegal during capture.
            if layer.w13_weight.device.type == "npu":
                _ensure_npu_subscribe_report(kt_current_stream(layer.w13_weight.device))

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
        """Attach KT at the correct side of the latest Ascend TP dispatcher.

        Ascend dispatch expands tokens before ``quant_method.apply``.  CPU MoE
        must consume the original token rows and be merged after finalize
        routing, so the legacy wrapper-only submit/sync placement is invalid.
        """
        if dispatcher.__class__.__name__ != "AscendTPDispatcher":
            return
        dispatcher.register_pre_dispatch_hook(self._ascend_pre_dispatch)
        dispatcher.register_post_combine_hook(self._ascend_post_combine)

    def _submit_raw(
        self,
        hidden_states: torch.Tensor,
        topk_output,
    ) -> None:
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

    def _ascend_pre_dispatch(self, dispatcher, hidden_states, topk_output):
        del dispatcher
        use_graph = (
            self.tp_rank == 0
            and self.wrapper is not None
            and _npu_use_graph_host_callback(hidden_states.device)
        )
        self._ascend_pending_hidden_states = hidden_states
        self._ascend_pending_graph = use_graph
        if use_graph:
            self._submit_raw_npu_graph(hidden_states, topk_output)
        else:
            self._submit_raw(hidden_states, topk_output)

        safe_ids, safe_weights = mask_cpu_expert_routing_npu(
            topk_output.topk_ids,
            topk_output.topk_weights,
            self.num_gpu_experts,
        )
        return hidden_states, topk_output._replace(
            topk_ids=safe_ids,
            topk_weights=safe_weights,
        )

    def _ascend_post_combine(self, dispatcher, hidden_states):
        del dispatcher
        original = self._ascend_pending_hidden_states
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
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Submit forward task to CPU (non-blocking)
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights, kt_current_stream_handle(x.device)
        )

    def sync(
        self, x: torch.Tensor, *, cpu_already_synced: bool = False
    ) -> torch.Tensor:
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
            return self.wrapper.copy_forward_output_to_device(x)
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

        # The latest AscendTP path submits CPU work in a pre-dispatch hook and
        # merges it in a post-combine hook.  Here only the resident NPU experts
        # run on already-expanded dispatcher output.
        if dispatch_output.format.is_ascend_tp():
            return self.gpu_method.apply(layer, dispatch_output)

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        use_npu_graph = (
            self.tp_rank == 0
            and self.wrapper is not None
            and _npu_use_graph_host_callback(x.device)
        )

        # Step 1: Submit CPU expert computation (non-blocking or captured host callback)
        if use_npu_graph:
            self._submit_cpu_npu_graph(dispatch_output, x)
        elif self.tp_rank == 0:
            self.submit(layer, dispatch_output)

        # Step 2/3: Run resident accelerator experts.  The all-CPU case must
        # not call grouped matmul with an empty weight tensor.
        if self.num_gpu_experts > 0:
            topk_ids = topk_output.topk_ids
            if x.device.type == "npu":
                masked_topk_ids, masked_topk_weights = mask_cpu_expert_routing_npu(
                    topk_ids, topk_output.topk_weights, self.num_gpu_experts
                )
                masked_topk_output = topk_output._replace(
                    topk_ids=masked_topk_ids,
                    topk_weights=masked_topk_weights,
                )
            else:
                masked_topk_output = topk_output._replace(
                    topk_ids=mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)
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
