# SPDX-License-Identifier: Apache-2.0
"""GPU expert placement for the KT hybrid CPU/accelerator MoE.

The KT wrapper keeps a subset of every MoE layer's experts resident on the
accelerator and offloads the rest to the CPU. Which experts stay resident is a
per-layer decision:

``prefix``
    The logical experts ``0 .. kt_num_gpu_experts-1`` stay resident. This is the
    historical placement and the default.
``frequency``
    The ``kt_num_gpu_experts`` most frequently activated experts of each layer
    stay resident, taken from an offline activation-frequency profile.

Both strategies are expressed as one boolean mask per layer plus the derived
logical-expert-id to resident-slot table, so routing, checkpoint loading and the
CPU kernel all read the same placement.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Slot value for a logical expert that is not resident on the accelerator.
CPU_EXPERT_SLOT = -1

# Placement is a load-time property of the process: both tables are built once
# from the server args, before any weight is loaded, and are read-only after.
_LAYER_MASKS: Optional[torch.Tensor] = None
_LOGICAL_TO_GPU: Optional[torch.Tensor] = None


class MoeLayout(msgspec.Struct, frozen=True):
    """The MoE geometry a placement needs, read off the HF config."""

    num_layers: int
    num_experts: int
    first_k_dense_replace: int
    moe_layer_freq: int


def is_moe_layer(layer_idx: int, layout: MoeLayout) -> bool:
    """Mirror of ``DeepseekV2ForCausalLM._is_layer_sparse`` for non-NextN layers."""
    return (
        layer_idx >= layout.first_k_dense_replace
        and layer_idx % layout.moe_layer_freq == 0
    )


def generate_prefix_masks(
    layout: MoeLayout, experts_per_moe_layer: int
) -> torch.Tensor:
    """Keep experts ``0 .. experts_per_moe_layer-1`` resident on every layer.

    Unlike the frequency placement below this does not gate on
    :func:`is_moe_layer`: a dense layer never builds a ``FusedMoE`` and so never
    reads its row, while DeepSeek's NextN module builds its (always sparse) MoE
    layer under ``layer_id == 0``, which ``is_moe_layer`` would classify as dense
    whenever ``first_k_dense_replace > 0``.
    """
    k = min(max(experts_per_moe_layer, 0), layout.num_experts)
    masks = torch.zeros(
        layout.num_layers, layout.num_experts, dtype=torch.bool, device="cpu"
    )
    if k > 0:
        masks[:, :k] = True
    return masks


def generate_frequency_masks_per_layer(
    activation_freq: torch.Tensor,
    layout: MoeLayout,
    experts_per_moe_layer: int,
) -> torch.Tensor:
    """Keep the top-``experts_per_moe_layer`` experts of each MoE layer resident."""
    k = min(max(experts_per_moe_layer, 0), layout.num_experts)
    masks = torch.zeros(
        layout.num_layers, layout.num_experts, dtype=torch.bool, device="cpu"
    )
    if k == 0:
        return masks
    freq_cpu = activation_freq.to(device="cpu", dtype=torch.float32)
    for layer_idx in range(layout.num_layers):
        if not is_moe_layer(layer_idx, layout):
            continue
        _, top = torch.topk(freq_cpu[layer_idx], k=k, largest=True, sorted=False)
        masks[layer_idx, top] = True
    return masks


def build_logical_to_gpu_index(masks: torch.Tensor) -> torch.Tensor:
    """``[num_layers, num_experts]`` bool -> resident slot per logical expert.

    Resident experts are packed in ascending logical order, which is the order
    the accelerator weight tensors are laid out in; a non-resident expert maps to
    :data:`CPU_EXPERT_SLOT`.
    """
    num_layers, num_experts = masks.shape
    out = torch.full(
        (num_layers, num_experts), CPU_EXPERT_SLOT, dtype=torch.long, device="cpu"
    )
    for layer_idx in range(num_layers):
        logical_ids = torch.nonzero(masks[layer_idx], as_tuple=False).view(-1)
        for slot, logical_id in enumerate(logical_ids.tolist()):
            out[layer_idx, logical_id] = slot
    return out


def load_activation_freq(path: str) -> torch.Tensor:
    """Load a ``[num_layers, num_experts]`` activation-frequency profile."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        for key in ("activation_freq", "freq", "data"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, torch.Tensor):
        raise TypeError(
            f"Expected an activation_freq tensor in {path}, got {type(data).__name__}"
        )
    return data.to(device="cpu", dtype=torch.float32)


def get_moe_layout_from_server_args(server_args: ServerArgs) -> MoeLayout:
    hf_config = server_args.get_model_config().hf_config
    # HF configs are third-party and genuinely heterogeneous here: the two
    # optional fields are absent on models that place a MoE block on every layer.
    num_layers = getattr(hf_config, "num_hidden_layers", None)
    num_experts = getattr(hf_config, "n_routed_experts", None)
    first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0) or 0
    moe_layer_freq = getattr(hf_config, "moe_layer_freq", 1) or 1
    if num_layers is None or num_experts is None:
        raise ValueError(
            "Cannot infer the MoE layout for KT expert placement: the model "
            "config has no num_hidden_layers / n_routed_experts."
        )
    if not isinstance(moe_layer_freq, int):
        raise ValueError(
            "KT expert placement needs a scalar moe_layer_freq, got "
            f"{moe_layer_freq!r}."
        )
    return MoeLayout(
        num_layers=num_layers,
        num_experts=num_experts,
        first_k_dense_replace=first_k_dense_replace,
        moe_layer_freq=moe_layer_freq,
    )


def ensure_kt_layer_masks(server_args: ServerArgs) -> None:
    """Build the process-wide placement tables once, before any layer is built."""
    global _LAYER_MASKS, _LOGICAL_TO_GPU
    if _LAYER_MASKS is not None:
        return
    if server_args.kt_weight_path is None:
        return

    layout = get_moe_layout_from_server_args(server_args)
    experts_per_moe_layer = server_args.kt_num_gpu_experts or 0
    strategy = server_args.kt_expert_placement_strategy

    if strategy == "frequency":
        if not server_args.kt_activation_freq_path:
            raise ValueError(
                "--kt-expert-placement-strategy frequency requires "
                "--kt-activation-freq-path"
            )
        if server_args.ep_size != 1:
            # Frequency placement picks a different expert subset per layer, so
            # a logical expert id no longer implies which EP rank owns it.
            raise ValueError(
                "--kt-expert-placement-strategy frequency requires "
                "--expert-parallel-size 1"
            )
        activation_freq = load_activation_freq(server_args.kt_activation_freq_path)
        if tuple(activation_freq.shape) != (layout.num_layers, layout.num_experts):
            raise ValueError(
                f"activation_freq shape {tuple(activation_freq.shape)} does not "
                f"match ({layout.num_layers}, {layout.num_experts})"
            )
        masks = generate_frequency_masks_per_layer(
            activation_freq=activation_freq,
            layout=layout,
            experts_per_moe_layer=experts_per_moe_layer,
        )
    else:
        masks = generate_prefix_masks(
            layout=layout, experts_per_moe_layer=experts_per_moe_layer
        )

    _LAYER_MASKS = masks
    _LOGICAL_TO_GPU = build_logical_to_gpu_index(masks)
    logger.info(
        "[KT] expert placement strategy=%s experts_per_moe_layer=%d "
        "moe_layers=%d resident_expert_slots=%d",
        strategy,
        experts_per_moe_layer,
        sum(1 for i in range(layout.num_layers) if is_moe_layer(i, layout)),
        int(masks.sum().item()),
    )


def get_layer_gpu_experts_mask(layer_idx: int) -> torch.Tensor:
    if _LAYER_MASKS is None:
        raise RuntimeError("KT layer masks not initialized; call ensure_kt_layer_masks")
    return _LAYER_MASKS[layer_idx]


def get_layer_logical_to_gpu_index(layer_idx: int) -> torch.Tensor:
    if _LOGICAL_TO_GPU is None:
        raise RuntimeError("KT layer masks not initialized; call ensure_kt_layer_masks")
    return _LOGICAL_TO_GPU[layer_idx]
