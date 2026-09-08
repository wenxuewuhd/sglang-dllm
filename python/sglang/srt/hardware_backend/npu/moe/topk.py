from typing import TYPE_CHECKING, Optional

import torch
from sgl_kernel_npu.norm.l1_norm import l1_norm

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import topk_ids_logical_to_physical
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    capture_routed_experts_if_allowed,
    select_experts,
)

if TYPE_CHECKING:
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutput


def _apply_routed_scaling_after_renorm(
    topk_weights: torch.Tensor,
    topk_config: "TopKConfig",
) -> torch.Tensor:
    """Mirror GPU post-renorm scaling when apply_routed_scaling_factor_on_output is set."""
    if (
        topk_config.renormalize
        and topk_config.apply_routed_scaling_factor_on_output
        and topk_config.routed_scaling_factor is not None
    ):
        return topk_weights * topk_config.routed_scaling_factor
    return topk_weights


def _append_fused_shared_slot(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_routed_experts: int,
    topk_config: "TopKConfig",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append the fused shared expert's slot to a routed-only top-k result.

    The vendor top-k ops (``npu_moe_gating_top_k`` and friends) select from the
    router's ``num_routed_experts`` logits and know nothing about the extra
    expert slot that shared-experts fusion appends to the MoE weight tensors.
    So we ask them for the routed ``top_k`` only and append the shared slot
    here, rather than the CUDA reference's trick of asking for one extra column
    and overwriting it (``biased_topk_impl``): overwriting would need the
    vendor op's renormalisation to run over the wrong column count.

    The weight is chosen so that the fused shared expert ends up contributing
    with weight exactly 1.0 after ``DeepseekV2MoE.forward_normal`` finishes,
    matching what the unfused ``shared_experts`` MLP would have added:

      * ``apply_routed_scaling_factor_on_output`` -> the routed weights already
        carry ``routed_scaling_factor`` and nothing scales the output later, so
        the shared slot's weight is 1.0.
      * otherwise (the Ascend default, see FusedMoE.
        ``should_fuse_routed_scaling_factor_in_topk``) ``forward_normal`` does
        ``final_hidden_states *= routed_scaling_factor`` on the *combined*
        output, so the shared slot goes in pre-divided by that factor.

    Same arithmetic as the CUDA path, which writes ``topk_weights[:, -1] =
    topk_weights[:, :-1].sum(-1) / routed_scaling_factor`` before a
    renormalisation that divides by exactly that sum.
    """
    n_fused = topk_config.num_fused_shared_experts
    n_tokens = topk_ids.shape[0]

    if n_fused == 1:
        shared_ids = topk_ids.new_full((n_tokens, 1), num_routed_experts)
    else:
        shared_ids = torch.arange(
            num_routed_experts,
            num_routed_experts + n_fused,
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        ).expand(n_tokens, n_fused)

    if topk_config.apply_routed_scaling_factor_on_output:
        shared_weight = 1.0
    else:
        shared_weight = 1.0 / float(topk_config.routed_scaling_factor or 1.0)

    # `new_full` + `cat` rather than `torch.nn.functional.pad`, which reads like
    # the cheaper way to say this.  It is not, and it is not even fewer kernels:
    # measured on one A3 die, TP1, bs=1, graph on, 42 MoE layers, pad lowers to
    # 84 MemSet (623.5 us/step) + 84 PadV3 (290.8 us/step) against this pair's
    # 84 Fill (~129) + 84 ConcatD (148.4) -- same 168 launches, 3.3x the device
    # time, and 32.357 vs 31.541 ms/step end to end (32.6 vs 31.5 ms/token wall,
    # at every one of concurrency 1/3/13/16).  Both spellings are bit-identical
    # in output (teacher-forced max|dlp| = 0.0).

    topk_ids = torch.cat([topk_ids, shared_ids], dim=1)
    topk_weights = torch.cat(
        [topk_weights, topk_weights.new_full((n_tokens, n_fused), shared_weight)],
        dim=1,
    )
    return topk_weights, topk_ids


def fused_topk_npu(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: "TopKConfig",
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional["ExpertLocationDispatchInfo"] = None,
    layer_id: Optional[int] = None,
) -> "TopKOutput":

    use_grouped_topk = topk_config.use_grouped_topk
    renormalize = topk_config.renormalize
    correction_bias = topk_config.correction_bias

    # The vendor top-k ops select among the router's logits only; the fused
    # shared expert's slot is appended afterwards by
    # _append_fused_shared_slot.  ``topk_config.top_k`` already includes it.
    num_fused_shared_experts = topk_config.num_fused_shared_experts
    routed_top_k = topk_config.top_k - num_fused_shared_experts

    # sqrtsoftplus (DSV4 noaux_tc): top-k over (scores + bias); weights from
    # un-biased scores. The custom op fuses softplus/sqrt/topk/gather/norm/cast.
    if topk_config.scoring_func == "sqrtsoftplus":
        routed_scaling_factor = (
            topk_config.routed_scaling_factor
            if topk_config.apply_routed_scaling_factor_on_output
            else 1.0
        )
        topk_weights, topk_ids, _ = torch.ops.custom.npu_moe_gating_top_k(
            x=router_logits.to(torch.float32),
            k=routed_top_k,
            bias=(
                correction_bias.to(torch.float32)
                if correction_bias is not None
                else None
            ),
            input_ids=None,
            tid2eid=None,
            routed_scaling_factor=float(routed_scaling_factor),
            norm_type=2,
        )
        topk_weights = topk_weights.to(torch.float32)

    # Fast path: simple top-k without grouped routing and bias
    elif not use_grouped_topk and correction_bias is None:
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(
            router_logits,
            k=routed_top_k,
        )

        if renormalize:
            topk_weights = l1_norm(topk_weights)
        topk_weights = topk_weights.to(torch.float32)

    # Support grouped top-k or correction bias or sigmoid or routed_scaling_factor
    elif (
        correction_bias is not None
        or topk_config.scoring_func == "sigmoid"
        or num_token_non_padded is not None
    ):
        topk_weights, topk_ids, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits.to(torch.float32),
            k=routed_top_k,
            bias=(
                correction_bias.to(torch.float32)
                if correction_bias is not None
                else None
            ),
            # num_expert_group and topk_group in some topk_config without group is None, (not supported by this ops)
            k_group=topk_config.topk_group if use_grouped_topk else 1,
            group_count=topk_config.num_expert_group if use_grouped_topk else 1,
            group_select_mode=(1 if use_grouped_topk else 0),
            renorm=renormalize,
            # 1 for sigmoid, 0 for softmax
            norm_type=(0 if topk_config.scoring_func == "softmax" else 1),
            routed_scaling_factor=(
                topk_config.routed_scaling_factor
                if topk_config.apply_routed_scaling_factor_on_output
                else 1
            ),
            eps=float(1e-20),
        )
        topk_weights = topk_weights.to(torch.float32)

    # torch native is not yet supported num_token_non_padded
    # Fallback to torch native implementation
    else:
        topk_config.torch_native = True
        return select_experts(
            hidden_states=hidden_states,
            layer_id=layer_id,
            router_logits=router_logits,
            topk_config=topk_config,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    if num_fused_shared_experts:
        topk_weights, topk_ids = _append_fused_shared_slot(
            topk_weights, topk_ids, router_logits.shape[-1], topk_config
        )

    if expert_location_dispatch_info is not None:
        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
    capture_routed_experts_if_allowed(topk_config, layer_id, topk_ids)

    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
