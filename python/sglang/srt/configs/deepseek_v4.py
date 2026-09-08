import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from transformers import PretrainedConfig

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils.common import is_npu

logger = logging.getLogger(__name__)


def try_detect_fp4_experts(model_path: str) -> Optional[bool]:
    """True = mxfp4-packed (U8/I8/F4), False = converted FP8 (F8_E4M3),
    None when the header isn't readable (HF slug not cached yet, etc.).
    Caller falls back to user default. Pure read; never mutates env.
    """
    from sglang.srt.model_loader.weight_utils import (
        probe_routed_expert_weight_dtype,
    )
    from sglang.srt.utils import find_local_repo_dir

    if os.path.isdir(model_path):
        local_path = model_path
    else:
        local_path = find_local_repo_dir(model_path)
        if not local_path or not os.path.isdir(local_path):
            return None

    try:
        dtype = probe_routed_expert_weight_dtype(local_path)
    except Exception as e:
        logger.warning("Failed to probe routed-expert dtype for %s: %s", model_path, e)
        return None
    if dtype is None:
        return None
    if dtype in ("U8", "I8", "F4"):
        return True
    if dtype == "F8_E4M3":
        return False
    logger.warning(
        "Unexpected routed-expert safetensors dtype=%s for DeepSeek V4", dtype
    )
    return None


@dataclass(kw_only=True)
class DeepSeekV4Config(PretrainedConfig):
    architectures: List[str]
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 0
    eos_token_id: int = 1
    ep_size: int = 1
    first_k_dense_replace: int = 0
    hidden_act: str = "silu"
    hidden_size: int = 4096
    index_head_dim: int = 128
    index_n_heads: int = 64
    index_topk: int = 512
    initializer_range: float = 0.02
    intermediate_size: int = 2048
    kv_lora_rank: int = 512
    max_position_embeddings: int = 65536
    model_type: str = "deepseek_v4"
    moe_intermediate_size: int = 2048
    moe_layer_freq: int = 1
    n_group: int = 8
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    norm_topk_prob: bool = True

    num_attention_heads: int = 64
    num_experts_per_tok: int = 6
    num_hidden_layers: int = 43
    num_key_value_heads: int = 1

    q_lora_rank: int = 1024
    qk_nope_head_dim: int = 448
    qk_rope_head_dim: int = 64

    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig)

    rms_norm_eps: float = 1e-6

    rope_scaling: Dict[str, float] = field(default_factory=dict)
    rope_theta: int = 10000

    routed_scaling_factor: float = 1.5
    # Ascend's router GEMM is a plain bf16 ``F.linear`` and returns bf16 logits.
    # The loss lands on a top-k, so it shows up as a discretely wrong expert
    # rather than a small error: measured at layer 3 / 8192 tokens, bf16 logits
    # give 0.9929 top-8 set overlap, 5.65% of tokens picking a different expert
    # set and the worst of them 34% off. ``router_fp32`` makes ``MoEGate``
    # allocate the router weight fp32 and take an fp32 ``F.linear``, which
    # restores the overlap to the bf16-input noise floor. CUDA already gets an
    # fp32 accumulation out of ``linear_bf16_fp32``, so it does not opt in.
    router_fp32: bool = field(default_factory=is_npu)
    scoring_func: str = "sqrtsoftplus"

    tie_word_embeddings: bool = False

    topk_group: int = 8
    topk_method: str = "noaux_tc"

    use_cache: bool = True
    v_head_dim: int = 512
    vocab_size: int = 129280
    o_lora_rank: int = 1024
    o_groups: int = 8
    window_size: int = 128

    compress_rope_theta: int = 40000
    compress_ratios: List[int] = field(default_factory=list)

    n_hash_layers: int = 3
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
