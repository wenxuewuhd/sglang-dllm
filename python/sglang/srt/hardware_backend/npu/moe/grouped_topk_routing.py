import os
from typing import Any, Optional, Tuple, Union

import torch


def _to_cpu_maybe_tensor(t: Any) -> Any:
    """Move tensor to CPU and detach for portability; leave non-tensors as-is."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().clone()
    return t


def save_grouped_topk_routing_inputs(
    dir_or_path: str,
    x: torch.Tensor,
    k: Union[int, torch.Tensor],
    *,
    bias: Optional[torch.Tensor] = None,
    k_group: int = 1,
    group_count: int = 1,
    group_select_mode: int = 0,
    renorm: int = 0,
    norm_type: int = 1,
    out_flag: bool = False,
    routed_scaling_factor: float = 1.0,
    eps: float = 1e-20,
    N: Optional[int] = None,
    block_top_m: Optional[int] = None,
    pool_delta: float = 0.0,
    load_lambda: float = 0.0,
    load_log: bool = True,
    load_from: str = "sigmoid",
    load_temp: float = 1.0,
    return_debug: bool = False,
    index: Optional[int] = None,
) -> str:
    """
    Save all inputs to grouped_topk_routing_ for later replay / golden comparison.
    Tensors are moved to CPU so the file can be loaded on any device.
    dir_or_path: directory (will write grouped_topk_inputs.pt or grouped_topk_inputs_{index}.pt) or full path.
    index: if not None and dir_or_path is a dir, filename becomes grouped_topk_inputs_{index}.pt.
    Returns the actual path written.
    """
    data = {
        "x": _to_cpu_maybe_tensor(x),
        "k": int(k.item()) if isinstance(k, torch.Tensor) else int(k),
        "bias": _to_cpu_maybe_tensor(bias) if bias is not None else None,
        "k_group": k_group,
        "group_count": group_count,
        "group_select_mode": group_select_mode,
        "renorm": renorm,
        "norm_type": norm_type,
        "out_flag": out_flag,
        "routed_scaling_factor": routed_scaling_factor,
        "eps": eps,
        "N": N,
        "block_top_m": block_top_m,
        "pool_delta": pool_delta,
        "load_lambda": load_lambda,
        "load_log": load_log,
        "load_from": load_from,
        "load_temp": load_temp,
        "return_debug": return_debug,
    }
    if os.path.isdir(dir_or_path):
        base = "grouped_topk_inputs"
        path = os.path.join(dir_or_path, f"{base}_{index}.pt" if index is not None else f"{base}.pt")
    else:
        path = dir_or_path if dir_or_path.endswith(".pt") else dir_or_path + ".pt"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(data, path)
    return path


def save_grouped_topk_routing_golden_outputs(
    dir_or_path: str,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    index: Optional[int] = None,
) -> str:
    """
    Save golden outputs (topk_weights, topk_ids) from grouped_topk_routing_.
    Tensors are moved to CPU for portability.
    dir_or_path: directory (will write grouped_topk_golden_outputs.pt or grouped_topk_golden_outputs_{index}.pt) or full path.
    index: if not None and dir_or_path is a dir, filename becomes grouped_topk_golden_outputs_{index}.pt.
    Returns the actual path written.
    """
    data = {
        "topk_weights": _to_cpu_maybe_tensor(topk_weights),
        "topk_ids": _to_cpu_maybe_tensor(topk_ids),
    }
    if os.path.isdir(dir_or_path):
        base = "grouped_topk_golden_outputs"
        path = os.path.join(dir_or_path, f"{base}_{index}.pt" if index is not None else f"{base}.pt")
    else:
        path = dir_or_path if dir_or_path.endswith(".pt") else dir_or_path + ".pt"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(data, path)
    return path


def _routing_debug_stats(topk_ids, mask_e, mask_s, k_val, N):
    """Return a dict of tensor stats for debugging (fully tensor-based)."""
    out = {"topk_ids": topk_ids, "mask_e": mask_e, "k_val": k_val}
    if mask_s is not None:
        out["mask_s"] = mask_s
    if N is not None:
        out["N"] = N
    return out


# ---------------------------------------------------------------------------
# torch_npu 接口列表 (更新时间 2025/12/23) — 与 MoE routing 相关的算子
#
# 【已用 / 可直接用】
#   - torch_npu.npu_moe_gating_top_k_softmax  对 gating 做 Softmax + topk
#   - torch_npu.npu_moe_gating_top_k         对 x 做 Sigmoid、分组排序、取前 k 个专家（本文件 fused_topk_npu 已用）
# 若 grouped_topk_routing_ 不开启 soft concentration，可优先走 npu_moe_gating_top_k 整段融合。
#
# 【列表中存在、可替代当前子算子的接口】
#   - torch_npu.npu_scaled_masked_softmax    计算「缩放 + mask 遮蔽」后的 Softmax，可试用于 load_from="softmax" 的 per-token 概率
#   - torch_npu.npu_one_hot                 用 index 生成 one-hot，可试用于从 S 构造 mask（需核对 on_value/off_value 与维度）
#   - torch_npu.scatter_update / scatter_update_  按轴 axis 与索引 indices 将 updates 写入 data，可试用于 mask_s（scatter ones at S）
#
# 【列表中未出现的接口】
#   列表内无独立 npu_sigmoid、npu_softmax、npu_topk，故 _npu_sigmoid_if_available / _npu_softmax_if_available
#   会回退到 torch；npu_topk 亦不存在，topk 保持 torch.topk。若后续版本增加上述算子，当前 getattr 会自动用上。
#
# 【其他相关】npu_moe_init_routing / npu_moe_compute_expert_tokens / npu_grouped_matmul / npu_moe_finalize_routing 等见 MoE 融合链路。
# ---------------------------------------------------------------------------

def _is_npu_tensor(x: torch.Tensor) -> bool:
    """Safe check for NPU tensor (torch_npu may not be imported)."""
    return getattr(x, "is_npu", False)


def _npu_sigmoid_if_available(x: torch.Tensor) -> torch.Tensor:
    """Use torch.ops.npu.npu_sigmoid when available, else torch.sigmoid (graph-friendly)."""
    op = getattr(torch.ops.npu, "npu_sigmoid", None)
    if op is not None and _is_npu_tensor(x):
        return op(x)
    return torch.sigmoid(x)


def _npu_softmax_if_available(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Use torch.ops.npu.npu_softmax when available, else torch.softmax."""
    op = getattr(torch.ops.npu, "npu_softmax", None)
    if op is not None and _is_npu_tensor(x):
        return op(x, dim=dim)
    return torch.softmax(x, dim=dim)


def _npu_topk_if_available(input_: torch.Tensor, k: int, dim: int = -1):
    """Use torch.ops.npu.npu_topk when available (return values, indices); else torch.topk.
    Call only after confirming API from torch_npu 接口列表 (e.g. 7.2.0 doc)."""
    op = getattr(torch.ops.npu, "npu_topk", None)
    if op is not None and _is_npu_tensor(input_):
        out = op(input_, k=k, dim=dim)
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            return out[0], out[1]
    return torch.topk(input_, k=k, dim=dim)


def grouped_topk_routing_(
    x: torch.Tensor,                         # [T, E]
    k: Union[int, torch.Tensor],             # int or 0-d tensor
    *,
    bias: Optional[torch.Tensor] = None,     # [E] or None
    k_group: int = 1,
    group_count: int = 1,
    group_select_mode: int = 0,              # 0: max-in-group, 1: top2-sum-in-group
    renorm: int = 0,                         # only support 0 per spec
    norm_type: int = 1,                      # 1: sigmoid, 0: softmax
    out_flag: bool = False,
    routed_scaling_factor: float = 1.0,
    eps: float = 1e-20,
    # ---------------- NEW: "soft" block concentration knobs ----------------
    N: Optional[int] = None,                 # target pool size (e.g., 16). If None -> baseline behavior.
    block_top_m: Optional[int] = None,       # if None: m = ceil(T*k/N)
    # Soft-bias (pool preference): score' = score + pool_delta for e in S
    pool_delta: float = 0.0,                 # small positive, e.g., 1e-3 ~ 1e-1 (depends on score scale)
    # Soft load penalty (encourage reuse): score' = score - lambda * log(1 + load_e)
    load_lambda: float = 0.0,                # small positive, start from 0, then try 1e-3 ~ 1e-1
    load_log: bool = True,                   # True: log(1+load), False: load
    # Soft load estimation uses probabilities from score:
    load_from: str = "sigmoid",              # "sigmoid" or "softmax"
    load_temp: float = 1.0,                  # temperature for load prob; >1 softer, <1 sharper
    # Optional debug stats (kept fully tensor-based)
    return_debug: bool = False,              # if True: returns an extra dict of tensor stats
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, dict],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
]:
    """
    Baseline grouped_topk_routing + an optional *soft* block-level concentration mechanism that is fully tensor-based
    (graph-friendly; no Python while/swap loops).

    If N is provided:
      1) Compute standard per-token eligibility mask via group selection (same as baseline).
      2) Compute a block-level expert pool S (|S|=N) using top-m-sum aggregation over tokens.
      3) Apply *soft* concentration:
          - pool bias: +pool_delta for experts in S
          - load penalty: -load_lambda * f(load_e) where load_e is a block-level soft usage estimate
      4) Run final per-token top-k over eligible experts using the adjusted scores.

    This does NOT hard-enforce union<=N, but will typically reduce union towards N with minimal accuracy loss
    compared to hard masking. All operations are fixed-shape tensor ops.

    Returns:
      topk_w:   [T, k] same dtype as x
      topk_ids: [T, k] int64
      norm_out: [T, E] same dtype as x (if out_flag=True)
      debug: dict of tensor stats (if return_debug=True)
    """

    # ------------------------
    # 0) Basic validation
    # ------------------------
    assert x.dim() == 2, f"x must be 2D [T,E], got {x.shape}"
    T, E = x.shape

    if isinstance(k, torch.Tensor):
        k_val = int(k.item())
    else:
        k_val = int(k)

    if renorm != 0:
        raise ValueError("renorm currently only supports 0 per spec.")

    if group_count <= 0:
        raise ValueError("group_count must be > 0.")
    if E % group_count != 0:
        raise ValueError(f"x.shape[-1]={E} must be divisible by group_count={group_count}.")
    group_size = E // group_count
    if group_size <= 2:
        raise ValueError(f"E/group_count must be > 2 per spec, got {group_size}.")

    if not (1 <= k_group <= group_count):
        raise ValueError(f"k_group must be in [1, group_count], got {k_group}.")

    max_k_allowed = (E // group_count) * k_group
    if not (1 <= k_val <= max_k_allowed):
        raise ValueError(f"k must be in [1, {max_k_allowed}] given group_count/k_group, got {k_val}.")

    if N is not None:
        N = int(N)
        if N <= 0:
            raise ValueError("N must be positive when provided.")
        if N > E:
            raise ValueError(f"N={N} cannot exceed num_experts E={E}.")

    if load_from not in ("sigmoid", "softmax"):
        raise ValueError("load_from must be 'sigmoid' or 'softmax'.")
    if load_temp <= 0:
        raise ValueError("load_temp must be > 0.")

    neg_large = -1e30  # scalar for _masked_scores (avoids 0-dim tensor alloc)

    # 硬编码走 Triton（无 fallback）：不需要 debug 时只走 Triton，报错即抛出
    #if not return_debug:
    #    from sglang.srt.hardware_backend.npu.moe import topk_triton
    #    result = topk_triton.grouped_topk_routing_triton_full(
    #        x,
    #        k_val,
    #        bias=bias,
    #        k_group=k_group,
    #        group_count=group_count,
    #        group_select_mode=group_select_mode,
    #        norm_type=norm_type,
    #        out_flag=out_flag,
    #        routed_scaling_factor=float(routed_scaling_factor),
    #        eps=float(eps),
    #        N=N,
    #        block_top_m=block_top_m,
    #        pool_delta=pool_delta,
    #        load_lambda=load_lambda,
    #        load_log=load_log,
    #        load_from=load_from,
    #        load_temp=load_temp,
    #    )
    #    topk_w, topk_ids, norm_out_opt = result
    #    if out_flag and norm_out_opt is not None:
    #        return topk_w, topk_ids, norm_out_opt.to(x.dtype)
    #    return topk_w, topk_ids

    # 以下仅当 return_debug=True 时执行（Python 路径，用于 debug 统计）
    # ------------------------
    # 1) norm(x) in fp32; prefer torch_npu ops when available
    # ------------------------
    x_fp32 = x.float()
    if norm_type == 1:
        norm_fp32 = _npu_sigmoid_if_available(x_fp32)
    elif norm_type == 0:
        norm_fp32 = _npu_softmax_if_available(x_fp32, dim=-1)
    else:
        raise ValueError(f"norm_type must be 0(softmax) or 1(sigmoid), got {norm_type}.")

    if bias is not None:
        assert bias.dim() == 1 and bias.numel() == E, f"bias must be [E], got {bias.shape}"
        norm_fp32 = norm_fp32 + bias.float().view(1, E)

    w_base_fp32 = norm_fp32
    # norm_out only when out_flag (avoid [T,E] copy when not needed)

    # ------------------------
    # 3) group reshape, 4) group score and select k_group groups -> mask_e
    # ------------------------
    norm_g = norm_fp32.view(T, group_count, group_size)
    if group_select_mode == 0:
        group_score = norm_g.max(dim=-1).values  # [T, G]
    elif group_select_mode == 1:
        group_score = torch.topk(norm_g, k=2, dim=-1).values.sum(dim=-1)  # [T, G]
    else:
        raise ValueError(f"group_select_mode must be 0 or 1, got {group_select_mode}.")

    _, sel_groups = _npu_topk_if_available(group_score, k_group, dim=-1)  # [T, k_group]
    mask_g = torch.zeros((T, group_count), device=x.device, dtype=norm_fp32.dtype)
    mask_g.scatter_(dim=1, index=sel_groups, value=1.0)
    mask_e = mask_g.unsqueeze(-1).expand(T, group_count, group_size).reshape(T, E)  # [T, E] float 0/1

    def _masked_scores(scores: torch.Tensor) -> torch.Tensor:
        return mask_e * scores + (1.0 - mask_e) * neg_large

    # Baseline path if no soft concentration requested
    if N is None or (pool_delta == 0.0 and load_lambda == 0.0):
        masked_scores = _masked_scores(norm_fp32)
        topk_w_fp32, topk_ids = _npu_topk_if_available(masked_scores, k_val, dim=-1)
        topk_ids = topk_ids.to(torch.int64)  # downstream expects int64

        denom = topk_w_fp32.sum(dim=-1, keepdim=True) + float(eps)
        topk_w_fp32.div_(denom).mul_(float(routed_scaling_factor))
        topk_w = topk_w_fp32.to(dtype=x.dtype)

        if return_debug:
            dbg = _routing_debug_stats(topk_ids, mask_e, None, k_val, N)
            if out_flag:
                return topk_w, topk_ids, norm_fp32.to(dtype=x.dtype), dbg
            else:
                return topk_w, topk_ids, dbg

        if out_flag:
            return topk_w, topk_ids, norm_fp32.to(dtype=x.dtype)
        else:
            return topk_w, topk_ids

    # ------------------------
    # Soft block concentration path (NPU-optimized: no transpose, float masks)
    # ------------------------
    eligible_score = _masked_scores(norm_fp32)

    # NPU: avoid math.ceil (graph break); use integer arithmetic
    if block_top_m is None:
        m = (T * k_val + N - 1) // N
        m = max(1, min(m, T))
    else:
        m = int(block_top_m)
        m = max(1, min(m, T))

    # NPU: pool build without transpose; topk(dim=0) on [T,E] gives top-m tokens per expert -> [m,E], then sum(0) -> [E]
    # (avoids transpose + topk on [E,T] which often falls back to AICPU)
    topm_vals = _npu_topk_if_available(eligible_score, m, dim=0)[0]   # [m, E]
    A = topm_vals.sum(dim=0)                                          # [E]
    S = _npu_topk_if_available(A, N, dim=-1)[1]                       # [N]

    # 2) Adjust scores: pool bias + load penalty (in-place to reduce allocations)
    # Pool bias: index-add only on S columns, no need to build full mask_s_f here.
    score_adj = eligible_score.clone()
    if pool_delta != 0.0:
        score_adj[:, S] = score_adj[:, S] + float(pool_delta)

    # Build mask_s_f only when needed (load penalty or debug); avoids E-sized scatter when only pool_delta.
    if load_lambda != 0.0 or return_debug:
        mask_s_f = torch.zeros((E,), device=x.device, dtype=norm_fp32.dtype)
        mask_s_f.scatter_(0, S, torch.ones(N, device=x.device, dtype=norm_fp32.dtype))

    if load_lambda != 0.0:
        s = score_adj if load_temp == 1.0 else score_adj / float(load_temp)  # [T,E]
        if load_from == "sigmoid":
            p = _npu_sigmoid_if_available(s)
            p = mask_e * p
        else:
            s_masked = mask_e * s + (1.0 - mask_e) * neg_large
            p = torch.softmax(s_masked, dim=-1)

        load_e = p.sum(dim=0)  # [E]
        if load_log:
            penalty = torch.log(1.0 + load_e)
        else:
            penalty = load_e

        pen = penalty.unsqueeze(0)
        outside_f = (1.0 - mask_s_f).unsqueeze(0)
        score_adj.sub_(float(load_lambda) * pen * outside_f)

    # score_adj is already masked; skip redundant _masked_scores
    _, topk_ids = _npu_topk_if_available(score_adj, k_val, dim=-1)  # [T,k]
    topk_ids = topk_ids.to(torch.int64)  # downstream expects int64

    # 4) Gather weights from baseline scores (NOT adjusted scores)
    topk_w_fp32 = torch.gather(w_base_fp32, dim=1, index=topk_ids)  # [T,k]

    # 5) Final renorm + scaling (same as baseline, in-place)
    denom = topk_w_fp32.sum(dim=-1, keepdim=True) + float(eps)
    topk_w_fp32.div_(denom).mul_(float(routed_scaling_factor))
    topk_w = topk_w_fp32.to(dtype=x.dtype)

    if return_debug:
        mask_s = (mask_s_f > 0)  # bool only for debug dict
        dbg = _routing_debug_stats(topk_ids, mask_e, mask_s, k_val, N)
        if out_flag:
            return topk_w, topk_ids, norm_fp32.to(dtype=x.dtype), dbg
        else:
            return topk_w, topk_ids, dbg

    if out_flag:
        return topk_w, topk_ids, norm_fp32.to(dtype=x.dtype)
    else:
        return topk_w, topk_ids
