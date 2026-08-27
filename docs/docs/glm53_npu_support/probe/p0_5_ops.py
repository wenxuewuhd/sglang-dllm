import torch, torch_npu, importlib
print("### 1. python 包")
for m in ("custom_ops","sgl_kernel_npu","attentions","torch_memory_saver","deep_ep"):
    try: importlib.import_module(m); print(f"  [OK]   import {m}")
    except Exception as e: print(f"  [FAIL] import {m}: {type(e).__name__}: {str(e)[:110]}")

print("### 2. torch.ops.custom.* （SGLang 昇腾路径依赖的 10 个）")
try: import custom_ops  # noqa  注册用
except Exception: pass
need = ["compressor","inplace_partial_rotary_mul","npu_hc_post","npu_hc_pre",
        "npu_mla_prolog_v3","npu_moe_gating_top_k","npu_quant_lightning_indexer",
        "npu_quant_lightning_indexer_metadata","npu_sparse_attn_sharedkv",
        "npu_sparse_attn_sharedkv_metadata"]
for n in need:
    print(f"  [{'OK   ' if hasattr(torch.ops.custom, n) else 'MISS '}] torch.ops.custom.{n}")

print("### 3. C1/C2 关键 kernel 可导入")
for mod, fn in [("sgl_kernel_npu.mamba.causal_conv1d","causal_conv1d_fn_npu"),
                ("sgl_kernel_npu.mamba.causal_conv1d","causal_conv1d_update_npu"),
                ("sgl_kernel_npu.fla.kda_gate","fused_kda_gate_npu"),
                ("sgl_kernel_npu.fla.kda_prefill","chunk_gla_fwd_o_gk_npu"),
                ("sgl_kernel_npu.fla.kda_chunk_delta_h","chunk_gated_delta_rule_fwd_h_npu"),
                ("sgl_kernel_npu.fla.kda_target_verify","kda_target_verify_npu")]:
    try:
        m = importlib.import_module(mod); getattr(m, fn)
        print(f"  [OK]   {mod}.{fn}")
    except Exception as e: print(f"  [FAIL] {mod}.{fn}: {type(e).__name__}: {str(e)[:90]}")
