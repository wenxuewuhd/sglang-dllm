"""P0.6 运行时 shape 探测 — GLM-5.3-Flash NoPE MLA on Ascend A3.

注意：本脚本在 torch_npu 2.7.1.post4 上跑（系统 py3.11），目标环境是 2.10.0.post4。
签名/约束可能有差异，结论需在 P0 venv 建好后复跑。
"""
import traceback
import torch, torch_npu  # noqa

DEV = "npu:0"
GLM = dict(kv_lora=512, qk_nope=256, qk_rope=0, v_head=256, n_heads=64, index_topk=2048)
DSV3 = dict(kv_lora=512, qk_nope=128, qk_rope=64, v_head=128, n_heads=128)


def sig(name):
    fn = getattr(torch_npu, name, None)
    if fn is None:
        return f"{name}: MISSING"
    d = (fn.__doc__ or "").strip().splitlines()
    return f"{name}:\n    " + "\n    ".join(d[:6])


def probe(label, fn):
    try:
        out = fn()
        print(f"  [OK]   {label} -> {out}")
        return True
    except Exception as e:
        msg = str(e).replace("\n", " ")[:260]
        print(f"  [FAIL] {label} -> {type(e).__name__}: {msg}")
        return False


print("=" * 78)
print("签名")
print("=" * 78)
for n in ("npu_sparse_flash_attention", "npu_kv_rmsnorm_rope_cache",
          "npu_kv_rmsnorm_rope_cache_v2", "npu_fused_infer_attention_score_v2"):
    print(sig(n)); print()

print("=" * 78)
print("A. npu_fused_infer_attention_score: MLA-absorbed decode, head_dim 变化")
print("=" * 78)
# MLA weight-absorbed decode: q=[B,N,1,kv_lora+rope], k=v=paged latent
for tag, D_ckv, D_rope, N in (("DSv3 512+64 / N=128", 512, 64, 128),
                              ("GLM  512+0  / N=64 ", 512, 0, 64),
                              ("GLM  512+0  / N=4  ", 512, 0, 4)):
    B, S_kv = 1, 128
    Dq = D_ckv + D_rope
    q = torch.randn(B, N, 1, Dq, dtype=torch.bfloat16, device=DEV)
    k = torch.randn(B, 1, S_kv, Dq, dtype=torch.bfloat16, device=DEV)
    v = k[..., :D_ckv].contiguous()
    probe(f"FIA {tag}", lambda q=q, k=k, v=v, N=N: torch_npu.npu_fused_infer_attention_score(
        q, k, v, num_heads=N, input_layout="BNSD", scale=1.0 / (Dq ** 0.5),
        num_key_value_heads=1, softmax_lse_flag=False)[0].shape)

print()
print("=" * 78)
print("B. npu_fused_infer_attention_score: 非 MLA, qk_head_dim=256 (GLM prefill 形态)")
print("=" * 78)
for tag, Dqk, Dv, N in (("DSv3 192/128", 192, 128, 16),
                        ("GLM  256/256", 256, 256, 4),
                        ("GLM  256/256 N=64", 256, 256, 64)):
    B, S = 1, 256
    q = torch.randn(B, N, S, Dqk, dtype=torch.bfloat16, device=DEV)
    k = torch.randn(B, N, S, Dqk, dtype=torch.bfloat16, device=DEV)
    v = torch.randn(B, N, S, Dv, dtype=torch.bfloat16, device=DEV)
    probe(f"FIA {tag}", lambda q=q, k=k, v=v, N=N, Dqk=Dqk: torch_npu.npu_fused_infer_attention_score(
        q, k, v, num_heads=N, input_layout="BNSD", scale=1.0 / (Dqk ** 0.5),
        num_key_value_heads=N, softmax_lse_flag=False)[0].shape)

print()
print("=" * 78)
print("C. npu_sparse_flash_attention: rope 是否真 Optional + head_dim=256")
print("=" * 78)
print(sig("npu_sparse_flash_attention"))
