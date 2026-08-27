import torch, torch_npu
DEV="npu:0"; T=8; KV=512
def mk(rope):
    kv = torch.randn(T,1,1,KV+rope, dtype=torch.bfloat16, device=DEV)
    gamma = torch.ones(KV, dtype=torch.bfloat16, device=DEV)
    cos = torch.randn(T,1,1,max(rope,1), dtype=torch.bfloat16, device=DEV)
    sin = torch.randn(T,1,1,max(rope,1), dtype=torch.bfloat16, device=DEV)
    idx = torch.arange(T, dtype=torch.int64, device=DEV)
    kc  = torch.zeros(4,128,1,max(rope,1), dtype=torch.bfloat16, device=DEV)
    ckv = torch.zeros(4,128,1,KV,          dtype=torch.bfloat16, device=DEV)
    return kv,gamma,cos,sin,idx,kc,ckv
for rope in (64, 0):
    for name in ("npu_kv_rmsnorm_rope_cache","npu_kv_rmsnorm_rope_cache_v2"):
        kv,gamma,cos,sin,idx,kc,ckv = mk(rope)
        if rope==0:
            cos = cos[...,:0].contiguous(); sin = sin[...,:0].contiguous(); kc = kc[...,:0].contiguous()
        try:
            r = getattr(torch.ops.npu,name)(kv,gamma,cos,sin,idx,kc,ckv,
                                            epsilon=1e-5, cache_mode="PA_BNSD", is_output_kv=True)
            print(f"  [OK]   rope={rope:<3} {name} -> {[tuple(t.shape) for t in r]}")
        except Exception as e:
            print(f"  [FAIL] rope={rope:<3} {name} -> {type(e).__name__}: {str(e).splitlines()[0][:150]}")
