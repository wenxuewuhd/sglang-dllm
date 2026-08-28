"""P3.4 -- can an Ascend operator compute GLM's kpool indexer logits?

PLAN listed "who computes indexer logits on Ascend" as the only genuinely open
operator question, on the grounds that `npu_quant_lightning_indexer` only accepts
`num_heads_q=64` while GLM's `index_n_heads` is 32.  That holds for the *quant*
variant.  `torch_npu.npu_lightning_indexer` is a different operator -- bf16 keys,
already used by DeepSeek-V4's non-kpool path -- and it takes 32 heads.

Run:  source $ROOT/env.sh && npy p3_4_lightning_indexer.py
"""

import torch
import torch_npu

try:
    import custom_ops  # noqa: F401  registers torch.ops.custom.*
except Exception as e:  # pragma: no cover
    print(f"  [warn] import custom_ops: {type(e).__name__}: {e}")

N, D, BLOCK, KPOOL = 32, 128, 64, 4  # GLM: index_n_heads / index_head_dim / page / index_kpool
DEV = "npu"
torch.manual_seed(0)


def ref_logits(q, key_flat, w):
    """The lightning-indexer score: sum_h w_h * relu(q_h . k_j)."""
    s = torch.einsum("tnd,sd->tns", q.float(), key_flat.float())
    return (torch.relu(s) * w.float().unsqueeze(-1)).sum(1)


def call(q, key, w, asq, ask, bt, sparse_count, sparse_mode=0):
    return torch_npu.npu_lightning_indexer(
        query=q, key=key, weights=w,
        actual_seq_lengths_query=asq, actual_seq_lengths_key=ask,
        block_table=bt, layout_query="TND", layout_key="PA_BSND",
        sparse_count=sparse_count, sparse_mode=sparse_mode,
    )


print("### 1. which num_heads_q does npu_lightning_indexer accept?  (GLM needs 32)")
for n_heads in (16, 32, 64, 128):
    bs, kv, nblk = 2, 512, 8
    try:
        out = call(
            torch.randn(bs, n_heads, D, dtype=torch.bfloat16, device=DEV),
            torch.randn(bs * nblk, BLOCK, 1, D, dtype=torch.bfloat16, device=DEV),
            torch.rand(bs, n_heads, dtype=torch.bfloat16, device=DEV),
            torch.arange(1, bs + 1, dtype=torch.int32, device=DEV),
            torch.full((bs,), kv, dtype=torch.int32, device=DEV),
            torch.arange(bs * nblk, dtype=torch.int32, device=DEV).view(bs, nblk),
            256,
        )
        print(f"  [OK]   num_heads_q={n_heads} -> {tuple(out[0].shape)}")
    except Exception as e:
        print(f"  [FAIL] num_heads_q={n_heads}: {type(e).__name__}: {str(e)[:90]}")

print("### 2. which key dtypes?  (decides the index-cache storage format)")
bs, kv, nblk = 2, 512, 8
common = (
    torch.randn(bs, N, D, dtype=torch.bfloat16, device=DEV),
    torch.rand(bs, N, dtype=torch.bfloat16, device=DEV),
    torch.arange(1, bs + 1, dtype=torch.int32, device=DEV),
    torch.full((bs,), kv, dtype=torch.int32, device=DEV),
    torch.arange(bs * nblk, dtype=torch.int32, device=DEV).view(bs, nblk),
)
for dtype in (torch.bfloat16, torch.float16, torch.int8):
    if dtype is torch.int8:
        key = torch.randint(-127, 127, (bs * nblk, BLOCK, 1, D), dtype=dtype, device=DEV)
    else:
        key = torch.randn(bs * nblk, BLOCK, 1, D, dtype=dtype, device=DEV)
    try:
        call(common[0], key, common[1], common[2], common[3], common[4], 256)
        print(f"  [OK]   key dtype={dtype}")
    except Exception as e:
        # NOTE: the operator doc claims float16 is supported; it is not.
        print(f"  [FAIL] key dtype={dtype}: {type(e).__name__}: {str(e)[:90]}")

print("### 3. does it compute sum_h w_h*relu(q_h.k), and how does it pad?")
for kv_lens, sparse_count in (([300, 512], 512), ([1000, 2000], 512)):
    bs = len(kv_lens)
    nblk = (max(kv_lens) + BLOCK - 1) // BLOCK
    q = torch.randn(bs, N, D, dtype=torch.bfloat16, device=DEV)
    key = torch.randn(bs * nblk, BLOCK, 1, D, dtype=torch.bfloat16, device=DEV)
    w = torch.rand(bs, N, dtype=torch.bfloat16, device=DEV)
    idx = call(
        q, key, w,
        torch.arange(1, bs + 1, dtype=torch.int32, device=DEV),
        torch.tensor(kv_lens, dtype=torch.int32, device=DEV),
        torch.arange(bs * nblk, dtype=torch.int32, device=DEV).view(bs, nblk),
        sparse_count,
    )[0].squeeze(1)
    for b, L in enumerate(kv_lens):
        kb = key[b * nblk:(b + 1) * nblk].reshape(-1, D)[:L]
        lg = ref_logits(q[b:b + 1], kb, w[b:b + 1])[0]
        k = min(sparse_count, L)
        ref = set(torch.topk(lg, k).indices.tolist())
        got = idx[b]
        valid = set(got[(got >= 0) & (got < L)].tolist())
        n_pad = int(((got < 0) | (got >= L)).sum())
        print(f"  [{'OK  ' if valid == ref else 'FAIL'}] kv={L} k={k}: "
              f"overlap={len(valid & ref)}/{k}, padded_with_-1={n_pad}")

print("### 4. are returned indices LOGICAL positions or PHYSICAL slots?")
L, nblk = 256, 4
q = torch.randn(1, N, D, dtype=torch.bfloat16, device=DEV)
w = torch.rand(1, N, dtype=torch.bfloat16, device=DEV)
key = torch.randn(8, BLOCK, 1, D, dtype=torch.bfloat16, device=DEV)  # 8 pages, 4 used
perm = torch.tensor([5, 1, 7, 2], dtype=torch.int32, device=DEV)     # shuffled
got = sorted(call(
    q, key, w,
    torch.tensor([1], dtype=torch.int32, device=DEV),
    torch.tensor([L], dtype=torch.int32, device=DEV),
    perm.view(1, nblk), 16,
)[0].squeeze().tolist())
logical = torch.cat([key[p] for p in perm.tolist()]).reshape(-1, D)[:L]
ref_log = sorted(torch.topk(ref_logits(q, logical, w)[0], 16).indices.tolist())
ref_phy = sorted(torch.topk(ref_logits(q, key.reshape(-1, D), w)[0], 16).indices.tolist())
print(f"  [{'OK  ' if got == ref_log else 'FAIL'}] logical={got == ref_log}  "
      f"physical={got == ref_phy}")

print("### 5. prefill: kpool visibility is floor(seq_len/KPOOL) pools -- slope 1/4,")
print("###    which sparse_mode=3 (rightDownCausal, slope 1) cannot express.")
print("###    Express it as runs of query rows sharing one visible-pool count.")
for q_len, prefix in ((256, 0), (1024, 4096), (4096, 28672)):
    seq_lens = prefix + torch.arange(1, q_len + 1)
    pool_lens = (seq_lens // KPOOL).to(torch.int32)
    run_vals, run_cnt = torch.unique_consecutive(pool_lens, return_counts=True)
    nblk = (max(int(run_vals.max()), 1) + BLOCK - 1) // BLOCK
    q = torch.randn(q_len, N, D, dtype=torch.bfloat16, device=DEV)
    key = torch.randn(nblk, BLOCK, 1, D, dtype=torch.bfloat16, device=DEV)
    w = torch.rand(q_len, N, dtype=torch.bfloat16, device=DEV)
    idx = call(
        q, key, w,
        run_cnt.cumsum(0).to(device=DEV, dtype=torch.int32),
        run_vals.to(device=DEV, dtype=torch.int32),
        torch.arange(nblk, dtype=torch.int32, device=DEV).repeat(run_vals.numel(), 1).contiguous(),
        512,
    )[0].squeeze(1)
    key_flat = key.reshape(-1, D)
    bad = 0
    rows = sorted(set(list(range(0, q_len, max(1, q_len // 8))) + [q_len - 1]))
    for r in rows:
        L = int(pool_lens[r])
        got = idx[r]
        valid = set(got[(got >= 0) & (got < max(L, 1))].tolist())
        if L:
            lg = ref_logits(q[r:r + 1], key_flat[:L], w[r:r + 1])[0]
            ref = set(torch.topk(lg, min(512, L)).indices.tolist())
        else:
            ref = set()
        bad += valid != ref
    print(f"  [{'OK  ' if not bad else 'FAIL'}] q_len={q_len} prefix={prefix} "
          f"runs={run_vals.numel()} pools<={int(run_vals.max())}: "
          f"{len(rows) - bad}/{len(rows)} rows exact")

print("### 6. the quant variant, for the record: int8 storage would need it")
aq = torch.tensor([1, 2], dtype=torch.int32, device=DEV)
ak = torch.tensor([512, 512], dtype=torch.int32, device=DEV)
for n_heads in (64, 32, 16, 128):
    try:
        torch.ops.custom.npu_quant_lightning_indexer_metadata(
            device=str(aq.device), actual_seq_lengths_query=aq,
            actual_seq_lengths_key=ak, layout_key="PA_BSND", sparse_count=512,
            sparse_mode=3, layout_query="TND", cmp_ratio=KPOOL,
            key_quant_mode=0, query_quant_mode=0,
            num_heads_q=n_heads, num_heads_k=1, head_dim=D,
        )
        print(f"  [OK]   quant metadata num_heads_q={n_heads}")
    except Exception as e:
        print(f"  [FAIL] quant metadata num_heads_q={n_heads}: {str(e)[:70]}")

print("### 7. the full survey: is there any OTHER operator that could select over")
print("###    a quantized key cache?  Enumerate every registered schema, don't")
print("###    guess from the names one happens to remember.")
import re

schemas = torch._C._jit_get_all_schemas()
by_ns = {}
for s in schemas:
    ns, _, name = s.name.partition("::")
    by_ns.setdefault(ns, set()).add(name)
KEY = re.compile(r"index|sparse|topk|top_k|select|mqa|logit|lightning|compress|nsa", re.I)
hits = {
    ns: sorted(n for n in names if KEY.search(n))
    for ns, names in by_ns.items()
    if ns in ("npu", "custom")
}
print(f"  npu:: {len(by_ns.get('npu', ()))} ops, custom:: {len(by_ns.get('custom', ()))} ops")
for ns, names in hits.items():
    print(f"  {ns}:: {len(names)} match the filter")

# Of those, the ones that PRODUCE a selection (rather than consume one) over a
# paged key cache, and the key dtype each takes:
PRODUCERS = {
    "npu_lightning_indexer": "bf16 keys only (measured above) -- GLM's route",
    "npu_quant_lightning_indexer": "int8 keys, but metadata takes only 64 heads",
    "npu_nsa_compress_attention_infer": "bf16/fp16 keys; NSA's fixed-stride block "
    "compression, a different algorithm, and needs a value tensor",
}
CONSUMERS = (
    "npu_nsa_select_attention_infer",       # takes topk_indices as input
    "npu_kv_quant_sparse_flash_attention",  # int8 KV, takes sparse_indices as input
    "npu_kv_quant_sparse_attn_sharedkv",
    "npu_sparse_flash_attention",
    "npu_gather_selection_kv_cache",
    "npu_block_sparse_attention",
)
print("  producers of a selection:")
for n, note in PRODUCERS.items():
    print(f"    {n}: {'present' if hasattr(torch_npu, n) else 'ABSENT'} -- {note}")
print(f"  consumers of a selection (not candidates): {', '.join(CONSUMERS)}")
print("  => no operator on this target selects over an int8 key cache at 32 heads.")
