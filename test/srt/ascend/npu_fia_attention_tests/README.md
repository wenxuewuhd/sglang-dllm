# NPU FIA (npu_fused_infer_attention_score) 测试

本目录用于 `torch_npu.npu_fused_infer_attention_score` 的专项测试，覆盖 chunk prefill、Page Attention (PA)、NPU graph capture、以及 **block-wise attention mask**（block 间 causal、block 内 full，参考 Block-wise Diffusion）。

## 设计要点

### 1. Chunk prefill 模拟
- **Chunk 固定 4096**；**序列长度**遍历：**2048, 4096, 8192, 16384, 32768**（2k～32k），Batch=1。
- **当 S ≤ 4096**：一次 FIA，Q/KV 长度均为 S（2048/4096 可进 NPU graph capture）。
- **当 S > 4096**（如 8k、16k、32k）：按 chunk 分多次 FIA。每次 Q = 当前 chunk（最多 4096），KV = 历史 prefix + 当前 chunk；block-wise mask 用 `q_offset` 表示 query 的全局位置，保证 block 间 causal。最后将各 chunk 输出 concat 得到 (S, N, D)。
- 与 LLaDA2.0-mini 的 MHA 配置一致。

### 2. Page Attention (PA)
- **block_size (page_size)**：CANN 文档要求 128 的倍数且 128≤block_size≤512，**不支持 2048**。测试中使用 **256**（或 512）作为 PA 的 block_size。
- 自建 dummy KV cache：形状 `(num_blocks, block_size, num_kv_heads * head_dim)`，按 block 填满；`block_table` 形状 `(B, max_blocks_per_seq)`，存 block id。

### 3. Block-wise attention mask（Block-wise Diffusion 风格）
- **Block size**：32（与 PA 的 block_size 区分：此处为 mask 的“块”大小）。
- **语义**：
  - **Block 间**：causal，即 block_i 的 query 只能看到 block_0,…,block_i 的 key，不能看后续 block。
  - **Block 内**：full attention，同一 block 内所有 query 可看该 block 内所有 key。
- **与 ascend_backend / CANN 一致**：`mask[i,j] = True` 表示该 (i,j) 被 mask（不参与计算，等价于 -inf）。ascend_backend 中 `generate_mask_flag` 为 `~tril()`，即上三角为 True（causal 时上三角不参与）；CANN sparse_mode 2/3/4 要求「下三角」为可计算区域，即上三角为 mask。本实现用 `block_id_q < block_id_kv` 得到“key 在更晚 block 则 mask”，与上述约定一致。
- 实现：`mask[i,j] = True` 当且仅当 `block_id(q_i) < block_id(kv_j)`；chunked 时通过 `q_offset` 表示 query 全局位置。
- FIA 使用 **sparse_mode=0**，传入 `(1, 1, Q_S, KV_S)` 的 `atten_mask`（CANN 支持 (1,Q_S,KV_S)/(B,1,Q_S,KV_S)/(1,1,Q_S,KV_S)）；传入前保证 `.contiguous()`（文档要求不支持非连续 tensor）。

### 4. 模型参数（LLaDA2.0-mini）
- 参考 `/data/y00359136/models/llada/LLaDA2.0-mini/config.json`：
  - `num_attention_heads`: 16
  - `num_key_value_heads`: 4
  - `head_dim`: 128
  - `hidden_size`: 2048

### 5. Torch golden
- 使用 PyTorch 实现相同 block-wise 规则的 attention：`scores.masked_fill(mask, -inf)` 后 softmax 再与 V 乘。
- 与 FIA 输出做数值对比（fp16 容差）。

### 6. NPU graph capture
- 使用 `torch.npu.graph` 捕获一次 FIA 调用（含 PA + block-wise mask），再 replay，与 eager 结果对比。

### 7. Input layout：TND（与文档、ascend_backend 对齐）
- **本测试统一用 TND**：query 形状 `(Q_S, N, D)` 即 (S, num_heads, head_dim)，无 B 维；KV 为 PA 排布 `(blocknum, block_size, KV_N*D)`，文档称 PA 下 key/value 的 `input_layout` 参数无效。
- **为何用 TND**：
  - 文档：TND 场景下 **必须传入** `actual_seq_lengths` 和 `actual_seq_lengths_kv`，且以二者元素个数作为 Batch。当前单 batch，即 `[query_len]` / `[kv_len]`，与实现一致。
  - 文档：page attention 下 query 的 `input_layout` 可为 **BNSD 或 TND**；TND 时 KV 支持 (blocknum, blocksize, H) 与 (blocknum, KV_N, blocksize, D)。
  - ascend_backend：**带 PA 的路径**（decode、dllm、forward_mtp 等）均用 `input_layout="TND"`；**不带 PA 的 extend**（逐 request 传稠密 k,v）用 `input_layout="BSND"`，且注释 "TND not supports q_heads!=k_heads" 仅指该非 PA 分支。本测试为 PA 分支，GQA 通过 `num_heads`/`num_key_value_heads` 指定即可，用 TND 符合文档与现有实现。
- **TND 约束小结（文档）**：actual_seq_lengths/actual_seq_lengths_kv 必传；D=128 时支持 TND（本测试 head_dim=128）；PA 时 block_size 需 16 对齐且 ≤1024（当前 256 满足）；不支持左 padding、tensorlist、prefix、伪量化等（本测未用）。

### 8. 入图（图模式）约束
- 文档明确：**「该接口支持图模式」**，未对 TND 或 PA 入图单独增加限制。
- 本测试仅在 **S ≤ chunk_len（4096）** 时做 NPU graph capture；多 chunk 时只跑 eager，不捕获图。

## 文件说明

- `README.md`：本说明。
- `test_npu_fia_blockwise_pa_graph.py`：主测例（chunk 4096、TND+PA、block-wise mask、graph capture、torch golden）。
- `llada2_mini_config.json`：LLaDA2.0-mini 的 config 副本（仅保留 attention 相关字段，便于本地无模型路径时使用）。

## 运行

```bash
# 在 NPU 环境下
cd /path/to/sglang-dllm
python -m pytest test/srt/ascend/npu_fia_attention_tests/ -v
# 或
python test/srt/ascend/npu_fia_attention_tests/test_npu_fia_blockwise_pa_graph.py
```

## 参考

- Ascend Extension for PyTorch 7.3.0：`npu_fused_infer_attention_score` 约束说明（block_size、sparse_mode、atten_mask shape）。
- `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`：FIA 在 sglang 中的调用方式。
- Block-wise Diffusion 示意图：block 间 causal、block 内 full attention。
