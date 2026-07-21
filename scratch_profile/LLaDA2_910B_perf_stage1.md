# LLaDA2.1-mini 910B3 大 batch 性能优化 — 阶段一收尾

分支 `llada-multi-batch-opt` ｜ 硬件 8×910B3 64GB ｜ 场景：RL rollout（大 batch，追吞吐）

## 0. 成果总览

| 里程碑 | FDFO@128 吞吐 | 相对起点 |
|---|---|---|
| 起点：sync@128（`--no-dllm-fdfo`, mem-frac 0.85） | 1348 tok/s | 1.0× |
| + 3 个 kernel commit（整步 -12%） | — | |
| **+ FDFO + mem-frac 0.75（部署配置）** | **~3000 tok/s** | **2.2×** |
| bs=256（追极限吞吐，+9%，代价 2× 延迟） | ~3230 tok/s | 2.4× |

精度全程无回归（gsm8k zero-shot 0.82–0.84）。

**最大的杠杆不是 kernel，是调度**：FDFO 消除 sync 的锁步长尾浪费，配合 mem-fraction 调优解 OOM，单这一项 +88%。kernel 优化合计约 +12%，是锦上添花。

## 1. 部署 Recipe（直接可用）

```bash
python -m sglang.launch_server \
  --model-path <LLaDA2.1-mini> \
  --trust-remote-code \
  --attention-backend ascend \
  --dllm-algorithm JointThreshold \
  --mem-fraction-static 0.75 \      # 关键：0.85 会 OOM@128（KV 池过配挤爆运行时）
  --max-running-requests 128 \      # 均衡点；追吞吐可 256（+9%）
  --dllm-fdfo                        # 关键：默认开 FDFO（不要加 --no-dllm-fdfo）
```

- **mem-fraction 0.75 而非 0.85**：0.85 时 KV 池 549k tokens（实际只需 ~70k，过配 8×），运行时内存仅剩 4.2GB，FDFO 持续满 batch 的峰值 activations 撑爆 → OOM。0.75 把运行时还到 10.35GB。
- **FDFO 而非 sync**：sync 整批锁步跑 ~17 步再等最慢那条，batch 越大长尾浪费越大；FDFO 连续补位近线性 scale。**注意**：FDFO 在 bs=1 反而慢一半（旧结论只适用单 batch），大 batch 完全相反。
- **测吞吐要 ≥1000 题、parallel≥128、预热后取稳态**：200 题喂不饱 batch（1141 vs 1348）；该 host e2e 吞吐噪声 ±10-15%，单次不可信。

## 2. 已落地的代码改动（3 commit）

| commit | 内容 | 收益 | 精度 |
|---|---|---|---|
| `9b233174ae` | dLLM 去噪 logits 路径：NPU 保持 bf16 不物化 fp32（解 bs≥112 OOM）+ 去噪 argmax/prob 单遍融合核 | 去噪 13→6ms，整步 -6% | argmax 逐位同，prob 误差 3.8e-7 |
| `22b8be7a14` | split_qkv 融合核 batch-gated 宽 grid（大 batch 消除逐行串行退化） | split_qkv 8.6→3.0ms，整步 -6% | Q/V 逐位同，K bf16 舍入 |
| `98deebca12` | init_forward_metadata 消 dLLM 路径的 2 个 D2H 同步（用 CPU 镜像值） | +6.8% | decode/prefill 路径逐位不变 |

新增文件：`hardware_backend/npu/norm/split_qkv.py`、`hardware_backend/npu/norm/argmax_softmax_prob.py`。
新增 env：`SGLANG_NPU_FUSED_QKV_MAX_TOKENS`（split_qkv 非融合 fallback 开关）。
pyproject.toml 是本地 NPU 环境改动，未提交。

## 3. Roofline 定位（击穿 H20 的依据）

- **胜负分界 AI ≈ 92.5 flops/byte** = H20 算力 148T ÷ 910B 带宽 1.6TB/s。MoE 的 AI ≈ 每专家 token 数 ≈ EP 域活跃 block 数。
- **MoE GMM 单项 910B 赢 1.48×(126blk) → 1.96×(512blk)**（全实测）。910B GMM 天花板 191 TFLOPS（60% 峰值）。
- bs=128 时 tokens/expert≈126 已进 crossover；bs=256 时≈252 进纯算力区 → **这就是 bs>128 收益递减的原因：device 变成 MoE GMM 算力受限**，正是"击穿算力甜点"的标志。
- attention 不是劣势：dLLM attention AI=128（GQA 原生不展开），H20 在此也算力受限，4TB/s 用不上。

## 4. bs=128 时间分布（优化后，真实 ~95ms/轮）

- **device busy ~74ms (78%)**：MoE GMM 38% + dense matmul 26% + MoE 胶水 13% + norm/rope 8% + 去噪核 6% + attention 5%。MoE GMM + dense 已近算力天花板。
- **host 空转 ~21ms (22%)**：init D2H 同步（已消 ~4ms）、每 MoE 层 aclnnCat（~6-8ms）、kernel launch+sync（~3ms）、scheduler 重组。因 dLLM 禁用了 overlap scheduler，这 21ms 全暴露成 device 空转。

## 5. 验证过不可行 / 收益不足的方向（避免重复踩）

- **CV 融合（多流 cube∥vector 重叠）：NPU 不可行**。实测两独立 matmul 双流 ratio=2.02（纯串行）、cube 31ms∥vec 8.8ms 双流 39.8ms（0 重叠）、graph capture 下也 0 重叠。910B3 torch_npu 下跨流 kernel 完全串行。Ascend 的 cube/vector 并行只能靠**单 kernel 内融合**（如 FlashAttention），不能靠多流。llada2 的 `_is_cuda` 门控是对的。**GPU 的 TBO/多流心智模型不迁移到 NPU。**
- **双流 MoE（shared∥router）**：shared expert 太小（~0.09ms），不值得。
- **去噪核再优化**：BLOCK_V=16384 已贴纯读地板（4.9ms vs 4.7ms），到头。

## 6. 未做 / 下一步（后续 session）

1. **部分 overlap（最大剩余杠杆 ~15%）**：把 host prep 藏进 device compute。dLLM 禁用了 overlap scheduler；现有 future_map 基建为 AR 单 token relay 建、dLLM 未接入。需写 **dLLM 专属双缓冲**，且先把 ScheduleBatch 的就地改 req 数组重构成 out-of-place（项目规则）。工程中等偏大，有 NPU graph+多流/KV 时序正确性风险。收益上界 ~15-18ms/轮，串行地板 ~2-4ms（input_ids token 依赖藏不掉）。
2. **长序列档（本阶段完全未做）**：attention FIA 效率是关键——seq4096 时 attention 占整步 33%，而 910B FIA 只跑 24% MFU，有 24%→50% 空间。长序列还涉及 EP8 释放显存、KV 预算等。这是独立的大主题。
3. W8A8：砍 MoE 带宽、拉开对 H20 算力差距（910B int8 收益 > H20）。需客户接受 int8 精度。

## 7. 测量方法学备注

- e2e 吞吐在该 host 噪声大（±10-15%），公平对比需：同 server 预热后取稳态、丢弃首次、多次取中位。
- kernel 级用 `scratch_profile/` 下的 `profile_once.sh`（单 block 解码）/ Level2 时间轴分析 host 间隙。dLLM 一次 run_batch 内含可变去噪迭代数，比较前必须按 GMM 计数（38/迭代）归一并核对迭代数一致。
- Level2 profiling 本身放大 host 间隙 ~1.5×，绝对 host 占比要打折看（47% profiled → ~22% 真实）。
