# GLM-5.3-Flash 昇腾适配

> **新 agent 从这里开始。** 把下面「启动提示词」整段贴给它即可。

---

## 启动提示词（复制以下全部内容）

```
⚠ 先看这条，否则第一条命令就会卡死：
本机 shell 继承了全局 HTTP_PROXY / HTTPS_PROXY，但该代理只对 GitHub / Anthropic 有效。
访问任何其他站点（pypi 华为源、hf-mirror、gitcode …）之前必须先
`unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`，否则一路超时。

你要接手 GLM-5.3-Flash 在昇腾 NPU（Atlas A3 / Ascend910_9362）上的适配。
仓库：当前目录（sglang-dllm fork，分支 glm53_dev）。

【第一件事：读文档，不要重新调研】
docs/docs/glm53_npu_support/ 下，按顺序：
  1. PLAN.md   —— 环境事实、算子结论、阶段计划。**每条结论都标了证据等级**
                  （实测 / 源码 / 推断），照着用，不要重新推导。
  2. SETUP.md  —— 环境搭建。换机才需要；坑都在里面，照做别自己摸索。
  3. operator_handoff/ —— 给算子团队的工单（含纯 torch 参考实现与 pytest）。
  4. probe/、tools/ —— 现成脚本，别重写。
GLM53_flash_ascend_support_assessment.html 是最初的评估报告，**多处已被实测推翻**，
以 PLAN.md 为准。

【环境】
source /mnt/workspace/y00359136/work/glm53_dev/env/env.sh   # 然后用 npy 代替 python
参考环境（HF golden，CPU）：$ROOT/.venv-ref/bin/python —— 绝不能装进 .venv-glm53

【当前进度】
P0 环境 / P1 分支合流 / P2 BF16 权重转换：全部完成，出口判据见 PLAN 表。
P3 逐模块对拍进行中：KDA ✅、mHC ✅（已接线并验数值）、NoPE MLA 部分、kpool **路线已定、待实现**。

【下一步：P3.4 kpool】
这是唯一的关键路径。**不需要新算子，也不需要 torch 兜底**——「昇腾侧由谁算 indexer
logits」这个曾经唯一开放的问题已经关闭：答案是 torch_npu.npu_lightning_indexer
（不是被排除的 npu_quant_lightning_indexer；DSv4 非 kpool 路径已在用）。它接受 GLM 的
32 头、只吃 bf16 的 key、返回逻辑位置、并把 top-k 一起融了。实测见
probe/p3_4_lightning_indexer.py，结论与用法见 PLAN §2.6。

所以索引缓存是 fp8 → **bf16**（不是 int8：int8 确实比 fp8 准，但目前没有算子能读它，
见 PLAN §2.7）。要做的四件事：
  ① index cache 存 bf16，退掉打包的 fp32 scale 区（mem_cache/index_key_cache.py）
  ② 加 IndexerKPool.forward_npu：decode 逐行、extend 按「可见 pool 数」分段
  ③ 选择之后的展开与尾部拼接，复用已实测逐位精确的 7 个 Triton kernel
  ④ 解掉 kpool_fp8_index.py:583/589、dsa_indexer_kpool.py:1766 的非 CUDA 硬拦
kpool 的 10 个 Triton kernel 里 7 个已实测可在 triton-ascend 上跑且逐位精确，
所以这从来不是"移植整个子系统"。

跑通之后按这个顺序验精度（快 → 慢）：
  tools/logit_check.py 的 teacher-forced logprob 对拍（秒级、无采样噪声、能定位）
  → prefill/decode 的 KL 一致性 → 最后才是 GSM8K + profiling。

【工作方式（用户明确要求）】
- PLAN.md 是活文档：完成一步就更新对应条目。**只记当前事实，不记怎么走到这里的**
  —— 过程看 git log。不要另开新文档。
- 实事求是：不确定就写「不确定」，并写清用什么动作能消解。证据不足不给结论。
  **每条判断都要标是「实测」还是「推断」**。
- 派 agent 干"输出量远大于结论量"的活（跑评测、扫清单、核实外部说法），
  别在主上下文里跑。派的时候把已知环境事实写进 prompt。
- 先讨论再动手：大的方案变更先说清楚。
- 涉及算子 handoff 的改动要主动 highlight。

【一条血的教训】
从源码或签名推断出来的"算子缺口"，本项目至今命中率 0/4 —— 四次都是算子本来就存在，
只是被默认参数、同名算子、另一个模型的属性，或者一个只差 quant_ 前缀的近名算子掩盖了
（npu_lightning_indexer vs npu_quant_lightning_indexer）。
**派人力之前先花十分钟把算子跑一次。**
反过来，"能跑但算错"才是真危险：见 PLAN §2.4 的四个陷阱。

【环境硬规则】
- 代理只对 GitHub / Anthropic 有效，其余先 unset。
- CANN 在 /home/developer/Ascend/ascend-toolkit/（不是 /usr/local/Ascend，那里只有 driver）。
- 认 SoC 用 torch.npu.get_device_name(0)；认 CANN 版本看 compiler/version.info。
- 绝对不要跑 pip install -e python/（会装 CUDA 变体顶掉 torch_npu）。
- 每条 pip 都要带 -i 华为源，且装 sglang 依赖必须带 constraints（见 SETUP §8.2）。
- --page-size：GLM 必须 64，DSv4 是 128。
```

---

## 文件索引

| 文件 | 用途 | 什么时候看 |
|---|---|---|
| **[SETUP.md](./SETUP.md)** | 环境搭建复现文档 | **换机第一件事**。照着做到「算子可见性验收」通过 |
| **[PLAN.md](./PLAN.md)** | 活的计划 + 全部核实结论 | 环境好了之后。§2 算子结论，§3 阶段计划 |
| `probe/` | 探测脚本 | `p0_5_ops.py` 验环境；`p0_6_*.py` 验算子 shape；`p3_4_lightning_indexer.py` 验 kpool 打分算子 |
| `launch_dsv4_a3.sh.example` | DSv4-Flash 起服务脚本（A3 TP16/DP16+DeepEP） | 冒烟 / 精度回归 |
| **[`operator_handoff/`](./operator_handoff/)** | **给算子团队的工单**：规格 + 纯 torch 参考 + pytest + 验收判据 | 派算子开发时 |
| 算子清单的对外呈现页 | https://claude.ai/code/artifact/54dbfb20-667f-465d-84c1-ea7d0cc1a827 | 发给下游同事时。**内容来自本目录，仓库为准**；要更新必须带上这个 URL，否则会新建一个而不是更新它 |
| `tools/fp8_to_bf16.py` | FP8 blockwise → BF16 逐 shard 反量化 | 换权重版本要重转时 |
| `tools/golden_kda.py` / `golden_mhc.py` | 从 HF 参考实现生成 CPU golden | 模块级数值对拍 |
| `tools/logit_check.py` | teacher-forced logprob 对拍（参考存盘、迭代秒级） | 改完接线快速验精度 |
| `env.sh.example` | 环境变量模板 | 复制到 `$ROOT/env.sh` |
| `GLM53_flash_ascend_support_assessment.html` | 最初的评估报告 | 参考。**若干判断已被推翻，见 PLAN.md §2.5** |

## 30 秒背景

GLM-5.3-Flash = 45 层混合架构（34 层 KDA 线性注意力 + 11 层 NoPE 稀疏 MLA），
mHC 取代残差，288 专家 MoE，官方 checkpoint 是 FP8 blockwise。
GPU 参考实现在 `upstream/xinyuan/glm-5.3-flash-support @ 0b9c38484e`（本地 git 对象里已有，不用联网）。

一期目标：**A3 单节点 TP16 / 纯文本 / BF16 打通 → W8A8(compressed-tensors) 闭环精度**。
多模态、MTP、长上下文 CP 一期都不做。

## 当前状态（2026-08-28）

| 阶段 | 状态 |
|---|---|
| P0 环境 / 算子可见性 / DSv4 冒烟与精度 | ✅ GPQA 73.74%（对标 73.23%） |
| P1 分支合流（rebase 到 `033446bb05`） | ✅ 回归 GPQA 73.23% |
| P2 FP8 → BF16 权重转换 | ✅ 599 GB，全量比对通过 |
| P3 逐模块对拍 | 进行中：KDA ✅ / mHC ✅ / NoPE MLA 部分 / kpool 路线已定、待实现 |
| P4 端到端 · P5 W8A8 · P6 性能 | 未开始 |

**当前关键路径**：P3.4 kpool —— 已无阻塞。A3 无 fp8，索引缓存改存 **bf16**
（不是 int8，理由见 PLAN §2.7），打分交给 `npu_lightning_indexer`。详见 PLAN §2.3、§2.6。

**给算子团队的工单**在 [`operator_handoff/`](./operator_handoff/) —— 四个原始需求
**全部撤销**（算子本就存在），只剩 `kv_rmsnorm_rope_cache` 支持 rope=0 这一项。
