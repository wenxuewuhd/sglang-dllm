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
  3. REGRESSION.md / SHARED_CHANGES.md —— 改完跑什么；动了共享代码记哪里。
  4. probe/、tools/ —— 现成脚本，别重写。
GLM53_flash_ascend_support_assessment.html 是最初的评估报告，**多处已被实测推翻**，
以 PLAN.md 为准。

【环境】
source /mnt/workspace/y00359136/work/glm53_dev/env/env.sh   # 然后用 npy 代替 python
参考环境（HF golden，CPU）：$ROOT/.venv-ref/bin/python —— 绝不能装进 .venv-glm53

【当前进度】
**整网已跑通**（2026-08-29，TP16 / 45 层 / 真实 HCCL / prefill + decode / 并发 ragged 批）。
**算子开发需求 0 项** —— 五条推断出来的缺口逐条上机核实，五条全部证伪，工单包已清空。
五类层（DSA / KDA / MoE / mHC / dense FFN）逐层对拍全部通过，回归脚本在 layer_check/。
NPU Graph 捕获：五类层各自 + 两个完整 decoder 层捕进同一个图 + 多 bs 共池 + 2 卡 HCCL，
全部验通 —— 但**还没在整网上开过**。

【下一步】
先读 RESUME.md，它是最新的接手指南（当前卡在哪、服务开没开、卡是不是共用的）。
顺序：拿到 logits 对拍的地板 → 判定 eager 基线 → 开 graph（判据是**逐位相同**）
→ 补 decode 与长上下文 → graph 下重做性能 → GSM8K。
⚠ GSM8K 在 eager 下不可行（480 ms/token），必须等 graph。
⚠ 改完跑什么，看 REGRESSION.md 的六级阶梯；动了共享代码，记 SHARED_CHANGES.md。

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
从源码或签名推断出来的"算子缺口"，本项目至今命中率 **0/5**。前四次都是算子本来就存在，
只是被默认参数、同名算子、另一个模型的属性，或者一个只差 quant_ 前缀的近名算子掩盖了
（npu_lightning_indexer vs npu_quant_lightning_indexer）。第五次不一样：算子限制是**真的**，
但那个调用点 GLM 根本走不到。
两条教训方向相反：**派人力之前先把算子跑一次**（杀掉前四个）；
**再确认模型真的会调它**（杀掉第五个）。
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
| **[REPRODUCE.md](./REPRODUCE.md)** | **从零复现整条链**：拉代码 → 建环境 → 转权重 → 起服务 → 验精度，每步都给了期望值 | **第一次接触这个项目就看这个** |
| **[SETUP.md](./SETUP.md)** | 环境搭建复现文档（REPRODUCE 的 §2 展开） | 建环境卡住时 |
| **[PLAN.md](./PLAN.md)** | 活的计划 + 全部核实结论 | 环境好了之后。§2 算子结论，§3 阶段计划 |
| **[REGRESSION.md](./REGRESSION.md)** | 回归阶梯：改完跑什么，各级覆盖什么、抓不到什么 | **每次改动之后** |
| **[SHARED_CHANGES.md](./SHARED_CHANGES.md)** | 共享路径改动台账：动了谁、影响谁、还欠什么回归 | 改到非 NPU 专属文件时 |
| `probe/` | 探测脚本 | `p0_5_ops.py` 验环境；`p0_6_*.py` 验算子 shape；`p3_4_lightning_indexer.py` 验 kpool 打分算子 |
| `launch_dsv4_a3.sh.example` | DSv4-Flash 起服务脚本（A3 TP16/DP16+DeepEP） | 冒烟 / 精度回归 |
| [`operator_handoff/`](./operator_handoff/) | **已归档**：四条需求全部撤销，不需要任何算子开发。留作「五条推断缺口为何逐条证伪」的记录 | 想知道某条为什么被撤销时 |
| [`layer_check/`](./layer_check/) | 逐层与整层对拍、图捕获验证、统一计时口径。**双参考法在 `tolerance.py`** | 改完要验数值时 |
| 算子清单的对外呈现页 | https://claude.ai/code/artifact/54dbfb20-667f-465d-84c1-ea7d0cc1a827 | 发给下游同事时。**内容来自本目录，仓库为准**；要更新必须带上这个 URL，否则会新建一个而不是更新它 |
| `tools/fp8_to_bf16.py` | FP8 blockwise → BF16 逐 shard 反量化 | 换权重版本要重转时 |
| `tools/golden_kda.py` / `golden_mhc.py` | 从 HF 参考实现生成 CPU golden | 模块级数值对拍 |
| `tools/golden_kpool_indexer.py` + `check_kpool_indexer_npu.py` | kpool indexer 的两段式对拍：CPU 出 fp32 参考，NPU 跑真算子比选择集合 | 改完 kpool 接线后回归 |
| `tools/logit_check.py` | teacher-forced logprob 对拍。`--streaming` 出 fp32 参考、`--emit-floor`/`--floor` 做**测出来的**判据、`--decode-tokens` 覆盖 decode、`--prompt-set long` 让稀疏选择真的生效 | 改完接线快速验精度 |
| `tools/run_gsm8k.py` | GSM8K 全量（thinking 口径，与 cookbook 的 97.50% 可比） | 出口判据 / P5 量化对账 |
| `tools/bench_graph_decode.py` | 图模式 decode 吞吐，prefill 与 decode 分开量 | 性能对比 |
| `tools/check_graph_padding.py` | 并发落在非捕获桶上时，padding 行会不会踩坏真实请求 | 动了图或 KDA 状态之后 |
| `tools/check_chunked_prefill.py` | 单条序列跨 forward 被切开之后还对不对（针探 + logprob 双探针）| 动了 KDA 状态传递或 kpool 增量写入之后 |
| `env.sh.example` | 环境变量模板 | 复制到 `$ROOT/env.sh` |
| `GLM53_flash_ascend_support_assessment.html` | 最初的评估报告 | 参考。**若干判断已被推翻，见 PLAN.md §2.5** |

## 30 秒背景

GLM-5.3-Flash = 45 层混合架构（34 层 KDA 线性注意力 + 11 层 NoPE 稀疏 MLA），
mHC 取代残差，288 专家 MoE，官方 checkpoint 是 FP8 blockwise。
GPU 参考实现在 `upstream/xinyuan/glm-5.3-flash-support @ 0b9c38484e`（本地 git 对象里已有，不用联网）。

一期目标：**A3 单节点 TP16 / 纯文本 / BF16 打通 → W8A8(compressed-tensors) 闭环精度**。
多模态、MTP、长上下文 CP 一期都不做。

## 当前状态（2026-08-29）

| 阶段 | 状态 |
|---|---|
| P0 环境 / 算子可见性 / DSv4 冒烟 | ✅ GPQA 73.74%（对标公开值 73.23%，198 题）|
| P1 分支合流（rebase 到 `033446bb05`） | ✅ 回归 GPQA 73.23% |
| P2 FP8 → BF16 权重转换 | ✅ 599 GB，全量比对通过 |
| P3 逐模块对拍 | ✅ 五类层（DSA / KDA / MoE / mHC / dense FFN）端到端全验 |
| **P4 端到端** | ✅ **闭环**。TP16 整网跑通；logprob 对拍 8/8 在实测地板内；**GSM8K 97.35%**（判据 97.50%）|
| **P6 NPU Graph** | ✅ 45 层整网捕获；同 batch 宽度下与 eager **逐位相同**；decode **约 8×** |
| P5 W8A8 | ☐ 未开始（**磁盘要先删 FP8 源**）|
| P6 性能优化 | ☐ 进行中：图下的排序要重做 |

**算子开发需求 0 项** —— 五条推断出来的缺口逐条上机核实，五条全部证伪，
`operator_handoff/` 已清空。

**当前关键路径**：图模式下的性能重排。两个已量出来的方向 ——
① **prefill 没进图**（GSM8K 那种短输出高翻台负载实测 875 个 prefill 批 / 每批仅 163 token，
而 `--disable-overlap-schedule` 下每次 prefill 都完全停住 decode）；
② **长上下文 64 并发就拐**（kpool 的 device 时间，图吃不掉，见 P6.7 / P6.10 / P6.11）。
