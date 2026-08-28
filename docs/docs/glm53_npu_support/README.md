# GLM-5.3-Flash 昇腾适配

> **新 agent 从这里开始。** 把下面「启动提示词」整段贴给它即可。

---

## 启动提示词（复制以下全部内容）

```
⚠ 先看这条，否则第一条命令就会卡死：
本机 shell 继承了全局 HTTP_PROXY / HTTPS_PROXY，但该代理**只对 GitHub / Anthropic 有效**。
访问任何其他站点（pypi 华为源、hf-mirror、gitcode、ports.ubuntu.com …）之前必须先
`unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`，否则一路超时到 timeout。
详见 SETUP.md §1 的站点/走法对照表。

你要接手 GLM-5.3-Flash 在昇腾 NPU（Atlas A3 / Ascend910_9362）上的适配工作。
仓库：当前目录（sglang-dllm fork，分支 glm53_dev）。

【第一件事：读文档，不要重新调研】
按顺序读，全部在 docs/docs/glm53_npu_support/ 下：
  1. SETUP.md  —— 环境搭建复现文档。前一轮踩过的坑都在里面，照做，别自己摸索。
  2. PLAN.md   —— 活的计划文档。§1 环境事实、§2 算子结论、§3 阶段计划(P0–P6)。
  3. probe/    —— 现成的探测脚本，别重写。
这些结论是逐条核实过的（读源码 + 上机实跑），带 file:line 证据。
GLM53_flash_ascend_support_assessment.html 是最初的评估报告，其中若干判断
**已被代码核实推翻**，以 PLAN.md §2.8 为准，不要照抄那份 html。

【当前状态（2026-08-28）】
- **环境已在 Ubuntu 24.04 上重建完毕，P0.1–P0.6 + P0.8 全部 PASS**。
  `source /mnt/workspace/y00359136/work/glm53_dev/env/env.sh` 之后用 `npy` 代替 `python` 即可干活。
- 24.04（glibc 2.39）**确认不需要 SETUP.md 附录 B 的 glibc 绕行**；vendor `.so` 全部正常加载。
- 两个模型权重都已下齐：`/mnt/workspace/models/{GLM-5.3-Flash, DeepSeek-V4-Flash-W8A8}`。
- 下一步是 **P0.7a**。

【接下来做什么】
✅ P0、P1 都已完成（环境 / 算子 / DSv4 冒烟 / GPQA 精度闭环 / rebase 到 `033446bb05`）。
现在从 **P3 逐模块对拍**开始，按 PLAN.md §3 往下走 P3 → P4 → P5 …

⚠ 磁盘：现在 915/984 GB，**剩 70 GB**。FP8 源（306 GB）仍保留，**等 P4 端到端验过再删**；
P5 的 W8A8（333 GB）那时本来也放不下它。DSv4 权重已删（元数据保留）。

（若又换机重建环境：照 SETUP.md 走一遍，24.04 的差异集中在附录 D。）

【工作方式（用户明确要求）】
- **PLAN.md 是活文档**：每完成一步就在对应条目标 [x] PASS 或 [!] FAIL，
  计划要改就直接改这个文件，并在 §5 变更日志加一行。不要另开新文档。
- **实事求是**：不确定就明确写「不确定」，并写清用什么动作能消解它。
  证据不足不要给结论。给判断时标注是「实测」还是「我的推断」。
- **打杂派 agent**：跑评测、扫清单、核实外部报告这类「输出量远大于结论量」的活，
  派 subagent 去做，别在主上下文里跑。派的时候把已知环境事实写进 prompt。
- **先讨论再动手**：大的方案变更先说清楚再执行。

【环境硬规则（会反复踩）】
- 代理 http://127.0.0.1:1056 **只对 GitHub / Anthropic 有效**。访问 pypi 华为源、
  hf-mirror、gitcode、ports.ubuntu.com 之前必须 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`，
  否则一路超时。
- CANN toolkit 在 `/home/developer/Ascend/ascend-toolkit/`，**不是** /usr/local/Ascend
  （那里只有 driver）。
- 认 SoC 用 `torch.npu.get_device_name(0)`，npu-smi 对 A2/A3 都显示 Ascend910。
- 认 CANN 版本看 `compiler/version.info`，**不要**信 ascend_toolkit_install.info 的包名。
- 绝对不要跑 `pip install -e python/`（会装 CUDA 变体顶掉 torch_npu，容器就废了）。

【一个关键判断，供你安排优先级】
PLAN.md §2.6 的结论是：BF16 首次打通**不被任何算子硬卡住** —— 确认要开发的 4 项
都有 torch 退路。所以别一上来就写 AscendC 算子，先用 torch 把精度打通（P3/P4），
算子优化放到 P6。
```

---

## 文件索引

| 文件 | 用途 | 什么时候看 |
|---|---|---|
| **[SETUP.md](./SETUP.md)** | 环境搭建复现文档 | **换机第一件事**。照着做到「算子可见性验收」通过 |
| **[PLAN.md](./PLAN.md)** | 活的计划 + 全部核实结论 | 环境好了之后。§2 算子结论，§3 阶段计划 |
| `probe/` | 探测脚本 | `p0_5_ops.py` 验环境；`p0_6_*.py` 验算子 shape |
| `launch_dsv4_a3.sh.example` | DSv4-Flash 起服务脚本（A3 TP16/DP16+DeepEP） | 冒烟 / 精度回归 |
| **[`operator_handoff/`](./operator_handoff/)** | **给算子团队的工单**：规格 + 纯 torch 参考 + pytest + 验收判据 | 派算子开发时 |
| `tools/fp8_to_bf16.py` | FP8 blockwise → BF16 逐 shard 反量化 | 换权重版本要重转时 |
| `tools/golden_kda.py` / `golden_mhc.py` | 从 HF 参考实现生成 CPU golden | 模块级数值对拍 |
| `tools/logit_check.py` | teacher-forced logprob 对拍（参考存盘、迭代秒级） | 改完接线快速验精度 |
| `env.sh.example` | 环境变量模板 | 复制到 `$ROOT/env.sh` |
| `GLM53_flash_ascend_support_assessment.html` | 最初的评估报告 | 参考。**若干判断已被推翻，见 PLAN.md §2.8** |

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
| P3 逐模块对拍 | 进行中：KDA ✅ / mHC ✅ / NoPE MLA 部分 / **kpool 阻塞** |
| P4 端到端 · P5 W8A8 · P6 性能 | 未开始 |

**当前阻塞点**：P3.4 kpool —— A3 无 fp8，索引缓存要改 int8。详见 PLAN §2.3。

**给算子团队的工单**在 [`operator_handoff/`](./operator_handoff/) —— 四个原始需求里
三个已撤销（算子本就存在），只剩 `kv_rmsnorm_rope_cache` 支持 rope=0 与 index cache 的 int8 化。
