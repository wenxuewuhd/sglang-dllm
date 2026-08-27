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

【当前状态】
- 上一轮在 Ubuntu 20.04 上做到 P0.5（算子可见性验收）通过。
- 现在换到了 Ubuntu 24.04 新镜像，环境需要照 SETUP.md 重建一遍。
- 24.04 是 glibc 2.39 / libstdc++ 13，**SETUP.md 附录 B 的 glibc 绕行整段跳过**，
  但要先 `ldd --version` 和 `strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.29`
  确认一下再跳。
- 权重需要重新下载（GLM-5.3-Flash FP8 官方权重 + DeepSeek-V4-Flash W8A8 冒烟用）。

【接下来做什么】
1. 照 SETUP.md 重建环境，跑通 probe/p0_5_ops.py（期望输出在 SETUP.md §7）。
   同时复跑 probe/p0_6_shapes.py 和 p0_6_rope0.py —— 上一轮是在 torch_npu 2.7.1 上
   跑的，2.10.0 上要重新确认。
2. P0.7a 冒烟：DeepSeek-V4-Flash W8A8 起服务 + 单条推理。
3. P0.7b 精度：GPQA-Diamond，non-thinking，repeat 3，多 batch 并发加速。
   通过后 DSv4 权重可删（回收 275 GB）。
4. 然后按 PLAN.md §3 往下走 P1 → P2 → P3 …

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
| `env.sh.example` | 环境变量模板 | 复制到 `$ROOT/env.sh` |
| `GLM53_flash_ascend_support_assessment.html` | 最初的评估报告 | 参考。**若干判断已被推翻，见 PLAN.md §2.8** |

## 30 秒背景

GLM-5.3-Flash = 45 层混合架构（34 层 KDA 线性注意力 + 11 层 NoPE 稀疏 MLA），
mHC 取代残差，288 专家 MoE，官方 checkpoint 是 FP8 blockwise。
GPU 参考实现在 `upstream/xinyuan/glm-5.3-flash-support @ 0b9c38484e`（本地 git 对象里已有，不用联网）。

一期目标：**A3 单节点 TP16 / 纯文本 / BF16 打通 → W8A8(compressed-tensors) 闭环精度**。
多模态、MTP、长上下文 CP 一期都不做。

## 当前状态（2026-08-27）

- **P0 环境**：Ubuntu 20.04 上验证到 P0.5 通过；**正在换 Ubuntu 24.04 重建**
- **P0.7 冒烟 + GPQA 精度**：未做
- **算子结论**：确认要开发 4 项、仍不确定 2 项、已排除 3 条路线 —— 详见 PLAN.md §2
- **关键判断**：BF16 首次打通**不被任何算子硬卡住**（PLAN.md §2.6 决策表）
