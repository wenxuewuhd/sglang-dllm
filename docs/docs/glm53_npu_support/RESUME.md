# 接手指南（2026-08-29 上午，整网已跑通）

新 session 从这里开始，然后读 `PLAN.md`。**上一份交接（20:27 那次中断）的内容已经全部消化，本文件是新的。**

---

## 一句话现状

**GLM-5.3-Flash 在 A3 上整网跑通了**（TP16、45 层、真实 HCCL、prefill + decode、并发 ragged 批）。
**算子开发需求 0 项**（五条推断缺口全部证伪，工单包已清空）。
**图捕获五类层 + 整层同图全部验通**，但**还没在整网上开过**。

## 服务现在还开着

端口 **30003**，eager（`--disable-cuda-graph`），日志
`$ROOT/run/glm_bf16_0905.log`。要停就 `pkill -f "[s]glang.launch_server"`
（**注意方括号**，否则会把自己的 shell 一起打掉，我踩过两次）。

⚠ **停服务后显存不是立刻回收的。** `bootstrap.py:339` 要求每卡空闲 ≥ 90%，
不满足直接 raise —— 停完不到一分钟就起会失败，我踩过。

## 卡是共用的

另一个用户（`l00960396`）会不定时起 16-die 的 DSv4 训练。
**GLM BF16 只能 TP16**（TP8 每卡要 74.9 GB，放不下），所以**整网必须独占整机**。
单卡的验证任务可以并行，但要先 `npu-smi info` 看一眼，并且**跑完让进程干净退出**。

---

## 当前正在做的事：logits 对拍（第 1 级回归）

见 `REGRESSION.md` 的六级阶梯。现在卡在**地板**上。

**已有**：`scratchpad/logit_ref_bf16.json`（CPU bf16 参考，8 个短提示）。
对服务的结果：`mean|dlp|` 0.013–0.25，`max` 2.97（都在 token 0），
**dNLL 很小且正负混杂**（-0.167 ~ +0.095）。

**结论：既不能说通过，也不能说失败** —— 缺地板。

**不要用工具里原来那句 `<1e-2`** —— 那是拍脑袋写的，已删。实测：MoE 路由在
fp32/bf16 之间会翻，layer 3 就 12.5%、layer 41 到 63.3%，**地板是离散的路由差异、
随深度涨到 1.8e-1 量级**。

### 下一步：用 trace 算地板（fp32 那条已放弃）

**fp32 整模型 `from_pretrained` 跑不完** —— 实测加载到 26% 吃掉 1.26 TB 并开始换页，已杀。

改走 `layer_check/trace_reference.py`：**流式**建模型（meta device 逐层物化再退回，
峰值只有一层），能同时出 fp32 与 bf16 两份 45 层 hidden state，
128 token 约 3.5 分钟、峰值 RSS 646 GB。
`harness.py` 的 `first_divergence` 能逐层报误差与该层地板。

⚠ trace 比的是 hidden state 不是 logprob，**量级可比、不能直接换算**。

`tools/logit_check.py` 已加 `--against`：`compare --ref A.json --against B.json`
可以两份参考互比（不打服务），这是算地板用的。

---

## 之后的顺序（已定）

1. **拿到地板，判定 eager 基线** ← 现在这里
2. **开 graph**，用 `logit_check.py --ref-source server` 录 eager 参考再对。
   **判据是逐位相同**，不是容差 —— 单层实测 replay 与 eager bit-identical，
   不等就是有东西被烘进图里
3. **补 decode 与长上下文**（回归阶梯第 2、4 级）
4. **GSM8K**：eager 下**不可行**（480 ms/token → 全量要 11 小时以上），必须等 graph

---

## 欠账

`SHARED_CHANGES.md` 有 5 条已改 + 3 条待决。其中：
- **DSv4 的 GPQA 回归没跑**（swiglu_limit 那条改动欠的，基线 73.23–73.74%）
- **待决 ②** `seq_lens_cpu_list` 在捕获时被烘死 —— GLM 逃过，
  但**对任何走 FIA 的非 DSA 昇腾模型是活的静默 bug**
- `git stash` 里有 kpool 共享路径的半成品（tail kernel，作者已判定原方案是错的；
  但**实测那 4.73 ms 全部来自被 clamp 的 gather load，去掉就是 5.557 → 0.282 ms**）

## 三条最贵的教训

1. **短 prompt 测不到 kpool** —— `seq_len < index_topk=2048` 时 indexer 直接全选。
   「Paris 答对了」只证明 45 层能串起来
2. **单层全绿不等于整网对** —— 整网拉通修的三个 bug 里两个是
   「顶层对象是包装，新方法加在了被包的那个上」，单层 harness 直接构造内层，**结构上发现不了**
3. **工具里拍脑袋的阈值比没有阈值更危险** —— 它让人对着错的基准下判断
