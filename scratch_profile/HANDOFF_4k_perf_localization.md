# Handoff:定位 4K/1.5K 下 910B vs H20 的 1.4× 差距

## 目标

在 4K in / 1.5K out(5632 token/req)场景下,H20(bs=72,开混批)输出吞吐 **1476 tok/s**,910B(bs=72,开混批)只有 **1058 tok/s**——**H20 快 1.40×**。用逐算子 profiling 把这个差距拆清楚:是哪些算子、什么机制导致的。**目前只有 e2e 吞吐,没有逐算子对照,归因未定。**

## 背景:整份分析报告已完成并提交

- 报告:`scratch_profile/LLaDA2_910B_vs_H20_report.md`(+ 对外版 `report_artifact.html`),分支 `llada-multi-batch-opt`,已 commit。
- 模型:LLaDA2.1-mini(dLLM,MoE,20 层 / 19 MoE 层 / 256 专家 top-8,block_size=32)。
- 硬件:910B3(320 TFLOPS bf16 / 1.6 TB/s / 64GB),H20(148 TFLOPS / 4.0 TB/s / 96GB)。
- 已确立的结论(勿重复推导):
  - 短到中等序列 + 大 batch:910B 略赢(gsm8k bs=128 +7%,512/2048 +10%)。
  - 小 batch:H20 赢(bs=1 1.9×、bs=8 2.5×)。
  - **4K/1.5K:H20 赢 1.4×(本任务要拆的就是它)**。
  - roofline:MoE 的 AI=t(每专家 token 数),dense 的 AI=行数,attention 的 AI=block×GQA组=128,vector≈1。910B 算力强(2.16×)、带宽/小算子弱。

## 已知的 e2e 数据(4K in / 1.5K out,均开混批)

| | bs | 输出吞吐 | 总吞吐 | Mean TPOT | Mean TTFT |
|---|---|---|---|---|---|
| H20 | 128 (无混批) | 1024 | 3784 | 123.6 ms | 8563 ms |
| H20 | 128 | 1593 | 5886 | 65.7 ms | 14190 ms |
| H20 | 72 | 1476 | 5452 | 40.7 ms | 8700 ms |
| **910B** | **72** | **1058** | **3909** | **48.1 ms** | **17000 ms** |

关键信号:同 bs=72,H20 每 token 快 1.18×(TPOT 40.7 vs 48.1),TTFT 910B 差 2×(17s vs 8.7s)。

## 机制(已从代码核实,别再猜错)

- **prefill 按 block_size=32 逐块处理**:`schedule_policy.py:766` `_rem_tokens = min(rem_dllm_tokens, dllm_block_size)`。所以 4096-prompt = 128 个 32-chunk 逐步 prefill,**不是一次 4096,也不是稀有的独立 forward**。
- **混批**把 prefill-chunk 和 decode-block 混进同一 forward,全是 32-token 行,按 token 数分不出 prefill/decode。
- 每请求 ≈ 128 prefill-chunk + 48 decode-block = 176 个 block-step(4K 输入的 prefill 开销远大于 512 输入的 16 chunk)。
- **910B FIA 是 packed 变长(TND)**:一个 forward 的 attention 时间 = 各行 KV 之和,不是 max×bs,**没有"被最长 req 拖累"的 straggler**(实测下面那版 FIA std 只有 1%)。

## 已抓的 910B profiling(有瑕疵,仅供参考)

`scratch_profile/profiles/trace_910b_4k_bs80/`(trace_view.json + kernel_details.csv)。**瑕疵:bs 是 80 不是 72**(launch 时 `--max-running-requests 80`,feeder 灌满到 MRR),且**采样的 12 个 forward 恰好都是 2560=80×32**(混批组成不明,可能 prefill-chunk 占比不确定)。逐算子(device 57.6ms/forward,bs=80,上下文 ~4.8K):

| 类别 | ms/fwd | 占比 |
|---|---|---|
| attention (FIA) | 16.31 | 28.3%(短序列 bs128 才 7%,膨胀 2.5×;mte2=0.89 带宽受限、~0.98 TB/s ~39% 效率)|
| dense matmul | 13.85 | 24.1% |
| MoE GMM | 10.70 | 18.6%(⚠ roofline 异常:显示 2.96 TB/s,超 1.6 规格,绝对值不可信)|
| MoE 路由 | 6.98 | 12.1% |
| 去噪+vector | 5.08 | 8.8% |
| norm/rope | 4.65 | 8.1% |

初步读数:attention 在长上下文下变最大项(28%),但 910B FIA ~39% 效率不算差(**比报告 §4.2 记的 24% 高**);H20 attention 算力受限,**两边可能相当**——若如此 attention 不是 910B 输的主因。**但没有 H20 逐算子,定不了。**

## 本任务要做的

1. **910B 精确 bs=72 重抓**:launch 用 `--max-running-requests 72`(不是 80),graph max_bs=72,4K/1.5K 工作负载,Level2 profiler。确认采样含稳态(上下文 ~4.8K)。
2. **H20 同 config 抓一版**(设备无关的 `capture_4k.py` 直接能跑,H20 上去掉 `--attention-backend ascend` 用 fa3)。
3. **两边逐算子对上**:dense / MoE GMM / attention / MoE路由 / vector 各类的 ms/forward 对比,找出 910B 在哪几类被拉开。重点看:是 MoE(bs=72 下 t 更小、更带宽受限)、还是 attention(长上下文)、还是 prefill-chunk 的开销。
4. 把结论写回报告 §2.5/§4.2(现在那里写的是"机理待 profiling",别改成没数据支撑的归因)。

## 工具 / 命令

- `scratch_profile/capture_4k.py`:发固定 4096-token input_ids + max_new 1536,灌满 batch,arm Level2 profiler。设备无关。
- `scratch_profile/analyze.py`:解析 msprof kernel_details.csv,出 per-forward 逐算子分类 + roofline(GMM 计数 38/forward 归一)。**注:H20 是 kineto 不是 msprof,analyze.py 不适用 H20;H20 用 perfetto/torch key_averages。**
- 910B launch(严格 bs=72,graph,Level2):见上一 session 给的命令,核心是 `--max-running-requests 72` + `--cuda-graph-config max_bs=72` + `SGLANG_NPU_PROFILER_LEVEL2=1` + `SGLANG_ENABLE_DLLM_MIXED_BATCH=1 SGLANG_DLLM_FDFO_STEPS_PER_ROUND=2`。
- 机器共用:`npu-smi info` 看每卡 HBM(别用 `head -1`,那只取 card0);只按端口精确杀进程;个别卡有 TSD 启动超时(E39007)故障,换卡。

## 已踩的坑(别重复)

- 把 FIA shape 的 `14520` 误读成"上下文 181"——错,上下文按 attention 时间(16ms → ~0.98TB/s)反推是 ~4.8K。
- 说"prefill 稀有 2%"、"prompt 一次 prefill 4096"——都错,prefill 是 block-cap 到 32 的。
- 把 4K 的 1.4× 归因给 FIA / prefill 成本 / 调度——都没数据支撑,已从报告删除,别再写回去,除非 profiling 证实。
