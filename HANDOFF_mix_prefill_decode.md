# Handoff: FDFO 合并 prefill + decode 到同一 forward

## 你的任务

LLaDA2.1-mini 在 910B3 NPU 上跑 RL rollout（大 batch、追吞吐）。上一阶段已经把 FDFO
大 batch 跑通（sync@128 的 1348 tok/s → FDFO@128 ~3000 tok/s，2.2×）。这一阶段做**继 FDFO
之后最大的一块优化**：把 dLLM 的 prefill 轮和 decode 轮合并到同一个 forward。

**先只做设计 + 小步验证，每步保精度。不要一上来大改。**

## 背景（已确认的事实，可直接采信）

### 问题：47% 的 forward 是浪费的 prefill 小轮

`python/sglang/srt/dllm/mixin/scheduler.py:214` 的 `_process_dllm_batches` 是 **prefill 优先、
二选一**：有新请求就整轮做 prefill（只处理新请求的首个 prompt block），没有才做 decode（去噪
所有活跃 block）。实测（1000 题 parallel 128 稳态）：

- **47% 的轮次是小 prefill 轮（<32 blocks）**——这一整个 forward 里那 ~86-128 个正在去噪的活跃
  block 完全不前进，但照样读全部 MoE 权重（低利用率）。
- 51% 是大 decode 轮（≥64 blocks）。
- 结果：batch 在 80-127 之间波动、打不满 128（admission 管线 lag）。

### 收益预估

合并后：batch 钉在满 128、消掉浪费 forward → 短 prompt（gsm8k）**+30~50%**；长 prompt（真实
rollout）**收益更大，可能接近 3×**（见下）。**实测才准，别信估计当结论。**

### 长序列实测（2026-07-21，坐实合并是压倒性第一优先级）

用 `sglang.bench_serving --dataset-name random --random-input-len 512 --random-output-len 2048
--num-prompts 256 --max-concurrency 128`（长 prompt + 长生成，贴近 RL rollout）实测 910B@128：

| 指标 | 短 prompt（gsm8k, ~100 tok） | **长 prompt（512 tok）** |
|---|---|---|
| prefill 小轮占比 | 47% | **65%** |
| decode batch 均值 | ~90 | **43 / 128** |
| decode batch 中位 | — | **1**（一半 forward 只处理 1 个 block！） |
| KV 池占用峰值 | — | **仅 30%** |
| 吞吐 | ~3000 tok/s | 1684 tok/s |

**关键判断**：
- **KV 不是瓶颈**（只用 30%）——不要往 KV 预留方向找。瓶颈是 prefill/decode 分离。
- **长 prompt 把 prefill 浪费从 47% 放大到 65%**，batch 均值从 ~90 掉到 43，中位掉到 1。长 prompt
  的 prefill 要占很多轮，而 prefill 轮里活跃去噪 block 完全不前进 → batch 严重填不满。
- **真实 RL rollout 是长 prompt/长生成，被这个 bug 伤得最狠**。合并把 batch 从均值 43 拉到满 128，
  长序列收益可能接近 3×，不止短 prompt 的 1.3~1.5×。
- vs H20：910B 长序列 1684 tok/s 是**被调度 bug 压的，不是硬件劣势**；H20 跑同一份代码有同样 bug，
  所以现在比长序列意义不大——**合并完成后再和 H20 对比才有意义**。

复现该测量：`--decode-log-interval 1` 起 server，跑上面的 bench，grep server 日志
`grep -oE "new-seq: [0-9]+"` 看 batch 分布，`grep "token usage"` 看 KV 占用。

### 三个障碍（按难度，已调研）

1. **去噪算法的 uniform-32 假设（最难，核心障碍）**：
   `python/sglang/srt/dllm/algorithm/base.py:109,136` 和
   `python/sglang/srt/dllm/algorithm/joint_threshold.py:37,140` 都做 `input_ids.view(B, 32)`，
   **假设整批每行都是 32-token block**。prefill 行是变长 prompt，reshape 直接崩。
   → 需要改成"只对 decode 子集（32-block 那些行）做去噪步，跳过 prefill 行"，传一个 row-mask。

2. **scheduler 合并 batch（中等）**：`_process_dllm_batches` 从 either/or 改成把 prefill_reqs +
   decode_reqs 加进同一个 adder/batch。KV 分配（`prepare_for_extend`）要同时处理 decode 的块槽
   复用 + prefill 的新槽分配 → out_cache_loc = concat(decode 固定槽, prefill 新槽)，可构造。

3. **init_forward_metadata（其实最小，别被误导）**：
   `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py:433`。混合 seq_lens 它能
   处理——**attention kernel 本来就支持变长**（`forward_dllm` 的 `npu_fused_infer_attention_score`
   用 `atten_mask=None` + TND 布局 + per-request `actual_seq_lengths`，见 :1886）。合并的唯一
   代价是丢掉 metadata-reuse（`mark_forward_metadata_ready`），但 FDFO 稳态下它本就几乎不触发
   （每轮 7-8 请求 commit，metadata 每轮都变），所以损失很小。

### 关键澄清

- **attention 不是障碍**：`atten_mask=None` 的全注意力 + TND 变长，prefill(query=prompt_len) 和
  decode(query=32) 能在同一个 kernel 调用里跑。这个已经确认。
- prefill 相位 = prompt 还没进 KV（变长 prompt chunk）；decode 相位 = prompt 在 KV、去噪 32-mask
  block。相位判定见 `python/sglang/srt/dllm/mixin/req.py:48-61`。
- normal sglang 的 mixed chunked prefill（prefill+decode 同批）**被专门为 dLLM 关了**
  （`server_args.py:6807-6811`）——这是个提示：合并方向就是给 dLLM 重新启用混批，当初可能是
  因为 uniform-block 假设保守关掉的。

## 建议的做法（小步、保精度）

1. **先读透数据流**：一个 FDFO 轮从 `get_new_batch_dllm` → `_create_dllm_batch` →
   `prepare_for_extend`（KV 分配）→ `model_runner.forward` → `_run_fdfo`（去噪 step）→
   `process_batch_result_dllm`（KV commit + req 数组回填）。搞清楚 prefill 行和 decode 行在每一
   步的差异，特别是 out_cache_loc / extend_range / KV 槽。

2. **设计 row-mask 方案**：让去噪 step 知道哪些行是 32-block（参与去噪）、哪些是 prefill（只算
   KV，不去噪）。`joint_threshold_update_step_vectorized` 要能处理不整齐的 batch。

3. **改 scheduler 合并**：`_process_dllm_batches` 把 prefill + decode 合进一个 batch。注意
   `ScheduleBatch` 的 out-of-place 规则（见 `.claude/rules/schedule-batch-out-of-place-mutation.md`）
   —— dLLM 现在大量就地改 req 数组（`dllm/mixin/scheduler.py:96-98,129-131`），合并可能需要先
   处理这个。

4. **必须配对齐测试**：合并前后同 prompt、temperature 0、逐 token diff 输出。精度不能变。
   参考内存里的 `npu-perf-measurement-methodology`：e2e 吞吐这台 host 噪声大，要预热取稳态、多次
   取中位；对比按 dLLM step 严格对齐。

## 环境 / 命令

- worktree: `/workspace/code/sglang-dllm-mix`，分支 `dllm-fdfo-mix-prefill-decode`，
  基于 `02412d271f`（已含上阶段全部优化）。`pyproject.toml` 已配 NPU 版（本地，不提交）。
- 起 server（复用上阶段脚本，注意 PYTHONPATH 指向本 worktree）：
  ```bash
  cd /workspace/code/sglang-dllm-mix
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  export PYTHONPATH=$PWD/python:$PYTHONPATH
  ASCEND_RT_VISIBLE_DEVICES=<空闲卡> python -m sglang.launch_server \
    --model-path /workspace/models/LLaDA/LLaDA2.1-mini/ --trust-remote-code \
    --attention-backend ascend --dllm-algorithm JointThreshold \
    --mem-fraction-static 0.75 --max-running-requests 128 --dllm-fdfo --port 31500
  ```
- 空闲卡：`npu-smi info` 看哪张 HBM 空（0/2/7 常有别人的进程，避开）。起服务前
  `python3 -c "import torch,torch_npu; print(torch.npu.mem_get_info(N))"` 确认卡空。
- 验证精度+吞吐（预热后取稳态，≥1000 题）：
  ```bash
  python -m sglang.test.few_shot_gsm8k --num-shots 0 --num-questions 1000 --parallel 128 --port 31500
  ```
  基线（合并前）：~3000 tok/s，精度 ~0.82-0.83。目标：吞吐显著提升、精度不变。

## 坑（上阶段踩过）

- `pkill -f 'launch_server'` 无匹配时返回 1、会中断 `&&` 复合命令。分开写，或 `|| true`。
- 起 server 后 graph capture 偶发失败（`Capture cuda graph failed`），重跑一遍一般就过。
- 测吞吐 200 题喂不饱 batch，用 ≥1000 题 parallel 128。
- 完整上阶段结论看 `scratch_profile/LLaDA2_910B_perf_stage1.md` 和内存
  `llada2-mini-910b-vs-h20-recipe`。
