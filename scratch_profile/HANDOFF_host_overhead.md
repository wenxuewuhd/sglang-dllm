# Handoff:定位 910B 剩余的 host 开销【已解,见 WORKING §3h;第一批优化已落地】

> **2026-07-27 已回答**:step 里 ~28ms 是裸露 host(device 空转)——
> batch 每 round 全量重建(最大项)+ 去噪状态 gather/scatter + 结果处理。等 forward ≈ kernel-sum →
> **graph 内空隙仅 2-4ms,"kernel launch 串行度"假设排除**。H20 推算裸露 host 更小 →
> 910B 为同一份 Python 付 ~1.8×(ARM CPU + transfer_to_npu 包装 + graph update 线程机制)。
> **判据落定:差在调度侧,可兑现**。优化排序见 §3h。下面原文保留作背景。

## 第一批优化已落地并验证(2026-07-27,commits `1b387db264` / `6a327bc0b2` / `48fe2df8fb`)

同卡同协议 A/B(4K/1.5K bench 400 prompts conc 72;gsm8k zero-shot 全量 1319):

| 指标 | 基线 | 优化后 |
|---|---|---|
| req/s | 1.21 | **1.25(+3.3%)** |
| 输出 tok/s | 1854 | 1917 |
| 稳态 step | ~100.0 ms | ~96.9 ms |
| scheduler 进程 CPU | 157% | **52%**(spin-wait→sleep-wait,省一个多 ARM 核) |
| gsm8k radix-on K=1 | 0.839 / 0.830(两次) | 0.828 / 0.836(两次)→ 噪声内 |
| gsm8k radix-off K=1 | 0.830 | 0.821(Δ=同代码重跑波动 ±0.009)→ 噪声内 |

**逐 commit 单点判决**(独立核账 agent,扣除采样伪影后与 wall −3.1ms 对平):
- `6a327bc0b2` alloc retained-block gather 向量化:**−1.9~−2.3 ms/round,唯一确定见效**,基线 F 段 100% 裸露、砍多少兑现多少。
- `1b387db264` K=1 走 batched 路径:净 ~−0.9(去噪发射 −1.65 / begin gather 前置 +0.74);同步原语从 spin 换 sleep,残差 +3~5ms 是否真实待 P1 打点分辨。
- `48fe2df8fb` seq_lens_cpu 复用:收益 ≈0(严格少做事,保留不记账)。

**⚠ 测量陷阱(新)**:py-spy 250Hz(4ms 栅格)对 ~100ms 强周期主循环会**频闪锁相**,把短相位段(如 2ms 的 process_batch_result)整段漏采,伪装成"优化收益"。修复:采样率与周期互质(如 `--rate 137`)+ `--idle`,并用"未改动代码段两份 profile ±15% 内"做自校验;round 速率必须在 py-spy 同窗口数 log 行,不能拿别的窗口平均。§3h 的绝对 ms 数受此影响整体偏大 ~15%(结构与排序不变)。

**剩余靶子(按序)**:F 段仍有 ~9-10ms 裸露(alloc 还剩 ~3.5-3.9 + prepare/fill_ids ~4)→ ④ 增量 batch 复用;graph-update `.join()` 5.2-5.9ms(与 forward 部分重叠,动之前先 P1 打点+msprof 看 device 起跑是否被 update 门控);begin gather 常驻化(+0.6ms,顺手做)。

## 要回答的问题

**910B 每 forward 的 kernel 工作量比 H20 少 11%(76.37 vs 85.88 ms),但 e2e 吞吐只打平(1.30 vs 1.32 req/s)。那 11% 丢在哪里?**

这是 4K/1.5K 分析里当前唯一的未解项。前置结论全部已定稿在 `WORKING_4k_selfconsistency.md`(§3g 是逐算子+吞吐总账,§0 是采样口径的坑),**先读那份**。

## 已经排除 / 已经解决的(别重复走)

- **radix 前缀缓存 = 之前最大的 host 项,已解决**。4K 长序列 + 各不相同的 prompt → 前缀命中率 0,但每个去噪步仍把 72 个请求各自 4K+ token 走一遍前缀树 → 占 scheduler CPU 的 **72%**(py-spy 实测)。`--disable-radix-cache` 后 host 208.6 → 21.2 ms/step,吞吐 1058 → 1706+ tok/s。**现在所有测量都必须带这个开关**,否则测的是 radix。
- **K=2 是 radix 开销的补丁,radix 关掉后要用 K=1**。K=2 的作用是把每轮 host 开销摊到 N 个 forward;host 降下来后它的代价(bs≥64 时 `check_early_exit=False` 无条件多跑一个 forward + 冻结批次推迟准入)变成净损失。两边实测 K=1 都赢(910B +10.7%、H20 +16.9%)。
- **不是去噪计算**。去噪阈值判定是 on-device kernel,py-spy 里 `fdfo_batched_end` 只占 ~3%。
- **1.4× 的硬件差距不存在**,已证伪(见 §3g),不要再往那个方向找。

## 当前推荐配置(已落地为脚本默认)

**radix off + K=1**,`launch_{npu,h20}_norad{,_prof}.sh` 默认就是这个,直接用即可。
- `SGLANG_DLLM_FDFO_STEPS_PER_ROUND` 的**代码默认本来就是 1**(`environ.py:429`),不需要改代码;K=2 通过 `FDFO_STEPS=2` 作为调优选项保留。
- `--disable-radix-cache` 只在脚本里传,**没有改 server_args 的全局默认**(它默认仍是开)。这是有意的:radix 对**真有共享前缀**的负载(统一 system prompt、few-shot、RL rollout)是有效的,只在长序列+各不相同 prompt 时是纯开销。若要改全局默认需另行评估。

## 已知的量(radix off 之前的旧口径,需要用 K=1 重测)

用当时的打点探针(K=2,radix off)测到:device ≈155 ms/step(=2 forward),**host ≈21 ms/step**,host 占 ~12%。
但这是 K=2 的数;**K=1 下每 step 只有 1 个 forward,host 的相对占比会翻倍**,需要重测。

## 工具

1. **打点探针(已从代码删除)** —— 每步 device/host 拆分。怎么加回来见 memory `dllm-step-timing-probe-howto`,原实现在 commit `a6b5c3ded0..26346b903d`(`git show` 可取回)。两个坑:device 用 `server_args.device`(不能用 `self.device`,init 顺序),forward 计数用 `model_runner.forward_pass_id`(不能用 `DllmAlgorithm._n_fwd`,那是本分支私有属性,别的 checkout 上会 AttributeError)。
   - ⚠ 探针本身对性能的影响**没有实测验证过**(当时准备测,用户让直接删了)。要用就先做 on/off 的吞吐对照。
2. **py-spy 采样 scheduler**(本 host 上可用,0.4.2):
   ```
   pgrep -f "sglang::scheduler"        # scheduler 是独立子进程,100% CPU 那个
   py-spy record --pid <PID> --duration 30 --rate 250 --format raw --output stacks.txt --nonblocking
   ```
   raw 格式是折叠栈,自己聚合 inclusive/self time 即可(注意文件可能有非 UTF-8 字节,用 `encoding="latin-1"` 读)。
3. **别用 profiler 量 host**。`bench_serving --profile` 不传 `with_stack`,服务端默认 **True** → 每个 op 做 Python 栈回溯 → host 被严重放大(那个"47ms GPU 空闲"就是伪值)。要抓 trace 用 `arm_profile.py`(传 `with_stack=False`),且**只信 kernel duration,不信 wall-clock/空隙**。

## 复现配置(两边已对齐)

```
# 910B                                        # H20
FDFO_STEPS=1 DEV=7 bash scratch_profile/launch_npu_norad.sh
FDFO_STEPS=1 DEV=0 bash scratch_profile/launch_h20_norad.sh

# bench(两边同参数,不带 --profile)
python -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port <31600|31400> \
  --dataset-name random --random-input-len 4096 --random-output-len 1536 --random-range-ratio 1 \
  --num-prompts 400 --max-concurrency 72 --seed 42
```
`--random-range-ratio 1` 必须带(否则长度是 [1,4096] 均匀随机,不是 4K 稳态)。真实语料很重要:合成 prompt 会让 MoE 少激活专家、系统性高估(见 §0)。

## 待查的假设(按优先级)

1. **kernel launch 串行度**。H20 每 forward 有 **1546** 个小 kernel(但 CUDA 可异步重叠),910B 只有 **230** 个——但 NPU 侧串行度更高。910B 的小算子(MoE glue 6.44 + norm/rope 4.36 = 10.8 ms)本身就比 H20(2.51 ms)慢 4.3×,如果它们还无法与其他 kernel 重叠,损失会放大。**先量 910B 的 device 空隙占比**。
2. **调度 Python 的剩余部分**。radix 拿掉后 `get_next_batch_to_run` 还剩什么?py-spy 重采一版(radix off + K=1)看新的热点分布。
3. **`process_batch_result_dllm` 的 72-req Python 循环**(每步逐请求提交 token / 更新 finish 状态 / 流式输出)。之前被 radix 盖住了,现在可能浮上来。
4. **torch_npu 的 patch 开销**:上一版 py-spy 里 `torch_npu/utils/patch_getenv.py`(每步读 env)+ `transfer_to_npu.py` 合计 ~4%,radix 拿掉后占比会变大。

## 判据

- 若 910B 的 device 空隙 / host 明显大于 H20 → 优化方向在调度侧,那 11% 的 kernel 优势可以兑现成 ~10% 吞吐。
- 若两边 host 相当 → 说明 910B 的 kernel 虽少但**串行度更高**(NPU 单流、无法重叠),那是架构性的,优化方向变成减少 kernel 数(融合小算子)。
