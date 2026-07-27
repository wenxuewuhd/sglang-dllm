# LLaDA2.1-mini:910B3 vs H20 分析报告(内部版)

> 目标:评估 LLaDA2.1-mini(dLLM,MoE,20 层 / 19 MoE 层 / 256 专家 top-8)在 Ascend 910B3 上能否、以及如何达到或超过 H20,面向客户 RL rollout 场景(允许大 batch,吞吐优先)。
>
> 数据口径:
> - 【910B 实测】= 910B3 上 msprof Level2 / e2e bench 实测。
> - 【H20 实测】= H20 上抓的 kineto trace;kernel 时间与构成为实测。
> - 【H20 roofline】= roofline 推算(无 trace 覆盖处),依赖对 H20 MFU 的假设。
> - 4K/1.5K 档已用两边同口径(radix off + K=1、真实 ShareGPT、稳态 arm)的干净数据对齐(§2.5);短序列 bs=128 逐类(§2.3)为早期非饱和 trace,方法已被 §2.5 取代,趋势为准。

---

## 结论

1. 大 batch 下 910B 与 H20 基本打平。短到中等序列 910B 略优:512in/2048out 输出吞吐 2591 vs 2364(+10%),gsm8k +7%(3667 vs 3412)。4K in/1.5K out 长序列在同 bs=72、正确配置(关 radix 前缀缓存 + STEPS_PER_ROUND=1)下也打平:910B 2004 vs H20 2032 tok/s,逐算子层面 910B 的 kernel 工作量还少 ~11%。H20 的整体小幅领先来自显存容量(96GB 可上 bs=128),不是单位算力或带宽——bs=128 以 TPOT +63% 为代价换总吞吐。
2. 小 batch 下 910B 明显落后:bs=1 时 910B 约为 H20 的 0.5×(312 vs 592),bs=8 约 0.4×。
3. EP(专家并行)通信代价过大,当前不划算:128 blocks/卡 实测 a2a 12.2ms/层,节点吞吐只有 4 张独立单卡的 1/4.6。

---

## 第一章 Roofline 分析

### 1.1 两颗芯片的算力 / 带宽配比

| | 算力 (bf16) | 带宽 (HBM) | 显存 | Ridge point (算力/带宽) |
|---|---|---|---|---|
| 910B3 | 320 TFLOPS | 1.6 TB/s | 64 GB | 200 flops/byte |
| H20 | 148 TFLOPS | 4.0 TB/s (有效 ~3.4) | 96 GB | 37 flops/byte |
| 比值 | 910B 快 2.16× | H20 快 2.5× | H20 大 1.5× | — |

算力比(2.16×)与带宽比(2.5×)接近互为倒数。因此:算力受限的算子 910B 快约 2.16×,带宽受限的算子 H20 快约 2.5×,计算与带宽各占一半的 workload 两者相当。胜负取决于给定 bs 下各算子落在 ridge 的哪一侧。

### 1.2 各算子的算术强度(AI)与瓶颈

算术强度 **AI = 计算量(FLOP) / 访存量(byte)**,决定算子落在 ridge 哪一侧:AI > ridge 算力受限,AI < ridge 带宽受限。ridge = 算力峰值 / 带宽(910B 320T/1.6 = 200,H20 148/4.0 = 37)。各算子 AI 的算法:

- **GEMM(有权重复用)**:`AI = 2·M·K·N / (K·N·2字节) = M`,即参与该权重的行数。MoE 的 M = 每专家 token 数 t;dense/lm_head 的 M = 总 token 数 = bs×block。
- **attention**:`AI = KV 复用次数 = block × GQA组`(dLLM:32 × 4 = 128;自回归 decode block=1 只复用 4)。
- **vector**:每元素读一次、写一次、O(1) 计算 → `AI ≈ 1`。

例(bs=128):MoE 的 t = 128×32×8/256 = **128** → AI=128;dense 的 M = 128×32 = **4096** → AI=4096;attention = 32×4 = **128**。

| 算子 | AI(算术强度) | 910B 瓶颈 (ridge 200) | H20 瓶颈 (ridge 37) | 相对 |
|---|---|---|---|---|
| MoE GMM | = t(每专家 token 数) | bs=128→t=128<200 带宽 | t=128>37 算力 | 随 t 变 |
| dense / lm_head | = M(行数)= bs×32 = 4096 | 4096>>200 算力 | 4096>>37 算力 | 910B ~2× |
| attention (FIA) | = 128(GQA 16Q/4KV 不展开) | 128<200 边缘 | 128>37 算力 | ~平 |
| vector(去噪规约/norm/rope/softmax) | ≈ 1(无权重复用) | 带宽 | 带宽 | H20 ~2.5× |
| MoE 路由(gating/dispatch/combine) | 小算子,访存/scalar | 访存 | 访存 | H20(fused routing) |

只有 MoE 的 AI 随 bs 变化:MoE 的 AI = 每专家 token 数 t ∝ bs(`AI = 2·t·H·I / 2·H·I = t`),所以打大 batch 只对 MoE 有意义。vector 与 attention 的 AI 都与 bs 无关,但值差别大:

- vector ≈ 1(rmsnorm/rope/softmax 无权重复用),带宽受限,H20 有利。
- attention ≈ 128,由 dLLM block(32)× GQA 组(16Q/4KV=4)的 KV 复用决定(自回归 decode block=1 仅复用 4,AI≈4 带宽受限;dLLM 的 block 把 AI 抬到算力区)。AI=128 时 attention 在 H20 上算力受限、910B 上带宽受限,两者相当。

**例:同一模型,blockwise dLLM vs 自回归(AR)的算术强度。** 差别只在每 forward 每请求处理的 token 数——AR decode = 1,blockwise dLLM = block(32),这把有权重复用的算子(dense/MoE)的 AI **×32**。

| 算子(AI 公式) | AR bs=1 | AR bs=128 | dLLM bs=1 | dLLM bs=128 |
|---|---|---|---|---|
| dense/lm_head(= M 行数) | 1 | 128 | 32 | **4096** |
| MoE GMM(= t) | 0.03 | 4 | 1 | **128** |
| attention(= 复用次数) | 4 | 4 | 128 | 128 |
| vector | ~1 | ~1 | ~1 | ~1 |

贴回 ridge(910B 200 / H20 37):dense 进算力区(AI>200)AR 需 bs≥200、**dLLM 只需 bs≥7**;MoE 甜点(t≥512)AR 需 bs≥16384、**dLLM 只需 bs≥512**;attention 从 AR 的 AI=4(带宽受限、favors H20)抬到 dLLM 的 128。**AR 小 bs 全是带宽受限(favors H20);dLLM 靠 block 把算子推向算力受限(favors 910B)——这是 dLLM 本身就更适合算力强芯片的根因。**

### 1.3 MoE 在两颗芯片上的瓶颈

| t (每专家 token) | 910B(ridge 200) | H20(ridge 37) |
|---|---|---|
| t=32 (小 bs) | 深度带宽受限,~16% 峰值 | 已过 ridge,~45% 峰值 |
| t=128 (单卡 bs=128) | 带宽受限,39% 峰值【910B 实测 126 TFLOPS】 | 算力受限,~88% 峰值【H20 实测 130 TFLOPS】 |
| t=512 (需 EP 堆) | 进算力区,~60% 峰值(189 TFLOPS) | 已算力饱和(roofline,无 trace) |

39% / 126T 可由带宽反推(用实际带宽,非理论峰值):

- MoE 每 forward 读约 30 GB 权重(19 层 × 256 专家 bf16,由 GEMM shape 得)。
- 实测 30.9ms,对应实际带宽 30 / 0.0309 = 0.97 TB/s。
- 带宽受限的有效算力 = AI × 实际带宽 = 128 × 0.97 = 124 TFLOPS,与实测 126T 一致,即 320T 的 39%。

注:实际带宽 0.97 TB/s 只有规格 1.6 的 61%(HBM 可达带宽一般为规格的 60-85%,此 256 专家分块访问模式为 61%,MTE2 load pipe 已 97% 占用)。按理论 1.6 算得 205T(64% 峰值);实测 39% 与之的差即带宽折损(64% × 61% = 39%)。ridge(200)按理论带宽算,落到实测 MFU 需乘此折损。H20 规格 4.0、有效 ~3.4(85%),两颗芯片的理论/实际带宽差都需计入。

bs=128 时 MoE 两者相当:910B 带宽受限 39%(126T)≈ H20 算力受限 88%(130T),绝对值接近,MoE 在 bs=128 未进入甜点(需 t≥512)。

注(MoE 测量口径):MoE 有效带宽对 prompt 多样性敏感——**合成/重复 prompt 会让专家激活不满、系统性高估 MoE 性能(实测可高估 1.4×)**,只有真实多样语料(ShareGPT)才读满 256 专家。4K decode 真实语料下 MoE 实测有效带宽 1.17–1.23 TB/s(规格 73–77%),比"深度带宽饥饿"略乐观。凡涉及 MoE 的 roofline 归因必须用真实语料。

### 1.4 小结

小 bs(≤128):MoE 在 910B 带宽受限(仅用 39% 算力),其 2× 算力优势用不出;vector(AI≈1)带宽受限,H20 有利;attention(AI≈128)两者相当。净结果是 910B 靠 dense GEMM 的算力优势扳平 vector 的带宽劣势,整体约 1.06× 的小幅领先或相当(见第二章实测)。

大 bs:只有把 t(每专家 token)推到 ≥512,910B 的 MoE 才进算力区、2× 算力优势才兑现。单卡 bs=128 只有 t=128,达不到 512;堆 t 需 DP+EP(见第三章)。H20 的 MoE 在 t=29 即饱和(ridge 37),打大 bs 对其 MoE 帮助很小,因此大 bs 是 910B 的杠杆、不是 H20 的。

超过 H20 的路径是把 MoE 推进算力区(大 t),让 910B 的 2× 算力压过它在 vector/attention 上的带宽劣势;小 bs 无此空间。

---

## 第二章 单卡优化与 bs=128 实测

### 2.1 单卡优化项

按收益排序(均【910B 实测】):

收益列统一为单卡吞吐提升(融合核的整步耗时下降按吞吐等价折算):

| 改进 | 机制 | 吞吐提升 |
|---|---|---|
| FDFO 调度 | 消除 sync 锁步的长尾浪费(整批等最慢那条),连续补位保持满 batch | +88%(sync@128 1348 → 2551 tok/s) |
| 混批 + K=2 | prefill/decode 合并进同一 forward,消掉 47% 的浪费小 prefill 轮;K=2 把轮界 host 开销摊到 2 个 forward | +27~29% |
| 去噪规约单遍融合核 | argmax+softmax-prob 一次读完 logits(去噪 13→6ms) | +6% |
| split_qkv 宽 grid 融合核 | 大 batch 消除逐行串行退化(8.6→3.0ms) | +6% |
| 消 init D2H 同步 | dLLM 路径用 CPU 镜像值,省 2 次 D2H | +6.8% |
| bf16 lm_head(不物化 fp32) | 解 bs≥112 的 logits OOM | 仅解 OOM |

CV 融合(多流 cube∥vector 重叠)在 NPU 不可行(实测证伪):910B3 在 torch_npu 下跨流 kernel 完全串行(双流 ratio=2.02,graph capture 下也 0 重叠)。GPU 的 TBO / 多流模型不迁移到 NPU;Ascend 的 cube/vector 并行只能靠单 kernel 内融合(如 FlashAttention)。因此 NPU 的优化方向是更少更胖的融合核(减 kernel 数 + 减冗余内存遍历),不是重叠。

### 2.2 最大 bs 的推导(HBM 限制)

显存分解(64GB,2.5K 序列口径,512in/2048out;更长的 5.5K 见 §2.5):

| 部分 | 大小 | 随 bs 变化 |
|---|---|---|
| 权重 | 30.3 GB | 固定 |
| CANN/驱动 overhead | 8.8 GB | 固定 |
| 固定合计 | 39.1 GB | — |
| decode graph(密桶 max_bs=128) | 9.76 GB | 随 max_bs 涨(160 桶要 12.21GB) |
| KV 池预留 | (prompt+max_new)×40 KiB/req = 2.5K 序列 ~100 MB/req | ∝ bs |
| activation | ~25 MB/req【实测】 | ∝ bs |

每长请求边际成本 = KV 100MB + activation 25MB = ~125 MB/req。

```
bs_max = (64 GB × 可用率 − 权重 − overhead − graph) / (KV_per_req + act_per_req)
```

【910B 实测】的边界:

- eager(无 graph):2.5K 序列 max bs ≈ 200(HBM 峰值 63.9GB,不 OOM),与理论 199 吻合。
- graph max_bs=128(密桶):graph 9.76GB,2.5K 序列 max bs ≈ 150。
- graph max_bs=160:graph 涨到 12.21GB,启动后运行时只剩 1.07GB,满负载 2.5K 序列贴边 63.0GB 并触发 retract,吞吐 2185 ≈ bs=128 的 2130,无收益。

单卡 2.5K 序列 bs 天花板约 160,但打大 batch 无收益:(a) graph 成本 ∝ max_bs 吃掉批头寸,(b) MoE 在 t=160 仍带宽受限(甜点需 t≥512,单卡达不到),attention/vector 随 bs 线性增长抵消。推荐单卡配置 = bs=128 + 密桶 graph + FDFO + 混批 + mem-fraction 0.75,不贴边、无 retract。

### 2.3 bs=128 逐类对比

每 forward:H20 ~90ms,910B ~85ms(1.06×,与吞吐比一致)。类别百分比来自 H20 trace(不依赖 forward 计数),套到 90/85ms 绝对值上算差值。

| 类别 | H20 ms | 910B ms | 差 | 瓶颈 | 相对 |
|---|---|---|---|---|---|
| dense matmul | 36.2 | 21.6 | +14.6 | 算力 | 910B 2×(bf16 峰值 2.16×,lm_head 269 vs 143 TFLOPS) |
| MoE GMM | 29.1 | 30.9 | −1.8 | H20 算力88% / 910B 带宽39% | ~平(148×0.88≈320×0.39≈128T) |
| 去噪规约(argmax+softmax) | 7.3 | 5.0 | +2.3 | vector | 910B(GPU fp32 未融合 vs 910B bf16 融合核) |
| 其他 vector(激活/cast/拷贝/逐元素) | 7.9 | 2.8 | +5.1 | vector | 910B(GPU 多出主要是 fp32 logits 的 cast/拷贝) |
| attention | 6.9 | 6.5 | +0.4 | 算力(AI=128) | ~平 |
| MoE 路由(gating/dispatch/combine) | 1.7 | 10.9 | −9.2 | vector/scalar | H20 5.5×(fused routing vs 910B aclnn 三段式) |
| norm/rope | 0.4 | 7.3 | −6.9 | vector | H20 15×(强 vector + 融合) |
| 合计 | ≈89.5 | ≈85.0 | +4.5 | | 910B 快 1.06× |

两颗芯片的优势在相反的算子上:910B 在大算力活上占优(dense GEMM +14.6ms,来自 2× bf16 算力;去噪靠融合核占优),H20 在小算子上占优(MoE 路由 −9.2 + norm −6.9 ≈ 16ms,fused routing + 强 vector)。MoE 相当,但机理相反:H20 算力受限 88%,910B 带宽受限 39%。

口径注:910B 列为 msprof 实测;H20 列为 H20 硬件 trace 的相对构成 × 90ms 锚点(单 kernel 时间为实测,90ms 锚点因该 capture 非饱和为估算)。净差 +4.5ms 能对上 90 vs 85,是自洽校验。

吞吐实测(bs=128,两颗芯片全实测):gsm8k 短序列 910B 3667 vs H20 3412(1.07×);长序列 512in/2048out 910B 2591 vs H20 2364(1.10×)。per-forward 比(1.06×)与吞吐比(1.07-1.10×)一致。完整 batch sweep 见 §2.4。

### 2.4 batch sweep 实测

换算:t(每专家 token 数)= parallel × block(32) × topk(8) / experts(256) = parallel,所以横轴 parallel 即 t。gsm8k 短序列输出吞吐(tok/s):

| parallel(= t) | H20 | 910B | 910B / H20 | H20 边际增益 | 910B 边际增益 |
|---|---|---|---|---|---|
| 1 | 592 | 312 | 0.53 | — | — |
| 4 | 1366 | 574 | 0.42 | — | — |
| 8 | 1923 | 772 | 0.40(H20 快 2.5×) | — | — |
| 16 | 2451 | 1461 | 0.60 | +27% | +89% |
| 32 | 2853 | 2190 | 0.77 | +16% | +50% |
| 64 | 3241 | 3043 | 0.94 | +14% | +39% |
| 128 | 3412 | 3667 | 1.07(910B 超过) | +5% | +20% |

精度全程持平(两家 0.82-0.84,无回归)。

配置口径:910B bs=1 用 sync(FDFO 在单请求下更慢——锁步长尾只在多请求时才被 FDFO 补位消除,单请求无可补且吃冻结开销);p4-128 用 FDFO+混批。每个点取各自最优配置。

两条曲线与第一章 roofline 一致:

- 小 batch(t < 37):H20 领先(bs=1 快 1.9×、p8 快 2.5×)。此时 MoE 在两颗芯片上都带宽受限(t < 两家 ridge),H20 的 2.5× 带宽在 MoE 上占优;vector/路由小算子 H20 也占优;tiny batch 下 H20 的低 launch 开销进一步占优。910B 的 dense 算力优势在小 GEMM 下发挥不出。
- H20 在 t≈37 越过 ridge,进入算力受限,吞吐增长放缓。边际增益从 p8 起单调衰减,p64→p128 仅 +5%,曲线趋平——H20 的 MoE 已 88% 峰值,无算力头寸。
- 910B 的 ridge 为 200,单卡 t≤128 全程带宽受限、仍有算力头寸,吞吐持续上升。p64→p128 仍有 +20%(叠加 FDFO 消锁步、dense GEMM 随 batch 变大),在 p128 超过 H20(3667 vs 3412,1.07×)。
- 交叉点约 parallel 100,即第一章"小 bs 无优势、大 bs 才可行"的实测边界。

长序列(512in/2048out,concurrency 128):910B 同样领先。

| 指标 | H20 | 910B |
|---|---|---|
| 输出吞吐 tok/s | 2364 | 2591(+9.6%) |
| 总吞吐 tok/s | 2978 | 3263 |
| 完成时长 | 435.7 s | 397.7 s |
| Mean TTFT | 11100 ms | 3819 ms |
| Mean TPOT | 46.0 ms | 47.0 ms |
| P99 E2E | 371.7 s | 150.0 s |

长序列 TPOT 接近相等(而非 910B 更快):attention 的 AI=128 两者相当(§1.3),它占比上升只是把一个相当的组件放大,稀释了 910B 在 dense 上的短序列优势;且 attention 在两颗芯片上都被 FIA 的 thin-M 低效拖累(§4.2),谁也没占到带宽便宜,稳态每 token 相当(47 vs 46,910B 微慢 1ms)。

若把 FIA 修到吃满带宽(§4.2),两颗芯片单 forward 都会缩,而 910B 的 kernel 工作量本就少(§2.5),会转为 910B 领先。

### 2.5 更长序列(4K in / 1.5K out)

序列 5632 token/req,多样 prompt(RL rollout 典型负载)。910B 受 64GB 显存限制 bs≈72,H20 96GB 可到 128。正确配置下(关 radix 前缀缓存 + STEPS_PER_ROUND=1,见 §2.6),同 bs=72 两颗芯片打平:

| | bs | 配置 | 输出吞吐 tok/s | 总吞吐 tok/s | req/s | Mean TPOT |
|---|---|---|---|---|---|---|
| 910B | 72 | radix off, K=1 | 2004 | 7349 | 1.30 | 25.0 ms |
| H20 | 72 | radix off, K=1 | 2032 | 7450 | 1.32 | 25.3 ms |

差 1.5%,基本相同。逐算子对照(两边 radix off + K=1,真实 ShareGPT,ms/forward):

| 类别 | 910B | H20 | |
|---|---|---|---|
| dense matmul | 12.70 | 26.02 | 910B 快 2.05× |
| MoE grouped gemm | 24.93 | 23.42 | ~平(H20 6%) |
| attention (FIA) | 22.67 | 24.12 | ~平(910B 6%) |
| MoE 路由 glue | 6.44 | 1.81 | H20 快 3.6× |
| norm/rope | 4.36 | 0.70 | H20 快 6.2× |
| 去噪/其他小算子 | 5.28 | 9.81 | 910B 快 |
| **kernel 合计** | **76.37** | **85.88** | **910B 少 11% 工作量** |

读数:910B 靠 dense 的 2× 算力优势,扳回了 MoE 路由 / norm 等小算子上的带宽劣势,kernel 总工作量还少 11%;attention、MoE 两大项打平。逐算子层面 910B 不输。

- H20 的整体小幅领先来自显存容量:96GB 可上 bs=128 再换一档总吞吐,代价是 TPOT 大幅上升(bs=128 vs 72,TPOT +63%);910B 64GB 上不去 bs=128。
- 未解项:910B 那 11% 的 kernel 优势没有完全转成吞吐,说明还有 host 侧(调度 Python 剩余部分)+ kernel launch 串行度的损失——H20 每 forward ~1546 个小 kernel 但可异步重叠,NPU ~230 个但串行度更高。方向在 host,不在算子。

### 2.6 radix 前缀缓存:长序列必须关

多样 prompt 的长序列 decode 里前缀命中率为 0,radix 前缀缓存的匹配 / 插入是纯开销,占 scheduler host CPU 的 ~72%;`--disable-radix-cache` 后 host 降 ~90%。叠加 `STEPS_PER_ROUND=2` 每轮多一次去噪 schedule 的开销,这两项都是设备无关的软件开销,吃掉了长序列吞吐,且对 host 占比更高的 910B 伤害更大——关掉后 910B req/s +88%(0.69→1.30),H20 +38%(0.96→1.32)。

配置建议:
- 多样 prompt 长序列(RL rollout):`--disable-radix-cache` + `SGLANG_DLLM_FDFO_STEPS_PER_ROUND=1`。
- 有共享前缀的短序列(gsm8k few-shot):保留 radix + K=2。

---

## 第三章 EP 的尝试与评估

### 3.1 为什么要进 EP

第一章的结论:超过 H20 需把 MoE 推进算力区(t≥512),但单卡 bs=128 只有 t=128。EP(专家并行)让每个专家收所有 DP 副本的 token:DP8+EP8、每卡 bs=128 → 每专家 t = 8×128 = 1024,理论上进入甜点。附带 EP 把每卡权重从 30.6GB 降到 3.8GB,腾出约 55GB 给 KV(长序列显存的解法)。

### 3.2 EP 实测结果【910B 实测,另一 session】

- DP8+EP8 跑通,gsm8k 0.835(单卡基线 0.834-0.845),KV 池每卡 0.55M → 1.04M tokens。
- MoE 进入甜点:GroupedMatmul 273-282 TFLOPS = 85-88% 峰值,t/expert=729(原预估天花板 191T/60%),架构假设成立。
- 显存释放兑现:长序列档单卡放不下时,EP 可用吞吐换容量。

### 3.3 通信代价的计算

一个 token 路由到 K=8 个专家,在 EP=P 下:

1. 落在几个 rank 上:`r(P) = P·(1−(1−1/P)^K)`。deep_ep 按目标 rank 发一份(不是按专家发 8 份),ep=4 时 8 个专家落在 3.6 个 rank 上。
2. 出卡比例:`(P−1)/P`(发给自己那份不过互联)。
3. 字节:dispatch = `r(P)·H·2B·(P−1)/P`,combine 原样返回,×2。
4. 可用带宽:P 卡组里每卡用 P−1 条 HCCS 链路,每链路每方向 15.6 GB/s。

| ep | 目标 rank r(P) | 出卡比 | 链路数 | 可用带宽 | 字节/token/层 | us/token/层 | T=4096 时 |
|---|---|---|---|---|---|---|---|
| 2 | 1.99 | 0.500 | 1 | 16 GB/s | 8.0 KB | 0.523 | 2.14 ms |
| 4 | 3.60 | 0.750 | 3 | 47 GB/s | 21.6 KB | 0.473 | 1.94 ms |
| 8 | 5.25 | 0.875 | 7 | 109 GB/s | 36.8 KB | 0.345 | 1.41 ms |

ep 越大,每 token 搬的字节越多(8→37 KB),但链路数涨得更快(1→7),所以每 token 反而更便宜。单节点只有 8 卡,没有 ep=16(再往上跨节点走 RoCE/IB,不能这样外推)。

口径:本表用每链路 15.6 GB/s 线性假设。HCCS 实测在 3 链路以上略有次线性,但当时其它影响未完全隔离,此处按线性取值。完整推导 + 脚本见 `LLaDA2_910B_dp_ep_findings.md` §2.6 / `scratch_dpep/ep_roofline.py`(换库/模型/硬件后重跑即重算)。

### 3.4 EP 收益的计算

每卡的 GMM 行数在 EP 和非 EP 下完全一样(`P rank × T token × K ÷ P 卡 = T·K`)。EP 换的只是"哪张卡的专家算哪些 token",没有减少工作量。它唯一改变的是这些行摊到的权重字节:1.61 GB → 1.61/P GB。

```
时间/层 = max( η_bw · 权重字节 / HBM带宽 ,  η_c · 行数·FLOP / 算力峰值 )
                ↑ EP 只动这一项                ↑ 这一项 EP 完全没动
```

| blocks/卡 | T | 行/层 | 权重 roofline | 算力 roofline | 非 EP | EP8 | 省下 |
|---|---|---|---|---|---|---|---|
| 16 | 512 | 4096 | 1.46 | 0.09 | 1.46 | 0.18 | 1.28 |
| 32 | 1024 | 8192 | 1.46 | 0.19 | 1.46 | 0.19 | 1.28 |
| 64 | 2048 | 16384 | 1.46 | 0.37 | 1.46 | 0.37 | 1.09 |
| 128 | 4096 | 32768 | 1.46 | 0.75 | 1.46 | 0.75 | 0.72 |
| 200 | 6400 | 51200 | 1.46 | 1.17 | 1.46 | 1.17 | 0.30 |

"非 EP"全程为 1.46 ms——它一直权重带宽受限、与 batch 无关。这是收益的天花板:EP 最多把 1.46 削到算力 roofline 为止。batch 越大算力项越高、能削的越少,200 blocks/卡时收益趋近 0。收益天花板 ~1.28ms/层(小 batch)由模型结构决定(每层 256 专家权重 1.61GB),与互联速度无关。

### 3.5 不同 batch 的净账(ep=8)

| blocks/卡 | 收益(均衡) | 收益(1.3× 不均) | a2a 地板 | 净(均衡) | 净(不均) |
|---|---|---|---|---|---|
| 16 | 1.28 | 1.28 | 0.18 | +1.11 | +1.11 |
| 32 | 1.28 | 1.22 | 0.35 | +0.92 | +0.87 |
| 64 | 1.09 | 0.98 | 0.71 | +0.39 | +0.27 |
| 87 | 0.96 | 0.80 | 0.96 | 0.00 | −0.16 |
| 128(实测点) | 0.72 | 0.49 | 1.41 | −0.69 | −0.92 |
| 200 | 0.30 | −0.05 | 2.21 | −1.91 | −2.26 |

收益是一条从 1.28 递减到 0 的曲线,代价是一条从 0 线性上升的直线,交叉点约 87 blocks/卡(带实测 1.3× 专家不均衡则 79)。低于它 EP 有收益、高于它 EP 净亏,128 在净亏一侧。

这是 roofline(理论地板)的交叉点。加上当前 deep_ep 实现的库开销 + skew,交叉点进一步降到 ≤21 blocks/卡(EP8),见 §3.6。所以 128 从任何口径看都在净亏侧。

### 3.6 实测 a2a 与 roofline 的差距

前几节均为 roofline 投影。实际跑的是 DP4+EP4,实测 a2a = 12.2ms/层,而它的理论地板(下表 ①)只有 2.84ms/层——实测是地板的 4.3 倍,多出的 9.4ms(②③④)是可优化的库开销 + skew,不是物理下限。整轮 95ms(单卡)→ 488ms,节点吞吐只有 4 张独立单卡的 1/4.6。

"EP 在 910B 上是否可行"这个问题,在当前实现下取决于实现开销而非 roofline。12.2ms/层 的四段拆解(ep=4 实测):

| 段 | ms/层 | 占比 | 说明 |
|---|---|---|---|
| ① 互联地板 | 2.84 | 23% | 理论最优字节 / HCCS 带宽,物理下限 |
| ② deep_ep 多搬字节 | 2.36 | 19% | 库按 (token,专家) 发 K/r(P) 倍字节(设计选择,可改) |
| ③ 算子相对自身字节开销 | 3.01 | 25% | MC2 dispatch 算子的 AICPU 编排/tiling 固定开销(不随字节变);几乎全在 dispatch,combine 打到自身字节地板 ≈0 |
| ④ rank 间等待 / skew | 3.99 | 33% | 负载不均衡(见下),a2a 是集合通信 barrier,整批等最慢卡 |

④ 负载不均衡实测(同一次 capture,DP4+EP4——4 张卡同时是 DP rank 和 EP shard)。行数 = 每卡本地专家(EP 分片)在 a2a dispatch 后处理的 token 数:

| 卡 | 行数 | 均分应为 | GMM 峰值% | ms/层 |
|---|---|---|---|---|
| 0 | 46707 | 35520 | 86% | 1.07 |
| 1 | 28785 | 35520 | 57% | — |
| 2 | 46785 | 35520 | 87% | 1.07 |
| 3 | 19803 | 35520 | 34% | 1.16 |

- 行数最多与最少差 2.4×(46785 vs 19803)。不均衡来自两处:各 DP rank 每轮 block 数不同(DP 侧,dLLM 调度器不对齐)+ 专家路由热度不均(EP 侧)。
- 行数最少的 rank 3 反而最慢(1.16 vs 1.07 ms/层)。因为 MoE 权重流式是每卡固定 0.40GB、与行数无关;行数少则摊不动这个固定权重读,退回带宽受限,每行更贵。轻载卡拉低整体,等待被放大。
- 修法:EPLB(sglang 有)+ 对齐各 rank 每轮 block 数,但只能部分回收(完美均衡也只省 ~0.66ms/层),且对齐用 padding 又浪费算力。属结构性问题,可部分优化。

即使换到 ep=8(最佳 EP 度数)+ 完美实现,128 blocks/卡 仍是净亏。下表是 ep=8 的 best-case roofline 投影(ep=8 有 7 条链路,地板 1.41,比 ep=4 的 2.84 更低):

| 情形(128 blocks/卡,ep=8 roofline) | a2a ms/层 | vs 收益 0.49 | forward 净开销 |
|---|---|---|---|
| 互联地板 | 1.41 | 亏 0.92 | 17.5 ms/forward |
| 互联翻倍(910C 口径) | 0.71 | 亏 0.21 | 4.0 ms/forward |

即便互联翻倍(910C 口径),128 blocks/卡 仍亏 0.21ms/层。EP 在 128 这个点上,即使是 roofline 也是净亏;有收益只能往小 batch 走(§3.5)。

### 3.7 结论

1. 大 batch(≥87 blocks/卡):roofline 上已是净亏,当前实现的 a2a 又是理论地板的 ~4 倍(9.4ms/层 库开销+skew)。不建议使用。
2. 小 batch(≲87 blocks/卡):roofline 上 EP 有收益,但当前实现的 a2a(隔离态就 8.21ms/层)会把收益吃光。除非先把库开销和 skew 压下去,否则小 batch 也无收益。
3. 互联翻倍(910C 口径)把交叉点从 87 推到 130(不均衡时 79→112),刚好把 128 挪到有收益一侧,但余量很薄,且不改变性质——收益天花板(1.46ms/层的权重带宽)由模型结构决定,与互联速度无关。

部署建议:

- 若用 EP,用 EP8(各维度均优于 EP2/EP4:权重分更细、"按专家发"浪费 4.0×→1.5×、可用链路 1→7)。
- 追吞吐用大 batch + 独立单卡 TP1 + FDFO(混批),不用 DP8+EP8。
- EP 目前唯一成立的价值是显存:长序列档单卡放不下时,用吞吐换容量。
- 已排除的路径(实测):low_latency(CANN 算子 1024 tokens tiling 失败 + 每 token 贵 6×)、换 deep_ep build(单机 A2 normal 无新算法)、通信/计算重叠(NPU 跨流串行)。

---

## 第四章 后续优化方向

dense 已算力饱和(78% 峰值、已快于 H20 2×),不需改动。发力点两类:算法/结构协同(把 workload 往 910B 算力强项推,§4.1)+ kernel 工程(FIA、融合核,§4.2-4.3)。

### 4.1 算法方向:把 workload 往算力密集推

第一章的结论是 910B 在算力上占优、在带宽和小算子上吃亏。所以算法层面最有效的方向是把 dLLM 的结构往"更计算密集、更少带宽/小算子"改,让 910B 的 2× 算力兑现。三个方向:

① 把 block 做大。dLLM 的 block_size 决定 attention 和 MoE 的算术强度:

- attention AI = block × GQA组(4)。block=32 → AI=128(带宽受限边缘);block=128 → AI=512 >> ridge 200,attention 进入算力区。
- MoE 的 t = block × bs × topk / experts。block=128、bs=128 → t=512,单卡即进入 MoE 甜点——第三章 EP 靠通信堆的 t,大 block 不用 a2a 就拿到。
- 参考:diffusion 版 Gemma 已做到 block_size=128。block 做大对 attention 和 MoE 同时更算力密集,是把 910B 优势兑现的最直接路径。代价:并行去噪的 block 内 token 更多,精度/收敛需验证。

② Blockwise MoE:压缩 block 内激活的专家数。一个 block 内的 token 语义相近(同一局部上下文),大概率路由到重叠的专家。把 block 的专家集合压到更少的 distinct 专家,可少读专家权重,直接降低 MoE 的权重带宽(§1.3 的 30GB/forward 瓶颈),即用路由结构去解 MoE 带宽问题。参考:已有开源 dmoe 做了这件事。

③ 全词表冗余裁剪。每去噪步对全 32 位置算 lm_head(10.3ms)+ 全词表(157k)归约(6ms),但后期步骤 still-masked 位置很少,大部分是冗余计算。只对 still-masked 位置算可大幅减少 dense + 去噪。障碍:JointThreshold 的 T2T 语义读所有非 mask 非 prompt 位置的概率(joint_threshold.py:95-98),需重新设计算法。收益上界大,工程量也大。

三条都是算法/结构改动,把 workload 往算力密集推,落在 910B 的强项、避开它的带宽/小算子弱项,比 kernel 优化更根本。

### 4.2 attention FIA 效率(长序列)

【910B 实测,4K decode】FIA 只跑到带宽 roofline 的 **32%**(有效 0.51 TB/s;下限 7.2 ms/forward vs 实测 22.7 ms),是长序列档最大的可回收项:修到吃满带宽 → 单 forward 76→~61 ms(**+25%**)。

**这不是带宽墙,是 kernel 实现低效,而且两颗芯片同病。** attention AI=128 < 910B ridge 200 属带宽受限,但 FIA 连带宽都没吃满:mac=0.25(cube 闲)+ mte2=0.93(载入忙但低效),两条 pipe 都空转。根因是 **thin-M**(TND 打包后每段只有 32 个 query 行,cube 的 M 维填不满)+ **online-softmax 的 QK→softmax→AV 串行气泡**,NPU 又不能靠多流重叠(§2.1)。**H20 的 flashinfer 在同一形状上更差**:有效 0.56 TB/s = 其 4 TB/s 规格的仅 **14%**(910B 是 32%)——所以 attention 上 H20 的带宽优势完全用不出,两边打平(§2.5)。

- 解法:算子团队优化 thin-M(M=32 × N≈4231)的 tiling + QK/softmax/AV 流水(大头 ~60%);或算法侧增大去噪块把 M 撑大(§4.1 block 做大)。
- KV `block_size=32` 另值 ~11%(block_table 间接开销),把 KV page 解耦到 128 即可消掉;算子内部本就按 128-tile 算。
- 复现/交付:`scratch_profile/fia_decode_bench.py`(整核 roofline)、`fia_page_probe.py`(内部 tile 粒度探针),固定 shape `q[2304,16,128] / kv[·,·,512]`。

### 4.3 更少更胖的融合核

NPU 的优化方向(第二章方法论):减 kernel 数 + 减冗余内存遍历。剩余目标:MoE 路由三段式融合(`npu_grouped_matmul_finalize_routing`,~3.5%)、norm/rope 进一步合并。单项收益都不大(个位数 %),但可累积。

### 4.4 硬件代际

910C 互联翻倍(218 GB/s)时,EP 在 128blk/卡接近打平,大 batch + EP 进入甜点才真正可行。当前 910B3 上 EP 净亏,需下一代硬件。

---

## 总结

| 维度 | 结论 |
|---|---|
| 小 bs(≤128) | 相当 / 小幅领先 1.06×。910B 靠 dense 算力扳平 vector/routing 的带宽劣势,MoE 未进甜点。 |
| 大 bs(单卡) | bs 天花板约 160,但无吞吐收益(MoE 仍带宽受限 + graph/KV 吃显存)。推荐 bs=128。 |
| 大 bs + EP | MoE 进甜点(85-88%),但 a2a 让节点吞吐降到 1/4.6,128blk 净亏。EP 目前只值得为显存用。 |
| 后续方向 | 算法层(§4.1):① block 做大(attention+MoE 同进算力区、单卡进甜点、免 a2a)② blockwise MoE(压 block 内专家、降 MoE 带宽)③ 全词表裁剪。 |
| 对 H20 的判断 | 小 batch H20 领先(bs=1 1.9×);大 batch 两者基本打平:短序列 910B +7%、512/2048 +10%,4K/1.5K 同 bs 打平(2004 vs 2032 tok/s)。H20 的整体小幅领先来自显存容量(96GB→bs=128,代价 TPOT +63%),不是单位算力/带宽;逐算子层面 910B kernel 工作量还少 11%。 |

配置注:多样 prompt 的长序列(RL rollout)用 `--disable-radix-cache` + `STEPS_PER_ROUND=1`(§2.5/§2.6);有共享前缀的短序列(gsm8k few-shot)保留 radix + K=2。
