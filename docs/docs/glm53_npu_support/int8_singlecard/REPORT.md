# GLM-5.3-Flash INT8 单卡（TP1）性能分析

2026-08-30，Atlas A3 单 die（`Ascend910_9362`），compressed-tensors W8A8-INT8，NPU Graph 开。
为「单卡 CPU MoE offload」项目做的前置调研。

**证据等级**：每条结论标 **[实测]** / **[读源码]** / **[推断]**。没测过的加速不写成结论。

---

## 0. 这份报告的构型，以及它让哪些结论失效

全量 288 专家的 INT8 权重是 **306.1 GiB [实测，逐张量读 safetensors header]**，单 die 64 GB 放不下。
所以跑的是**只保留前 16 个专家**的裁剪 checkpoint，路由自然变成 **top-8 of 16**。

裁法：离线重写 checkpoint（`tools/prune_experts_int8.py`），丢掉 `experts.{E≥16}`、
把 `text_config.n_routed_experts` 改成 16、`mlp.gate.weight` 和 `e_score_correction_bias`
截断到 `[:16]`。**没有改路由代码** —— top-8 of 16 是配置的自然结果，不是 hook 出来的。

内存账（独立复算，与交接文档一致到小数点后一位）：

| 保留专家 | 权重 GiB [实测] |
|---|---|
| 288 | 306.10 |
| 32 | 47.75 |
| 16 | **31.61** |
| 8 | 23.53 |

非专家部分 15.46 GiB，每个专家 1.0091 GiB（42 个 MoE 层 + 1 个 MTP 层 × 3 个矩阵 × 4096×2048 int8）。

### 哪些结论因此不可外推

- ❌ **专家负载分布**。top-8 of 16 和 top-8 of 288 是完全不同的分布。
- ❌ **`npu_grouped_matmul` 的 `group_list` 形状分布**。group_list 长度就是专家数（16 vs 288），
  而 gmm 的性能对它敏感。**大 batch 下尤其不可外推**：16 个专家时 token 很快就把每个专家填满，
  288 个专家时每组小得多。本报告里所有 **bs>1 的 MoE 数字都带这个偏差**。
- ✅ **bs=1 的 MoE 权重流量可以外推**。top-k=8 与专家总数无关，
  所以 bs=1 每 token 读的仍然是 8 个专家的权重，和 288 专家时**逐字节相同 [推断，但由 §2 的
  实测 gmm 时间证实：gmm1 实测 107.3 µs，正好是 8 个专家 128 MiB 在 1.25 TB/s 下的 107.4 µs]**。
- ❌ **精度**。输出是坏的，本轮不验，也不该拿这份构型说任何精度的事。

### 服务构型 [实测]

`--tp-size 1 --page-size 64 --context-length 32768 --max-running-requests 16
--mem-fraction-static 0.80 --disable-radix-cache --moe-a2a-backend none`，图开、prefill 不捕获。
`ASCEND_RT_VISIBLE_DEVICES=0`（另一 session 用 die 14/15）。

- 权重 31.50 GB，加载 36 s
- **Mamba state pool：16 槽 2.34 GB = 每槽 146 MB**。TP16 下是每 die 8.75 MB/槽 —— **TP1 贵 16 倍**
  （64 个头而不是 4 个）。这是单卡并发的第一道墙：128 槽要 18 GB
- KV pool 16.28 GB / 1,241,728 token
- 图捕获 6 个桶 `[1,2,4,8,12,16]`，15.2 s，图池 0.37 GB

---

## 1. 一句话

**单卡 bs=1 decode 起始 42.8 ms/token，权重带宽 roofline 是 17.94 ms，即 2.39× [实测]。
六条优化之后是 33.3 ms（−22.1%），见 §6。**
时间的 39% 在 KDA、28% 在 MoE，而 **MoE 读专家权重的那两个 `GroupedMatmul` 已经贴在
带宽地板上（1.03×）** —— 注意这 1.03× 只属于那两个 kernel，不是整个 MoE 家族（家族是 1.56×，
差额几乎全是路由簿记，见 §2）。
起始一步跑 **3578 个 kernel**，现在 **2570 个**；两端 device 时间都与墙钟**逐点吻合**
（起始 42.823 ms profiler vs 42.8 ms 墙钟，现在 33.348 vs 33.5）
—— **图模式下 host 侧已经没有气泡可挤**，省下的每一毫秒都必须来自少发或发得更省的 kernel。

---

### 吞吐全貌 [实测，图模式，prefill 与 decode 分开量，贪心 + `ignore_eos`]

构型 A（仓库原样，e16）。**这台机器共用，采数时 15 个 die 空闲、无人训练。**

| 并发 | 短上下文 13 tok<br>ms/token | 短 合计 token/s | 长上下文 3256 tok<br>ms/token | 长 合计 token/s |
|---|---|---|---|---|
| 1 | **42.8** | 23.3 | 43.6 | 22.9 |
| 2 | 46.8 | 42.8 | 48.2 | 41.5 |
| 4 | 47.7 | 83.9 | 52.2 | 76.7 |
| 8 | 62.2 | 128.6 | 61.9 | 129.2 |
| 16 | 75.4 | **212.1** | 118.2 | 135.4 |

- **bs=1 → bs=4 只慢 11%**：典型的带宽/固定开销主导区，和 §3 的 roofline 一致
- **长上下文在 16 并发就拐了**（8 → 16 时 129 → 135 token/s，而单请求 61.9 → 118.2 ms）。
  与 TP16 上「长上下文 64 并发拐点」同源（PLAN P6.7/P6.10/P6.11 的 kpool device 时间）。

  ⚠ **机制现在有了，但是别人量的，不是本部署的数** [实测，来自长上下文那条线的 TP8 部署]：
  decode 一步要给 **n/4 个 pool 打分**，而真正读的 KV 被 `topk=2048` **封顶**。
  所以 decode 该线性于 n 而不是二次 —— 他们在 32 k→1 M 上量到
  27.5 / 28.2 / 28.9 / 30.2 / **32.9 ms/token**，拟合 `≈ 27.3 + 5.4e-6 · n`，
  **每翻一倍约 +1.1 ms，稳定**。prefill 那边才是真二次（整条约 `n²/8` 次打分，1 M 处二次项占 43%）。

  **对本部署的含义**：把那个拟合套到本节两个工况上，线性项是
  **0.0001 ms（13 tok）和 0.018 ms（3256 tok）** —— 比单次 profile 几百微秒的噪声底还小两个量级 —— 所以上面那个 16 并发拐点**不是**这条线性项造成的，
  别把两件事混起来。本部署没有量过长上下文，**这一段不能当作本部署的结论引用**。
- 并发上限 16 不是 KV 卡的（用了 1.24 M token 里的一小半），
  是 **KDA mamba state pool**：TP1 每槽 **149.8 MiB**（`conv_state 0.08 + ssm_state 2.26 GB` ÷ 16 槽）

  ⚠ **这个结论有明确的适用范围，别照抄** [实测，两个部署对照]。
  mamba 池是**每槽定额**，KV 是**每 token**（**14084 B/token**，实测
  `16.31 GB ÷ 1243456`），所以谁卡住并发**随上下文长度换人**：

  | | mamba 每槽 | 交叉点（KV 追平 mamba） |
  |---|---|---|
  | **本部署 TP1** | 149.8 MiB | **≈ 11150 token** |
  | 对照 TP8（另一条线实测） | 20 MiB | ≈ 1490 token |

  本节两个工况（13 tok 和 3256 tok）**都远在 11150 以下**，所以「mamba 卡并发」在这里成立。
  但超过约 11 k token 之后卡并发的就是 KV 了 —— 长上下文那条线在 1M 上量到
  每请求 13.75 GiB、并发上限 1，那边**扣 mamba 是白扣**。
  **同一句话在两个工况下一句对一句错，所以引用时必须带工况。**

  ⚠ 14084 B/token 这个常数**在三个互相独立的部署上整除到同一个值**
  （本部署、TP8 INT8、TP16 BF16），拆解也对得上：
  11 个 DSA 层 × (`kv_lora` 512 + `index_head_dim` 128) × 2 B = 14080。
  **三点独立印证比任何单点测量都强**，这个数可以直接拿去做容量估算。
- **prefill 约 1040–1090 token/s**（8192 token 一个 chunk，调度器自己的每批计时；
  墙钟法一致：16 × 3256 = 52096 token 用 48.81 s = 1067 token/s）。
  ⚠ **这个数偏乐观**：16 个专家时 prefill 的 gmm 分组比 288 专家时大得多、效率更高。

**对照 TP16 INT8（PLAN P6.15，别人的机器时间，构型不同）**：bs=1 28.9 ms/token。
单卡只慢 **1.48×**，远好于「算力少 16 倍」的朴素预期 —— 因为 bs=1 是带宽而不是算力受限。

---

## 2. bs=1 每步的时间分布 —— **起点（构型 A）** [实测]

⚠ **这一节和 §3 / §4 量的是起点 42.823 ms，不是现在。** 现状（构型 G，33.348 ms）
的同一张分解在 **§8.2**，六条优化各拿到多少在 **§6**。先看起点是因为**优化的理由在起点里**；
但引用数字时别引错了构型 —— §7b.7 就是这么作废过一次测量的。

kernel 级 profiler（`torch_npu.profiler` Level1 + PipeUtilization），20 个 decode step，
通过服务的 `/start_profile` 采（TP1 单 rank 下服务级 profiling **没有**把服务打挂）。
归属方法见 `tools/attribute_kernels.py`：整网是一张捕获好的图，profiler 无法告诉你 kernel
来自哪个模块，但**每个层族每步的调用次数是唯一的**（34 KDA / 11 DSA / 42 MoE / 3 dense /
45×2 mHC），按次数归属，歧义的按 shape 显式列规则。

| 层族 | ms/step | 占比 | kernel/step | µs/kernel | 每层 |
|---|---|---|---|---|---|
| **KDA（34 层）** | **16.824** | **39.3%** | 850 | 19.8 | 495 µs |
| **MoE（42 层）** | **12.041** | **28.1%** | 798 | 15.1 | 287 µs |
| **DSA（11 层）** | **6.693** | **15.6%** | 1012 | 6.6 | 608 µs |
| mHC（45 层 × 2 站点） | 3.627 | 8.5% | 180 | 20.2 | 40 µs/站点 |
| 未归类（小算子尾巴） | 1.697 | 4.0% | 511 | 3.3 | |
| lm_head | 0.930 | 2.2% | 1 | 930 | |
| dense FFN（3 层） | 0.616 | 1.4% | 72 | 8.5 | 205 µs |
| 其余 per-layer / 全局 | 0.396 | 0.9% | 153 | | |
| **合计** | **42.823** | 100% | **3578** | 12.0 | |

**每个 KDA 层 25 个 kernel、每个 MoE 层 19 个、每个 DSA 层 92 个。**
（构型 G 是 **8 / 16 / 83** —— 全量对照见 `data/kernel_attribution_cfgG.txt`。）

### ⚠ 这张表的「ms/step」和 §3 的「roofline ms」不能直接比

上表是**实测的全部 kernel 时间**，§3 是**只读权重的理论地板**。以 MoE 为例对账 [实测]：

| MoE 家族拆解（bs=1） | ms/step | kernel/step | §3 地板 | 实测/地板 |
|---|---|---|---|---|
| `GroupedMatmul` ×2（读 routed 专家权重） | **6.986** | 84 | 6.773 | **1.03×** |
| `QuantBatchMatmul` ×2（shared expert） | 1.635 | 84 | 0.847 | 1.93× |
| `MatMulV2`（router gate） | 0.264 | 42 | 0.079 | — |
| **路由 / 簿记**（其余 13 组算子） | **3.156** | **588** | ~0 | — |
| **合计** | **12.041** | 798 | 7.699 | **1.56×** |

簿记的大头：`MoeInitRoutingV3` 585 µs、`ConcatD` 568、`ClipByValueV2` 279 + 272、
`DequantSwigluQuant` 260、`Cast` 242、`MoeFinalizeRoutingV2` 239、`MoeGatingTopK` 194。
**每个 MoE 层 14 个簿记 kernel。**

→ 正确的说法是：**MoE 的权重读取没得优化（1.03×），但 MoE 家族整体是地板的 1.56×，
差额 4.3 ms 里 3.2 ms 是路由簿记、0.8 ms 是 shared expert 那两个被 launch 开销主导的小 GEMM。**
同样的拆解对 KDA 见 §8.1、对 DSA 见 §8.2。

---

## 3. Roofline：bs=1 必须读多少字节 [实测，逐张量]

按算子**实际需要**的权重（不是常驻量）算，1.25 TB/s 实测读写带宽：

| | 每层 MiB | MiB/token | ms @1.25 TB/s | 占流量 |
|---|---|---|---|---|
| **KDA ×34（BF16，未被量化）** | 262.7 | **8932.5** | **7.493** | **41.8%** |
| **MoE routed top-8 ×42** | 192.2 | **8074.5** | **6.773** | **37.8%** |
| DSA/MLA ×11 | 142.3 | 1565.8 | 1.313 | 7.3% |
| lm_head | — | 1210.0 | 1.015 | 5.7% |
| MoE shared expert ×42 | 24.0 | 1009.3 | 0.847 | 4.7% |
| dense FFN ×3 | 144.1 | 432.3 | 0.363 | 2.0% |
| MoE gate ×42 | 2.3 | 94.5 | 0.079 | 0.4% |
| mHC + layernorm | 1.5 | 68.2 | 0.057 | 0.3% |
| **合计** | | **21387 (20.9 GiB)** | **17.941** | |

**实测 42.8 ms ÷ roofline 17.94 ms = 2.39×。**

⚠ 这张表 2026-08-30 修过一次算术错误（初版：routed 7886.7、gate 233.1、dense 288.2、
合计 17.759 ms、2.41×）。两个 bug：① 每专家字节除以 43 层却乘了 42 层
（第 43 个 MoE 层是不建的 MTP 层）；② 分桶用 `'mlp.gate' in rest`，
把 dense 层的 `mlp.gate_proj` 也算进了 MoE gate。结论方向未变。

### 最反直觉的一条：KDA 根本没被量化 [实测]

这份 W8A8 checkpoint 里，34 个 KDA 层的 `q/k/v_proj [8192,4096]`、`o_proj [4096,8192]`
**全是 BF16**（逐张量读 dtype，不是读 config 的 `ignore` 列表）。每层 262 MiB，34 层 8.72 GiB。
DSA 层的 `kv_b_proj [32768,512]` 同样是 BF16。

**于是 bs=1 每 token 权重流量的 41.8% 完全没有被 INT8 覆盖。**

精确的对比（别把这两个数说混）：

| | 每 token | 占流量 |
|---|---|---|
| KDA 34 层 | 8932.5 MiB | **41.8%** |
| MoE routed（top-8） | 8074.5 MiB | 37.8% |
| MoE 全部（routed + shared + gate） | 9178.4 MiB | **42.9%** |

**KDA 比 MoE routed 专家大 11%；和整个 MoE 块基本打平（41.8% vs 42.9%）。**
一句话版本：**这 34 个没被量化的层，单独就和整个 MoE 一样贵。**

关键的算术在**每层**：KDA 每层 **262.7 MiB 全量读**，MoE 每层 **187.8 MiB = 288 个专家里的 8 个（2.8%）**。
所以「MoE 有 288 个专家所以它最大」这个直觉在 bs=1 是错的 ——
MoE 常驻 283.9 GiB 每 token 只读 7.7 GiB，KDA 常驻 8.7 GiB **常驻即流量**。

⚠ **这不是转换时漏了，是照搬厂商的决定。** 由 glm53_graph_perf session 查保留下来的 FP8
元数据独立核实 **[实测]**：KDA 层的 q/k/v/o_proj **一个 `weight_scale_inv` 都没有**（DSA 层的
q_a/kv_a/o_proj 有），而 `modules_to_not_convert` 是**逐层精确点名**的 —— `o_proj` 出现在
**34/34 个 KDA 层、0/11 个 DSA 层**。不是笼统的模式匹配，是刻意逐层挑出来排除的。

**注意区分两个不能互换的数**：BF16 权重占**常驻**的 4.2%（12.90/305.7 GiB），
但占 **bs=1 每 token 流量**的 42%。因为 MoE 常驻 288 个专家却只读 8 个。

---

## 4. kernel 级大头：实测耗时 vs 自己的地板 [实测]

每个 kernel 的地板由**它自己搬的字节**决定，不是全局平均：
`traffic = 输入张量字节 + 输出张量字节`（profiler 的 `Input/Output Shapes` 列即使不开
`record_shapes` 也是填的），除以实测的 1.25 TB/s。
**流量 <16 MB 的 kernel，耗时不由它搬的字节解释，对地板的比值没有意义**，
这些行不给倍数，只给判定。

⚠ **这里曾经写着「由约 13.5 µs 的固定 launch 开销主导」，那句话是错的**，见 §7b.14。
本机 device 侧单 kernel 的实际下限是 **1.3–1.5 µs**（本 profile 里最小的
`Cast` / `Mul` / `BroadcastTo`，标量形状）。**「launch 主导」这个标签的意思是
「它的时间不由它的字节解释」，不是「其中 13.5 µs 是 launch 开销」。**生成脚本 `tools/kernel_roofline.py`，全量输出在
`data/kernel_roofline.txt`。

```
=== bs=1 decode step: 42.823 ms device time, 3578 kernel launches, floor 1.25 TB/s ===
kernel                     n  us/step     %  cum%  us/call       MiB  floor us  x floor  bound
----------------------------------------------------------------------------------------------
MatMulV2  KDA qkv         34   7823.3 18.3% 18.3%    182.1     192.1     161.1    1.13x  bandwidth
GroupedMatmul  MoE gate_up 42  4516.5 10.5% 28.8%    107.3     128.0     107.4    1.00x  bandwidth [8/16 专家]
HcPre                     90   2995.1  7.0% 35.8%     32.9       1.5       1.3        —  compute
GroupedMatmul  MoE down   42   2469.6  5.8% 41.6%     58.4      64.0      53.7    1.09x  bandwidth [8/16 专家]
MatMulV2  KDA o_proj      34   2026.9  4.7% 46.3%     59.0      64.0      53.7    1.10x  bandwidth
fused_sigmoid_gating_...  34   1128.8  2.6% 48.9%     33.1       0.0       0.0        —  compute
MatMulV2  f_a/g_a/indexer 90   1036.3  2.4% 51.4%     11.5       1.0       0.8        —  launch
QuantBatchMatmulV3 shared 42    954.5  2.2% 53.6%     22.7      16.0      13.4    1.69x  above floor
MatMulV2  lm_head          1    930.1  2.2% 55.8%    930.4    1210.3    1015.3    0.92x  bandwidth
QuantBatchMatmulV3 DSA o  11    717.2  1.7% 57.4%     65.6      64.0      53.7    1.22x  bandwidth
BroadcastTo  零 rope      11    701.0  1.6% 59.1%     63.7     151.6     127.2    0.50x  bandwidth
QuantBatchMatmulV3 shared 42    680.8  1.6% 60.7%     16.1       8.0       6.7        —  launch
HcPost                    90    631.9  1.5% 62.1%      7.0       0.1       0.1        —  launch
Range                    100    600.7  1.4% 63.5%      6.4       0.0       0.0        —  launch
MoeInitRoutingV3          42    585.3  1.4% 64.9%     13.8       0.0       0.0        —  launch
ConcatD  MoE              42    567.9  1.3% 66.2%     13.4       0.0       0.0        —  launch
Slice  KDA                34    522.3  1.2% 67.5%     15.4       0.7       0.6        —  launch
ReduceSum  KDA            34    507.4  1.2% 68.6%     14.8       0.5       0.4        —  launch
MatMulV2  f_b/g_b         68    477.4  1.1% 69.8%      6.9       2.0       1.7        —  launch
IndexPutV2  KDA           34    428.9  1.0% 70.8%     12.6       0.8       0.7        —  launch
MatMulV2  b_proj          34    424.9  1.0% 71.8%     12.5       0.5       0.4        —  launch
Cast  KDA qkv fp32        68    413.9  1.0% 72.7%      6.2       0.4       0.4        —  launch
batch_matmul_transpose_0  11    388.3  0.9% 73.6%     35.3       0.0       0.0        —  compute
RmsNorm                   91    363.4  0.8% 74.5%      4.3       0.0       0.0        —  launch
Index  KDA                34    334.3  0.8% 75.3%      9.4       0.6       0.5        —  launch
SparseFlashAttention      11    328.0  0.8% 76.0%     29.8       4.0       3.4        —  compute [<=2048 选中 token]
ClipByValueV2             68    319.7  0.7% 76.8%      5.0       0.0       0.0        —  launch
ConcatD  KDA              34    297.3  0.7% 77.5%      8.7       0.8       0.6        —  launch
... 其余 120 组: 9651.4 us/step (22.5%)
```

### 按「什么在限制它」汇总整步 [实测]

| | ms/step | 占比 | kernel/step |
|---|---|---|---|
| **带宽受限**（≥16 MB，实测 <1.5× 地板） | **20.022** | **46.8%** | 215 |
| **launch 主导**（<16 MB 且 <27 µs/call） | **16.662** | **38.9%** | **3163** |
| compute / 固定成本主导（<16 MB 但 ≥27 µs/call） | 4.899 | 11.4% | 147 |
| 高于地板（≥16 MB 但 ≥1.5× 地板） | 1.241 | 2.9% | 53 |

**这是整份报告最该记住的一张表**：**一半的时间花在 3163 个搬不到 16 MB 的小 kernel 上。**
带宽那 46.8% 里的大头（KDA qkv 1.13×、MoE gmm 1.00×、lm_head 0.92×）**已经没有余地**，
真正的余地在那 38.9%。

### 逐条要点

1. **MoE 的 `GroupedMatmul` 已经贴地板**：gmm1 实测中位 107.3 µs，
   8 个专家 128 MiB 在 1.25 TB/s 下是 107.4 µs —— **1.00×**。
   这同时证明 **gmm 只读被选中的 8 个专家**，不是全部 16 个。
   **MoE 的权重读取在单卡上没有性能可捡了，只有搬走它这一个选项。**
2. **两条低于 1.0× 的行不是测量错误，是 L2**：`lm_head` 0.92×、
   零 rope 的 `BroadcastTo` **0.50×**。后者写 151.6 MiB，
   而 **L2 是 168 MB** —— 它整份都落在 L2 里，根本没打到 HBM。
   这正是 §6.1 那个「修掉它省下的比它自己还多 3 倍」的机制证据。
3. **`npu_dynamic_quant` 的 140 个 kernel 不是 bs=1 的头号开销 [实测，纠正一条推断]**。
   交接文档基于 TP16 的「源码 + 计数」推断这 140 个 kernel 值 +3.5 ms/token。
   本轮在 TP1 图模式下实测：同类 kernel 123 个（按实际建起来的层数：11 DSA×4 + 42 shared×2 +
   3 dense×2；MTP 层不建），**合计 382 µs/step = 0.9%**，每个 2.8 µs。
   **图模式把 launch 开销吃掉了，剩下的就是这么多。**
   ⚠ 本条只推翻「140 个 kernel 在 TP1 图模式下值 3.5 ms」这个机制解释，
   **不推翻 TP16 上 INT8 比 BF16 慢 3.5 ms 这个现象本身** —— 那是别处的原因，本轮没测。
4. **`Cast` 是最大的隐形开销**：一步 568 次、1.47 ms、3.4%，全部是 dtype 转换。
   其中 KDA 的 `"1,24576,3"` bf16↔fp32 四组共 660 µs/step，来自 KDA 的 fp32 工作集绕行。
5. **`SparseFlashAttention` / `LightningIndexer` 的地板必须人工修正**：
   算子拿到的是整个 paged KV cache（声明 2577 MiB），实际只读 indexer 选中的
   ≤2048 个 token（约 4 MiB）。不修正的话它们会显示成「0.01× 地板」这种无意义的数。
   修正在 `kernel_roofline.py` 的 `OVERRIDE_BY_TYPE` 里，显式列着。

## 5. 哪些开销在 bs=1 主导、在大 batch 会摊薄 [实测]

**这是对 offload 项目最重要的一节** —— offload 的工作点就是小 batch。
同一台机器、同一份 checkpoint、bs=1 与 bs=16 各采一次 profile：

| 层族 | bs=1 ms/step | bs=16 ms/step | bs=16 每 token | **摊薄倍数** |
|---|---|---|---|---|
| mHC | 3.627 | 4.096 | 0.256 | **14.2×**（几乎完美） |
| dense FFN | 0.616 | 0.760 | 0.048 | 13.0× |
| **MoE** | **12.041** | **15.314** | **0.957** | **12.6×** |
| DSA | 6.693 | 9.010 | 0.563 | 11.9× |
| **KDA** | **16.824** | **37.895** | **2.368** | **7.1×（最差）** |
| 合计 | 42.823 | 72.507 | 4.532 | 9.45× |

（16 是完美摊薄的上限。）

⚠ **「摊薄」是每 token 口径，每步延迟一分不降 —— mHC 甚至还涨了**（3.627 → 4.096 ms/step）。
这一列现在有机制了 [实测，`tools/hcpre_microbench.py`]：`HcPre` 的耗时在 **M ≤ 16 上几乎是平的**
（M=1 是 28.4 µs，M=16 是 30.5 µs，**M=1 是 M=16 的 98%**），大 M 才线性
（`T(M) ≈ 19.7 + 0.224·M` µs，边际追平固定项要到 **M ≈ 85**）。

**所以 mHC 那个「14.2× 几乎完美」不是它变高效了，是它压根没变** ——
分母涨了 16 倍而分子没动。`--max-running-requests 16` 这个可部署区间**整个落在平段里**，
`HcPre` 是每步的常数，不是 bs=1 的伪影。
反过来，**prefill（M=1024~8192）每 token 只要 0.23 µs，可忽略；
MTP / 投机解码的 HcPre 是白送的**（M 从 1 涨到 16 不要钱）。

**结论**：

- **MoE 是最容易摊薄的大项**（12.6×）。它在 bs=1 占 28%，在 bs=16 只占 21%。
  原因很直白：一个 batch 里所有 token 共读同一批专家权重。
  ⚠ **但这条被 16 专家构型放大了** —— 288 专家时 batch 内 token 会散到更多专家上，
  摊薄倍数会比 12.6 低。**这个数字是乐观的上界。**
- **KDA 是唯一摊薄不掉的大项**，而且它在 bs=16 涨到 **52.3%**。
  根子在 `fused_sigmoid_gating_delta_rule_update`：33.1 → 333.1 µs，
  **16 倍 token 只换来 10 倍**。线性注意力的状态更新是每 token 各更新一份
  `[heads, head_dim, head_dim]` 的递归状态，**本来就不可摊薄**。
- **专家总数在 bs=1 几乎不影响，在 batch 下强烈影响 [实测，对照实验]**。
  常驻专家从 16 换成 32（其余不变）：**bs=1 只慢 1.0%（39.8 → 40.2 ms），
  bs=16 慢 12.6%（72.9 → 82.1 ms）**。
  bs=1 时无论常驻多少，读的都是 top-8，所以只多了 `group_list` 变长的那点开销；
  batch 下 16 个 token × top-8 = 128 次选择散到 32 个专家上而不是 16 个，
  **同样的 token 要读近两倍的专家权重**。
  → **§0 那条警告到此有了数量级**：288 专家时 bs=16 最多能激活 128 个不同专家，
  是 16 专家构型的 8 倍流量。**bs>1 的 MoE 数字一律不可外推；bs=1 的可以。**
- **对 offload 的直接含义**：小 batch 下 die 是**带宽饥饿**的，
  而 MoE 恰好是唯一贴在带宽地板上、且最容易摊薄的那一块。
  把它搬到 CPU 意味着用**远低于 1.25 TB/s** 的链路去换 HBM 容量，
  且在小 batch 下**没有 batch 可以帮你摊薄搬运成本**。

---

## 6. 做了哪些优化，各自实测收益

****六条落地**：四条逐位/逐 token 不变，两条走形状地板判据。** 基线链条一次只动一个变量；每一步都有 profiler 的
device 时间和墙钟两个独立仪器，**每一步都吻合** —— 这不是巧合，§1 说过这条链路 100% device-bound。

| 构型 | device ms/step | 墙钟 ms/token | kernel/step | 相对上一行 |
|---|---|---|---|---|
| **A** 仓库原样（e16） | 42.823 | 42.8 | 3578 | — |
| **B** = A + §6.1 零 rope 不再每层物化 | 39.741 | 39.8 | 3556 | −7.2% |
| **C** = B + §6.2 KDA conv state fp32 + padding clamp | 38.727 | 38.6 | 3352 | −2.5% |
| **D** = C + §6.3 MoE 两个循环不变量 | 37.679 | 37.6 | 3220 | −2.7% |
| **E** = D + §6.4 KDA 小投影融合 | 36.802 | 36.6 | 3084 | −2.3% |
| **F** = E + §6.5 conv 池翻面 → AOT 算子 | 34.023 | 34.1 | 2710 | −7.6% |
| **G** = F + §6.6 DSA 元数据每步算一次 | **33.348** | **33.5** | **2570** | **−2.0%** |
| | **−22.1%** | **−21.7%** | **−1008** | |

bs=16 同链条：75.4 → 72.9 → 71.4 → 70.3 → 69.6 → 60.1 → **59.0 ms/token（−21.8%）**。
**bs=16 的收益比 bs=1 还大** —— 卷积是每 token 的活，从来摊薄不掉。
构型 F→G 在四个并发上一致下移（34.1→33.5、41.9→40.9、57.6→56.6、60.1→59.0），
**四点同向是这条 −0.68 ms 不是噪声的第三个仪器**（§7b.11：单次 profile 噪声底是几百微秒）。

**精度**：§6.1、§6.3、§6.6 端到端逐 token 相同（3 条提示 × 64 token，3/3）；
§6.2 由 `check_kda` 判定 **18/18 张量逐位相同**（float64 `==`）。
只有 §6.4 / §6.5 是非逐位的，走 teacher-forced 的形状地板判据。**没有一条动了数值。**

### 6.1 修掉 NoPE 零 rope 被静默物化 [−3.08 ms/step，−7.2%]

`ascend_backend.py` 的 `_nope_zero_rope`：GLM 是 NoPE 模型，但
`npu_sparse_flash_attention` 拒绝缺省的 rope、零宽度的 rope 和除 64 外的一切宽度，
所以那里喂一个全零 rope。原实现用 **stride-0 的 `expand`** 想省掉那份缓存，
并在 docstring 里写了 ⚠：「算子文档说不支持非连续输入，所以这个 aliasing 是观察到的行为」。

**那条 ⚠ 兑现了，但不是以算错的方式，而是以静默变慢的方式。** torch_npu 调 aclnn 前把输入
变连续，于是这个 expand **每次调用都被真的物化**：`[1,64,1,64] → [19403,64,1,64]`，
`aiv_mte3_ratio = 0.945`（纯 store），每个 DSA 层每步写 **159 MiB 的零**。

**最值钱的性质：它是 O(KV pool)。** pool 从 124 万开到 152 万 token，这一项 0.70 → **1.34 ms/step**；
bs=1 和 bs=16 几乎一样贵，**大 batch 摊不掉**。

**收益是它自己那 0.80 ms 的 3.9 倍**，差额的机制 **[推断]** 是 L2 冲刷（151.6 MiB 对 L2 的 168 MB），
证据是**分布形状而不是均值**：相邻 KDA qkv matmul 的中位数只降 2.4%（182.1 → 177.8 µs），
每步总和却降 23%（7823 → 6022 µs）—— 长尾消失了。看整个分布更清楚：

| KDA qkv GEMM（34 次/步，同一 shape） | 中位 | 均值 | p90 | max |
|---|---|---|---|---|
| 构型 A | 182.1 | **230.1** | **338.0** | 340.5 |
| 构型 B = A + §6.1 | 177.8 | 177.1 | 183.3 | 189.8 |
| 构型 G（现在） | 175.2 | 174.3 | 179.1 | 185.8 |

**构型 A 是双峰的**：约三分之一的 KDA 层付 ~340 µs 而不是 ~182 µs。
§6.1 之后这个峰整个消失，而中位数几乎没动 —— **这不是让 GEMM 变快，是不再拖慢它**。
⚠ 机制（L2 冲刷）仍是 [推断]，没有直接量过 L2 命中率；**被证实的是「尾巴消失」这个现象**。

**判据（给别的配置复核用）**：profile 里 `BroadcastTo` 的
`"1,64,1,64" → "<pages>,64,1,64"` 这一组应当完全消失。

### 6.2 让 KDA conv 的 decode 快路径变得可达 [−1.01 ms/step，−2.5%]

`_causal_conv1d_decode` 有一条快路径和一条 dtype 不匹配时的绕行。GLM 的 conv **权重**是 fp32
（`glm5_next.py` 的 `params_dtype`），而 conv **state** 默认 bf16，**于是快路径永远走不到**，
每步都走 gather → cast → conv → cast → scatter。`SGLANG_MAMBA_CONV_DTYPE=float32` 让两者一致。

被删掉的 kernel 在 profile 里一一对得上（这是它不是噪声的证据）：
`GatherV3 34→0`、`Cast 68→0`、`ScatterUpdate 34→0`、`Range 100→66`、`ClipByValueV2 68→34`，
**每个 KDA 层 7 个**。代价：conv state 显存 0.08 → 0.16 GB。

⚠ **「fp32 state 数值更好」这个直觉是错的，实测证伪。** conv 窗口存的是 bf16 投影输出的
**拷贝而不是计算结果**，所以 bf16 往返本来就是无损的 —— 实测 `max|t − bf16(t)| == 0.0`。
**这条改动的理由只能是 kernel 数，不能是精度。**

⚠ **走快路径必须补一个 clamp，否则是静默的正确性 bug。** 快路径原样传 `cache_indices`，
而 padding 行是 `-1`；`causal_conv1d_update_npu` 虽然有 `pad_slot_id` 参数，
**但在我们走的那条腿上根本不看它**（`cache_seqlens` 与 `num_accepted_tokens` 都是 None 时
它落到裸的高级索引，而 `torch_causal_conv1d_update_npu` 连这个参数都没有）。
于是 `-1` 被当作负下标**绕到最后一个 mamba 槽**，而分配器保留的是**槽 0**。
结果是把一个活着的请求的 conv state 静默写坏 —— continuous batching 下可达，但不可按需复现。

### 6.3 停止每步重建两个循环不变量 [−1.05 ms/step，−2.7%]

两个 kernel，存在的唯一目的是撤销上一行刚做过的事。

**shared expert 的 clamp 折叠**：非对称 swiglu clamp 原本是「切两半 → 各自 clamp → 拼回去」，
而 `[rows, 2I]` 的两半是**跨步视图**，等于两趟跨步遍历加一次拷贝。仓库里已经有
`apply_swiglu_limit_`，它把不对称性折进广播界向量、对连续缓冲区一次 clamp
（docstring 记着 186 → 589 GB/s、逐位相同），而且它的 `lru_cache` key 和 routed 路径撞得上，
**零额外显存**。

```
ConcatD [1,2048]×2   42 → 0    −508.4 µs
ClipByValueV2 标量界  84 → 0    −275.1 µs
ClipByValueV2 向量界   0 → 42   +113.3 µs
                              ───────── −670 µs, −84 kernel
顺带白捡 dense FFN 3 层（走同一段代码）  −72 µs, −9 kernel
```

⚠ 那个 `ConcatD` **从来不是带宽问题**：实测每次 **13.5 µs 搬 8 KB**，起了 **37 个向量核**
而 `aiv_vec_time` 只有 **0.02 µs**。纯块启动开销。**在图模式下，「起了几个核」比「搬了多少字节」
更能解释小算子的耗时** —— 这是本轮反复出现的形状。

**router gate 权重的 cast 提到 load 期**：非 CUDA 分支刻意用 fp32 算 logits（bf16 路由会
整个丢掉专家而不只是掉精度），但它把**权重**的 cast 放在了 forward 里，于是一个循环不变量
每层每步转一次。改成按参数的 storage 指针 + version 缓存，权重更新 / LoRA / EPLB 重排会
自动失效。`Cast [16,4096] 42 → 0，−249.3 µs`。288 专家下这个 cast 是每层 7 MB 而不是 128 KB，
**只会更省**（代价是常驻 +99 MB）。

⚠ **一条没解释掉的**：hoist 之后 gate matmul 自己慢了 **+75.7 µs**（同样 42 次、同形状、同 dtype），
吃掉这条三分之一的收益。可能是局部性，也可能是噪声。**如实记下，没有替它编解释。**

### 6.4 让 KDA 小投影融合 [−0.877 ms/step，−2.3%]

`glm5_next.py` 的 `do_fuse_qkvbfg` 原本门在 `quant_config is None` 上 —— 它问的是
「模型有没有量化配置」，该问的是「**这几层是不是被量化了**」。GLM 的 W8A8 checkpoint 里
`q/k/v/b/f_a/g_a/f_b/g_b_proj` **八个全在 `modules_to_not_convert` 里**，厂商连
**融合模块名本身**也列了进去 —— 读起来像是预期过这条路。而挡路的只有那个条件：
`MergedColumnParallelRepeatedLinear` 本来就收 `quant_config`，加载器的八个 shard 映射也都在。

**每个 kernel 都对得上账，而不是只给一个净收益**：

```
MatMulV2 [1,4096;24576,4096]  34 → 0    −5966 µs   ┐ 被替换
MatMulV2 [1,4096;24896,4096]   0 → 34   +6010 µs   ┘ 宽 2.5 MiB，只贵 44 µs
MatMulV2 [1,4096;128,4096]    90 → 22    −764 µs   ← 68 个 KDA 的 f_a/g_a，22 个 DSA 留着
MatMulV2 [1,128;8192,128]     68 → 0     −482 µs   ┐ 换成
BatchMatMulV2 [2,1,128;…]      0 → 34    +394 µs   ┘ 一次批量调用
MatMulV2 [1,4096;64,4096]     34 → 0     −421 µs   ← b_proj
                                        −1229 µs 的矩阵乘
```

谓词**刻意保守**：只认明文 `ignore`（正则不认）、只认精确名，未知配置一律回落到今天的行为
（不融合，慢但正确）。⚠ 两个条件都必需但**理由不同**：**子模块**在 ignore 里 → 这几层没被量化，
**所以融合是正确的**；**融合名**在 ignore 里 → 融合模块自己不会去要 checkpoint 里不存在的
int8 权重。第二条从调用点读不出来。

⚠ **对多卡线的推论（是推断，不要外推）**：BF16 下 `quant_config is None`，所以它**本来就走融合路径** ——
INT8 相对 BF16 白付了这 1.68 ms。这是「TP16 bs=1 INT8 慢 3.5 ms」那条**机制已被撤回**的观测的
一个候选解释。⚠ **只在 TP1 上量过**，TP16 下 KDA 被 16 分，这些矩阵乘小得多，占比未知。
**可证伪的判据**：TP16 上 BF16 与 INT8 各采一份 bs=1 profile，数
`MatMulV2 [1,4096;128,4096]` 的次数 —— BF16 应当没有，INT8 应当有 68/step。

---

### 6.5 把 KDA 的 conv 池翻成 window-major，接上 AOT 算子 [−2.78 ms/step，−7.6%]

**这一条是最大的。** KDA 的 decode 卷积原本是**每层九个 torch 算子**，根因是布局：

| | conv 池布局 | 能到达的入口 |
|---|---|---|
| GDN（`is_kda=False`） | **window-major** | `torch.ops.npu.causal_conv1d`（AOT，一个 kernel） |
| **KDA（`is_kda=True`）** | **channel-major** `[17, 24576, 3]` | `causal_conv1d_update_npu` —— **带 torch 回退的那个** |

`sgl_kernel_npu/mamba/causal_conv1d.py:1379`：`cache_seqlens` 与 `num_accepted_tokens`
都为 None 时（**正是我们**）落进 `torch_causal_conv1d_update_npu`，Triton kernel 走不到。

⚠ **PLAN §2.5 曾把这条列在「已排除（不要再做）」下，并写着「decode 已在用」—— 后半句是错的。**
已更正（`811de9e6a9`）。**挂在「别再看了」下面的错误是最贵的一类：没人会去复查它。**

**逐 kernel 对账**：

```
Slice / ReduceSum / IndexPutV2 / Index / Cast /
ConcatD / BroadcastTo / Mul / Swish        各 34 → 0     −3167 µs
causal_conv1d_4                              0 → 34       +545 µs
                                                        ─────────
                                                          −2622 µs
```

整步 36.802 → **34.023 ms**，kernel 3084 → **2710**。

#### 实际是 9 处消费点，不是任务书列的 7 处

多出来的两处**都是同一个形状：只在旧布局下碰巧成立的代码**（见 §7b.9）。
`conv_states_shape` 把原始池形状暴露出去，而 `_init_track_conv_indices` 读它的 `[-1]`
当卷积窗口长度 —— 翻面后会读到 **24576 而不是 3**，不报错。
另一处是 `check_kda.py` 的 `RankRunner.states()`，**同一个文件里的第三个硬编码**。

#### conv 权重必须降到 bf16，但 cast 放在 NPU 侧

算子硬要求 weight / activation / conv state 三者同 dtype ∈ {bf16, fp16}，fp32 是
**干净的 host 侧拒绝**。但 cast 放进 `_get_conv_weights_t` 而不是参数本身 ——
改 `params_dtype` 会**把 CUDA 一起改了**，而 CUDA 的 `kda_backend.py` 在 conv 权重非 fp32 时
**静默地**（返回 False）放弃它的 K3 融合。那个缓存转置本来就每层做一次，所以 cast 免费。

#### `x.contiguous()` 不是装饰，而且它暴露了单变量 A/B 的盲区

融合投影（§6.4）把 qkv 作为一个宽输出的**最后一维前缀切片**交出来 ——
**bs=1 时它是连续的（只有一行），bs≥2 才不是**。所以它只在图捕获走到 bs≥2 的桶时才炸，
而且**两个调用点各炸一次**（decode 一次、extend 一次）。

⚠ **agent 的单层验证 24/24 全绿是完全正确的** —— 它的 worktree 基线在融合提交之前，
未融合路径下 qkv 是全新的连续张量。**单变量 A/B 保证归因干净，但交互只有集成时才看得见。**

算子是**拒绝**（`RuntimeError: x must be contiguous`）而不是按错误的步长读。
⚠ **这是运气**：这条改动本来就不是逐位相同的，如果它静默读错，我大概率发现不了。
本项目已知的四个算子约束里**三个是静默的**（`causal_conv1d_update` 的 layout、
`clamp_limit` 被忽略、int64 下标算错），响亮的只有 dtype 那一个。

#### 验证

- **AI core trap = 0**，padded batch（并发 **3 / 13**，非桶值）全过
- 单层 **24/24 在预算内**，`out.decode` 与 `state.ssm.final` 四个用例**全部改善**
- 端到端 teacher-forced **mean|dlp| 8.05e-03**（对构型 D），地板 0~2.6e-2
- ⚠ **未验证**：MTP / speculative 快照路径、mask-track 散写分支 —— 见 §8

### 6.6 DSA 的 batch 元数据每步算一次，而不是每层算一次 [−0.675 ms/step，−2.0%]

`kpool_indexer_npu.py`。11 个 DSA 层，每层进来都从 `forward_batch` 重新推一遍
**同一份**东西：`seq_lens_row`、`req_index_row`、`pool_lens_row`、`cu_seqlens_q`、
`block_table`。**这五样在一个 forward 内是常量** —— 它们只依赖 batch 的形状，不依赖层。
于是同一串 index 算术被完整重跑了 11 遍，profiler 里表现为 DSA 家族每层 97 个 kernel
而 `SparseFlashAttention` 本身只占 32 µs。

改成 `KPoolNPUIndexerMixin._step_metadata()`：每个 forward 算一次，11 层共用。
缓存 key 是 `(seq_lens.data_ptr(), seq_lens._version, block_tables.data_ptr(),
block_tables._version, batch, batch_size)`，外加「层号必须比上次大」来识别新的 forward。
**只在 decode 路径用**，extend 路径原样不动。

⚠ **`_version` 是这里唯一防原地写的东西。** 光比 `data_ptr` 不够 —— 图模式下 batch 张量
是复用的固定缓冲，指针每步都一样而内容每步都变。`_version` 在 `copy_` 之后会自增，
所以它能抓到原地更新；但它抓不到通过别的视图绕过版本计数的写。这条改动能成立，
**依赖「decode 阶段这几个张量只被 `copy_` 更新」这个前提**，前提变了它就会静默用旧值。

#### 收益从哪来

DSA 家族 6.253 → 5.307 ms，kernel 1067 → 913（**每层 97 → 83 个**）。
**cube 类那 143 个 kernel 一个没少，时间也没变**（3.002 → 3.021 ms，差 19 µs 在噪声里）
—— 省下的 946 µs 全部来自非 cube 的索引算术。这个「该动的动了、不该动的没动」
是归因干净的判据，比总时间下降本身更有说服力。

#### 验证

- 端到端**逐 token 相同**：3 条提示 × 64 token，**3/3 完全一致**
  （对构型 F 的 `ids_clipswiglu.json`，§7.7 已判定它与构型 F 逐位相同）
- 并发 1 / 3 / 13 / 16 **四点一致下移**，其中 3 和 13 是非桶值（padded replay）
- ⚠ **未验证**：MTP / speculative。缓存 key 里的 `cu_seqlens_q` 建立在
  「每请求一行、每行一个 token」上，**投机解码一步多 token 会打穿这个前提**。
  这与 `kpool_decode_update_index_cache` 的 ring 槽假设是**同一个错误前提的两处表现**，
  要改就一起改，或者在 MTP 路径上直接绕过这个缓存（它只值 0.68 ms，正确性优先）。
  已把这条交给接手 MTP 的 session。

---

---

## 7. 试过、测了、不落地的

**和成功的一样写在这里，而且都写在了会被再次尝试的那个源码位置**，因为它们从源码看都"显然对"。

### 7.1 量化 KDA 投影 —— 测到 −12.1%，**已封闭，不要再试**

⚠ 性能探针，非可交付。**决定经 glm53_graph_perf session 转达**（与我在 §3 的顾虑一致；
建议读者向用户复核一次）。KDA qkv 融合投影 7823 → **2717 µs/step**
（96 MiB int8，实测中位 79.7 µs 对地板 76.8，**1.04×**），o_proj 2027 → 1192，
整步 42.823 → 37.653 ms。

#### ⛔ 封闭的理由：**模型作者自己排除了它们** [实测，读原始发布的 config]

⚠ **本条曾经写成「这是怀疑不是解释」，那句话是错的，现在有硬证据。**

`/mnt/workspace/models/GLM-5.3-Flash/config.json`（**原始 FP8 E4M3 发布**，不是本项目的转换产物）的
`quantization_config.modules_to_not_convert` 共 **1509 条**，其中：

- **34 个层**各自点名列着 `self_attn.` 下的
  `q_proj`、`k_proj`、`v_proj`、`o_proj`、`qkv_proj`、`fused_qkvbfg_a_proj`、
  `f_a_proj`、`f_b_proj`、`g_a_proj`、`g_b_proj`、`b_proj`
- **34 正好等于本模型的 KDA 层数**（45 = 34 KDA + 11 DSA），**无一例外**
- 同一批里还有 `A_log` / `dt_bias` / `q_conv1d` / `k_conv1d` / `v_conv1d` / `o_norm`

**所以 KDA 投影是 bf16 不是转换时的疏漏，是训练这个模型的人逐层逐模块做的决定。**
本轮量化它 = 做一件模型作者明确不做的事，而**我们没有任何手段判它坏没坏**
（KDA 是递归路径，权重误差沿 SSM 状态随序列累积，症状是长序列上的缓慢漂移，
GSM8K 那种几百 token 根本看不见）。

**决定：封闭。用户已明确指示不要动。** 这不是「等条件成熟再做」，是不做。

✅ **后续**：为了 §8 的 conv 池翻面，单独验了 conv **权重** fp32→bf16（不是投影）：
5 个用例 × 3 层，**30/30 张量在预算内，最差 0.33×（预算 1.00×）**，
而且 `state.ssm.final` 跨 256 → 4096 → 32768 是 3.722e-3 → 3.546e-3 → **3.516e-3**，
**平到略降，不累积**。所以「KDA 沾 bf16 就危险」这个笼统说法**不成立**，
危险的是**投影**（大矩阵、误差进入递归），不是 **conv 权重**（4 抽头、窗口是拷贝）。

### 7.2 `custom::npu_dequant_swiglu_clamp_quant` —— **它根本不 clamp**

目标是省掉 routed 路径的预 clamp（278.9 µs/step）。仓库 docstring 明确写着
「mode 1 下精确复现参考实现，所以 int8 路径可以去掉预 clamp 直接调它」。**那句话是错的。**

实测（A3，`[8,4096]` 与 `[128,4096]` bf16，输入放大 48× 让 limit 真正咬到）：

| 和哪种语义比 | max\|Δint8\| | 不同元素 |
|---|---|---|
| 参考（gate max=L, up ±L） | 126 | 8772/16384 |
| 两半都 ±L | 126 | 8772 |
| 只 clamp gate / 只 clamp up / gate±L,up max=L | 112–119 | 7760–8024 |
| **完全不 clamp** | **0** | **0/16384** |

`swiglu_mode` 取 **0/1/2/3 结果完全一样** —— `clamp_limit` 和 `swiglu_mode` 都被静默忽略。

**照那句 docstring 做，会静默删掉每个 routed 专家的 swiglu clamp**，而 clamp 是模型定义的
一部分（`KT_DISABLE_SWIGLU_CLAMP` 存在就是为了让"关掉它"是显式的）。

⚠ **它为什么骗得过随手一验**：真实激活 `max|gate_up| ≈ 2.17` 对 limit 10，**clamp 从不触发**，
不放大输入两者就是逐位相同。**任何以后验这个算子的人必须强制让 clamp 触发**
（`check_dense_ffn.py --scale-input 48` 就是干这个的）。已改正 docstring。

### 7.3 把 `routed_scaling_factor` 折进 topk —— 生效了，但收益低于噪声

`npu_moe_gating_top_k` 本来就收这个参数。**我怀疑的失败模式（缩放被 renorm 吃掉）实测不成立** ——
它在归一化**之后**乘（返回权重比值精确 2.5、专家集合不变，renorm 开关两种都验了）。
机制也确实生效：**`Muls [1,4096]` 42 → 0，−73.2 µs，−42 kernel**，和预测一致。

**但整步 device 时间只动了 −0.003 ms** —— 那 73 µs 完全在轮间噪声里。
而输出**不再逐位相同**（3 条提示里 2 条从 token 0–1 分叉，这是次 ulp 的重结合被 MoE 路由
翻转放大的典型形状）。**证明它落在精度地板内是一项真正的研究，0.19% 付不起这个价。回退。**

⚠ 姊妹项（shared 加法折进 `npu_moe_finalize_routing` 的 `skip1`，96 µs）**没做**：
要把 `shared_output` 穿过 dispatcher，且 `_shared_expert_tp1` 下加法必须在 all-reduce
**之后**，`skip1` 在那里是错的。

### 7.4 `TASK_QUEUE_ENABLE=2` —— 图模式下根本起不来 [P6.9 就此关闭]

```
RuntimeError: Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture,
              please export TASK_QUEUE_ENABLE=1/0.
```
抛在 `torch_npu/npu/graphs.py:625` 的 `capture_begin()`，第一个 bs 桶刚开始捕获时。
**torch_npu 的硬约束，响亮地失败，不会静默降级。** 与 §1 一致：device 时间已等于墙钟，
host 侧本来就没有气泡。

### 7.5 调低 sinkhorn 迭代数 —— 有效，但只值 0.7%

`hc_sinkhorn_iters` 20 → 1 **确实生效**（`HcPre` 中位 30.34 → 26.42 µs），整步
39.741 → 39.445 ms。**19 次迭代总共只值 347 µs/step。**
→ `HcPre` 那 2.4 ms **不是 sinkhorn**，是算子在 M=1 下的固定开销（26 µs 搬 1.5 MiB = 地板的 20 倍）。
**这个旋钮改数值、不改性能，别动。**

### 7.6 AOT `torch.ops.npu.causal_conv1d_update` —— 存在、可调、**又错又慢**

- **按它自己 assert 字符串写的 layout 算出来是错的，不报错。** 脉冲响应（weight 第 k 抽头设 `10^k`，
  输出数字直接读回抽头编号）显示它把 4 抽头窗口读成 `[S1, S2, S2, x]` —— **只有两个不同的
  历史值**。真实约定是 `[cache_len, WIDTH, dim]`（**width 行，不是 width−1**）+ 调用方每次
  `torch.roll(+1)`，**未文档化**。
- **比它本该替换的 torch 回退慢 2.5×**（b=1：689 vs 270 µs），约 0.3 GB/s。

**该用的是另一个**：见 §8。

---

### 7.7 把 shared expert 的 clamp+swiglu 融成一个算子 —— 逐位相同、少 45 个 kernel、**反而更慢**

`torch_npu.npu_clipped_swiglu`（四个参数全传）与「`apply_swiglu_limit_` + `npu_swiglu`」
两步式**逐位相同**（`max|dlp| = 0.0`，端到端），而且两个反向对照都有牙：
**全默认参数 max|Δ| = 156**（就是 `PLAN.md` §2.4 记的那个 109× 陷阱），**对「完全不 clamp」max|Δ| = 2.9e4**。

**但它更贵**：

| | µs/次 | ×42/步 |
|---|---|---|
| `npu_swiglu` | 4.27 | 179.4 |
| `ClipByValueV2`（向量界 clamp） | 2.64 | 110.9 |
| **两步合计** | **6.91** | **290.3** |
| **`ClippedSwiglu`（融合）** | **8.22** | **345.2** |

融合掉 45 个 kernel（42 shared expert + 3 dense FFN），**净 +65.6 µs**；整步 34.023 → 33.997，
是噪声。**回退。**

⚠ **这条推翻了本报告一直在用的启发式。** §4 和 §8.3 都说「杠杆是减 kernel 个数」——
这里减了 45 个反而更慢。**准确的说法是：杠杆是减少总的固定成本，而 kernel 个数只是它的代理，
当替换进来的 kernel 每次启动做的活不等价时，这个代理就失效。**

⚠ 与 §7.2 的直接对照值得记住：**同一族、名字都带 clamp、一真一假** ——
`custom::npu_dequant_swiglu_clamp_quant` 静默忽略 `clamp_limit`，
而 `torch_npu.npu_clipped_swiglu` 参数传对就精确。**必须逐个验，不能类推。**
⚠ 而且**只做正向比对是证明不了算子在做那件事的** —— §7.2 那次我就是这么被骗的；
这次加的两个反向对照（默认参数、不 clamp）才是让结论站得住的东西。

---

## 7b. 测量与工具的坑 —— 每一条都让一次测量作废过

**这些不是花絮。** 本轮七条结论里有三条是先被一个错误测量指到反方向、再被纠回来的。

### 7b.1 检查工具静默地测不了它被叫来测的东西（同一个文件，两处）

`check_kda.py` 在 `cache_params` 里硬编码 `dtype=Mamba2StateDType(conv=torch.bfloat16, ...)`，
**对 `SGLANG_MAMBA_CONV_DTYPE` 完全免疫** —— 而那正是决定 §6.2 走哪条分支的唯一旋钮。
它还在 `:127` 硬编码 `f32(...)` 的 conv 权重，**于是也验不了 §8 的 fp32→bf16**。

**一个 checking tool 里硬编码一个 dtype 是 bug，两个是 pattern。** 两处都是靠先问
**「这个 harness 真的在跑我要改的那件事吗」** 发现的，不是靠看绿灯。
现在两处都走生产的解析方式 / 由 CLI 控制。

**判据**：改一个旋钮之前，先证明 harness 会因为它而走到不同的代码 —— 用调用计数，不用假设。
（本轮的做法：在 `_causal_conv1d_decode` 上挂计数器，确认 128/128 次调用从绕行分支翻到快路径。）

### 7b.2 我的 commit message 写了一句不实的话

`7688ca5f8d` 的结尾写着「check_kda 现在和生产一样解析 state dtype」。**它没有** ——
那次提交只动了两个文件，`check_kda.py` 不在其中。成因：两处修改写在同一个 patch 脚本里，
第一处 assert 失败、脚本中止，第二处没写盘；我重跑了第一处，然后**照着打算做的事写 commit
message，而不是照着 diff 写**。`git show --stat` 一行就能抓到。已修（`8d054a123e`）并把经过写进提交。

### 7b.3 并发扫描用「桶大小」就测不到 padded replay

图的 decode 桶是 `[1,2,4,8,12,16]`。我最初的扫描用 **1 / 4 / 16 —— 恰好全是桶值，一行 padding 都没有**，
所以 §6.2 那个 padding 静默写坏 conv state 的 bug **在我所有 bench 里都不会暴露**。
另一条线用 1/8/16/32/64/128，同样全是桶值，同样测不到。
**现在规则是扫描必须混非桶值，固定 1 / 3 / 13 / 16。**

### 7b.4 「相减法」量 decode 对冷启动是**单向**敏感的

`bench_graph_decode.py` 用「先跑 `max_new_tokens=1` 拿 prefill 墙钟，再相减」分离 decode。
冷服务的第一发 prefill 花了 **1.88 s** 而不是 0.39 s，这 1.5 s **全部落在被减去的那一项里**。
我因此一度报出「KDA 量化 1.39×」，**真实是 1.14×**。**它只会让结果偏乐观。**
修法：先发一次丢弃的 warmup（已进 `ab_tp1.sh` 与共享的 `bench_graph_decode.py`）。

### 7b.5 `npu-smi` 的空闲检查少写一个 `\s*` 就恰好漏掉忙的卡

`npu-smi` 把已用 MiB 按固定宽度对齐：空闲 die 打印 `2880 / 65536`，
**忙的打印 `33861/ 65536`，斜杠前没有空格**。于是 `grep -oP '\d+(?= / 65536)'`
**恰好漏掉所有忙的 die**，列表整体前移一位，检查读到另一块卡的空闲数字并放行 ——
直接撞进加载权重时的 OOM，**看起来像显存不够，其实是被插队**。
实测：一块 die 在 33861 MiB 时，严格写法返回 15 个数，带 `\s*` 的返回 16 个。

### 7b.6 别编辑正在运行的 shell 脚本

bash 按字节偏移增量读脚本。在 `ab_tp1.sh` 跑着的时候给它打补丁，它接着从新文件的旧偏移读下去，
把一行注释当命令执行了，一次 A/B 白跑。

### 7b.7 换配置就不能沿用旧基线

（来自 glm53_graph_perf 的实测教训，同样适用于本线。）TP 宽度变了规约顺序就变，精度也不同，
硬套旧 golden 会测出一堆差异然后误以为是自己改坏了。做法是**在同一配置上自己造 before/after**
（`git worktree add --detach <改动前的提交>` 起一份基线）。本报告的每条 A/B 都是这么做的：
一次只动一个变量，golden 取自紧邻的上一个构型。

### 7b.8 验收只覆盖你去读的东西 —— 「逐位相同」看不见被写坏的旁路

重构 KDA conv 路径时，agent 在**正要被删掉的那条路**上找到一个实测 bug：
`causal_conv1d_fn_npu` 在**一个 extend batch 内混合 `has_initial_state`** 时会写坏 conv state。

```
全 True   state err 0        全 False  state err 0
F,T,F,T   state err 5.844    T,F,T,F   state err 5.984     (|state| ≈ 4.2)
```

**输出是对的，只有 state 回写坏。** 触发条件是**一批里同时有 prefix-cache 命中和冷请求**。

⚠ **`check_kda` 抓不到它** —— 基线 6/6 全绿，因为它只比对 golden 那个槽位。
这条比 bug 本身值钱：它说明本项目那套「逐位相同」信号有一个**确定的盲区形状** ——
**它排除不了「写坏了一块暂时没人读的东西，直到某个路径开始读它」。**
kpool 那条线的逐位验证是同一类结构（比对的是 logprob，不是所有被写的内存）。

⚠ **两条线的精度数字都没有被污染，但都是碰巧**：全部启动脚本带 `--disable-radix-cache`
（两边都核过**服务端实际生效的参数**，不是只看脚本），于是 `has_initial_state` 每批一致为 False，
触发形状从未出现。**那个 flag 在那里是为了让测量可复现，跟这个 bug 毫无关系。**
而「上生产第一件事就是打开 prefix cache」恰恰是给它上膛的动作 ——
所以 PLAN P6.2 记的不是「已知缺陷」，是一条禁令：**修好之前不要打开 radix cache 跑精度评测**
（glm53_graph_perf，`37250a2e43`）。

#### 这不是 conv state 的个案，是一个通例

本轮又独立撞到同形状的第二例，所以把它提炼出来。

**通例**：一条改动动了 N 个量，而你的判据只读其中 M 个。**M 个全绿说明不了另外 N−M 个。**
危险的地方在于，被漏掉的那些通常**不是随机漏的** —— 它们恰好是「当前这条路不读、所以想不起来」的量，
而「当前这条路不读」正是它们能被写坏而不被发现的原因。

| 例 | 判据读了什么 | 漏了什么 | 为什么当时没发现 |
|---|---|---|---|
| conv state（本节上文） | golden 槽位的输出 | **非 golden 槽位的 state 回写** | `--disable-radix-cache` 让触发形状从未出现 |
| W8A8 权重重映射（本轮在做的 shared expert 融合） | 权重张量本身 | **`weight_scale`** | per-channel scale 装错只表现成「精度差一点」，不崩 |

**第二例还多一层**：`w13` 的 gate/up 拼接顺序由 `switch_w13` 这个 **flag** 决定
（`fused_moe_triton/layer.py:669`），不是固定约定。一个「元素全对、顺序反了」的权重
**shape 对、元素集合对、norm 对、sum 对** —— 所有偷懒判据全部通过。
所以判据必须是**分别** `torch.equal(w13[slot][0], gate)` 和 `torch.equal(w13[slot][1], up)`，
**不许用 `sorted` / `norm` / `set` 去比**。

**可操作的那一条**：写判据之前先列「这条改动写了哪些量」，再逐个问「我的判据读它吗」。
**列不出来就说明还没想清楚改了什么。**

### 7b.9 「只在某个布局下碰巧成立」是这类重构的标准失败形态

conv 池翻面时，`ascend_kda_backend` 的 `conv_states_shape` 覆盖把原始池形状暴露出去，
而 `_init_track_conv_indices` 读它的 `[-1]` 当**卷积窗口长度** —— 这只在 channel-major 下成立。
翻面后它会读到 **24576 而不是 3**，不报错、全错。

**它不是写错了，它依赖了一个没人声明过的巧合。** 找它的办法不是读代码找 bug，
是**列出所有依赖布局的消费点，逐个问「这一行为什么成立」** ——
任务书列了 7 处，实际是 9 处，多出来的两处都是这个形状（另一处是 `check_kda.py`
的第三个硬编码）。

### 7b.10 非逐位改动的对拍必须 teacher forcing，否则数字是假的

验证 §6.4 时我第一次用的是「两个构型各自自由生成，然后比 logprob」。**一旦贪心路径分叉，
比的就是不同 token 的 logprob，那个数字没有意义** —— 而它长得像个数值结论。

| 提示 | 自由生成（错） | teacher forcing（对） |
|---|---|---|
| 0 | 4.622e-03 | 8.546e-04 |
| 1 | **2.849e-01** ← 像是超地板 11× | **1.911e-02** ✅ |
| 2 | 1.724e-02 | 4.192e-03 |
| 合计 | 1.022e-01 | **8.051e-03** |

**15 倍的差异纯粹来自测量方法。我差点用它把一条好改动判死。**
正确做法是让两个构型**给同一串 token 打分**（`input_ids = prompt + 固定输出`，
`max_new_tokens=0`，`return_logprob=True`）。

⚠ 这条不是本线独有：`tools/logit_check.py` 的 `_compare_decode` 有**同一个 bug** ——
它按全长 `range(n)` 求 `max|dlp|` 而不是分叉前的公共前缀，所以在分叉输入上会报出同样性质的
大数字。由 glm53_graph_perf 修于 `d77916eb96`，并把「分叉在第几个 token」和「公共前缀上的
max|dlp|」拆成两个数 —— **离散信号和连续信号塞进一个数字里就是在骗自己**。

### 7b.11 单次 profile 的噪声底是几百微秒

同样 90 次调用、同样形状的 `HcPre` 在两次 profile 之间无缘无故差过 226 µs。
**所以本报告不靠单个时间数字下结论**，而是靠：① 两个独立仪器（profiler device 时间 / 墙钟）
是否吻合；② **kernel 计数变化**（`ConcatD 42→0` 这种不是噪声能产生的）。
§7.3 就是被这条否掉的 —— 机制生效、kernel 少了 42 个，但整步只动 −0.003 ms。

---

### 7b.12 空闲检查是快照，而这台机器上还有第三个用户

conv 池翻面的服务级验证第一次跑失败了，**不是改动的问题**：die 0 的空闲检查通过（3135 MiB），
服务开始加载权重，**同一分钟另一个用户起了 16 卡 DSv4 训练**（每 die 55 GB），
我在 MoE 的 `create_weights` 处 OOM。

这是交接文档第 3 条坑的字面复现。**它最贵的地方不是浪费一次启动，是它长得像自己的 bug** ——
我差点去查 conv 池改动是不是把显存算错了。证据链得自己拼：
`Load weight begin. avail mem=55.96 GB`（正常 60.80）、对方 run 目录名的时间戳、
`npu-smi -t proc-mem` 里那个不属于我们的 PID。

已在 `ab_tp1.sh` 里加了一道 fail-fast：抓 `Load weight begin. avail mem=` 那一行，
低于 58 GB 就立刻中止并打印占卡的进程，**并明说这是碰撞不是回归**。

⚠ **按端口 pkill 的约定只解决两条 Claude 线之间的冲突。这台机器上的第三个用户不在任何约定里** ——
对他只能检测，不能协调。

### 7b.13 归因规则会活得比它描述的那个事实更久，然后开始凭空造时间

`attribute_kernels.py` 里只有一个 shape 是两个家族共用的
（`MatMulV2 "1,4096;128,4096"`，两边同一个 shape、同一个耗时），规则按调用次数拆：
**68 个 KDA（f_a/g_a）+ 22 个 DSA（indexer wk、kpool compress gate）= 90**。
每个家族分到 `us * k / per_step`。

**§6.4 把 KDA 的 f_a/g_a 融掉了**，从构型 E 起这一组每步只跑 22 次。
规则不知道，照旧给 KDA 记 `68/22` 份 —— **凭空多出 0.800 ms 和 68 个 kernel 的 KDA 时间**。

**工具当时把证据打在屏幕上了，只是没人做减法**：家族和 34.148 ms 对整步 33.348 ms，
家族 kernel 数 2638 对实际 launch 数 2570。两个数字都差得正好等于那一行。
构型 F 的 §8.2 我手工发现并改对了（KDA 10.434），但**没回去修工具**，
于是构型 G 又踩了一次同一个坑。

修法（`806e7f5ad7`）不是把比例调对，是**让它算不出来就报错**：拆分里允许一个家族标成
`REST` 吃余数，固定份额加余数必须等于实测到的次数，否则退出并说「模型在规则底下变了」。
全链条 A/D/E/F/G 重跑，五份 profile 的家族和现在同时对上整步和 launch 数，
构型 A 与报告已引用的数字逐位一致（42.823 ms / 3578 kernel / KDA 16.824）。

**一般化的那条**：任何「按写死的比例拆分」的归因规则，在优化改变了被拆分的对象之后
都会开始撒谎，而且**是往「你刚优化的那个家族还很胖」的方向撒谎** ——
正好是最容易被当成真的方向。**加总校验（§8.2）不是形式，它就是抓这个的。**

### 7b.14 一个从单次观测提拔上来的常数，让报告自相矛盾了一整轮

`kernel_roofline.py` 里有 `LAUNCH_US = 13.5`，注释写着「每个 kernel 约 13.5 µs 的固定
launch 开销」，§4 也照抄了这句。**这个数来自本报告 §6.3 里一次具体观测** ——
那个 `ConcatD` 实测每次 13.5 µs 搬 8 KB。**一个 kernel 的实测值被提拔成了全机的物理常数。**

**它和同一页上另一个数字直接打架，而且我一直没做这个除法**：

```
§8.4：launch 主导 10.179 ms / 2177 个 kernel = 4.7 µs/kernel
§4  ：每 kernel 约 13.5 µs 固定 launch 开销
若属实：2177 × 13.5 = 29.4 ms，而整步只有 33.348 ms
```

**两句话不可能同时成立**，而且一句除法就能看出来。

实测的真值（本 profile 自己就有，我从没去看过最小的那几行）：**最小的 kernel 是
1.3–1.5 µs**（标量形状的 `Cast` / `Mul` / `BroadcastTo`），`HcPost` 6.30 µs，
1.5 MiB 的纯流式读约 8.4 µs（`tools/hcpre_microbench.py` 独立实测）。

**修法**：常数改名 `SMALL_OP_US` 并把理由写实 —— 它是 `<16 MB` 组内部
「小算子 / 真在干活」的经验分界（27 µs），**不是 launch 地板的两倍**。
分桶数字一个没变（那条界本来就只在 <16 MB 组内部划分）。

⚠ **对结论的影响**：六条优化的收益全是**实测**的，不是从 13.5 µs 推出来的，所以**结论不受影响**。
受影响的是**解释**：「launch 主导」这个标签的意思是**「它的时间不由它的字节解释」**，
**不是「其中 13.5 µs 是 launch 开销」**。前者支持「减少总固定成本」这个杠杆，后者会让人
去算一笔根本不存在的账。

**和 §7b.13 是同一个病**：§7b.13 是一条写死的拆分比例活得比事实久，这条是一个单点观测被当成常数。
**两条的共同解药也一样 —— 加总校验。** 除一下就露馅了。

## 8. 还没做的，以及两条**没验证**的

### 8.1 ⚠ 两条改动的一部分路径本部署验证不了

**这不是「没时间」，是这个部署没有能触发它们的负载。** 写在这里而不只写在 commit 里，
因为 `git log` 会把它埋掉，而下一个人是从 PLAN / RESUME 进来的。

| 路径 | 为什么验不了 | 什么条件下能验 |
|---|---|---|
| **MTP / speculative 快照路径**（`ascend_kda_backend.py` 约 700–780） | 这个部署不跑 MTP / spec decode，没有负载能走到它。两处改动在代码里都标了 `UNVERIFIED` | 起一个带 MTP / spec decode 的配置；或构造一个直接驱动快照路径的单层 harness。**已交给接手 MTP 的 session**（`glm53_longctx`），他们的 D② 负载正好能触发 |
| **mask-track 散写分支**（`has_mamba_track_mask`） | `check_kda` 用 `enable_mamba_extra_buffer=False` 建池，且从不设 `mamba_track_mask`，**这条分支从未执行过** | 让 harness 构造带 track mask 的池 |

⚠ mask-track 那条**转而验了它依赖的不变量**：算子的 state 回写与 `x[L-3:L]` 在 L=64/256/8192
下**逐位相同**，而那正是散写要写的东西。**构造不出那条路径时，验它必须成立的性质，
比不验强，也比假装验过诚实** —— 但它不是端到端测试，不要当成端到端测试引用。

⚠ 另有一条**与本线改动无关、但被这次重构顺带发现的实测缺陷**，见 §7b.8：
`causal_conv1d_fn_npu`（**正在被删掉的那条路**）在一批内混合 `has_initial_state` 时写坏 conv state。
PLAN P6.2 已就此记了一条禁令：**修好之前不要打开 radix cache 跑精度评测**。

#### ⚠ 但 conv 池翻面这个方向本身，上游已经有先例 [读源码，本轮独立复核过]

§6.5 把 KDA 的 conv 池从 `(slots, conv_dim, kernel_size)` 翻成
`(slots, kernel_size, conv_dim)`。这**不是本轮发明的约定** ——
`mem_cache/memory_pool.py:795-799`，speculative 的 dense conv 中间缓冲
**早就在按同一个方向翻**：

```python
(conv_shape[1], conv_shape[0]) if _is_npu and cache_params.is_kda else conv_shape
```

而同文件 `:127` 的 `conv_window_dedup_enabled` 判据是
`not is_npu and not is_cpu and not is_kda and topk <= 1` ——
**GLM 在 NPU 上被 `is_npu` 和 `is_kda` 两条各自否掉**，所以 dedup 那条路走不到，只走 dense。
它的 docstring 把理由写明了：「KDA transposes the window before conv so the overlapping
`as_strided` layout would corrupt stores」。

**这条降低但没有消除 §8.1 那个风险**：两边方向一致，意味着快照路径如果对不上会**响亮地错**
（形状不匹配）而不是静默错位。但**方向一致 ≠ 已经验过**，那两处 `UNVERIFIED` 仍然是 UNVERIFIED。

⚠ 另有三处 `device="cuda"` 写死（`memory_pool.py:473 / 745 / 813`）。
`:473` 只在 dedup 路径、NPU 不可达；**`:745` 和 `:813` 在 NPU 上可达**，
一开 MTP 就会撞。本线不动它们（没有负载能验），已交给 `glm53_longctx`。

### 8.2 剩下的目标（构型 G = 33.348 ms 上重算，并做了加总校验）

**「簿记」在这里有一个明确的定义**（上一版没写下来，见 §7b.13 的教训）：
**一个层族里非 cube 类（不是 `AI_CORE` / `MIX_AIC`）、且不是该层类型专用厂商算子的部分。**
按这个定义重算，五项之和 **9.304 ms = 整步的 27.9%**，未超 100%，无重复计数；
层族分解配平到小数点后三位
（MoE 11.091 + KDA 10.401 + DSA 5.306 + mHC 3.736 + 其他 2.246 + dense 0.568 = **33.348**，
kernel 数同时配平：672 + 272 + 913 + 270 + 377 + 66 = **2570** = 实际 launch 数）。

| | ms/step | 构型 F | 判断 |
|---|---|---|---|
| **`HcPre`** | **2.969** | 2.969 | ⚠ **现在最大的单项**，但**不是「算子写得差」** —— 33.0 µs 里约 **25.6 µs 是一个 `(1,16384)×(16384,24)` fp32 GEMV**，而同 shape 的 stock `aclnnMatmul` **一样是 25–26 µs**。**慢的是这个 shape，不是这个算子**。已刻画完毕并给出厂商问法，见 **§8.3**。**本线改不动** |
| **DSA 的簿记** | **2.286** | 3.252 | §6.6 砍掉 966 µs（每层 84 → 70 个非 cube kernel）。**剩下的大部分是每层必须做的活**：ring 槽维护、选中 token 的散读散写。**按目标分工归 glm53_graph_perf**（P6.7 / P6.10 / P6.11） |
| **MoE 的路由簿记** | **2.170** | 2.171 | ⚠ **拆开之后大部分动不了**：`MoeInitRoutingV3` 594 + `DequantSwigluQuant` 255 + `MoeFinalizeRoutingV2` 247 + `MoeGatingTopK` 188 + `Cast[1,8]` 59 = **1343 µs（62%）是本质的厂商算子**；routed 预 clamp 278 **没有融合路径**（§7.2）；shared clamp+swiglu 282 **融了反而更慢**（§7.7）；`Add`+`Muls` 160 **低于噪声**（§7.3）。**真正可动的约 108 µs** |
| **shared expert 两个量化 GEMM** | **1.678** | 1.666 | 对地板 1.73× / launch 主导。搬 8–16 MB 属固定成本主导区，约 0.8 ms 可动 |
| KDA 的簿记 | **0.201** | 0.201 | §6.5 之后基本清空（曾是 3.383）—— 那 3.2 ms 几乎全是被拆开的卷积 |

⚠ **「X 占了 N 毫秒」是观测，「X 里有 M 毫秒不必发生」才是待办。**
上一版把 MoE 路由簿记 2.171 ms 整个写成目标，拆开之后真正可动的只有 108 µs。
DSA 那 2.286 ms 同理 —— 别再当成 2.286 ms 的空间。

⚠ **加总校验这个动作要保留，而且要连 kernel 数一起校验。** 只对时间不对个数的话，
§7b.13 那个凭空造出 0.8 ms + 68 个 kernel 的归因 bug 会再躲过一轮。
另一条线的 P6.7 就是同一笔账从「函数」和「阶段」两个角度各量过一次，
在待办里变成两条独立条目。**合并的判据是数字对得上，不是名字像。**

### 8.3 `HcPre` 已刻画完毕 —— 结论是「不是我们的」，但现在有了具体问法

工具 `tools/hcpre_microbench.py`（不起服务、不需要 checkpoint、只依赖 torch_npu + custom_ops，
约 90 s 跑完），全量输出 `data/hcpre_microbench.txt`。以下全部 **[实测]**，
30 次调用 p50、至少 10 次预热，run-to-run 抖动 ±1~2 µs。

**① 33 µs 是什么** —— 用同包里的拆分算子把内部拆开（三者数值与融合版一致）：

| | µs |
|---|---|
| `HcPreInvRms` | 3.74 |
| **stock matmul `(1,16384)×(16384,24)` fp32** | **25.6** |
| `HcPreSinkhorn` | 10.40 |
| 三者之和（未融合） | 39.5 |
| **`HcPre`（融合）** | **28.2** |

**融合本身已经省了 11.3 µs。剩下的 25 µs 就是那个 GEMV，而 stock matmul 同 shape 一样慢。**
PipeUtilization 拆 M=1：cube 标量 10.6 + vector 标量 5.7，MAC 只有 **1.35**，
`aic_scalar_ratio 0.62`、`aiv_vec_ratio 0.011` —— **标量/控制受限，不是带宽也不是 MAC。**

**② 能不能少调几次** —— **不能** [读源码 `communicator_mhc.py:105-116`]：
两个站点**串行依赖**（`attn_to_mlp` 先 `hc_post` 再 `hc_ffn_pre`），层与层也是链。
**层内合不了，跨层也合不了，除非改模型。** 上限还是量了：
90 个独立 GEMV 2.252 ms，一次 90 组的 `bmm` 0.247 ms → **9.1×**。
**那 2.97 ms 里约 90% 是「不能批」的代价，在当前 dataflow 下一分钱拿不到。**

**③ 有没有别的算子** —— 有，**但慢 3.1×** [实测]。CANN 自带
`aclnnMhcPreSinkhorn`（有 ascend910_93 二进制，没有 torch 绑定，用 ctypes 调通，
数值与 `npu_hc_pre` 一致到 5e-7）：M=1 时 **86.7 µs** 对 27.5 µs，还要 304 MiB workspace。
`npu_mhc_pre` / `npu_mhc_sinkhorn` 有 torch 绑定但**这套 CANN 上没有对应 kernel 二进制**，调用一律 561103。
**厂商的自研定制算子已经是快的那个。**

**④ 敏感度** [实测]：`hidden` **只接受 4096 / 7168**，`hc_mult` **只接受 4**，
`hc_fn` 必须 fp32、`x` 必须 bf16（换 bf16 也没用，同 shape stock bf16 matmul 28.3 µs 反而更慢）；
90 个不同 `fn`（135 MiB）对同一个 `fn` 是 28.32 vs 28.10 µs，**不是 cache 问题**；
expert 数 / `n_hash_layers` **根本不是它的输入**。
`hc_sinkhorn_iters` 0.224 µs/迭代，20 次 = 4.5 µs = **16%** ——
**独立复现了 §7.5**（服务侧 30.34 → 26.42 = 3.9 µs），两条线交叉验证通过。

#### 给厂商的一句话

> `HcPre` 的成本 = 一个 `(1,16384)×(16384,24)` fp32 GEMV。这个 shape 在 A3 上跑到
> **60 GB/s（峰值的 5%）**，而且**不是这个算子的问题** —— 同 shape 的 stock `aclnnMatmul`
> 一样是 25–26 µs，而同样 1.5 MiB 的纯流式读只要 8.4 µs。算子内部 62% 的 AI Core 时间在标量单元，
> MAC 只干了 1.35 µs 的活，向量单元利用率 1.1%。**要么优化「N 极小的 fp32 GEMV」这一类 tiling，
> 要么给这个算子一个 grouped/batched 形态（批 90 组实测能快 9.1×）。sinkhorn 只值 16%，别往那儿查。**

### 8.4 优化空间的形状没变

构型 G（`tools/kernel_roofline.py`，全量在 `data/kernel_roofline_cfgG.txt`）：

| | ms/step | 占比 | kernel/step | 起点（构型 A） | 能不能动 |
|---|---|---|---|---|---|
| 带宽受限（≥16 MB） | **17.363** | **52.1%** | 204 | 20.022 / 215 | ❌ 已贴墙（MoE gmm 0.99× 地板、KDA qkv 1.07×、lm_head 0.91×） |
| **launch 主导（<16 MB）** | **10.179** | **30.5%** | **2177** | 16.662 / 3163 | ✅ **空间一直在这里** |
| compute / 固定成本 | 4.539 | 13.6% | 136 | 4.899 / 147 | 部分（`HcPre` 2.969 在其中，占了大半） |
| 高于地板（≥16 MB） | 1.267 | 3.8% | 53 | 1.241 / 53 | shared expert 那两个 GEMM |

**六条优化几乎全打在 launch 主导那一栏：16.662 → 10.179 ms、3163 → 2177 个 kernel，
拿走了整步 9.475 ms 里的 6.483 ms。**

带宽那一栏也掉了 2.659 ms，但**同样不是把任何 GEMM 变快了**，它两笔都可以对上账：
删掉零 rope 的 `BroadcastTo` 本身 **0.701 ms**（11 个 kernel，正好是少掉的那 11 个），
加上 KDA qkv GEMM 的**长尾消失** **1.898 ms**（7823 → 5926 µs，而中位数只从 182.1 走到 175.2，
见 §6.1 那张分布表）。剩下 60 µs 在噪声里。
**没有一条优化让某个 kernel 算得更快 —— 拿到的全是「少发」和「不再互相拖慢」。**

**带宽那 52.1% 现在是硬地板，而且是封死的。**

唯一能动它的杠杆是「少读字节」，而字节的大头是 KDA 投影的 bf16（占 bs=1 权重流量 42%，见 §3）。
**那条路已经封闭** —— 不是因为难，是因为**模型作者在原始 FP8 发布里点名排除了全部 34 层的
KDA 投影**（§7.1）。量化它就是推翻模型作者的决定，而本部署没有任何手段能判它坏没坏。

**所以：单卡 NPU 侧的带宽账到此为止。** 剩下的只有把活搬走（offload / 多卡），
那是容量问题不是速度问题，不属于本报告。

⚠ **但「减 kernel 个数」是代理不是目标，见 §7.7**：把 shared expert 的 clamp+swiglu 融成
一个算子减掉了 45 个 kernel，**却因为融合 kernel 单次更贵而净慢 65.6 µs**。
**准确的说法是：杠杆是减少总的固定成本。** kernel 个数通常是它的好代理，
因为每个 kernel 都带固定开销 —— 但当替换进来的那个每次启动做的活不等价时，代理就失效。

## 对外页面（别新建，要更新那一个）

本报告的对外呈现在**算子缺口那一页的第二部分**：
<https://claude.ai/code/artifact/54dbfb20-667f-465d-84c1-ea7d0cc1a827>

⚠ **更新时必须把这个 URL 传给 Artifact 工具**，否则会新建一页而不是更新它。
页面的第一部分是算子清单（权威来源 `PLAN.md` §2.5 / §4），第二部分是本报告的摘要
（八～十二节）。**以仓库为准** —— 页面只放结论和判据，数据表的全量在本目录 `data/`。

⚠ 用户名下还有一页叫「GLM-5.3-Flash 单卡方案」
（`1df75bd5-2186-4fb6-a02c-d45db9cdc7f9`），**名字和这条线撞，但不是这轮的产物**，别更新错。

---

## 9. 复现

```bash
source /mnt/workspace/y00359136/work/glm53_dev/env/env.sh
cd /mnt/workspace/y00359136/work/glm53_dev/wt-int8-singlecard/docs/docs/glm53_npu_support

# 1) 裁一份 16 专家的 checkpoint（39 s，31.5 GiB，写在 / 上，不写 /mnt/workspace）
$VENV/bin/python tools/prune_experts_int8.py \
    --dst /var/tmp/glm53/GLM-5.3-Flash-W8A8-e16 --experts 16

# 2) 起 TP1 服务（die 0）
EXPERTS=16 bash $ROOT/run/launch_glm_w8a8_tp1.sh

# 3) 分阶段量 decode（prefill 与 decode 分开；先发 warmup）
$VENV/bin/python tools/bench_graph_decode.py --port 30013 \
    --concurrency 1,4,16 --decode-tokens 128 --pools short

# 4) kernel 级 profile + 按层族归属
$VENV/bin/python tools/profile_server_decode.py --port 30013 \
    --concurrency 1 --steps 20 --out /var/tmp/glm53/prof/bs1
$VENV/bin/python tools/attribute_kernels.py --profile /var/tmp/glm53/prof/bs1 --steps 20
$VENV/bin/python tools/kernel_roofline.py   --profile /var/tmp/glm53/prof/bs1 --steps 20

# 5) A/B 一个变量（自动等 die 空闲、warmup、跑 bench）
NAME=tqe2 TQE=2 bash $ROOT/run/ab_tp1.sh
```

新增工具：`tools/prune_experts_int8.py`、`tools/quantize_kda_int8.py`、
`tools/profile_server_decode.py`、`tools/attribute_kernels.py`、`tools/kernel_roofline.py`、`$ROOT/run/ab_tp1.sh`、
`$ROOT/run/launch_glm_w8a8_tp1.sh`。

**服务级 profiling 在 TP1 / bs=1 / 单 rank 下是可用的 [实测]** —— 交接文档记录的
「16 rank 全部 SIGSEGV、数据不可读」在这个规模下没有复现。采了 6 次，服务一次都没挂。
用的是 `profile_by_stage=False`、`record_shapes=False`、`activities=["CPU","GPU"]`、20 步。
（`Input Shapes` 列即便不开 `record_shapes` 也是填的，所以按 shape 归属不需要冒那个险。）
