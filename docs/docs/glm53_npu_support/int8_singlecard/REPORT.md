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

**单卡 bs=1 decode = 42.8 ms/token，权重带宽 roofline 是 17.94 ms，即 2.39× [实测]。**
时间的 39% 在 KDA、28% 在 MoE，而 **MoE 读专家权重的那两个 `GroupedMatmul` 已经贴在
带宽地板上（1.03×）** —— 注意这 1.03× 只属于那两个 kernel，不是整个 MoE 家族（家族是 1.56×，
差额几乎全是路由簿记，见 §2）。
一步跑 **3578 个 kernel**，device 时间与墙钟**逐点吻合**（42.823 ms profiler vs 42.8 ms 墙钟）
—— **图模式下 host 侧已经没有气泡可挤**。

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
  与 TP16 上「长上下文 64 并发拐点」同源（PLAN P6.7/P6.10/P6.11 的 kpool device 时间）
- 并发上限 16 不是 KV 卡的（用了 1.24 M token 里的一小半），
  是 **KDA mamba state pool**：TP1 每槽 146 MB，是 TP16 每 die 的 16 倍
- **prefill 约 1040–1090 token/s**（8192 token 一个 chunk，调度器自己的每批计时；
  墙钟法一致：16 × 3256 = 52096 token 用 48.81 s = 1067 token/s）。
  ⚠ **这个数偏乐观**：16 个专家时 prefill 的 gmm 分组比 288 专家时大得多、效率更高。

**对照 TP16 INT8（PLAN P6.15，别人的机器时间，构型不同）**：bs=1 28.9 ms/token。
单卡只慢 **1.48×**，远好于「算力少 16 倍」的朴素预期 —— 因为 bs=1 是带宽而不是算力受限。

---

## 2. bs=1 每步的时间分布 [实测]

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
**流量 <16 MB 的 kernel 由约 13.5 µs 的固定 launch 开销主导，对地板的比值没有意义**，
这些行不给倍数，只给判定。生成脚本 `tools/kernel_roofline.py`，全量输出在
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

基线链条（一次只动一个变量）：

| 构型 | device ms/step [profiler] | 墙钟 ms/token [bench] | 相对基线 |
|---|---|---|---|
| **A** e16，仓库原样 | **42.823** | **42.8** | — |
| **B** = A + §6.1 零 rope 修复 | **39.741** | **39.8** | **−7.2%** |
| **C** = A + §6.2 KDA 量化 | **37.653** | **37.7** | **−12.1%** |
| **D** = B + C | — | **35.8** | **−16.4%** |

bs=16 同样的链条：A 75.4 → B 72.9 → C 65.5 → **D 65.1 ms/token**。

**推荐落地的是 B**（只有 §6.1 那一处修复）。§6.2 的 KDA 量化**测了但不采纳**，理由见 §6.2。

两个仪器（profiler device 时间、墙钟）**逐点吻合**，这不是巧合 —— §1 说过这条链路 100% device-bound。

### 6.1 修掉 NoPE 零 rope 被静默物化 [实测，−3.08 ms/step = −7.2%]

`hardware_backend/npu/attention/ascend_backend.py` 的 `_nope_zero_rope`：
GLM 是 NoPE 模型（`qk_rope_head_dim=0`），但 `npu_sparse_flash_attention` 拒绝缺失的 rope、
零宽度的 rope 和除 64 以外的一切宽度，所以那里喂了一个全零 rope。
为了不真的分配一份「装满零的第二套 paged cache」，原实现用了 **stride-0 的 `expand`**，
并且在 docstring 里写了一条 ⚠：「算子文档说不支持非连续输入，所以这个 aliasing 是
观察到的行为，不是承诺的行为」。

**那条 ⚠ 兑现了，但不是以算错的方式，而是以静默变慢的方式。**
torch_npu 在调 aclnn 之前把输入变连续，于是这个 expand **每次调用都被真的物化**：

- `BroadcastTo`，in `[1,64,1,64]` bf16（16 KiB）→ out `[19403,64,1,64]`（**159 MiB**），
  `aiv_mte3_ratio = 0.945`（纯 store，不算不搬别的）
- 每个 DSA 层每步一次 × 11 层 = **每 token 写 1.75 GiB 的零**
- 直接成本 **0.80 ms/step**（k 侧 701 µs + q 侧 96 µs）

**最值钱的性质：它是 O(KV pool)，不是 O(batch) 也不是 O(seq_len)。**
把 KV pool 从 124 万 token 开到 152 万 token，这一项就从 0.70 → **1.34 ms/step [实测]**。
bs=1 和 bs=16 几乎一样贵（0.70 → 1.03 ms），**大 batch 摊不掉**。

**改法**：不 expand，把全零张量按 `(device, dtype, 完整 shape)` 缓存一次，所有 DSA 层共用
（`_zero_rope()`）。数值恒等 —— 零就是零。代价是一次性 159 MiB HBM（随 pool 大小缩放）。

**实测收益 −3.08 ms/step，是它自己那 0.80 ms 的 3.9 倍。**
差额的机制 **[推断]**：那 159 MiB 几乎正好是 L2 的大小（**168 MB**），
每个 DSA 层把它写一遍就把 L2 冲干净了，殃及相邻的带宽受限 kernel。
支持这个推断的是**分布形状而不是均值**：KDA qkv matmul 的**中位数只降了 2.4%（182.1→177.8 µs），
但每步总和降了 23%（7823→6022 µs）** —— 修复前有一条长尾（均值 230 µs 远高于中位 182），
修复后均值与中位数重合。34 个 KDA 层与 11 个 DSA 层交错，挨着 DSA 的那些 KDA 层在付这个钱。

**判据（给 TP16 复核用）**：profile 里 `BroadcastTo` 的
`"1,64,1,64" → "<pages>,64,1,64"` 这一组**应该完全消失**。

### 6.2 把 KDA 投影量化成 INT8 —— 测了 −12.1%，**决定不采纳**

⚠ **这是性能探针，不是可交付的量化，而且已决定本轮不做**（决定经 glm53_graph_perf
session 转达，与我在 §3 提的顾虑一致；建议由本报告读者向用户复核一次）。
没有校准、没有验精度。它回答的只是「把那 8.93 GiB 砍半值多少时间」，得到的是一个**上界**。

**为什么不采纳 —— 主要不是收益问题，是「测不出它坏」**：
KDA 是线性注意力的递归路径，权重误差会沿 SSM 状态**随序列累积**，
而不像标准注意力每步从 KV 重新读。厂商同时排除了 `A_log` / `dt_bias` / `conv1d`
——正是 HF 用 `_keep_in_fp32_modules_strict` 强制留 fp32 的那批递归参数。
⚠ **投影本身没有那个约束，所以这是怀疑，不是解释。**
但正因为是怀疑，代价不对称：**如果它成立，症状是长序列上的缓慢漂移，
而 GSM8K 那种几百 token 的题目看不见** —— 一条「基准全过」的坏优化比一条明显坏的更危险。

真要做的正确顺序（留给以后）：量化出权重 → GSM8K 每侧 2 轮（BF16 基线已有）
→ **再加一个长上下文判据**，否则测不到那个失效模式。

做法：`tools/quantize_kda_int8.py` 对 34 个 KDA 层的 `q/k/v/o_proj` 做
对称 per-output-channel min-max int8（就是这份 checkpoint 对其他所有 Linear 已经声明的方案），
并把它们从 `quantization_config.ignore` 里删掉。checkpoint 8.50 → 4.25 GiB，权重 31.50 → 28.03 GB。

| kernel | 前 | 后 | roofline |
|---|---|---|---|
| KDA qkv `[24576,4096]` | 7823 µs/step（bf16，192 MiB） | **2717 µs/step**（int8，96 MiB） | 76.8 µs，实测中位 79.7 → **1.04×** |
| KDA o_proj `[4096,8192]` | 2027 µs/step | **1192 µs/step** | 25.6 µs，实测中位 34.3 → 1.34× |

**整步 42.823 → 37.653 ms（−12.1%），墙钟 42.8 → 37.7 独立吻合。**
叠加 §6.1 之后墙钟 **35.8 ms/token（对 A −16.4%）**。

**这两个数的用途是给「量化 KDA」这件事标价，不是给它背书。**

---

## 7. 试过但没用的，以及踩到的坑

**和成功的一样有价值。**

### 7.1 只把 `q/k/v_proj` 从 `ignore` 里删掉 —— 不生效，而且看起来生效了

第一次尝试 §6.2 时，**o_proj 量化了，qkv 没有**，整层还是 bf16，墙钟看不出来。
机制 **[读源码 + 实测复现]**：sglang 把三个 KDA 投影建成一个融合的 `qkv_proj`，
而 `should_ignore_layer()` 只在 `proj_name in fused_mapping **and layer_name not in ignore**`
时才把融合名展开回 q/k/v 逐个检查。厂商的 `ignore` 列表里**直接写了融合模块名
`model.layers.{L}.self_attn.qkv_proj`**，前置条件不成立 → 不展开 → 精确匹配命中 → 整层被忽略。

**必须连融合名一起删。** 症状识别：profile 里那一项还是 `MatMulV2` + `DT_BF16;DT_BF16`。
接手的人一定会再踩一次。

### 7.2 「KDA 量化带来 1.39×」—— 我自己报错过一次，是冷启动伪像

一度量到 bs=1 从 42.8 → 30.7 ms/token。**是假的。**
`bench_graph_decode.py` 用「先跑 `max_new_tokens=1` 拿 prefill 墙钟，再跑 129 相减」的办法
分离 decode。冷服务的第一发 prefill 花了 **1.88 s** 而不是 0.39 s，
这 1.5 s **全部落在被减去的那一项里**，把 decode 显得快了 28%。

**这个相减法对冷启动是单向敏感的 —— 只会让结果偏乐观。**
修法：先发一次丢弃的 warmup 请求（已加进 `run/ab_tp1.sh`；
glm53_graph_perf 也已把 warmup 提交进共享的 `bench_graph_decode.py`，commit `44e2c3f4cb`）。
加 warmup 后真实数是 42.8 → 37.7（1.14×），**与 profiler 的 device 时间逐点吻合**。

**教训**：任何「A/B 只测了一次、而且 A 和 B 的服务新旧程度不同」的加速，先怀疑它。

### 7.3 `npu-smi` 的空闲检查写法：一个字符决定它挡不挡得住

`npu-smi info` 把「已用 MiB」字段按固定宽度对齐，所以空闲 die 打印
`2880 / 65536`，而**忙的 die 打印 `33861/ 65536`，斜杠前没有空格**。

于是 `grep -oP '\d+(?= / 65536)'` 会**恰好漏掉所有忙的 die**，
返回 15 个数而不是 16 个，列表整体前移一位，检查读到的是**另一块 die 的空闲数字**并放行 —— 
直接撞进加载权重时的 OOM。**[实测：一块 die 在 33861 MiB 时，严格写法返回 15 个数，
带 `\s*` 的写法返回 16 个]**

交接文档里的一行式是对的（用了 `\s*`）；我自己的脚本第一版写错了，已修，
并在 `run/ab_tp1.sh` 里把原因写在注释里。

### 7.4 `TASK_QUEUE_ENABLE=2` —— 在图模式下根本起不来 [实测，P6.9 就此关闭]

PLAN P6.9 记的是 eager 时代实测的 **DSA decode 1.74×，零代码改动**，
并注明「图模式下可能已被吃掉，要重测」。重测结果比「被吃掉」更干脆：**服务起不来。**

```
RuntimeError: Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture,
              please export TASK_QUEUE_ENABLE=1/0.
[ERROR] ERR00007 PTA feature not supported
```

抛在 `torch_npu/npu/graphs.py:625` 的 `capture_begin()`，
即第一个 bs 桶的捕获刚开始时（`decode_cuda_graph_runner.py` → `npu_cudagraph_backend.capture_one`）。
**这是 torch_npu 的硬约束，不是调参能绕的**，而且它**响亮地失败**，不会静默降级。

→ **图模式下 TQE 只能是 1 或 0，P6.9 在图模式下不存在。**
这与 §1 的观察一致：device 时间已经等于墙钟，host 侧本来就没有气泡可挤。

### 7.5 调低 sinkhorn 迭代数 —— 有效，但不值得（0.7%）[实测]

`hc_sinkhorn_iters` 20 → 1（改 config，其余不变）。**知识点是它确实生效了**，
不是「改了没反应」：`HcPre` 中位 30.34 → 26.42 µs、每步 2741 → 2394 µs。

但整步只从 39.741 → **39.445 ms（−0.7%）**，墙钟 39.8 → 39.3。
**19 次迭代总共只值 347 µs/step**，摊到 90 个站点是每站点每次迭代 0.04 µs。

→ **decode 下 sinkhorn 几乎免费**（与 PLAN P6.1 eager 时代的「decode 下占 0%」一致，
现在在图模式下也成立）。`HcPre` 那 2.4 ms **不是 sinkhorn**，是 mHC pre 本身的固定开销：
M=1 时 26 µs 搬 1.5 MiB，是带宽地板的 20 倍，纯固定成本主导。
**这个旋钮改数值、不改性能，不要动它。**

### 7.6 编辑正在运行的 shell 脚本

bash 按字节偏移增量读脚本文件。在 `ab_tp1.sh` 跑着的时候给它打补丁，
它接着从新文件的旧偏移读下去，把一行注释当命令执行了（`Measured: No such file or directory`），
一次 A/B 白跑。**别改正在跑的脚本。**

---

## 8. 剩下最值钱的（按实测大小排，都还没做）

构型 B 之后，42.8 → 39.8 ms/step。剩下的大头：

1. **KDA 的簿记开销 ≈ 4.2 ms/step（10%）[实测，按差额]**。
   KDA 16.82 ms 里，两个大 matmul 占 9.85、delta-rule kernel 1.13、四个小投影 1.69，
   **剩下 4.16 ms 是 Slice / ReduceSum / IndexPut / Cast / Index / ConcatD / BroadcastTo /
   Mul / ScatterUpdate / GatherV3 / Swish 这一堆小向量算子**，
   **每个 KDA 层 25 个 kernel**。其中 `"1,24576,3"` 那四组 bf16↔fp32 转换共 660 µs/step，
   来自 KDA 的 fp32 工作集绕行（PLAN 记的 conv 权重 fp32 / cache bf16 那条）。
2. **DSA 的簿记开销 ≈ 5 ms/step（12%）[实测]**。
   `SparseFlashAttention` 本身只要 29.8 µs/层、`LightningIndexer` 17.1 µs/层，
   而整个 DSA 族 6.69 ms、**每层 92 个 kernel**。
   与 BF16 时代的诊断同源：**「DSA 慢不是因为注意力慢，是因为注意力周围的簿记」**，
   在 INT8 单卡图模式下**依然成立**。P6.7 / P6.10 / P6.11 那几条仍然值钱。
3. **`Cast` 合计 1.47 ms/step（3.4%）**，一步 568 次，纯 dtype 转换。
3b. **MoE 的路由簿记 3.16 ms/step（7.4%）**，每层 14 个 kernel，地板是 0 —— 见 §2 的对账表。
4. **shared expert 的两个量化 matmul 是 launch 受限的**：
   `"1,2048;128,128,…"` 搬 8 MiB 却用 16.1 µs，是地板的 2.4×（<16 MB 的 kernel 由固定开销主导）。
   42 层 × 2 个 = 1.64 ms/step，地板只要 0.85（1.93×）。
5. **mHC 的 `HcPre` 2.4 ms/step**：M=1 时 26 µs 搬 1.5 MiB，是地板的 20 倍，
   **且 §7.5 已证明这不是 sinkhorn**，是算子本身的固定开销。

**不值得再看的**：MoE 的 `GroupedMatmul`（1.00× 地板）、`lm_head`（0.96×）、
KDA 的两个大 matmul（1.13× / 1.15×）—— 这四项合计占一步的 39%，**已经贴在带宽线上**。

⚠ 归属表里有 **4.0% 的「未归类」**（511 个每步调用次数不落在任何层族倍数上的小 kernel）。
上面的排序对这部分是保守的。

---

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
