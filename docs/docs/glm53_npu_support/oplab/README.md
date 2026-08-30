# oplab — 两个可独立运行的单层算子性能用例

给昇腾算子优化团队。覆盖 GLM-5.3-Flash 单卡 INT8 decode 里 **KDA** 和 **DSA**
两个层族的算子序列，**不起服务、不加载 checkpoint、不 import 模型代码**。

| 文件 | 内容 |
|---|---|
| `bench_kda_layer.py` | KDA（线性注意力）单层，入图，扫序列长度、扫并发槽位 |
| `bench_dsa_layer.py` | DSA（稀疏注意力）单层，入图，扫序列长度 |
| `baseline_kda.txt` / `baseline_dsa.txt` | 在本机一张空闲 A3 die 上的参考输出 |

---

## 0. 一句话结论

| 层族 | 层数 | 用例单层 p50 | ×层数 | 整网实测 | 偏差 |
|---|---|---|---|---|---|
| **KDA** | 34 | **292.0 µs** | **9.928 ms** | 10.357 ms | **−4.1%** |
| KDA（真跑 34 个不同层） | 34 | — | **9.712 ms** | 10.357 ms | −6.2% |
| **DSA**（n=512） | 11 | **453.5 µs** | **4.988 ms** | 5.143 ms | **−3.0%** |
| DSA（真跑 11 个不同层，各自的池子） | 11 | — | **5.338 ms** | 5.143 ms | **+3.8%** |

两个都在「20% 以内算正常」的范围里，而且**逐算子对得上**：shape 字符串与
`../int8_singlecard/data/kernel_attribution_cfgI.txt` 完全一致。

单层版都比整网**略便宜**（−3% ~ −4%），因为单层跑没有和一步里的 MoE / mHC / head
抢内存系统 —— `hcpre_microbench` 上是同一个形状：微基准 28–30 µs，服务里 33.0 µs。
**这不是需要「调平」的误差，是两种测量的定义差。**

而「真建 N 个不同的层串起来跑」这一档（`family`）把整网的 −4.1% / −3.0% 变成
**KDA −6.2%、DSA +3.8%** —— 两个方向相反。所以「单层 ×N 偏低是因为 cache 太热」
这个看起来很合理的解释**被自己的数据否掉了**（KDA 那边真跑 34 层反而更便宜）。
没有替它编第二个解释。

---

## 1. 怎么跑

```bash
source /mnt/workspace/y00359136/work/glm53_dev/env/env.sh
export PYTHONPATH=/mnt/workspace/y00359136/work/glm53_dev/wt-int8-singlecard/python:$PYTHONPATH
cd docs/docs/glm53_npu_support/oplab

ASCEND_RT_VISIBLE_DEVICES=0 python bench_kda_layer.py
ASCEND_RT_VISIBLE_DEVICES=0 python bench_dsa_layer.py
```

⚠⚠ **`PYTHONPATH` 必须用 `:$PYTHONPATH` 追加，不能覆盖。** `env.sh` 里 CANN 的
`set_env.sh` 往 `PYTHONPATH` 塞了 `.../cann-9.2.0/python/site-packages` 和
`.../op_impl/ai_core/tbe`。覆盖掉它，`npu_format_cast` 会在
`LazyInitAclops` 里炸出 `error code 500001` + `No module named 'tbe'`——
错误信息里没有一个字提到 `PYTHONPATH`。本用例开发时踩了这个，记在这里。

分段跑：

```bash
python bench_kda_layer.py --sections layer          # 只做回归判据
python bench_kda_layer.py --sections refs           # 只做带宽地板对照
python bench_kda_layer.py --sections family         # 真的建 34 个层（~11 GB）
python bench_dsa_layer.py --sections sweep          # 序列长度扫描
python bench_dsa_layer.py --ref-seq 32768 --sections layer
```

显存：KDA `layer/sweep/refs` < 2 GB，`family` ~11 GB；
DSA `layer/sweep/refs` ~2.2 GB（KV 池按部署的 1.25 M token 全量开），`family` ~19 GB。

**跑之前先 `npu-smi info` 看你那张 die 是不是空的**（见 §7 第 9 条）。

---

## 2. 这两个用例 import 了什么

只依赖 `torch` / `torch_npu` / `sgl_kernel_npu`，**外加几个 sglang 树里的
kernel 函数**（不是模型代码）。如实列出：

| 用例 | 从 sglang 里 import 的 | 是什么 |
|---|---|---|
| KDA | `sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update` | Triton kernel + 它的 launcher |
| KDA | `sglang.kernels.ops.attention.fla.fused_norm_gate.layer_norm_gated_fwd` | Triton kernel + 它的 launcher |
| DSA | `...hardware_backend.npu.attention.kpool_indexer_npu.compress_pool_bf16` | 纯 torch，pool 压缩 |
| DSA | `...hardware_backend.npu.attention.kpool_indexer_npu.hadamard_transform_npu` | 纯 torch，fp32 Hadamard |
| DSA | `...layers.attention.dsa.kpool_fp8_index.expand_pooled_groups_to_topk` | 纯 torch，pool→token 展开 |
| DSA | `...layers.attention.dsa.kpool_fp8_index.append_kpool_tail_to_topk` | Triton kernel |

**没有 import 的**：任何 model 文件、attention backend 类、scheduler、memory pool、
ForwardBatch、ServerArgs。DSA 用例里 `kpool_decode_update_index_cache` 那段
（ring 更新 + 压缩写 cache）是**照抄成本地代码**的，不是调用 —— 因为原函数挂在
memory pool 类上。抄的来源写在代码注释里。

如果算子团队想彻底断开 sglang：上面四个纯 torch 函数不到 60 行，可以直接内联；
两个 Triton kernel 就得连 `.py` 一起拷。

---

## 3. KDA 用例覆盖了什么

**34 层，每层 8 个算子（见 §5.1 的更正：其实是 9 个），合计 272 个 kernel / 10.357 ms。**

序列（`glm5_next.py: Glm5NextLinearAttention.forward`，融合投影那条腿）：

```
x [1,4096] bf16
 1 fused_qkvbfg_a_proj   W[24896,4096] bf16        MatMulV2 "1,4096;24896,4096"
   split → qkv[1,24576] | beta[1,64] | f_a,g_a[1,256]
 2 fused_fg_b_proj       bmm [2,1,128]×[2,8192,128] BatchMatMulV2
   → forget_gate[1,8192], norm_gate[1,8192]
 3 clamp(cache_indices, min=0)                     ClipByValueV2 "1;;"
 4 torch.ops.npu.causal_conv1d(run_mode=1)         causal_conv1d_4 (+ 内部 2 个 Cast)
   x[1,24576], w[4,24576] bf16, state[17,3,24576] bf16 window-major
 5 fused_sigmoid_gating_delta_rule_update(is_kda=True)
   q,k,v[1,1,64,128] · a[1,8192] · b[1,1,64] · ssm[17,64,128,128] fp32
 6 FusedRMSNormGated(128, "sigmoid")               layer_norm_gated_fwd_kernel
 7 o_proj                W[4096,8192] bf16         MatMulV2 "1,8192;4096,8192"
```

结构常数全部来自 `config.json`：hidden 4096，`linear_attn_config` 的
`num_heads 64 / head_dim 128 / short_conv_kernel_size 4 / gate_lower_bound -5.0`，
`kda_layers` 34 个。融合投影 24896 = 24576 (q|k|v) + 64 (beta) + 128 (f_a) + 128 (g_a)。

**KDA 的投影是 bf16 不是 INT8** —— W8A8 checkpoint 的 `ignore` 里列了全部八个子模块
加融合模块名，所以走 `MatMulV2` 而不是 `QuantBatchMatmulV3`。这是作者的决定，
不是漏掉（REPORT.md §6.4）。用例照此建权重。

---

## 4. DSA 用例覆盖了什么

**11 层，每层 81 个 kernel，合计 891 / 5.143 ms。** 用例跑出 93 个（差异见 §5.2）。

```
A 隐向量  DynamicQuant → QuantBatchMatmulV3 "1,4096;64,256,16,32;2048;1"
          split 1536 q_lora | 512 kv_lora；RmsNorm(1536)、RmsNorm(512)
          q_b_proj  QuantBatchMatmulV3 "1,1536;512,96,16,32;16384;1"
          W^UK 吸收  BatchMatMulV2 "64,1,256;64,512,256" → q 进 512 维隐空间
B 索引器  wq_b 1536→4096   MatMulV2 "1,1536;4096,1536"
          wk   4096→128    MatMulV2 "1,4096;128,4096"  + LayerNormV3(128)
          gate 4096→128    MatMulV2 "1,4096;128,4096"   ← 两个同 shape，n=22/step
          weights_proj 4096→32 fp32  MatMulV2 "1,4096;32,4096"
          fp32 Hadamard-128  MatMulV2 "32,128;128,128" / "1,128;128,128"
          4 token 压成 1 个 pool     ReduceMax/ReduceSum "1,4,128;1"
          ring + index cache 写      ScatterNdUpdate ×3
          npu_lightning_indexer(sparse_count=512) → 512 个 pool id
          512 pool → 2048 token      Add "1,512,1;4" + BroadcastTo "1,512,1;3"
          + 3 列 tail                → topk_indices [1,2051] int32
C 注意力  kv_buffer[loc] = k_nope    IndexPutV2 "1246656,1,512;..."
          npu_sparse_flash_attention(sparse_mode=3, attention_mode=2,
              layout TND / PA_BSND, sparse_block_size=1)
D 出口    W^UV 吸收  torch.ops.npu.batch_matmul_transpose → batch_matmul_transpose_0
          o_proj     QuantBatchMatmulV3 "1,16384;128,1024,16,32;4096;1"
```

池子按部署全量开：KV `[1246656,1,512]` bf16（PA_BSND 视图 `[19479,64,1,512]`），
index-K `[19480,64,1,128]` bf16，两者差一页是有意的。

---

## 5. 已知的对不上的地方（如实记，没有调参去凑）

### 5.1 KDA 家族其实是每层 9 个 kernel，不是 8 个

用例每层稳定跑出 **9** 个 kernel，比 cfgI 的 KDA 段多一个
`Cast "1"`（int32，1 个元素，1.2 µs）。

不是用例多做了事：**`torch.ops.npu.causal_conv1d` 自己会发两个 Cast** ——
一个是 `query_start_loc`（2 个元素 → `Cast "2"`），一个是 `cache_indices`
（1 个元素 → `Cast "1"`）。独立探针验过，跟外面加不加 `clamp` 无关。

那 cfgI 里它去哪了？在 **`--- unclassified ---` 段**：

```
84.7 us/step  n=59  1.4 us  Cast  AI_VECTOR_ "1"
```

`attribute_kernels.py` 是**按调用次数归属**的（34 次 → KDA）。这个 shape 被 KDA 的
34 次和别处的 25 次混在一起变成 59，于是整组落进 unclassified。

**结论：KDA 的真实开销是 10.357 + 34×1.4 ≈ 10.405 ms，不是 10.357。**
差 0.05 ms，不影响任何结论，但**基于计数的归属工具会漏掉与别人共享 shape 的算子**，
这条对以后读 attribution 表的人有用。

### 5.2 DSA 用例是 93 个 kernel，cfgI 是 81 个

多出来的 12 个，逐个有交代，没有一个是「多算了一遍」：

| 多出来的 | 原因 |
|---|---|
| `Range ";;"` ×1 | `expand_pooled_groups_to_topk` 每次现建 `arange(4)`。仓库后来把这个 arange 提出去了（commit `c69883df97`「two aranges per DSA layer that are the same arange every time」），cfgI 是提出去之后的树 |
| `FloorMod "1;1"` ×4、`BroadcastTo ";1"` ×4 等 | 本地抄写 `kpool_decode_update_index_cache` 时没有复用它的 `_decode_arange` 缓存 |
| `Index "1,16384;..."` 而不是 `"1,512;..."` | 用例默认把 block table 按 1M 上下文开（`--seq-lens` 最大 1048576 → 16384 页）；cfgI 是 `--context-length 32768` → 512 页。**同一个算子，宽度不同** |

**这 12 个合计约 20 µs/层**，也就是说用例的 453.5 µs 里有 ~4% 是它自己的簿记开销；
扣掉之后是 ~433 µs，对 467.6 的偏差从 −3.0% 变成 −7.4%。两个数都在容忍范围内，
**这里给出两个而不是挑一个。**

### 5.3 ⚠ 最重要的一条：cfgI 的 DSA 数字是**短上下文**数字

`tools/profile_server_decode.py` 的默认是 `--prompt-tokens 13`。cfgI 那份
attribution 是在 13 token 提示 + 若干解码 token 上采的，**序列长度是几百，不是 32k**。

所以 `DSA = 5.143 ms/step` 这个数**只在短上下文成立**。用例扫出来：

| n | µs/层 | ×11 (ms) | 相对 n=128 |
|---|---|---|---|
| 128 | 436.3 | 4.799 | +0.0% |
| 512 | 455.7 | 5.013 | +4.4% |
| 1 024 | 471.1 | 5.182 | +8.0% |
| 4 096 | 510.6 | 5.616 | +17.0% |
| 32 768 | 516.4 | 5.680 | +18.4% |
| 131 072 | 524.0 | 5.763 | +20.1% |
| 1 048 576 | 590.8 | 6.499 | **+35.4%** |

**引用 5.143 ms 的时候必须带上「短上下文」这个限定。** 在 32k（服务默认的
`--context-length`）上，同一份 DSA 是 5.680 ms（+10%）；在 1M 上是 6.499 ms（+26%）。

### 5.4 `SparseFlashAttention` 对 cfgI 是 1.39×（n=512 时 45.1 vs 32.5）

用例 n=512 时 45.1 µs，cfgI 32.5 µs。看 §5.3 的曲线，SFA 在 n=128 是 26.4 µs，
n=512 是 45.4 —— **cfgI 的 32.5 落在 n≈200~300 之间**，与「13 token 提示 +
若干解码 token」完全一致。
所以这不是用例算错了，是两边的 n 不同。**这是本轮最强的一条交叉验证**：
一个我们本来不知道的参数（采数时的序列长度）被算子时间本身反解出来了。

### 5.5 几个比 cfgI 便宜的算子

`BatchMatMulV2 "64,1,256;64,512,256"` 0.71×、`batch_matmul_transpose_0` 0.55×、
`MatMulV2 "1,1536;4096,1536"` 0.66×、`ClipByValueV2 "1;;"` 0.40×。
**没有去解释它们**：单层跑的内存系统状态跟整网不同，而这几个都是小算子，
整网里更容易被邻居拖慢。方向一致（全部偏便宜），量级也一致，所以按 §0 的
「测量定义差」处理，不当成算子差异。

---

## 6. 看到什么算正常 / 哪些算子不用优化 / 哪些是真候选

### 6.1 正常范围

* **KDA**：`layer` 段 285–300 µs/层，×34 = 9.7–10.2 ms。
  `MatMulV2 "1,4096;24896,4096"` 应该在 **164–172 µs**。
* **DSA**（`--ref-seq 512`）：445–460 µs/层，×11 = 4.9–5.1 ms。
* 两个用例的 shape 字符串必须和 cfgI **逐字相同**。不同就是用例建错了，
  不是机器慢了 —— 特别注意 DSA 那三个 `QuantBatchMatmulV3` 必须显示成
  `128,1024,16,32` 这种四维 NZ 形状（见 §7 第 10 条）。
* **p90 / p50 < 1.1**。分布一散就是 die 上有别人。

### 6.2 已经贴在带宽地板上，不用看了

`refs` 段用四个对照给出判决，而不是只给一个比值：

| 算子 | 权重 | 冷跑 p50 | 1.25 TB/s 地板 | 1.40 TB/s 地板 | 比值 |
|---|---|---|---|---|---|
| KDA `fused_qkvbfg_a_proj` `[1,4096]×[24896,4096]` | 194.5 MiB | 146.1 µs | 163.2 µs | 145.7 µs | **1.00×** |
| KDA `o_proj` `[1,8192]×[4096,8192]` | 64.0 MiB | 55.8 µs | 53.7 µs | 47.9 µs | 1.16× |
| DSA `o_proj` INT8 `16384→4096` | 64.0 MiB | 55.9 µs | 53.7 µs | — | **1.04×** |

（1.40 TB/s 是 GEMM 在本机实测能到的读带宽，见 §7.12；1.25 是 REPORT.md 沿用的
保守值，对 reduce 类算子合适，对 GEMM 偏低。）

**KDA 那两个 GEMM 加起来是 KDA 的 76%、整个 decode step 的 25%。它们已经比
「把同样的字节纯读一遍」还快**：194.5 MiB 的 `w.sum()` 要 **167.7 µs**，
同样字节的 GEMM 只要 **146.1 µs**。算子团队在这上面拿不到东西。唯一的杠杆是
**少读字节**（换更小的 dtype）或**少调用**，两者都是模型层面的改动。

DSA 侧同理：三个 `QuantBatchMatmulV3` 都在 0.93–1.12× cfgI，`o_proj` 冷跑
1.04× 地板，**INT8 NZ 已经是快路径**。另外两个大池子读也贴墙：
KV 池 1.19 GiB 读 1100.5 µs（1.08× 地板）、index 池 0.30 GiB 读 269.8 µs（1.06×）。

### 6.3 真正值得看的

按「不是带宽地板 + 绝对时间够大」排：

1. **`fused_sigmoid_gating_delta_rule`（KDA，34×34.6 µs = 1.18 ms/step）**。
   它读的是 `[64,128,128]` fp32 状态 = 4.19 MiB，地板 3.4 µs。**34.3 µs 是地板的
   10×。** 这是本轮 KDA 侧最大的非带宽项。Triton kernel，`grid=(1,4,64)`，
   `num_warps=1`。
2. **`SparseFlashAttention` 饱和后的 ~99 µs**（DSA，n≥4k，11×99 = 1.09 ms/step）。
   它读 2048 个 token × 512 维 bf16 = **2 MiB**。用例专门测了同样字节的
   gather 对照（`refs` 段的 `KV pool read 2048 tokens`）：**6.95 µs**。
   **也就是说 SFA 有 ~92 µs 不是在读 KV。14× 于同字节的 gather。**
   这是本轮 DSA 侧最大的非带宽项，而且它在 n≥4k 之后**恒定**，
   所以省下来的每一微秒在所有上下文长度上都算数。
3. **`LightningIndexer`（DSA）**。短上下文 15.4 µs，1M 上 81.1 µs。
   **它是整个 DSA 唯一的 O(n) 项**（见 §6.4），长上下文那条线的斜率全在这里。
   ⚠ 但它在 1M 上**已经接近带宽地板**：262144 个 pool × 128 × bf16 = 64 MiB，
   1.25 TB/s 地板 53.7 µs，实测 81.1 = **1.51×**。所以这里的杠杆不是 kernel
   调优，而是**少读字节**：index cache 换 fp8（现在是 bf16，仓库里有
   `f06266470a npu: store the DSA index-K cache as bf16` 这条反向改动的历史）
   或者把 `index_kpool` 从 4 调大。两者都是**模型/配置**决定，不是算子决定。
4. **`causal_conv1d_4`（KDA，11.9–15.9 µs，见 §7.13）**。读 conv 窗口 2.39 MiB → 地板 2.0 µs，
   但真正被摸到的只有 1 个 slot 的 [3,24576] = 144 KiB。**接近纯固定开销。**
5. **DSA 的 ~48 个小算子合计 114.6 µs/层 = 1.26 ms/step（占 DSA 的 25%）**，
   单个都在 1–8 µs。**这一堆是「总固定成本」问题，不是带宽问题** ——
   参见 §7 第 6 条：减少 kernel 个数是代理不是目标。

### 6.4 两个层族对序列长度的依赖完全不同 —— 这是这对用例最大的价值

**KDA 是 O(1)，而且是结构性的 O(1)。** 用例先证明再测量：
`sweep` 段打印每个长度下所有输入张量的 shape 指纹，四档**逐字节相同**
（KDA 的 decode 状态是固定的 `[64,128,128]` fp32 + `[3,24576]` conv 窗口，
序列长度不进入任何 shape、stride 或循环边界）。然后照样测：

| n | 1 024 | 4 096 | 32 768 | 131 072 |
|---|---|---|---|---|
| µs/层 | 293.3 | 293.5 | 297.9 | 300.0 |
| ×34 (ms) | 9.972 | 9.977 | 10.130 | 10.201 |

最大 +2.3%，**这就是本机的噪声底**，不是长度效应。

> ⚠ 只测不证是不够的：四个完全一样的负载测出来一样，本身什么也没证明。
> 所以用例把「什么都没变」这件事**打印出来**。

**DSA 是两段：注意力本体被 topk 封顶，索引器 O(n)。** 用例把这两个机制分开了：

| n | 128 | 512 | 1 024 | 4 096 | 32 768 | 131 072 | 1 048 576 |
|---|---|---|---|---|---|---|---|
| `SparseFlashAttention` | 26.4 | 45.4 | 59.8 | 99.1 | 99.1 | 99.0 | **100.1** |
| `LightningIndexer` | 15.4 | 16.6 | 17.2 | 17.9 | 23.5 | 31.3 | **81.1** |

* SFA 一路涨到 n≈4k **就再也不动了** —— `index_topk=2048` 的封顶，实测到了。
* LightningIndexer 从 32k 到 1M 涨 3.2×，n 涨 32×。**给 n/4 个 pool 打分。**

n ≥ 4096（注意力已饱和）的最小二乘：**DSA 家族 ≈ 5.640 + 8.22e-7 · n ms/step**。
多卡 TP8 那条线拟的是**整步** `27.3 + 5.4e-6 · n`。两个不是同一个量
（一个是 TP1 的 DSA 家族，一个是 TP8 的整步），**它们的比值不能直接当证据**。

⚠ 还有一条**没解释掉的**：`BatchMatMulV2`、`batch_matmul_transpose_0`、
`MatMulV2 "1,1536;4096,1536"` 这三个 shape **不依赖 n**，却在 n=1M 时各涨了
3–5 µs（21.4→25.1、10.8→15.8、12.9→15.4）。最可能是索引器扫 1M 个 pool 把 L2
冲干净了，但**没有直接量过 L2 命中率，所以这是推断，不是结论**。

### 6.5 并发（不是长度）才是 KDA 的另一个轴

ssm 状态是 **4.19 MiB / 槽 / 层**，TP1 下 16 槽就是 2.34 GB。
`slots` 段扫 2/17/65/129 槽（8 MiB → 516 MiB 每层）：

| slots | 2 | 17 | 65 | 129 |
|---|---|---|---|---|
| µs/层 | 291.1 | 291.1 | 290.2 | 294.8 |

**平的。** 也就是说 REPORT.md §6.1 那条「L2 冲刷」机制在**单层隔离测量里复现不出来**
——池子再大，本层只摸 1 个槽。⚠ 这不证伪 §6.1（那条是整网 34 层交替时的现象），
但它说明**光把池子开大不会让 KDA 变慢**，变慢需要真的有别的东西在冲 L2。

`family` 段就是去测这个的：真建 34 个**不同的**层串起来跑一张图。结果
**9.712 ms，比「单层 ×34」的 9.928 还便宜 2.2%** —— 所以
**「单层 ×34 偏低是因为 cache 太热」这个假设被证伪了。** 剩下的 −6% 只能归给
与一步里其它层族的共存，跟 `hcpre_microbench` 的 28–30 vs 33.0 同源。

---

## 7. 踩坑清单

前 8 条来自本项目此前的实测（都写在 `int8_singlecard/REPORT.md` 里），
后 4 条是做这两个用例时新踩的。

1. **`tensor.is_cuda` 在这台机器上返回 `True`。** `torch_npu.contrib.transfer_to_npu`
   补的是属性本身，**连 import 之前建的张量也变**。判平台用 `device.type`，
   永远不要用 `is_cuda`。
   ⚠ `bench_dsa_layer.py` **必须** import `transfer_to_npu`（见第 10 条），
   所以这个用例里 `is_cuda` 是坏的。KDA 用例没 import，是好的。**同一个 repo 里
   两个文件行为不同——这正是不能靠 `is_cuda` 的理由。**

2. **「每 kernel 13.5 µs 固定 launch 开销」是错的。** 这台机器 device 侧单 kernel
   下限实测 **1.3~1.5 µs**（标量 Cast/Mul），`HcPost` 6.3 µs。13.5 是一次具体
   观测被误当成机器常数。本用例的 `refs` 段每次都重新测这个下限
   （`trivial torch.add on 1 element`，实测 1.4–1.7 µs），**因为它跟构型走，
   不是常数**。

3. **本后端上贪心输出不是 batch 不变的**，而且同一进程、同一宽度、重复跑都可能不同
   ——不确定的是调度器的组批。**batch > 1 上不存在可复现基线**，别用「输出逐位相同」
   当判据；宽度 1 是稳定的。（这两个用例不做精度判据，只做性能。）

4. **`causal_conv1d` 的 AOT 算子要求输入 contiguous，且响亮报错**
   （`x must be contiguous`）；但同族的 `causal_conv1d_update` 的 layout 约束是
   **静默**的。本项目已知四个算子约束里**三个是静默的**。
   在 KDA 用例里 `qkv` 是 24896 宽输出的**末维前缀切片**：bs=1 时恰好连续，
   bs≥2 不连续 —— 所以 `.contiguous()` 那行不是装饰。

5. **`custom::npu_dequant_swiglu_clamp_quant` 的 `clamp_limit` 和 `swiglu_mode`
   参数被静默忽略** —— 实测取 0/1/2/3 结果完全一样，且与「完全不 clamp」逐位相同。
   **要验这个算子必须放大输入让 clamp 真的咬到。**（不在这两个用例的范围内，
   但同一批算子里，列出来提醒。）

6. **`npu_clipped_swiglu` 融合版逐位相同、少 45 个 kernel、却净慢 65.6 µs。**
   「减少 kernel 个数」是**代理不是目标**；杠杆是减少**总固定成本**，
   当替换进来的 kernel 单次做的活不等价时代理就失效。看 §6.3 第 5 条的时候
   请带着这条。

7. **`torch.nn.functional.pad` 在这里降解成 `MemSet` + `PadV3`** —— 同样的 launch 数、
   3.3× device 时间。

8. **`HcPre` 类的算子跨服务重启会差 10%**（同一 shape 同一次数，33.0 vs 29.9 µs）。
   **单次测量的噪声底是几百微秒，别拿单次结果下结论。** 这两个用例默认
   30 次取 p50 并同时打 p90/max，就是为了这条。

--- 以下 4 条是做这两个用例时新踩的 ---

9. **die 被别人占着，测出来的数会整体膨胀 ~1.7×，而且没有任何报错。**
   实测（2026-08-31，die 0 上有别人的 8 die 服务、die 1 上有别人的 16 卡训练）：

   | | 空闲 die | 有人的 die 0 | 有人的 die 1 |
   |---|---|---|---|
   | `MatMulV2 "1,4096;24896,4096"` p50 | **166 µs** | 287 µs | 296 µs |
   | 同一算子 p90 | 168 | 387 | 551 |

   **1.7× 正好落在「20% 以内正常 / 2× 说明 shape 错了」这个判据的缝里**，
   所以两个用例都内置了哨兵，**两个独立判据，因为它们失效的时机不同**：

   * **中位数**对已知 shape 的固定算子 > 1.25×  →  die 整段时间都忙。
   * **离散度** p90/p50 > 1.25×  →  die 是**阵发性**忙。

   第二个不是锦上添花。2026-08-31 现场量到：一次 DSA 跑出 **+41.4%**，
   而它最大的那个 GEMM 中位数只有 **1.12×** —— 伤害全在尾巴上
   （p50 71 µs / p90 223 µs）。**只看中位数会整个漏掉。**
   这正是本项目最重要的那条证据（GEMM 中位数只降 2.4% 而 p90 从 338 掉到 183）
   的同一个教训，现在焊进了工具里。

   命中时会打印 `!! THIS DIE IS SHARED` 加上最差的四个算子，让你先去看
   `npu-smi info` 而不是去怀疑算子。

10. **`npu_format_cast` 转 FRACTAL_NZ 是个静默 no-op，除非先开 internal format ——
    而那个开关在 import `transfer_to_npu` 之前根本不存在。**

    ```python
    from torch_npu.contrib import transfer_to_npu   # 必须先 import
    torch.npu.config.allow_internal_format = True   # 否则这个属性都没有
    ```

    不开的话：权重停在 ND，profiler 打出 `"1,16384;16384,4096;4096;1"` 而不是
    `"1,16384;128,1024,16,32;4096;1"`，`o_proj` **103 µs 而不是 58 µs**，
    stderr 上只有一行 `Warning: Cannot create tensor with internal format`。
    **本用例现在会主动 `get_npu_format` 复查并在不是 NZ 时直接退出** ——
    因为这是第 4 条说的那种静默约束里最贵的一个。

11. **在一个 profiler session 里按「每个 case 均分 kernel 行数」来切分是错的，
    而且是静默错的。** 只要两个 case 的 launch 数不同，一个 case 的 kernel 就被
    算到隔壁头上。实测后果：一次 68 MiB 的读被报成 **7.06 µs = 它自己带宽地板的
    0.12×** —— 一个物理上不可能的数，但表格里看不出来。
    两个用例改成**在 case 之间插一个独有 shape 的哨兵 kernel**（`[1357]` 的
    `add_`），按哨兵切分，并且在哨兵个数对不上时**拒绝出数**而不是继续。

12. **这台机器的 L2 有 ~168 MiB，足以把一整个权重装进去；重复读同一个 buffer
    量出来的带宽是假的。** 实测 KDA `o_proj` 权重 64 MiB：

    | | p50 | 该权重的带宽地板 | 比值 |
    |---|---|---|---|
    | 热跑（同一个权重 30 次） | 29.8 µs | 53.7 µs | **0.56×** |
    | 冷跑（6 份不同权重轮转，> 2×L2） | 54.7 µs | 53.7 µs | 1.02× |

    **热跑比它自己的 HBM 地板还快 1.8 倍。** 服务里不是这个状态：一步要把 34 层
    不同的权重从 L2 前面流过去。`refs` 段现在**热冷都给**，
    并且**任何低于 1.0 的比值都当成警告而不是结论**。

    ⚠ **这条我自己先踩了一次，记下来。** 第一版用「> 2×L2」做冷跑轮转，
    194.5 MiB 的权重只需要 2 份 = 389 MiB，对 168 MiB 的 L2 仍然留了不少复用，
    量出 146.9 µs 就直接反推「≥1.39 TB/s，所以 1.25 常数偏低」。
    **判据应该是「把轮转做大，答案不再动」，不是「轮转 > 2×L2」。**
    改成 4 份（778 MiB）重测：**146.1 µs**，确实不动了 —— 结论碰巧站得住，
    但当时的证据不足以支持它。

    结论的正确版本，用两个独立测量钉住：

    | 测量 | 字节 | p50 | 达到的带宽 |
    |---|---|---|---|
    | GEMM `[1,4096]×[24896,4096]`，4 份轮转 | 194.5 MiB | 146.1 µs | **1.40 TB/s** |
    | `w.sum()` 同样的字节 | 194.5 MiB | 167.7 µs | 1.22 TB/s |
    | DSA KV 池整读 1.19 GiB（远超任何复用） | 1217.4 MiB | 1100.5 µs | 1.16 TB/s |

    **1.25 TB/s 不是一个数，是两个**：GEMM 走的读路径能到 ~1.40 TB/s，
    reduce 类走的能到 ~1.16–1.22 TB/s。用例打印地板时仍用 1.25（跟
    REPORT.md 一致，好比对），**但 §6.2 的判决用 1.40 复核过**，
    结论是那两个 GEMM 比 1.25 的表看起来**更贴墙**。

13. **同一张空闲 die、同一个脚本、隔几分钟跑两次，单个小算子能差 34%。**
    实测 `causal_conv1d_4`：一次 **11.89 µs**，另一次 **15.88 µs**（两次都是
    30 replay 的 p50，两次的 p90/p50 都 < 1.05，所以不是尾巴，是整个分布挪了）。
    `BatchMatMulV2` 同样在 8.66 和 11.58 之间跳。
    **这是第 8 条在这两个用例上的具体形态**：p50 稳、p90 稳，但**跨 run 不稳**。
    所以判一个算子改动有没有效，必须**同一次 run 里 A/B**，不能拿今天的数
    去比昨天的数。大 GEMM 不受影响（`MatMulV2 "1,4096;24896,4096"` 三次分别是
    166.4 / 165.2 / 168.3 µs，1.2% 内）——**受影响的正好是那些「固定开销主导」
    的小算子，也就是 §6.3 里真正值得优化的那些。**

14. **一次 AI core 异常会往你的当前目录扔 200 MB。** 本用例开发时踩到一个
    `aicore exception (507015)`，CANN 在 **cwd** 下建了
    `extra-info/data-dump/0/exception_info.<...>`（204 MB）+ 几个算子二进制，
    另外每次跑都会掉一个 `fusion_result.json`。
    **`/mnt/workspace` 只剩 17 GB，而且这些文件是 `-r--------`，`ls` 不留神就漏了。**
    跑完记得 `rm -rf extra-info fusion_result.json`，或者干脆在
    `/var/tmp/glm53/` 下建个空目录 `cd` 进去再跑
    （用例本身的 profiler 输出已经写在 `/var/tmp/glm53/oplab/`，可以用
    `OPLAB_PROF_DIR` 改）。

---

## 8. 测量纪律（用例里已经做了的）

* **入图。** `torch_npu.npu.NPUGraph()` + `torch_npu.npu.graph(g)` 捕获，然后
  `g.replay()`。eager 的墙钟在这里是 **686 µs/层**，图 replay 是 **304 µs** ——
  eager 量的是 host，不是 device。`bench_kda_layer.py --sections eager` 会把这个
  差打出来。
* **捕获前先热身 ≥10 次。** Ascend 上第一次跑一个新 shape 要付编译和 tiling 选择，
  Triton kernel 还要 JIT。冷着捕获会把编译一起捕进去。
* **图捕获里不能有 `int(tensor)` / `.item()` / `.nonzero()` / `bool(t.any())`。**
  真实代码里这些都被刻意消掉了（DSA 的 run 分段、kpool 的 closing 掩码都是
  「全算 + 掩码」而不是「筛选」，就是为了这个），两个用例也一样。
* **读 profiler 的 device `Duration(us)`，不读墙钟。** 墙钟只在 `eager` 段出现，
  而且只是为了说明它不能用。
* **30 次取 p50，同时打 p90 和 max。** 本项目最关键的一条证据就是分布形状：
  一个 GEMM 中位数只降 2.4% 而 p90 从 338 掉到 183 —— 只看均值会把真相盖掉。

---

## 9. 参考文件

* `../int8_singlecard/data/kernel_attribution_cfgI.txt` — 两个用例的回归目标，
  全量 kernel 清单
* `../int8_singlecard/REPORT.md` — 构型 I 的来历，§6.1–§6.5 是 KDA 的五条优化
* `../tools/hcpre_microbench.py` — 同一类交付物的前作（`HcPre` 单算子），
  这两个用例的 `refs` 段就是抄它的
* `../tools/profile_server_decode.py` — cfgI 的采数驱动，
  **`--prompt-tokens` 默认 13**，见 §5.3
* `../layer_check/check_kda.py` / `check_dsa.py` — 精度侧的单层用例
  （这两个用例只管性能）

⚠ **本目录下还有两个文件不是这两个用例的一部分**，是另一个 session 在同一个
worktree 里放的（时间戳 08-30 23:52/23:53，早于本 README）：
`regress_against_network.py` + `reference_inventory_cfgI.json`。
从它的 docstring 看是**按算子清单（而不是按总时间）**核对单层用例与整网的差异 ——
和这里的 §5 是同一件事的另一种做法，**没有合并，也没有互相验证过**。
用之前先跟那个 session 对一下，别假设两边的口径一致。
