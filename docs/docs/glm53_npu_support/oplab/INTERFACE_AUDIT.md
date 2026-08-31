# INTERFACE_AUDIT — 用例调的算子入口，和整网是不是同一个？

`bench_kda_layer.py` / `bench_dsa_layer.py` 已经过**组成回归**（KDA 9/9 组、DSA 60/61 组，
`regress_against_network.py`）：算子类型、input shape 字符串、每层调用次数都和整网 profile 逐项一致。

组成回归**看不见** stride、contiguity、storage offset、npu_format 和可选参数取值。
同一个 shape 字符串可以对应不同的 stride，同一个 `[k,n] int8` 可以是 ND 也可以是 NZ。
本文档补的就是这一层：**在算子边界上把张量属性打出来对一遍。**

---

## 0. 怎么测的（每条结论的证据等级）

三种证据，本文每条结论都标注了用的是哪种：

| 标记 | 含义 |
|---|---|
| **实测(整网)** | 在真实 server 进程里，包一层 `torch.ops.npu.*` / sglang python 函数，在算子被调用的那一刻打印 `shape / stride / dtype / is_contiguous / npu_format / storage_offset`。 |
| **实测(用例)** | 同一个探针，同一份代码，跑 `bench_*_layer.py` 的 `body()`。 |
| **读源码** | 只读了代码，没上机。 |

探针 `opprobe.py` / `hlprobe.py` 与运行记录在 `/var/tmp/glm53/opaudit/`（临时目录，不入库）。

**整网这一侧跑的是真 server**，不是单层 harness：

```
GLM-5.3-Flash-W8A8-e16, --tp-size 1 --page-size 64 --context-length 32768
--max-running-requests 16 --mem-fraction-static 0.80 --disable-radix-cache
ASCEND_RT_VISIBLE_DEVICES=3   (A3 单 die)
```

即 `int8_singlecard/launch_glm_w8a8_tp1.sh.example`，然后发一条 24 token prompt /
24 token decode 的 `/generate`，把 prefill 和 decode 都走一遍。

### 两个必须说清楚的测量口径

1. **`torch.ops.npu.*` 那一层的探针和 NPU graph capture 不共存。**
   包一层 python callable 之后，capture 会在 torch_npu 的 `__torch_dispatch__` 里挂掉
   （`'SimpleOperatorEntry' object has no attribute '__name__'`）。
   所以**厂商算子的张量属性是在 `--disable-cuda-graph`（eager）下测的。**
   已确认 `ascend_kda_backend.py` / `kpool_indexer_npu.py` 的 decode 路径**没有任何
   graph/eager 分支**（`grep is_capturing|capture_mode|graph_mode` 只命中一个
   `get_cuda_graph_seq_len_fill_value`）→ 算子边界上的 layout 与是否 capture 无关。**（读源码）**

2. **eager 下 block table 是按 batch 现算的，宽度会比整网小。**
   为此另跑了一次 **graph 模式** server，探针换成只包 sglang 的 **python 函数**
   （`select_pools`、`AscendAttnBackend.forward_sparse`、两个 Triton 入口），
   完全不碰 torch dispatch，capture 正常。block table 的结论来自这一次。**（实测(整网)，graph）**

---

## 1. 总表

| 算子 | 谁能优化 | 整网入口 | 用例入口 | 同构？ | 优化后能否直接接线 | 判据 |
|---|---|---|---|---|---|---|
| `torch.ops.npu.causal_conv1d` | **厂商**（`sgl_kernel_npu` .so） | 直调，在 `ascend_kda_backend.py:406 _causal_conv1d_decode`（extend 在 `:342`） | 直调 | ✅ 是（唯一差异：`conv_states` 的 npu_format 0 vs 2，已实测无影响） | ✅ **换 .so 直接生效** | 实测(整网+用例)，另做了 format A/B 实测 |
| `torch.ops.npu.batch_matmul_transpose` | **厂商**（`sgl_kernel_npu` .so） | 直调，`deepseek_v2_attention_mla_npu.py:533 forward_dsa_core_npu` | 直调 | ✅ 完全一致（3 个张量 shape/stride/dtype/format 全同） | ✅ **换 .so 直接生效** | 实测(整网+用例) |
| `torch.ops.npu.npu_quant_matmul` | **厂商**（torch_npu / CANN） | **经 wrapper**：`compressed_tensors_w8a8_int8.py:204 → NPUW8A8Int8DynamicLinearMethod.apply (linear_method_npu.py:148)` | 直调 | ✅ 是（含权重 **FRACTAL_NZ(29)**，3 个 DSA shape 全对上） | ✅ **换 kernel 直接生效** | 实测(整网+用例) |
| `torch.ops.npu.npu_dynamic_quant` | **厂商** | 同上 wrapper，`linear_method_npu.py:147` | 直调 | ✅ 完全一致 | ✅ **直接生效** | 实测(整网+用例) |
| `torch_npu.npu_sparse_flash_attention` | **厂商** | 直调，`ascend_backend.py:1226 forward_sparse` | 直调 | ✅ 是（含 block table `[1,512]`、`sparse_mode=3`、`attention_mode=2`、零 rope 宽 64） | ✅ **直接生效** | 实测(整网 eager + graph 双测 + 用例) |
| `torch_npu.npu_lightning_indexer` | **厂商** | **经 wrapper**：`kpool_indexer_npu.py:766 forward_npu → select_pools() → :211` | 直调 | ✅ 是（含 pooled block table `[1,128]`、`sparse_count=512`、`sparse_mode=0`、weights **fp32**） | ✅ **直接生效** | 实测(整网 eager + graph 双测 + 用例) |
| `torch_npu.npu_rms_norm` | **厂商** | **经 2 层 wrapper**：`RMSNorm(BaseFusedOp).forward → forward_npu (layernorm.py:615)` | 直调 | ✅ 是（含 stride 2048 的切片视图和 `off=1536`） | ✅ 直接生效，**但见 §4.3 的 `npu_add_rms_norm` 分支** | 实测(整网+用例) |
| `torch_npu.npu_scatter_nd_update_` | **厂商** | **经 wrapper**：`memory_pool_npu.py:869 set_index_k_bf16` / `:995 kpool_decode_update_index_cache` | 直调 | ✅ 是（两种形态 `[N,1,128]` 与 `[76,1,128]` 都对上，索引 int64） | ✅ **直接生效** | 实测(整网+用例) |
| `fused_sigmoid_gating_recurrent` (Triton) | **只有我们** | `TritonKDAKernel.decode` (`kda_triton.py:138`)，经 `KDAKernelDispatcher` | 直调同一个函数 | ✅ bs=1 一致；**bs>1 不一致**（见 §4.4） | n/a — 厂商改不了 | 实测(整网+用例) |
| `fused_norm_gate` (Triton) | **只有我们** | `FusedRMSNormGated.forward → rms_norm_gated → layer_norm_gated_fwd` | 直调 `layer_norm_gated_fwd` | ✅ 完全一致（含 `x` 的 npu_format 30 和 `g` 的 `off=8192`） | n/a — 厂商改不了 | 实测(整网+用例) |

---

## 2. 结论（先给答案）

### 2.1 优化完可以**直接接回整网**（改 .so / 换 kernel 二进制即生效，不用动用例也不用动模型代码）

**8 个厂商算子全部可以。** 逐个在算子边界实测过张量属性，没有一个是"相邻但不相同"的调用：

- `causal_conv1d`、`batch_matmul_transpose` — 我们自己的 `sgl_kernel_npu` .so，整网**直调**，用例**直调**，同一个 `torch.ops` 注册名。换 .so 立刻生效。
- `npu_quant_matmul`、`npu_dynamic_quant`、`npu_rms_norm`、`npu_scatter_nd_update_`、`npu_lightning_indexer` — 整网经过 sglang 的 python wrapper，用例直调裸算子。**但 wrapper 是纯透传**：实测两侧到达算子的张量属性逐字段相同，wrapper 末端就是同一个 `torch.ops.npu.*`。厂商换实现，整网自动受益。
- `npu_sparse_flash_attention` — 整网直调，用例直调，18 个参数逐个对上。

**一句话交付给算子团队：这两个用例可以直接当作优化目标，优化结果换 .so / 换 kernel 二进制即可回到整网，不需要我们改代码。**

### 2.2 需要**我们改代码**才能接的

**没有厂商算子落在这一类。** 但有两个算子**厂商根本改不了**：

- `sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent`（config I 里 35.0 us × 34 层 = **1.19 ms/step**，KDA 第 3 大项）
- `sglang.kernels.ops.attention.fla.fused_norm_gate`（3.6 us × 34 = 0.12 ms/step）

这两个是**我们自己的 Triton kernel**，源码在 sglang 树里（`python/sglang/kernels/ops/attention/fla/`）。
厂商拿到 `bench_kda_layer.py` 会连着这两个一起测到，但**它们不在厂商的交付范围内**——
改进只能由我们做（或者由厂商提供一个 Ascend 原生替代，那是新算子，需要我们改
`kda_triton.py` 的 dispatcher 接线，不是换 .so）。

**交付时必须和算子团队说清楚这条边界**，否则 1.19 ms/step 的那一项会落在无人认领的地方。
参考：GDN 已经有这个先例——`gdn_triton.py:22` 在 NPU 上把
`fused_sigmoid_gating_delta_rule_update` 换成了 `sgl_kernel_npu.fla` 的 NPU 版；
**KDA 没有这个 NPU override**（`kda_triton.py:17` 无条件用 Triton 版），
所以 KDA 这一项今天 100% 是我们的 Triton。**（读源码 + 实测确认整网走的就是 Triton 版）**

### 2.3 用例**代表不了整网**的地方

**在 bs=1 这个目标口径上，没有。** 十个入口全部同构。

但有 **3 条口径边界**必须写进交接说明，否则会被误读成"用例覆盖了整网"：

1. **用例只有 bs=1，整网在 concurrency>1 时张量会变成非连续。**
   **实测(整网, graph)**：graph 在 bs=16/12/8/4/2/1 各 capture 一份。在 bs=16 那份里，
   `fused_sigmoid_gating_delta_rule_update` 的 `q/k/v` 是
   `(1,16,64,128) stride (393216, 24576, 128, 1)` — **非连续**，token 维 stride 24576
   （q/k/v 是同一条 packed qkv 行的三个交错切片，offset 0 / 8192 / 16384）。
   用例的是 `(1,1,64,128) stride (8192,8192,128,1)` — **连续**。
   把整网 `ascend_kda_backend.py:278-281` 那段表达式原样重放，B=1 得到
   `s(8192,8192,128,1) contig=True`（和用例逐字节相同），B=16 得到
   `s(393216,24576,128,1) contig=False`（和整网相同）——**差异纯粹来自 batch，不是来自用例写错了。**
   影响面：**只落在我们自己的 Triton kernel 上**。厂商算子在 bs>1 时仍然拿到连续输入
   （`causal_conv1d` 前面的 `.contiguous()` 在 bs>1 时会真的拷贝一次）。
2. **厂商算子的属性是在 eager 下测的**（原因见 §0）。block table 宽度另用 graph 模式的
   python 层探针补测过；其余字段依赖"decode 路径无 graph/eager 分支"这个**读源码**结论。
3. **KV / index 池大小两侧不同**（整网 19737/19738 页 vs 用例 19479/19480 页）。
   这是 `--mem-fraction-static` 算出来的，不是接口问题；stride 和 layout 完全一致。

---

## 3. 已知的坑，逐条核对

| # | 坑 | 结论 | 判据 |
|---|---|---|---|
| 1 | KDA conv 池必须是 window-major `(slots, kernel_size-1, conv_dim)` | ✅ **两侧都是**。整网 `(17, 3, 24576) stride (73728, 24576, 1)`，用例**逐字段相同**。用例没有喂 channel-major。 | 实测(整网+用例) |
| 2 | `causal_conv1d` 要 contiguous（响亮）；`causal_conv1d_update` 的 layout 约束是静默的 | ✅ 两侧 `x` 都是 `(1,24576) stride (24896,1) is_contiguous=True`。**注意：`.contiguous()` 在 bs=1 是 no-op**（dim0 size=1 时 PyTorch 判连续），所以算子实际收到的行 stride 是 **24896 而不是 24576**——**用例精确复现了这一点**。另外：**整网的 KDA/DSA decode 路径根本不调 `causal_conv1d_update`**（探针包了它，整网 93 条记录里 0 次命中），那个静默约束不在这条路上。 | 实测(整网+用例) |
| 3 | `npu_quant_matmul` 权重可能是 NZ | ✅ **两侧都是 NZ(29)**。整网 `(4096,2048)/(1536,16384)/(16384,4096) int8 npu_format=29`，用例三个 shape 逐字段相同。用例的 `nz_int8()` 里那三行 `transfer_to_npu` + `allow_internal_format=True` 是必需的，整网在 `hardware_backend/npu/utils.py:156` 做了同样的事。 | 实测(整网+用例) |
| 4 | DSA 的 SFA / lightning_indexer 吃分页 KV，用例的 block table / page 布局对不对 | ✅ **对**。graph 模式实测：`forward_sparse` 的 `block_tables = (1,512) int32`，`select_pools` 的 `block_table = (1,128) int32` —— 和用例的 `self.block_tables`（`context_len/page = 32768/64 = 512` 宽）、`self.pooled_bt`（`512/4 = 128` 宽）**完全一致**。KV cache 两侧都是 `PA_BSND` 的 `(pages, 64, 1, 512)`，index cache 是 `(pages, 64, 1, 128)`。 | 实测(整网, graph + 用例) |
| 5 | `index_kpool = 4`（不是 64） | ✅ 两侧一致。`sparse_count=512`（=2048/4）、pooled block table 宽 128（=512/4）、tail ring `(76,1,128)`（=19×4）——**三个独立的地方都印证 kpool=4**。 | 实测(整网+用例) |

---

## 4. 逐条展开有问题 / 需要说明的

### 4.1 `causal_conv1d` — 唯一一个实测到的属性差异，已证明无害

| 参数 | 用例 | 整网 | |
|---|---|---|---|
| `x` (arg0) | `(1,24576)` stride `(24896,1)` bf16 contig | `(1,24576)` stride `(24896,1)` bf16 contig | 同 |
| weight (arg1) | `(4,24576)` stride `(24576,1)` bf16 contig | 同 | 同 |
| `conv_states` | `(17,3,24576)` stride `(73728,24576,1)` bf16 contig **npu_format=2 (ND)** | 同 shape/stride/dtype，**npu_format=0 (NCHW)** | **差异** |
| `bias` | `None` | `None` | 同 |
| `query_start_loc` | `(2,)` int32 | `(2,)` int32 | 同 |
| `cache_indices` | `(1,)` int32 | `(1,)` int32 | 同 |
| `activation_mode` / `pad_slot_id` / `run_mode` | `1 / -1 / 1` | `1 / -1 / 1` | 同 |

**为什么差**：整网的 conv 池是 `_init_npu_conv_state`（`memory_pool_npu.py:25`）分配的
**4 维** `[layers, slots, window, channels]`，backend 取 `layer_cache.conv[0]` 得到一个 3 维视图，
视图继承基张量的 format → NCHW(0)。用例直接 `torch.zeros(slots, 3, 24576)`，3 维 → ND(2)。
**torch_npu 是按基张量的 rank 打 format tag 的**，两者内存布局字节级相同（stride 一样、都连续）。

**实测证伪它有影响**（`/var/tmp/glm53/opaudit/fmt_ab.py`，同一份初始 state、同一份输入）：

```
network conv_states fmt=0 stride=(73728, 24576, 1) contig=True
bench   conv_states fmt=2 stride=(73728, 24576, 1) contig=True
outputs bitwise equal : True
states  bitwise equal : True
round 0  wall us/call  network(fmt0)=79.00   bench(fmt2)=78.23
round 1  wall us/call  network(fmt0)=79.56   bench(fmt2)=103.75
round 2  wall us/call  network(fmt0)=73.94   bench(fmt2)=71.83
```

输出和写回的 state **逐 bit 相同**，wall clock 没有系统性差异（host-bound，噪声 ±30%）。
→ **判定：可忽略。厂商照用例优化不会优化错对象。**

同一类 format tag 差异还出现在 `initial_state_source`（整网 30/NCDHW ← 5 维池的视图，
用例 0/NCHW ← 4 维张量）和 `layer_norm_gated_fwd` 的 `x`（两侧**都是** 30）。
落在厂商算子边界上的只有 `conv_states` 这一处，已单独验过。
**其余进厂商算子的 format 两侧完全一致**：`index_k_cache` 两侧都是 NCHW(0)，
`key_rope` 两侧都是 NCHW(0)，`scatter_nd_update_` 的目标 buffer 两侧都是 NCHW(0)，
`npu_quant_matmul` 的权重两侧都是 **NZ(29)**。

### 4.2 经 wrapper 的 5 个算子 — wrapper 是纯透传，实测确认

用例直调裸算子，整网经过一层 sglang python：

| 算子 | 整网的 wrapper 链 | 实测结果 |
|---|---|---|
| `npu_quant_matmul` / `npu_dynamic_quant` | `NPUCompressedTensorsW8A8Int8.apply_weights` → `NPUW8A8Int8DynamicLinearMethod.apply` | 参数逐字段相同，含 `pertoken_scale` fp32 `(1,)`、`bias=None`、`output_dtype=bfloat16` |
| `npu_lightning_indexer` | `KPoolNPUIndexerMixin.forward_npu` → `select_pools()` | 10 个参数逐字段相同 |
| `npu_rms_norm` | `RMSNorm(BaseFusedOp).forward` → `forward_npu` | 参数逐字段相同 |
| `npu_scatter_nd_update_` | `NPUDSATokenToKVPool.set_index_k_bf16` / `kpool_decode_update_index_cache` | 两种形态都逐字段相同 |

**对"换 .so 能不能生效"的影响：没有。** wrapper 里没有 layout 转换、没有 dtype 转换、
没有参数重排，末端是同一个 `torch.ops.npu.*` 调用，vendor 换实现整网自动走新实现。

`npu_rms_norm` 这一条最漂亮，因为它证明用例连**切片视图**都复现了：

```
q_a_layernorm  用例: (1,1536)   stride (2048,1)   bf16 contig
               整网: (1,1536)   stride (2048,1)   bf16 contig          ← 同
kv_a_layernorm 用例: (1,1,512)  stride (2048,512,1) bf16 contig off=1536
               整网: (1,1,512)  stride (2048,512,1) bf16 contig off=1536 ← 同，含 storage offset
```

两侧的 stride 都是 **2048**（不是 1536），storage offset 都是 **1536** ——
因为两边都是从同一条 `fused_qkv_a_proj` 的 `[.,2048]` 输出上 `split` 下来的。

### 4.3 `npu_rms_norm` 的一个分支：`npu_add_rms_norm`

`RMSNorm.forward_npu`（`layernorm.py:601-615`）有两条路：

```python
if residual is not None:
    out, _, residual_out = torch_npu.npu_add_rms_norm(residual, x, self.weight.data, self.variance_epsilon)
    return out, residual_out
return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]
```

**实测**：DSA 层的 `q_a_layernorm` / `kv_a_layernorm` 两次调用都走**无 residual** 分支
（探针记到的全是 `npu_rms_norm`，参数只有 3 个）。
**但模型里其它带 residual 的 RMSNorm 会走 `npu_add_rms_norm`，那是一个用例完全没碰的算子。**
厂商优化 `npu_rms_norm` 不会让这些受益。
（config I 里 attention 前后的 layernorm 被 mHC 融合成 `HcPre`，不是 RmsNorm，
所以这一条在当前配置下影响很小 —— **读源码 + config I 归因表**，没有单独实测。）

### 4.4 两个 Triton kernel — 厂商改不了，且 bs>1 时用例代表不了整网

**`fused_sigmoid_gating_delta_rule_update`**

整网入口：`AscendKDAAttnBackend.forward_decode` → `self.kernel_dispatcher.decode(...)`
→ `TritonKDAKernel.decode`（`kda_triton.py:138`）→ 同一个 Triton 函数。
NPU 上 `TritonKDAKernel.supports_packed_decode = not is_cpu() and not is_npu() and not is_xpu()`
= **False**，所以走的是非 packed 的 `decode()`，正是用例调的那个。**（读源码 + 实测确认）**

三侧对比（A=用例 bs=1；B=真 server graph capture bs=16；C=`layer_check/check_kda.py`
驱动的**真生产 backend** `AscendKDAAttnBackend.forward_decode`，bs=1，TP16 rank0 → 4 heads）：

| 参数 | A 用例 (TP1, bs=1) | B 真 server (TP1, bs=16) | C 生产 backend (TP16 rank0, bs=1) |
|---|---|---|---|
| `q` | `(1,1,64,128)` s`(8192,8192,128,1)` **contig** off=0 | `(1,16,64,128)` s`(393216,24576,128,1)` **非连续** off=0 | `(1,1,4,128)` s`(512,512,128,1)` **contig** off=0 |
| `k` | 同上 off=8192 | 同上 off=8192 | 同上 off=512 |
| `v` | 同上 off=16384 | 同上 off=16384 | 同上 off=1024 |
| `b` | `(1,1,64)` s`(24896,24896,1)` contig off=24576 | `(1,16,64)` s`(398336,24896,1)` **非连续** off=24576 | `(1,1,4)` s`(4,4,1)` contig off=0 |
| `a` | `(1,8192)` contig | `(16,8192)` contig | `(1,512)` contig |
| `cu_seqlens` | `(2,)` int32 | `(17,)` int32 | `(2,)` int32 |
| `initial_state_source` | `(17,64,128,128)` fp32 contig **fmt=0** off=0 | 同 shape/stride，**fmt=30**，**每层一个 offset**（0 / 17825792 / 35651584 …） | `(2,4,128,128)` fp32 contig **fmt=30** |
| `A_log` / `dt_bias` / `lower_bound` / `is_kda` / `use_qk_l2norm_in_kernel` / `softplus_*` | `(1,1,64,1)` fp32 / `(8192,)` fp32 / `-5.0` / `True` / `True` / `1.0, 20.0` | **完全相同** | **完全相同**（`A_log`/`dt_bias` 按 4 heads 缩放） |

**C 列是关键的一列**：它是真的生产 backend（`AscendKDAAttnBackend.forward_decode`
→ `KDAKernelDispatcher.decode` → `TritonKDAKernel.decode`）在 bs=1 下跑出来的，
不是重放的表达式。它的 `q/k/v` 是 **contiguous、token 维 stride == q_dim、
offset 0 / q_dim / 2·q_dim** —— 和 A 列**结构完全相同**（A 列 q_dim=8192 是 64 heads，
C 列 q_dim=512 是 TP16 的 4 heads）。**bs=1 下用例是对的。**

两处需要说明：

- **`b`**：C 列 off=0 stride 4，A/B 列 off=24576 stride 24896。这里**用例对、C 列不对**——
  `check_kda.py` 的 harness 用的是未融合的投影，而部署的 `glm5_next` 走 fused
  `qkvbfg_a_proj`（24896 宽），`b` 是它上面的一个切片。**A 列和 B 列（真 server）一致**，
  说明用例复现的是部署形态。
- **`initial_state_source` 的 fmt=30**：B 列和 C 列都是 30，A 列是 0。两次独立测量都指向
  §4.1 描述的同一类"视图继承基张量 rank tag"现象。只影响我们自己的 Triton kernel。

把整网 `ascend_kda_backend.py:278-281` 那四行原样重放，可以看到 B 列的非连续形态纯粹来自 batch
（`/var/tmp/glm53/opaudit/qkv_stride.py`）：

```
B=1   q (1, 1, 64, 128) s(8192, 8192, 128, 1) contig=True  off=0 | k off=8192 | v off=16384
B=16  q (1, 16, 64, 128) s(393216, 24576, 128, 1) contig=False off=0 | k off=8192 | v off=16384
```

**→ bs=1（部署口径）用例和整网结构一致（A 列 vs C 列，实测）；bs>1 的非连续形态用例产生不出来。**
这不是用例写错了，是它按设计只跑 bs=1。要覆盖 bs>1，用例需要新加一档。

**`fused_norm_gate` / `layer_norm_gated_fwd`**

整网入口：`glm5_next.py:652 self.o_norm(core_attn_out, norm_gate)` →
`FusedRMSNormGated.forward` → `rms_norm_gated(...)` → `layer_norm_gated_fwd(...)`。
用例直调 `layer_norm_gated_fwd`。**实测在 `layer_norm_gated_fwd` 边界上：**

| 参数 | 用例 | 整网 (graph, bs=1) |
|---|---|---|
| `x` | `(64,128)` s`(128,1)` bf16 contig **fmt=30** | `(64,128)` s`(128,1)` bf16 contig **fmt=30** |
| `g` | `(64,128)` s`(128,1)` bf16 contig **off=8192** | `(64,128)` s`(128,1)` bf16 contig **off=8192** |
| `weight` | `(128,)` bf16 | `(128,)` bf16 |
| `bias` / `activation` / `eps` / `is_rms_norm` | `None` / `'sigmoid'` / `1e-05` / `True` | 相同（整网另外显式传 `residual=None, residual_dtype=None`，是默认值） |

**逐字段相同，包括 format 30 和 `g` 的 storage offset 8192。**
（64 = 1 token × 64 heads；graph 在 bs=16/12/8/4/2/1 各 capture 一份，
`x` 分别是 `(1024/768/512/256/128/64, 128)`，用例对应的是 bs=1 那份。）

---

## 5. 实测对比了哪些张量属性

每个张量都打了 **7 个字段**：`shape` / `stride` / `dtype` / `is_contiguous` / `npu_format` /
`storage_offset` / `numel`；非张量参数打字面值。

- 整网：**93 条** `torch.ops.npu.*` 边界记录（真 server，eager，去重后的不同调用签名）
  + **36 条** python 层记录（真 server，graph 模式）
  + **5 条** python 层记录（`layer_check/check_kda.py` 驱动的生产 backend，bs=1）。
- 用例：**14 条** `torch.ops.npu.*` 记录 + **2 条** Triton 入口记录，同一份探针同一套字段。

**发现的不一致，穷举如下（只有 3 类，全部已定性）：**

| # | 不一致 | 性质 | 处置 |
|---|---|---|---|
| 1 | `conv_states` 的 `npu_format`：整网 0 (NCHW) / 用例 2 (ND) | 池是 4 维分配后切片 vs 直接 3 维分配，torch_npu 按基张量 rank 打 tag；stride 与连续性完全相同 | **实测输出与 state 逐 bit 相同、无时间差** → 忽略 |
| 2 | `initial_state_source` 的 `npu_format`：整网 30 (NCDHW，5 维池的切片，且每层带 offset) / 用例 0 (NCHW) | 同上一类 | 只进**我们自己的** Triton kernel，不影响厂商交接 |
| 3 | 池大小：整网 KV 19737 页 / index 19738 页；用例 19479 / 19480 | `--mem-fraction-static` 算出来的，stride、layout、format 全同 | 非接口问题 |

**没有发现的（逐条确认为"两侧一致"）**：
布局顺序（window-major / channel-major）、dtype、连续性、storage offset、
NZ 权重格式、block table 宽度、page 布局、`index_kpool`、所有可选参数取值
（`run_mode` / `activation_mode` / `pad_slot_id` / `sparse_mode` / `attention_mode` /
`sparse_block_size` / `sparse_count` / `layout_query` / `layout_kv` / `layout_key` /
`output_dtype` / `is_kda` / `lower_bound` / `use_qk_l2norm_in_kernel` / `softplus_*` /
`is_rms_norm` / `activation` / `eps`）。

---

## 6. 复现

```bash
source /mnt/workspace/y00359136/work/glm53_dev/env/env.sh
export PYTHONPATH=/var/tmp/glm53/opaudit:$WT/python:$PYTHONPATH

# 用例侧
OPPROBE=1 OPPROBE_TAG=bench_kda OPPROBE_OUT=.../bench_kda.jsonl \
ASCEND_RT_VISIBLE_DEVICES=3 python drive_bench.py kda

# 整网侧（厂商算子，eager）
OPPROBE=1 ... launch_glm_w8a8_tp1.sh --disable-cuda-graph   # 见 launch_probe_server.sh
curl .../generate -d '{"text":"...","sampling_params":{"max_new_tokens":24}}'

# 整网侧（block table + Triton，graph）
HLPROBE=1 ... launch_glm_w8a8_tp1.sh                        # 不包 torch.ops，capture 正常
```

探针、驱动脚本和原始 jsonl 在 `/var/tmp/glm53/opaudit/`：
`opprobe.py`（torch.ops 层）、`hlprobe.py`（python 层）、`drive_bench.py`、
`drive_bench_hl.py`、`fmt_ab.py`（format A/B）、`qkv_stride.py`（表达式重放）、
`network.jsonl`、`hl_network.jsonl`、`bench_{kda,dsa}.jsonl`、`hl_bench_kda.jsonl`。

## 7. 还没验的

- 厂商算子的张量属性是 **eager** 下测的；graph 下只用 python 层探针补测了 block table
  和两个 Triton 入口。"decode 路径无 graph/eager 分支"是**读源码**结论，没有做
  graph 下的 `torch.ops` 层实测（探针与 capture 不共存，见 §0）。
- `npu_add_rms_norm`（§4.3）没有实测，只读了源码和 config I 归因表。
- bs>1 的厂商算子边界只测到 graph capture 记录的那几档（16/12/8/4/2/1），
  没有做 bs>1 的**稳态** decode。
- 真 server 的 `fused_sigmoid_gating_delta_rule_update` **bs=1 那一档没抓到**
  （探针的去重上限被 bs=16 的 34 层各自不同的 state offset 占满）。bs=1 的形态改用
  `layer_check/check_kda.py` 驱动真生产 backend 补测（§4.4 C 列），那是 TP16 rank0 的
  4 heads 形状，结构一致但数值规模不同。
- 试过重跑真 server 去补 bs=1 那一档（探针上限提到 3000），但另一位用户的 16 卡
  torchtitan 训练作业在审计后半程反复占用 die 3（~54 GB），两次都在加载阶段 OOM。
  这一档因此停留在 §4.4 的 C 列（生产 backend，TP16 rank0）。**这台机器是共用的**，
  重跑前先 `npu-smi info -t proc-mem -i 1 -c 1` 看 die 3 是不是空的。

## ⚠ 一条审计之外的线索：厂商已经有 KDA 递归核，而我们没接

`fused_sigmoid_gating_delta_rule_update` 是 KDA 第三大项（**1.19 ms/step，整步的 3.8%**），
上表把它标成「只有我们能优化」。**那一半对，一半漏了。**

`sgl_kernel_npu/fla/fused_sigmoid_gating_recurrent.py` 里**有一个 NPU 版本**
（`fused_sigmoid_gating_delta_rule_update_npu`），而且 **GDN 在 NPU 上已经切过去了**：

```python
# srt/layers/attention/linear/kernels/gdn_triton.py:21-27
if is_npu():
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update_npu)
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
```

**KDA 那一侧（`kernels/kda_triton.py:17`）没有这个 override**，直接 import 了 sglang 的 Triton 版。
`sgl_kernel_npu/fla/` 下还有 `kda_gate.py` / `kda_chunk_delta_h.py` / `kda_prefill.py` /
`kda_target_verify.py` —— **厂商是有 KDA 支持的**，只是 decode 这一支没接。

⚠ **但这是线索不是结论，两条限定必须一起读**：

1. `kda_triton.py:26-29` 的注释表明 NPU 走 fallback 是**明确的选择**而不是疏漏：
   `supports_packed_decode = not is_cpu() and not is_npu() and not is_xpu()`，
   注释说 XPU/CPU/NPU 都走 `fused_sigmoid_gating_delta_rule_update` 这条非 packed 路径。
2. **GDN 和 KDA 不是同一个算子。** 同名的厂商 kernel 对 KDA 是否语义正确**没有验过** ——
   厂商为 KDA 单独准备了 prefill / target-verify / gate，却没有单独的 decode update，
   这本身可以有两种解释（共用是对的 / 还没做）。

**该怎么查**：拿 `oplab/bench_kda_layer.py` 换掉那一个调用，先对逐位/形状地板判据，
再看时间。**这是本轮唯一一条「厂商的东西已经在盘上、我们没接」的候选，值 1.19 ms。**

