# GLM-5.3-Flash 昇腾（Atlas A3 / Ascend910_9362）适配计划

> 活文档。**只记当前事实与计划，不记怎么走到这里的** —— 过程看 `git log`。
> 每条结论都标了证据等级：**实测**（本机跑过）/ **源码**（读代码或头文件得出，未执行）/ **推断**。
> 最后更新：2026-08-28

---

## 0. 目标与已定决策

| 项 | 决定 |
|---|---|
| 一期目标 | GLM-5.3-Flash **纯文本**在 A3 上跑通并闭环精度 |
| BF16 部署形态 | **单节点 TP16**（16 die × 64 GB） |
| 量化格式 | **compressed-tensors W8A8-INT8**（weight per-channel + act per-token dynamic） |
| 精度基准 | **HF `transformers==5.16.1` 的 `glm5_next`**（纯 torch、CPU 可跑、四模块全覆盖） |
| 多模态 / MTP / 长上下文 CP | 一期**不做** |
| DSv4 权重 | P1.2 后已删（275 GB safetensors；元数据保留） |

---

## 1. 环境事实（全部实测）

### 硬件与系统
- **SoC `Ascend910_9362`（A3，不是 A2/910B）** —— 认型号只能用 `torch.npu.get_device_name(0)`，
  `npu-smi` 对 A2/A3 都显示 `Ascend910`。所有 sgl-kernel-npu 包选 **a3** 档
- 16 die × 64 GB HBM；CPU 320 核 / 内存 1.8 TB
- **Ubuntu 24.04.3 / glibc 2.39** → SETUP 附录 B 的 glibc 绕行**整段不需要**
- **A3 没有 fp8**：`bishengir-compile` 无法 lower e4m3（`unsupported datatype for arith::TruncFOp`），
  torch 侧 `x.to(torch.float8_e4m3fn)` 直接触发 device 异常。
  **连分配都不行**：`torch.zeros(4, dtype=torch.float8_e4m3fn, device="npu")` →
  `aclnnInplaceZero failed, 161002`（实测）。所以这不只是 kernel 语言问题，
  任何**实体化** fp8 张量的路径都会死，跟 triton 无关

### 软件栈
- CANN **组件实为 9.1.0**（外层包名标 9.2.0，不可信）；toolkit 在 `/home/developer/Ascend/ascend-toolkit/`
- Python 3.12.9 / torch **2.10.0** / torch_npu **2.10.0.post4** / triton-ascend **3.2.2**（带 `cann` 后端）
- sgl-kernel-npu **20260826**，`cann9.1.0-a3-aarch64` + `py312`
- `ROOT=/mnt/workspace/y00359136/work/glm53_dev/env`，`source $ROOT/env.sh` 后用 `npy` 代替 `python`
- 参考环境 `$ROOT/.venv-ref`：transformers 5.16.1 + CPU torch。**绝不能装进 `.venv-glm53`**（sglang 钉 5.12.1）

### 磁盘
- `/mnt/workspace` 984 GB，已用 915 / **余 70 GB**
- GLM FP8 306 GiB（保留至 P4 验过）+ BF16 **599 GB**
- ⚠ P5 的 W8A8 约 333 GB：`643 + 333 = 976 / 984`，**必须先删 FP8 源**

### 模型结构
- 45 层 = **34 KDA**（`linear_attention`）+ **11 DSA**（`deepseek_sparse_attention`，层号 3,7,11,…,43），另有第 45 层 MTP
- FFN：前 3 层 dense + 后 42 层 MoE（288 routed + 1 shared，top-8）
- **NoPE MLA**：`qk_nope=256, qk_rope=0, v=256, kv_lora=512, q_lora=1536, heads=64`
- KPool indexer：`index_n_heads=32, index_head_dim=128, index_topk=2048, index_kpool=4`，压缩 + 总选尾部
- KDA：`num_heads=64, head_dim=128, conv_kernel=4, gate_lower_bound=-5.0`
- mHC：`hc_mult=4, sinkhorn_iters=20, hc_eps=1e-6`；`rms_norm_eps=1e-5`（DSv4 是 1e-6）
- `swiglu_limit=10.0`；**TP16 整除性全部 OK**（64/64/32/288 都能被 16 整除）

### 服务配置（踩过的）
- **`--page-size` 必须是 64** —— DSA pool 有 `assert self.page_size == 64`。DSv4 用的是 128，照搬会启动失败
- 纯 TP16 权重 **37.25 GB/die**（= 599/16，无复制）；DP-attention 会把 attention/dense 每 rank 各存一份
- DSv4 的 TP16/DP16+DeepEP 配方见 `launch_dsv4_a3.sh.example`；GLM 的见 `$ROOT/run/launch_glm_bf16.sh`

### 权重
- `/mnt/workspace/models/GLM-5.3-Flash`（FP8，62 shard）与 `-BF16`（转换产物）
- **版本已核实**为 `zai-org/GLM-5.3-Flash` revision **`c5b82b63e37b`**（71/71 文件 + 62/62 分片 size 一致；非 sha256）

---

## 2. 算子结论

### 2.1 三个来源，别混淆

| 来源 | 命名空间 |
|---|---|
| torch_npu 原生 | `torch_npu.npu_*` |
| CANN vendor 算子包 | `torch.ops.custom.*`（需先 `import custom_ops`） |
| sgl-kernel-npu | `sgl_kernel_npu.*`（triton-ascend 实现） |

本仓库的 KDA / DSA 路径**主要走后两者**。「某个 `torch_npu.npu_xxx` 不支持某特性」**推不出**「这是缺口」。

> ⚠ **同名不同物**：`torch.ops.npu.compressor`（DSv4 用）与 `torch.ops.custom.compressor` 不是一个东西，
> 本 build 只有 `custom` 命名空间有；`sgl_kernel_npu` 的 causal_conv1d 与 `torch_npu.npu_fused_causal_conv1d`
> 也不是一个东西（后者 K 固定为 3）。

### 2.2 已验证可用（对 golden 实测，不是只验形状）

| 算子 | 误差 / 噪声地板 |
|---|---|
| `causal_conv1d_fn_npu` / `causal_conv1d_update_npu`（含 conv_state 回写） | **逐位精确** |
| `fused_kda_gate_npu`（prefill 与 decode 两种形状） | 0.37× |
| KDA chunk 链（`solve_tril`+`recompute_w_u`+`chunk_gated_delta_rule_fwd_h`+`chunk_gla_fwd_o_gk`） | 0.30× |
| KDA decode（`fused_sigmoid_gating_delta_rule_update`）、`o_norm` | 通过 |
| `npu_hc_pre` / `npu_hc_post`（mHC） | post 8.9e-5 / comb 6.1e-6 / collapsed 4.6e-3，均在地板内 |
| `npu_clipped_swiglu`（**四个参数全传时**） | **逐位精确** |
| FIA `npu_fused_infer_attention_score`：BSND prefill + MLA-absorbed decode | 通过 |
| MoE `npu_moe_gating_top_k`（expert 集合 2048/2048 一致）、`npu_grouped_matmul` | 通过 |
| kpool 的 10 个 Triton kernel 中的 **7 个**（top-k、展开、尾部、plan/layout） | **逐位精确** |
| `torch_npu.npu_lightning_indexer`（**bf16**，`n_heads=32`）prefill + decode | 对真实权重的 golden **落在噪声地板上**，见 §2.8 |

### 2.3 确认要开发 / 要改

| # | 项 | 结论 |
|---|---|---|
| **1** | **kpool index-K cache: fp8 → bf16** | A3 无 fp8，4 个 compress-write Triton kernel 因此无法编译。改存 **bf16** 后其中 3 个可编译且全程对得上（第 4 个见 §2.4）。选 bf16 不是折中：它既是精度天花板，**也是唯一能被消费的格式**（§2.7）。纯仓库侧改动，**不需要厂商算子** |
| ~~2~~ | ~~`npu_kv_rmsnorm_rope_cache` 支持 rope=0~~ | **已撤销** —— 算子确实不支持 rope=0（那条实测没被推翻），但**调用点不在 GLM 的路径上**。见 §2.5 |
| ~~3~~ | ~~全零 rope 的 workaround~~ | **已实现**（`ascend_backend.py` 的 `_nope_zero_rope`）。实测：`query_rope`/`key_rope` 签名是 Optional，但**缺省、0 宽、16、32、128 全部报错，只收宽度 64**。全零 rope 数值正确（对 torch MLA 参考 rel 3e-3，即 bf16 输出舍入）。**一页零页用 stride-0 `expand` 铺满整个 cache**，与真实零 cache 逐位相同 —— 总共 8 KB，而不是多一份约 10% 的 KV。⚠ 算子文档说不支持非连续输入，所以这是**观察到的行为、不是承诺的行为**；哪天 CANN 不认 stride-0 就退回分配真张量 |

### 2.4 陷阱（能跑但算错 / 名实不符）

| 陷阱 | 表现 |
|---|---|
| **FIA 在 TND 布局省略 `num_key_value_heads`** | 错 **200×**，**不报错**。BSND 下同一默认值却是对的。已修 `ascend_backend.py` 的 prefill 调用点（全文件唯一漏传的） |
| **`npu_clipped_swiglu` 的默认参数** | 四个默认值（`alpha=1.702, limit=7.0, bias=1.0, interleaved=True`）**对 GLM 全错**，只传部分错 109× |
| **Hadamard-128 能不能删，取决于存什么** | 正交归一、q/k 同旋转、点积不变，所以**在 bf16 下确实可删**；**一旦量化就不能删** —— 它正是 int8 优于 fp8 的原因（旋转后 kurtosis≈3.0）。既然一期走 bf16，**保留它**（保留同样中性，且 3 个 kernel 里它本来就正常）；只有撞上 §4 那个 triton-ascend codegen 缺陷才删 |
| **`ue8m0` scale 舍入** | 对浮点格式免费，对 int8 要付一个真实 bit（32k 重合 99.18% → 98.84%） |
| **`npu_sparse_flash_attention` 的 sparse_indices 契约比「有效值在前」更严，失败方式也更坏** | 之前记的是「不是前缀就静默返回 0」。**实测在真实形状下不对**：decode@32k、2048 个有效索引时，把 `-1` 挪到 slot 0 得到的是 **rel 1.23e-1 的静默错误非零值**，不是 0。位置扫描显示 `-1` 落**偶数槽**会改变输出（rel 0.15–0.29，静默），落**奇数槽**则逐位相同。「静默返回 0」只在**第一个分块整块无效**时出现（小形状上复现）。所以契约是某种与**槽位奇偶 / 分块对齐**相关的东西，前缀恰好满足它。**我们的链路产出严格前缀所以安全**，但任何自己拼索引的地方风险比原记录高得多。机制**未查清**（要看 `aclnnSparseFlashAttention` 的 tiling 或问厂商）|
| **Hadamard 在 bf16 里做** | CUDA kernel 把 bf16 读进 **fp32 寄存器**再变换（`hadamard_jit.cuh:150` 的 `float x_vals[..]`），Triton 的 `_hadamard128` 同样作用在 fp32 accumulator 上。在 bf16 里做要 round 7 次而它们 round 1 次，**不报错**，只是悄悄挪走一批 pool —— 32k 下选择重合掉 0.0006（实测）。这个 bug 我写过一次，被端到端对拍抓出来 |
| **MoE 的路由在 fp32 与 bf16 之间会翻，从 layer 3 就开始** | 实测：top-8 集合不同的 token 占比 layer 3 为 12.5%、layer 41 达 **63.3%**。后果不是"精度差一点"，而是**双参考法的地板从第一个 MoE 层起就由离散的路由差异主导，不再是舍入**——地板从 9.5e-3 涨到 1.8e-1。**深层的宽地板不能当成"宽误差可接受"的依据**；注入 5% 误差实测只有 layer 7–25 测得出来，layer 26+ 测不出。这是**验收方法本身的边界**，不是某个算子的问题 |
| **NPU 的 bf16 矩阵乘不是 batch-shape 不变的** | 同一份输入只改 M（4096 行 vs 4080 行），过 `wk`+`k_norm` 后 **5/4080 行**差 1 个 bf16 ulp，gate 差 6/4080（实测）。根因在 torch_npu 的 matmul tiling，不在业务代码。**后果：NPU 上任何 prefill-vs-decode 的逐位一致性断言都不成立**，包括 P4 打算用的 KL 一致性 —— 只能定阈值，不能要求 bit-exact |
| **KDA prefill 的 Triton autotune** | `kda.py:214` 的 24-config `do_bench` 扫描挂死 AI core（die 4/6/8 上 3/3 复现）；单独钉住任一 config 都能跑。⚠ 实际服务未触发，列为**上线前须确认** |

### 2.5 已排除（不要再做）

| 项 | 为什么 |
|---|---|
| compressor 的 LayerNorm 变体 | **GLM 从不调 vendor `compressor`**（全仓两处引用均在 DSv4 路径）；其 index-K LayerNorm 是独立模块、压缩前施加、从未融合 |
| K=4 causal conv1d | `sgl_kernel_npu` 的 Triton kernel `KERNEL_WIDTH` 1–6 全有，decode 已在用 |
| mHC pre/post | 算子已存在且 DSv4 在用，已接线并验过 |
| GLM 版 clipped SwiGLU | `npu_clipped_swiglu` 参数传对时逐位精确 |
| bf16 输出的 `DequantSwigluClampQuant` | 同上，不需要 |
| `aclnnMixedQuantSparseFlashMla` | `rope_head_dim` 只能是 64；且 AscendC 只有 arch35，**A3 无二进制** |
| `MlaPrologV3` 走 rope=0 | `ropeSin/ropeCos/krCache/queryRopeOut` 全部非 Optional |
| `npu_kv_rmsnorm_rope_cache` 支持 rope=0（原 OP-3）| 全仓只有两个调用点。① `deepseek_v2_attention_mla_npu.py:93`，在 `forward_mha_prepare_npu` 的 yarn 分支里 —— 而 GLM 每个 MLA 层 `use_dsa=True`（**实测**：用仓库自己的 `get_config` 读真实权重目录，`is_deepseek_dsa == True`），`handle_attention_ascend` 的两条腿都返回 `DSA_NPU`，这个函数**整个进不去**；就算进去了，`glm5_next.py:665` 传 `skip_rope=True` 且 `qk_rope_head_dim=0`，`deepseek_v2.py:1913` 让 `rotary_emb = None`，会在算子前 11 行就 `AttributeError`。② `mla_preprocess.py:331`，被默认关闭的 `SGLANG_NPU_USE_MLAPO` 门控，且本机 `npu_mla_prolog_v3` MISS。**证据等级：源码 + config 实测，未在活服务上观测**；P4.1 起服务时顺手确认 |
| `npu_quant_lightning_indexer` 用于 GLM | metadata **只接受 `num_heads_q=64`**，GLM 是 32（64 OK；16/32/128 全 FAIL，两轮独立实测）。这堵死了 int8 索引缓存的消费者，见 §2.7 |

### 2.6 仍未验证

- **`torch.ops.custom.compressor`** 与 **`npu_quant_lightning_indexer`** 的 kernel —— 都需要活的 pool/page-table，
  独立 harness 驱动不了，返回不透明错误
- `npu_sparse_attn_sharedkv` 的 kernel（metadata 能建，kernel 需活 pool）
- `npu_lightning_indexer` 在 **NPU Graph 捕获**下能否用；以及它相对 CUDA fp8 路径的**性能**（都未测）

> **已关闭：昇腾侧由谁计算 indexer logits。** 答案是 `torch_npu.npu_lightning_indexer`
> —— 不是被排除的 `npu_quant_lightning_indexer`，是 DSv4 非 kpool 路径已在用的那个 bf16 算子
> （`dsa/dsa_npu_indexer.py:25`）。它接受 GLM 的 32 头，算的正是 `Σ_h w_h·relu(q_h·k_j)`，
> 并把 top-k 一起融了。全部实测，探针见 `probe/p3_4_lightning_indexer.py`：
> - `num_heads_q`：16/32/64 OK，128 FAIL
> - key dtype：**只收 bf16**；fp16 和 int8 都被拒（**算子文档写「支持 bfloat16 和 float16」，与实现不符**）
> - 返回的是**逻辑序列位置**（用打乱的 block_table 验证），正是 kpool 展开步骤要的
> - `actual_seq_lengths_key < sparse_count` 时用 `-1` 补齐
> - prefill 的可见性斜率是 1/4（`floor(seq_len/kpool)` 个 pool），`sparse_mode=3`（rightDownCausal，
>   斜率 1）表达不了；把「可见 pool 数相同」的连续 query 行分成一段、每段一个 TND batch、配
>   `sparse_mode=0`，逐行精确。4096 query × 8192 pool 用 3.6 ms

---

### 2.7 int8 索引缓存：更准，但目前没有消费者

**int8 比它要取代的 fp8 更准**，这个结论**没有被推翻**，两轮都复现：键重构误差
低 4.2×（0.0067 vs 0.0267）。它回答的是「必须存量化格式时选哪一种」。
§2.8 那张表把 bf16/int8/fp8 放在同一根轴上，bf16 又比 int8 好 1.7×。

但一期不走 int8，因为**没有算子能读它**（两条都实测）：

| 候选消费者 | 结论 |
|---|---|
| `npu_lightning_indexer` | 只收 bf16，int8 被拒 |
| `npu_quant_lightning_indexer` | metadata 只接受 `num_heads_q=64`，GLM 是 32 |
| `npu_nsa_compress_attention_infer` | 也只收 bf16/fp16；且是 NSA 的定步长块压缩，算法不同、还要 value |

这不是凭印象列的：**全量枚举过** `npu::`（399）与 `custom::`（24）的所有算子 schema，
按 index/sparse/topk/select/mqa/logit/lightning/compress/nsa 过滤（命中 35 + 12），
再区分「产出选择」与「消费选择」。产出选择的只有上表三个。
探针第 7 节可复现。**注意这是按名字过滤的枚举，不是穷尽证明** ——
若某个名字不沾这些词的算子也能做选择，结论要改。

唯一的绕法是每层每次前向把整个 cache 反量化成 bf16 —— 多一份全尺寸 buffer、多一趟 dequant，
精度还不如直接存 bf16（那张表里 overlap=1.0 的基准**就是 bf16**）。

**int8 降级为 P5 显存预案**：bf16 每槽 256 B，int8 128 B（约 704 vs 352 B/token，
按 11 个 DSA 层折算，**推断**，未在活服务上量过）。真要启用，
`operator_handoff/specs/op1_kpool_topk_transform.md` §3 那三个条件依然成立，
但**先要解决消费者**。

---

### 2.9 P3.4 的接线地形（调研结论）

| 问题 | 结论 |
|---|---|
| 11 个 DSA 层由谁服务 | **`AscendAttnBackend`**，外面包 `AscendKDAHybridLinearAttnBackend`（`attention_registry.py:503-542`） |
| `get_indexer_metadata()` | **返回 `None`** —— `AscendAttnBackend` 不定义它，落到 `base_attn_backend.py:302` 的 ABC 默认。**kpool 那套 metadata 在 NPU 上一个都不存在** |
| 能不能改用 `DeepseekSparseAttnBackend` | 不能。① NPU 无条件把 `attention_backend` 钉成 `"ascend"`（`npu/utils.py:51-64`），DSA 那个 override 根本不触发；② 就算强行选上，`dsa_backend.py:615` 无条件调 `torch.cuda.get_device_capability()`，构造就死（实测）；③ 再修好也只拿到全零的 `kpool_write_plan`（`kpool_plan.py:741` 在 `not is_cuda()` 时直接 return）和一堆 `None` 的 `pooled_*`（`kpool_plan.py:586`） |
| KV pool 是哪个 | `HybridLinearKVPool` 包一个**普通的** `DSATokenToKVPool`。GLM 是 mambaish，而配置器里每个 NPU 分支都带 `and not self.mambaish_config`，所以全被跳过（`kv_cache_configurator.py:992/1006/1033` → `:1059` 命中） |
| 接口在哪加 | `memory_pool.py:3813` 的 `elif use_dsa:` —— 三个分支里**唯一没有 `_is_npu` 分支**的那个（MHA 在 `:3779`、MLA 在 `:3853` 都有）。这就是接缝 |
| pool 的 9 个方法 | 全都存在。tail buffer（bf16）、`slots_per_page`、`page_size`、`_is_layer_owned` 直接可用；**只有 `set_index_k_scale_buffer` 和 `kpool_decode_update_index_cache` 因 fp8 不可用** |
| 拿到 topk 之后谁做稀疏注意力 | `torch_npu.npu_sparse_flash_attention`（`ascend_backend.py:1146`）。契约（实测）：`[T, 1, K]` int32、**逻辑 token 位置**（不是物理槽位）、`-1` 补齐、**有效值必须在前**（见 §2.4）、不要求排序 |
| 已有的先例 | `dsa/dsa_npu_indexer.py:24` 的 `DSANPUIndexerMixin.forward_npu` —— 非 kpool 的 DSA indexer 在 NPU 上早就跑通了，用的正是 `npu_lightning_indexer`，而且**完全不碰 `get_indexer_metadata`** |

> 附带一个可能有用的观察（**未采用，记着**）：`npu_sparse_flash_attention` 的
> `sparse_block_size` 是**选择粒度**（仓库一律传 1），索引 `j` 选逻辑区间
> `[j*s, (j+1)*s)` 并按 `actual_seq_lengths_kv` 截尾。也就是说传 `sparse_block_size=index_kpool`
> 就能**直接喂 pool id**，省掉展开这一步。尾部语义对不对没验，先不动。

---

### 2.8 P3.4 的数值门槛：已通过（实测）

layer 3、真实权重、真实 hidden states（embed + layer 0–2 真跑）、32768 真实 token，
每个长度取最后 512 个 query 行，只比 pooled 部分（尾部不在范围内）。
**参考先对 HF 校准过**：seq=4096 的第 4095 行（1024 pool，k=512，选择真正起作用），
本地 fp32 流水线选出的 pool 集合与 `Glm5NextTextIndexer.forward` **完全一致**（overlap 1.000000）。

选中 pool 集合与 **fp32 参考**的重合率（按行取均值）：

| seq_len | pool 数 | **bf16（真算子）** | int8 absmax/127 | int8+ue8m0 | fp8 e4m3+ue8m0 |
|---|---|---|---|---|---|
| 2048 | 512 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| 4096 | 1024 | **0.99816** | 0.99538 | 0.99384 | 0.98267 |
| 8192 | 2048 | **0.99738** | 0.99302 | 0.99124 | 0.97375 |
| 32768 | 8192 | **0.99641** | 0.99021 | 0.98791 | 0.96181 |
| 32768 | score mass | **0.999995** | 0.999954 | 0.999923 | 0.999252 |
| 32768 | 被丢掉的最差 pool 距 top-k 分数跨度 | **0.0045** | 0.0198 | 0.0258 | 0.0889 |

> ⚠ 这里的基准是 **fp32**，OP-1 §2 那张表的基准是 **bf16**，所以同一个 fp8 在两张表里
> 是 0.9618 和 0.9653 —— 不矛盾，是分母不同。2048 处选择不起作用（512 pool 选 512）。

**按 ACCEPTANCE §A 的双参考法**：R32 = fp32 未旋转、fp32 算；R16 = bf16 旋转后的 q 与 pooled key、
fp32 累加、纯 torch。候选**正好落在地板上，SLACK = 1.0**：

| seq_len | 地板 logits rel-L2 | 地板 选择重合 | 候选（真算子） |
|---|---|---|---|
| 2048 | 3.21e-3 | 1.00000 | 1.00000 |
| 4096 | 3.85e-3 | 0.99816 | 0.99816 |
| 8192 | 3.70e-3 | 0.99738 | 0.99738 |
| 32768 | 3.16e-3 | 0.99641 | 0.99641 |

比表面看到的更强：**同样的 bf16 输入下，算子的选择与 fp32-torch 参考逐行逐位相同** ——
四个长度、512 行、共 262144 个选中槽位，**0 行有差异**。算子在 bf16 存储之上没有再引入误差。
**decode 与 prefill 在全部 32 个格子里选择完全相同**，说明「按可见-pool 分段」这个 prefill
写法在真实数据上也是精确的。

**没验的**：只有 layer 3、只有一个 prompt、只有每个长度最后 512 行；尾部不在范围内；
int8/fp8 是在 host 上重建后以 bf16 交给算子的（算子拒 int8），量的是量化误差而非真实 int8 缓存；
用的是单位 block table，没有活的 pool；**没有端到端精度**。

---

## 3. 阶段计划

| 阶段 | 状态 | 出口结果 |
|---|---|---|
| **P0 环境** | ✅ | 24.04 上重建；算子可见性 21 项仅 `npu_mla_prolog_v3` MISS（非阻塞）；DSv4 冒烟 + **GPQA 73.74%**（三轮 74.24/75.25/71.72，对标 73.23%） |
| **P1 分支合流** | ✅ | rebase 到 GPU 参考实现 **`033446bb05`**（tag `glm53-gpu-ref-033446bb`），19 commit / 2 冲突；回归 **GPQA 73.23%**（差 −0.50pp） |
| **P2 BF16 权重** | ✅ | 62/62 shard → **599 GB**；首 shard **27.6 亿元素逐位比对 0 处不一致**；名称/形状/dtype 全量核对通过 |

### P3 · 逐模块对拍 ☐
- [x] **P3.1 KDA** —— `attention_registry.py` 加 NPU 分支；`_flat_kda_gate` 修布局契约（Kimi 是 head-split，GLM 已 flat）。数值已验
- [x] **P3.2 mHC** —— `_mhc_pre_dispatch`/`_mhc_post_dispatch` 加 NPU 分支走 `npu_hc_pre/post`。
      **接线四坑**：kernel 内部已乘 2（外面不能再乘）、`norm_eps` 传 1e-5、输入必须 4-D、权重必须 fp32；
      `npu_hc_post` 的 `post` 必须 2-D
- [x] **P3.3 NoPE MLA** —— **已对拍**（随 DSA 整层一起，见下）。原「剩余」三项逐条核实后
      **都不需要做**：`trans_rope_weight(w,0)` 的 7 个调用点**全在** `mla_preprocess.py` 的
      MLAPO 里，被默认关闭的 `SGLANG_NPU_USE_MLAPO` 门控且本机 `npu_mla_prolog_v3` MISS，
      GLM 进不去（加 assert 也是死代码）；`fused_split_qk_norm` 被 `has_rope` 挡住；
      「20+ 处 split 早退」**真实路径上只有 2 处**（`q.split([256,0])` 与
      `latent_cache.split([512,0])`），两处都产出 0 宽张量并在交给算子前降成 `None`。
      其余 split 全在 GLM 不走的分支里。**KV buffer 二元组语义是真问题**，见 §4
- [x] **P3.5 DSA 整层（注意力本体）** —— **已对拍，注意力本体干净**。真实部署形状
      （TP16 rank0：4 个 MLA 头、32 个 indexer 头、`o_proj` 部分和、page 64、context 32k、
      decode batch 16）。参考先对 HF `Glm5NextTextAttention` 校准过（rel 1.2e-6，cos=1.0）。
      **判据用受控对比**：把「选择差异」摘掉之后，30+ 个格子的 rel 全部落在噪声地板上
      （ratio 0.97–1.05）；(a) 端到端与 (b) 受控之间的差距，完全能被单独量出的
      「选择差异成本」解释 —— **问题在选择（bf16 索引存储的已知代价），不在注意力**。
      KV cache 写入也验了（rel 2.56e-3，地板 1.66e-3）。
      回归脚本：`layer_check/{tp_fixture,reference_mla_math,reference_dsa,check_dsa}.py`
  - [ ] 未验：**不是真的 TP16**（形状真、但单 die、无 HCCL、`o_proj` 部分和未规约）；
        只有 layer 3 一条 prompt；11 个 DSA 层共享一个 pool；NPU Graph；MTP；
        radix 前缀复用；混合 batch
- [x] **P3.4 kpool indexer** —— **端到端跑通并对齐**。形态见 §2.9，数值见 §2.8。
      回归脚本：`tools/golden_kpool_indexer.py`（CPU 出 fp32 参考）+
      `tools/check_kpool_indexer_e2e_npu.py`（真机跑真实 `IndexerKPool` + 真实
      `NPUDSATokenToKVPool`，约 4 分钟）
  - [x] `rotate_activation` 的 NPU 分支；`hadamard_transform_npu` 用 matmul
        （实测 fp32 matmul 就是 fp32 精度，无降精度模式，且比蝶形快 16×）
  - [x] `compress_pool_bf16`、`NPUDSATokenToKVPool`（bf16 索引缓存 + decode 写入）
  - [x] `KPoolNPUIndexerMixin.forward_npu` —— 绕开 `BaseIndexerMetadata`
  - [x] **端到端对齐**（layer 3 真权重、打乱物理页、对 fp32 参考）：

        | seq_len | extend pool overlap | decode | 地板 |
        |---|---|---|---|
        | 2048 | 1.00000 | 1.00000 | 1.00000（全选，不具判别力）|
        | 4096 | 0.99787 | — | 0.99816 |
        | 8192 | 0.99722 | **0.99722** | 0.99737 |
        | 32768 | **0.99616** | — | 0.99641 |

        差地板 0.00025，因为这轮 q/wk/gate 全是设备上的 bf16 矩阵乘而非 CPU fp32。
        **decode 与 extend 在 8192 上数值完全相同**；decode 写入的 pool 与 extend 写入的
        **逐位相同**。键侧单独验过：写进缓存的 pooled key 对 fp32 参考 rel-L2 0.00328、
        cos 最小 0.999989 —— 压缩+旋转+页寻址整条链都对，不只是"选出来的集合像"
  - [x] **输出契约**（全部 32768 行逐行验）：int32、宽 2051、`-1` 补齐、
        **有效值严格是前缀**、有效个数 == `min(pool_len,512)*4 + seq_len%4` 逐行精确
  - [x] `append_kpool_tail_to_topk` 这个 Triton kernel 在 Ascend 上**真的跑起来了**
        —— 事先怀疑会炸，实际没有。又一次"推断出的缺口"落空
  - [ ] **仍未验证**：ACL/graph capture（§2.9 有 host sync，**推断**会失败）；
        TP>1；多 DSA 层共享 pool；未对齐 chunk 起点与 radix 前缀复用；
        overlap scheduler 下 `seq_lens_cpu` 领先设备张量一步的场景；只测了 layer 3、一条 prompt
  - [ ] **接 spec decode 前必须先解决**：`kpool_decode_update_index_cache` 假设每请求一行，
        MTP 一次多 draft token 会让同一 `req_pool_index` 的多行抢同一个 ring 槽。
        共享 CUDA kernel 有 `kpool_max_closed_pools` 那套多 token 逻辑，NPU 这条没有
  - [ ] DSA 注意力本体还缺全零 rope 的接线（§2.3 第 3 条），否则拿到 topk 也跑不出注意力

- [x] **Dense FFN** —— 端到端已验（真实 TP16 每卡形状，M=1/16/8192），最差 0.66× 预算。
      两处与预期不同：dense FFN **根本不调 `npu_clipped_swiglu`**（走 `chunk`+`clamp`+`npu_swiglu`），
      所以那个 109× 的默认参数陷阱在这条路上不适用；而且**真实输入下 clamp 从不触发**
      （max|gate_up| = 2.17，limit 是 10），要验 clamp 必须放大输入
- [ ] **P3.5 出口判据** —— 四模块逐层 golden 对齐

### P4 · BF16 端到端 ☐
- [ ] P4.1 TP16 / 32K / 纯文本 / 关 NPU Graph 启动
      **关 graph 只是 bring-up 顺序，不是终态** —— 开 graph 是 P6.6，一定要做。
      P4 关掉的理由：捕获会把「算错」变成更难查的「算错」（静态 shape/地址、
      host 侧控制流被烘进图里或直接打断捕获），而 kpool 这条路上到处是 host 侧控制流
      （`_compress_write_extend` 在 python 循环里读 `extend_seq_lens_cpu[i]`；
      `moe_runner/ascend.py:270` 的 D2H 同步）。先拿到一个可调试的正确基线。
      另外 `npu_lightning_indexer` 在捕获下能否用**尚未验证**（§2.6）；
      decode 侧形状是静态的（每 batch 一行、`sparse_count` 固定），**推断**可捕获，
      prefill 的分段数随位置变化但 prefill 本来就不捕获
- [ ] P4.2 **出口判据**：**GSM8K 97.50%**（GPU 分支 cookbook 口径：全 1319 题、stop rate 100%、4×GB300 TP4/EP4）
      - ⚠ **该数字是 thinking 打开测的**（`temperature=1.0, top_p=0.95, max_tokens=32768`，`sgl-eval run gsm8k --thinking`）
      - ⚠ cookbook 的速度数字带 `SGLANG_SIMULATE_ACC_LEN=3`，**只能当吞吐口径**
- [ ] **迭代期用更快的信号**（不是最终判据）：`tools/logit_check.py` 的 teacher-forced logprob 对拍
      （一次前向、无采样噪声、能定位）；以及 prefill-vs-decode 的 KL 一致性

### P5 · W8A8 compressed-tensors ☐
- [ ] P5.1 recipe：weight per-channel + act per-token **dynamic**（静态会被 raise）
- [ ] P5.2 ignore list 照搬 checkpoint 的 `modules_to_not_convert`：KDA 34 层全部、indexer 全套、`hc_*` 全部、所有 norm/embed/router
- [ ] P5.3 288 专家校准（覆盖度是主要风险）
- [ ] P5.4 **出口判据**：精度回归到 BF16 基线 1% 以内
- ⚠ `INF_NAN_MODE_FORCE_DISABLE=1` **必须设**，否则 W8A8 溢出产生 NaN

### P6 · 性能 ☐（可与 P3–P5 并行）
排序按实测/静态估算的影响。**前三项都不是算子开发，算子已存在。**
- [ ] P6.1 **mHC**（已在 P3.2 接线，待 profiling 确认收益）—— 原 torch 路径每 forward **90 次调用 / ~12,600 次 launch**
- [ ] P6.2 **KDA prefill conv1d** —— `causal_conv1d_fn_npu` 内部退回 `F.conv1d`（`sgl_kernel_npu` 上游的实现选择）；
      且 Ascend 后端拆成 3 次调用而共享后端只做 1 次
- [ ] P6.3 **MoE SwiGLU clamp** —— 现在是 2×clamp + `cat` + `npu_swiglu` 四个 kernel，可换成一个 `npu_clipped_swiglu`
- [ ] P6.4 DeepEP-normal 的 D2H 同步（`moe_runner/ascend.py:270-274`，prefill 每 forward 42 次）—— 修在第三方 wheel 里，先 profiling
- [ ] P6.5 NoPE 未融合的 split+RMSNorm（与 P3.3 同源，一起做）；顺带删掉那个看起来是死代码的 `q.clone()`
- [ ] **P6.7 kpool indexer 的 expand+tail**（实测，单层单 4096-chunk 的最大单项）——
      `expand_pooled_groups_to_topk` 中间物化了 `[4096, 512, 4]` 的 int64（67 MB）再 reshape，
      占 6.3 ms / 单层。11 个 DSA 层 → 每个 4096-token chunk 约 69 ms。**在共享代码里**
      （`kpool_fp8_index.py:379`），改动会影响 CUDA 路径，先 profiling 再决定
- [ ] P6.6 NPU Graph —— **decode 路径已扫清 host 同步**（实测 `timing.count_syncs`：
      kpool 的 decode 缓存更新 **0 次/调用**）。做法：decode 跳过 `visible_pool_runs`
      （每行本来就自成一段），并把缓存更新改成无分支——**不过滤行，而是给被屏蔽的行
      一个 scratch 目标**。⚠ 这里有个真陷阱：屏蔽行的 `req_pool_indices` 会被 clamp，
      padding 行通常带 0，于是和真实请求 0 撞同一个槽位，**重复下标的写入顺序未定义，
      真实行的写入可能被覆盖**——而这恰恰只在图捕获（padding batch）下发生。
      所以索引缓存多分配一页、tail ring 多分配一行，专供屏蔽行落地。
      **extend 仍有同步**（`visible_pool_runs` 里的 `int(...max())`、
      `_kpool_extend_rows_npu` 的 host 侧构造），但 prefill 本来就不捕获，不需要动
- ⚠ **所有性能数字目前都是静态推算**，端到端跑通后必须用 profiling 重排序

---

## 4. 待决与已知缺陷

- [ ] **P5 的磁盘**：BF16(643) + W8A8(333) = 976/984 GB → 必须先删 FP8 源
- [ ] **索引缓存改 bf16 后显存翻倍**：每槽 256 B（打包 fp8 是 128+4=132 B）。
      按 11 个 DSA 层折算约 704 vs 363 B/token，相对 MLA KV 的约 11.3 KB/token 是 +3% 量级。
      **这是从 `mem_cache/index_key_cache.py:33-38` 的 buffer 形状推算的，未实测**；
      P4 起服务时量一次。真顶不住的退路见 §2.7
- [ ] **两个会挡住对拍的精度缺陷**（发现但未修）：
      ① DeepEP routed 专家路径**静默丢掉 `swiglu_limit=10.0`**（`moe_runner/ascend.py:114-118` 只读
      `gemm1_clamp_limit`，GLM 为 None）→ 同层内 shared 专家 clamp 而 routed 不 clamp；
      ② NPU router GEMM 走 bf16（`deepseek_v2.py:567-568`），而 GLM 配置是 `moe_router_dtype: float32`
- [ ] **DSv4 的一处潜在 bug**：bf16 fallback 读 int8 buffer 却**不施加 scale**（`ascend_dsv4_backend.py:637-641`），
      仅因 `:685` 无条件强制 int8 而不可达
- [ ] **triton-ascend 的 `_hadamard128` codegen 缺陷**（UB 越界，上下文相关）—— 值得上报。
      在三个 compress kernel 内部正常，独立跑必挂；第 4 个 kernel
      （`_kpool_write_tail_and_maybe_compress_kernel`）只在 `num_draft_tokens >= 3` 触发它。
      **一期不在关键路径上**：那是 MTP，一期不做；真撞上就删 Hadamard（bf16 下可删，见 §2.4）
- [ ] `deep_ep` 的打包 bug 已用 `.pth` 绕过，可向上游反馈
- [ ] **`get_kv_buffer()` 返回的二元组语义随 pool 类而变，而 `forward_sparse` 只按一种解读**
      （源码）。`NPUMLATokenToKVPool:615` 返回 `(k[512宽], v[rope宽])` = (nope, rope)；
      而 `NPUDSATokenToKVPool` 继承的是**共享的** `MLATokenToKVPool:4437`，返回
      `(整块 [N,1,kv_cache_dim], 同一块的 [...,:kv_lora_rank] 切片)` = (融合, nope)。
      **GLM 没事纯粹因为 `qk_rope_head_dim == 0`**——融合 buffer 恰好就是 512 宽，
      而 `k_pe` 反正被零页替换。**换一个带 rope 的 DSA 模型走同一个 pool，
      `key_rope` 会被喂进 nope 切片，宽度和内容都错且不报错。**
      这不是 bug，是一个没写下来的跨类不变量；建议在 pool 基类上显式化，
      或让 DSA pool 覆盖 `get_kv_buffer` 返回 `(nope, rope)`
- [ ] **`do_cp_balance_attn`（`ascend_backend.py:937/959`）有与已修的 `forward_sparse`
      完全相同的 3-D/PA_BSND 缺陷**：开 prefill CP 就会撞 `561002`。一期不开 CP，
      驱动不了因此没改——**开 CP 前必须先修**
- [ ] **`NPUMLATokenToKVPool.set_kv_buffer:679` 在检查 `cache_v is None` 之前就
      `cache_v.to(...)`**（源码）。GLM 走的是共享实现所以碰不到，但这条路真被走到会
      直接 `AttributeError`
- [ ] **预热**：DSA 单层 decode 首次 45.3 ms、稳态 5.6 ms（实测），说明有明显的
      tiling/编译预热。真实启动脚本带 `--skip-server-warmup`，**P4 起服务前复查**
- [x] ~~能不能让 qk_rope=0 走非 yarn 分支~~ —— **问题是空的**：那条分支 GLM 根本进不去，
      不需要改条件。OP-3 已撤销（§2.5），**工单包清空**。已在
      `deepseek_v2_attention_mla_npu.py` 的分支上方留了一行说明这个跨文件不变量
