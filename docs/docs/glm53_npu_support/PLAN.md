# GLM-5.3-Flash 昇腾 (Atlas A3 / Ascend910_9362) 适配计划

> 活文档。每完成一步就在对应条目标 `[x] PASS` 或 `[!] FAIL`，FAIL 的直接在本文件改计划。
> 最后更新：2026-08-27

---

## 0. 目标与已定决策

| 项 | 决定 | 决定时间 |
|---|---|---|
| 一期目标 | GLM-5.3-Flash **纯文本** 在 A2 上跑通并闭环精度 | 2026-08-27 |
| BF16 部署形态 | **单节点 TP16**（16 die × 64 GB = 1024 GB） | 2026-08-27 |
| 量化格式 | **compressed-tensors W8A8-INT8**（weight per-channel + act per-token dynamic） | 2026-08-27 |
| 磁盘策略 | FP8 → BF16 **逐 shard 转换、转完即删源 shard** | 2026-08-27 |
| 精度基准 | CPU 上跑 GPU 分支参考实现取 golden | 2026-08-27 |
| 多模态 (ViT/video) | 一期**不做** | 2026-08-27 |
| MTP / NextN | 一期**不做** | 2026-08-27 |
| 长上下文 CP | 一期**不做**，只承诺 32K | 2026-08-27 |

---

## 1. 已核实的环境事实

### 硬件
- **SoC = `Ascend910_9362`（910_93 = Atlas A3）**，不是 A2/910B。
  - 证据：`torch.npu.get_device_name(0)` → `Ascend910_9362`；`npu-smi info -t board -i 0` → `Product Name: IT22HMDA_4_S`, `PCI Device ID 0xD803`；`/dev/davinci0..15`
  - ⚠ 2026-08-27 修正：初版按 A2/910B 判断，错误。所有 sgl-kernel-npu 包必须选 **a3** 档
- **16 × Ascend910 die**（npu-smi: 8 NPU × 2 chip），每 die **64 GB HBM** → 合计 1024 GB
- CPU **320 核**，内存 **1.8 TB**
- 磁盘 `/mnt/workspace` 984 GB 总量 / **677 GB 可用**（306 GB 已被 FP8 权重占用）
  - 其他挂载：`/usr/.devenv` 465 GB 可用、`/home` 179 GB 可用

### CANN（关键：标称与实际不一致）
- `ascend_toolkit_install.info` 写 `version=9.2.0`，路径 `/home/developer/Ascend/cann-9.2.0`
- **但所有组件的 `version.info` 实际都是 `9.1.0`**（timestamp 20260715）：
  `compiler=9.1.0`、`opp=9.1.0`、`bisheng-compiler=9.1.0`、`hccl=9.1.0`、`runtime/ge/metadef=9.1.0`
- → **本机等价于 CANN 9.1.0。"9.2 还是 9.1" 这个问题不存在，我们就在 9.1.0 上。**
- driver 在 `/usr/local/Ascend/driver`；toolkit 在 `/home/developer/Ascend/ascend-toolkit/`
- **`opp/vendors/` 初始为空** → `torch.ops.custom.*` 全部缺失。已通过 `--install-path` 装到独立目录 `opp_custom/` 解决（不污染共享 toolkit）
- ⚠ **glibc 门槛**：sgl-kernel-npu 的预编译 `.so` 需 **GLIBC ≥ 2.34 + GLIBCXX ≥ 3.4.29**（CI 用 Ubuntu 22.04 编）。
  - 与 CANN 版本**无关** —— `cann9.0.0` 档实测同样需要 2.32/2.34
  - Ubuntu 20.04（glibc 2.31）上需附录 B 的独立 loader 绕行（已验证可行）
  - **Ubuntu 24.04（glibc 2.39 / libstdc++ 13）无此问题**
- ⚠ 修正：系统 python3.11 里**已装 torch 2.7.1+cpu / torch_npu 2.7.1.post4**，`source /home/developer/Ascend/ascend-toolkit/set_env.sh` 后可 import，`device_count=16`。
  → **现在就能做运行时算子探测**，不必等新 venv。（初版写"torch_npu 未安装"，是漏了 set_env.sh）

### sgl-kernel-npu 发布矩阵（github.com/sgl-project/sgl-kernel-npu，最新 tag `20260826`）
只发 **cann9.0.0** 和 **cann9.1.0** 两档，**没有 9.2.0**。本机（**a3** 档，非 910b）对应这三个包：

| 包 | 文件名 | 大小 |
|---|---|---|
| custom-ops（提供 `torch.ops.custom.*`） | `custom-ops-20260826-torch2.10.0-cann9.1.0-**a3**-aarch64.zip` | 56.5 MB |
| ops-transformer | `ops-transformer-20260826-torch2.10.0-cann9.1.0-**a3**-aarch64.zip` | 6.3 MB |
| sgl_kernel_npu（python） | `sgl-kernel-npu-20260826-torch2.10.0-**py312**-cann9.1.0-**a3**-aarch64.zip` | 12.7 MB |

**约束（与安装文档矛盾，以 release 为准）**：
- cann9.1.0 档 → **Python 3.12**（cann9.0.0 档才是 py311）
- torch **2.10.0** 固定

### 网络
- 代理 `http://127.0.0.1:1056` **只对 GitHub / Anthropic 有效**
- 其他站点必须先 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`：
  pypi.tuna ✅ / mirrors.huaweicloud.com(ascend) ✅ / hf-mirror ✅ / gitcode ✅ / modelscope ✅ / pypi.org ✗
- chatgpt.com 两种方式都不可达

### 代码基线
- fork 分叉点 `eea2e5d6e5`（2026-08-13），我们有 **18 个 NPU commit / 18 个文件**（DeepSeek-V4-Flash + KT CPU offload）
- **GPU 参考实现已在本地 git 对象**：`0b9c38484e`（ref `upstream/xinyuan/glm-5.3-flash-support`），143 文件 / +16,647 行
  - 比我们的 base 多 **697 个 commit**，改动 2519 个文件
  - 与我们 18 个文件的**重叠只有 10 个** → rebase 可行
- commit message 标注：verified on 4×GB300 (TP4/EP4) 与 8×H100 (TP8/EP8)

### 权重
`/mnt/workspace/models/GLM-5.3-Flash`，62 个 safetensors，306 GiB

| dtype | 体积 | tensor 数 |
|---|---|---|
| F8_E4M3 | 292.80 GiB | 37338 |
| BF16 | 12.90 GiB | 1141 |
| F32 | 0.07 GiB | 37629 |

- BF16 展开后 **598.6 GiB ≈ 643 GB** → TP16 下每 die 37.4 GiB，余 ~26 GiB
- **37338 个 fp8 权重的 `weight_scale_inv` 全部与权重同分片，0 个跨分片** → 逐 shard 独立反量化安全

### 模型结构（来自 config.json，已核）
- 45 层 = 34 KDA(`linear_attention`) + 11 `deepseek_sparse_attention`（层号 3,7,11,…,43）
- FFN：前 3 层 dense + 后 42 层 MoE（288 routed + 1 shared，top-8，`moe_intermediate_size=2048`）
- NoPE MLA：`qk_nope_head_dim=256, qk_rope_head_dim=0, v_head_dim=256, kv_lora_rank=512, q_lora_rank=1536, num_attention_heads=64`，`mla_use_nope=true`
- KPool indexer：`index_n_heads=32, index_head_dim=128, index_topk=2048, index_kpool=4, index_kpool_compress=true, index_kpool_always_select_tail=true`
- KDA：`num_heads=64, head_dim=128, short_conv_kernel_size=4, gate_lower_bound=-5.0`
- mHC：`mhc=true, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6`
- `swiglu_limit=10.0`，`max_position_embeddings=1048576`
- **TP16 整除性全部 OK**：64 / 64 / 32 / 288 都能被 16 整除，无需 pad

---

## 2. 算子现状与缺口

> 本章只讲算子。每条给**明确结论**；证据不足的一律标 **❓不确定**，并写清用什么动作消解。
> 证据来源：CANN 9.1.0 组件的 aclnn 头 / op-info json / AscendC 源码 / `libopapi_transformer.so` 字符串表；
> `torch_npu 2.7.1.post4` 运行时签名与实跑；本仓库昇腾代码；GPU 参考实现 `0b9c38484e`。
> ⚠ 运行时探测跑在 torch_npu **2.7.1**，目标是 **2.10.0**。P0 venv 建好后需复跑。

### 2.1 一条方法论前提（影响很多结论）

昇腾侧算子有**三个来源**，混淆它们会得出错误结论：

| 来源 | 命名空间 | 本机状态 |
|---|---|---|
| torch_npu 原生 | `torch_npu.npu_*` | ✅ 已装（2.7.1.post4） |
| CANN vendor 自定义算子包 | `torch.ops.custom.*` | ❌ **未装**（`opp/vendors/` 空）→ P0.3 |
| sgl-kernel-npu python 包 | `sgl_kernel_npu.*` | ❌ **未装** → P0.3 |

**本仓库的昇腾 KDA / DSA 路径主要走后两者，不走 torch_npu 原生。**
所以"某个 `torch_npu.npu_xxx` 不支持某特性"**不能**直接推出"这是缺口"。

### 2.2 已确认可用（不需要开发）

| # | 算子 / 能力 | 结论 | 证据 |
|---|---|---|---|
| A1 | **NoPE MLA 的 attention core** | ✅ **实测可跑**。`npu_fused_infer_attention_score` 接受 `qk=256/v=256/N=64`（prefill 形态，→`[1,64,256,256]`）与 `kv_lora=512 + rope=0/N=64`（MLA-absorbed decode，→`[1,64,1,512]`） | 本机实跑，`probe/p0_6_shapes.py` |
| A2 | **稀疏 attention 的 rope 可选** | ✅ 签名确认：`npu_sparse_flash_attention(..., Tensor? query_rope=None, Tensor? key_rope=None, ...)`。**rope=0 直接不传，不需要"传 64 维全零"的 workaround** | torch_npu 运行时 doc。⚠ 实现在 vendor 包，**尚未实跑** |
| A3 | **pooled-key 打分（KPool 的打分环节）** | ✅ **vendor 包已提供且我们已在用**：`torch.ops.custom.npu_quant_lightning_indexer(..., cmp_ratio=4, sparse_count=2048, sparse_mode=3)` 就在 DSv4 生产路径里 | `ascend_dsv4_backend.py:1002-1021` |
| A4 | **KPool 的压缩写 cache** | ✅ vendor `torch.ops.custom.compressor` 已带 **fused norm + RoPE**，且 AscendC 实现（`AddApeToScore → ColumnSoftMax(逐 head-dim 列) → Mul → ColumnSum`）与 GLM 的 kernel **同构**。差异只有 norm 类型 → 见 B1 | `ascend_dsv4_backend.py:411-430`；AscendC `compressor/arch22/` |
| A5 | **mHC pre / post** | ✅ 算子存在：`torch.ops.custom.npu_hc_pre` / `npu_hc_post`，DSv4 在用。GLM-5.3 侧只缺 dispatch 分支（**代码工作，非算子**） | `kernels/ops/layernorm/mhc.py:1605`；`models/deepseek_v4.py:1713` |
| A6 | **MoE router / GMM** | ✅ `npu_moe_gating_top_k`、`npu_grouped_matmul` 已绑定，A2/A3 支持 | torch_npu doc |
| A7 | **Hadamard-128 可省略** | ✅ 它是正交归一阵（`* 0.08838834764831845` = 1/√128），q 与 k 同时旋转 → 点积严格不变。走 bf16 indexer 时**数学等价，可整体删掉** | `kpool_fp8_index.py:862-870`；`dsa_indexer_kpool.py:656` |

### 2.3 确认要新开发

| # | 算子 | 结论 | 证据 |
|---|---|---|---|
| B1 | **compressor 的 LayerNorm 变体** | 需要开发，但**范围很小**：vendor `compressor` 的 fused norm 是 **RMSNorm**（DSv4 用），GLM 的 index-K norm 是**真 LayerNorm（减均值 + bias）**。是"给已有算子扩一个 norm 类型" | `layers/layernorm.py:974-1014` GLM 走 `F.layer_norm(bias=True)`；`ascend_dsv4_backend.py:411` 传 `_fused_norm_weight_fp32` |
| B2 | **GLM 版 clipped SwiGLU（非对称 clamp）** | 需要开发。`npu_clipped_swiglu` 是 **gpt-oss 语义**（`alpha=1.702, bias=1.0, interleaved=True`），GLM 是 `silu(clamp(gate, max=L)) * clamp(up, -L, L)`。本仓库早已踩过并用 `torch.clamp_` 兜底 | torch_npu 签名；`glm5_next.py:139-143`；`hardware_backend/npu/moe/activation.py:93-96,116` |
| B3 | **KPool 的 pool→raw 展开 + 尾部追加** | 需要开发（或用 torch 实现）。这部分是 GLM 特有的索引后处理，`compressor` / `lightning_indexer` 都不负责 | `kpool_fp8_index.py:379-401 expand_pooled_groups_to_topk`、`:421+ append_kpool_tail_to_topk` |

### 2.4 不确定项的消解结果（2026-08-27 二次核实）

**消解手段**：C1/C2 靠**读 wheel 里的 Python/Triton 源码**（不需要执行）；C4 靠**运行时实跑**（torch_npu 2.10 已可用）；
C3/C5 依赖 vendor 包，被 GLIBC 挡住（见 §1 环境）。

| # | 问题 | **结论** | 证据 |
|---|---|---|---|
| C1 | K=4 stateful causal conv1d 缺不缺 | ✅ **不缺，解决**。`sgl_kernel_npu.mamba.causal_conv1d` 是 **triton-ascend** kernel，`KERNEL_WIDTH` 是 `tl.constexpr`，**1/2/3/4/5/6 全有显式分支**（`==4` 在 363-400 与 444-458 两处），`width, _ = weight.shape` 从权重读，非硬编码 | `sgl_kernel_npu/mamba/causal_conv1d.py:66,387,449,550` |
| C2 | featurewise-gate KDA 缺不缺 | ✅ **不缺，解决**。`fused_kda_gate_npu(gate, A_log, head_dim, gate_bias, lower_bound)`：`gate` 是 `[tokens, heads*head_dim]` 的**逐通道** gate（`heads=A_log.numel()`，`heads*head_dim==hidden` 有断言），`gate_bias.numel()==hidden` 也是逐通道，且**带 `lower_bound` 参数**。与 GLM 的 `f_b_proj→[T,64*128]` + `dt_bias[8192]` + `gate_lower_bound=-5.0` 完全对得上。prefill 侧 `chunk_gla_fwd_o_gk_npu` 的 `gk` 也是逐通道命名 | `sgl_kernel_npu/fla/kda_gate.py:74-94`、`fla/kda_prefill.py:295` |
| C4 | `npu_kv_rmsnorm_rope_cache_v2` 支持 rope=0 吗 | ❌ **不支持，解决**。**实跑验证**：`rope=64` 时 v1/v2 都 [OK]；`rope=0`（cos/sin/k_cache 传 0 宽）时 v1/v2 **都 RuntimeError**。→ D2 拆 `rmsnorm + reshape_and_cache` 是确定要做的活 | `probe`，`aclnnKvRmsNormRopeCache*` 报错 |
| C3 | `MlaPreprocess` 的 `rope_dim=0` 是否合法 | ⏸ **仍不确定，但已降级**。`mla_preprocess` **不在 `torch.ops.npu` 里**（torch_npu 2.10 无此绑定），它来自 vendor 包 → 被 GLIBC 挡住，现在探不了。**但它是性能项**（`SGLANG_NPU_USE_MLAPO` 默认关），不阻塞 BF16 打通 | `torch.ops.npu` 无 `mla_preprocess`；`mla_preprocess.py:386` 调 `torch.ops.npu.mla_preprocess` |
| C5 | 分组 top-k 是否要自研 | ⏸ **很可能不用，但要跑起来才能确认**。`aclnnQuantLightningIndexer` 有 `cmpRatio` + `sparseCount` + `sparseIndicesOut/sparseValuesOut`，且我们 DSv4 路径已用 `cmp_ratio=4`。**未确证的一点**：`cmp_ratio=4` 时 `sparseIndicesOut` 返回的是 **pool 下标**还是 **raw token 下标**。DSv4 是 compress-then-attend（拿到就直接喂 `cmp_ratio=4` 的 sparse attn，不展开），GLM 是 pool-score-then-expand（要展开回 raw token） | `aclnn_quant_lightning_indexer.h`；`ascend_dsv4_backend.py:1013-1021` |

### 2.4.1 vendor 包的算子分布（2026-08-27 实测解包）

装包时才发现算子分散在两个 vendor 里，且**只有部分编译了 `ascend910_93`（我们的 SoC）**：

| 包 | 提供的算子 | 编进 `ascend910_93` 的 |
|---|---|---|
| **custom-ops** | `dequant_swiglu_clamp_quant, gather_selection_kv_cache, **hc_post, hc_pre, hc_pre_inv_rms, hc_pre_sinkhorn**, indexer_compress_epilog, inplace_partial_rotary_mul, kv_compress_epilog, moe_gating_top_k_hash, moe_init_routing_group_quant, partial_rotary_mul_quant, rms_norm_dynamic_quant, scatter_nd_update_asc, swiglu_clip_quant, swiglu_group_quant` | `dequant_swiglu_clamp_quant, gather_selection_kv_cache, **hc_post, hc_pre, hc_pre_inv_rms, hc_pre_sinkhorn**, inplace_partial_rotary_mul, moe_gating_top_k_hash, rms_norm_dynamic_quant, scatter_nd_update_asc, swiglu_clip_quant` |
| **ops-transformer** | `compressor, quant_lightning_indexer(+metadata), sparse_attn_sharedkv(+metadata)` | 全部有 |
| **attentions**(wheel) | `ada_block_sparse_attention, laser_attention, sparse_block_estimate` | 全部有 |

→ **mHC 的 `hc_pre`/`hc_post` 确认为 A3 编译**（A5 结论坐实）。
→ **A5-only（有头文件但无 a3 kernel）**：`indexer_compress_epilog`、`kv_compress_epilog`、`moe_init_routing_group_quant`、`partial_rotary_mul_quant`、`swiglu_group_quant`。
→ ⚠ **但这三个包的 `.so` 在本机全部因 GLIBC 无法 dlopen**（已实测，见 §1）。

### 2.5 已排除的路线（不要再走）

| 路线 | 为什么排除 |
|---|---|
| `aclnnMixedQuantSparseFlashMla` | 约束串 `rope_head_dim should only be 64, but got %lld` → **0 非法**；且 AscendC 只有 `arch35`，未在任何 SoC 的 ops-info 注册 → **A3 无二进制** |
| `aclnnMlaPrologV3WeightNz` 直接 rope=0 | `ropeSin`/`ropeCos`/`krCacheRef`/`queryRopeOut` 全部**非 Optional**，无放宽变体 |
| "传 64 维全零 RoPE" 的 workaround | 不必要 —— 见 A2，rope 参数本就是 Optional |

### 2.6 决策表：要开发 / 不确定的，BF16 首次打通能不能用 torch 绕过

> "用 torch 绕过" = 先用 SGLang 里现成的 PyTorch 参考实现或手写 torch 算子把功能跑通，**只求精度对齐、不求性能**，把 AscendC 开发推到 P6。
> ⚠ 这一列是**我的判断**，不是实测结论。标注了把握程度。

| 项 | 状态 | BF16 打通能否用 torch 绕过 | 怎么绕 / 代价 | 把握 |
|---|---|---|---|---|
| **B1** compressor LayerNorm 变体 | 确认要开发 | **能** | index-K 的 LayerNorm 用 `F.layer_norm` 在算子外面先做，再把已 norm 的 K 喂给 `compressor`（norm 权重传单位向量 / 或走不带 norm 的路径）。**前提是 vendor `compressor` 允许"norm 已在外面做过"**——这点❓不确定，要看它的参数是否可绕过内建 norm | 中 |
| **B2** GLM 版 clipped SwiGLU | 确认要开发 | **能，且已有先例** | 本仓库 `moe/activation.py:116` 对 DSv4 就是用独立 `torch.clamp_` 兜底的，照抄即可。纯 torch，无精度损失 | **高** |
| **B3** KPool pool→raw 展开 + 尾部追加 | 确认要开发 | **能** | GPU 侧这两个函数本质是 `arange/gather/where`（`expand_pooled_groups_to_topk` 就是 `.expand(-1,-1,pool_size)`）。用 torch 重写很直接，只是 decode 时每步都跑会慢 | **高** |
| **C1** K=4 causal conv1d | ❓不确定缺不缺 | **能（若真缺）** | prefill 用 `torch.nn.functional.conv1d` 保正确；decode 需要维护 K-1=3 个 token 的 state，用 torch 手写 rolling buffer。慢但可跑 | **高** |
| **C2** featurewise-gate KDA | ❓不确定缺不缺 | **能（若真缺）** | SGLang 有纯 torch/triton 的 KDA 参考路径（`layers/attention/linear/kernels/`）。但 triton 在昇腾要靠 triton-ascend，❓可用性未验证；退到纯 torch 会**非常慢**（34 层线性注意力） | 中 |
| **C3** MlaPreprocess rope_dim=0 | ❓不确定 | **不需要绕** | 融合 prolog 默认关（`SGLANG_NPU_USE_MLAPO`），BF16 打通走非融合路径。这是纯性能项 | **高** |
| **C4** kv_rmsnorm_rope_cache_v2 支持 rope=0 | ❓不确定 | **能** | 拆成 `npu_rms_norm` + 手写 `reshape_and_cache`（或 `index_copy_`）。就是 P3.3 那块改造 | **高** |
| **C5** 分组 top-k | ❓不确定要不要自研 | **能** | 先解掉 `kpool_fp8_index.py:588` 的非 CUDA 硬拦，用 `torch.topk` 选 512 个 pool。prefill 上 logits 矩阵大，会慢，但功能正确 | **高** |

**汇总判断（我的看法，非实测）**：
**BF16 首次打通不被任何算子硬卡住** —— 8 项里 6 项有高把握的 torch 绕过路径，2 项（B1 / C2）是中等把握。
最大的不确定是 **C2**：如果 sgl_kernel_npu 的 KDA 不支持 featurewise gate，34 层线性注意力退到纯 torch 会慢到影响调试效率（不是跑不通，是跑得难受）。**P0.5 应该优先验 C2 和 C1。**

**DSv4 冒烟（P0.7）不涉及以上任何一项** —— DSv4 用的是 RMSNorm compressor + 有 RoPE 的 MLA + DSv4 自己的激活路径，只需要 vendor 包装对。


### 2.7 SGLang 侧改造点（**不是算子问题**，但同属 NoPE 工作面）

这几条是纯代码改造，无算子风险，列在这里免得被 §2.2 的"算子可用"误导成"没事干"：

| # | 断点 | 说明 |
|---|---|---|
| D1 | `trans_rope_weight(w, 0)` **会静默损坏权重** | `mla_preprocess.py:59-63`。Python 里 `-0 == 0`，`weight[..., -0::2, :]` 是**整个张量 stride-2 取**（不是空张量），`weight[..., -0:, :] = ...` 是**全量覆写**。本期虽不走这条路（融合 prolog 默认关），但**谁打开 `SGLANG_NPU_USE_MLAPO` 就是无声精度事故** → 加 assert |
| D2 | `npu_kv_rmsnorm_rope_cache` 是结构性依赖 | `deepseek_v2_attention_mla_npu.py:93`，一个算子同时做 RMSNorm(kv_a) + RoPE(k_pe) + 写 ckv_cache + 写 k_rope_cache。rope=0 时输入形状 / `k_rope_cache` / `cos,sin` 全部失配 → 拆成 `npu_rms_norm` + reshape_and_cache（除非 C4 成立） |
| D3 | 20+ 处 `q.split([qk_nope_head_dim, qk_rope_head_dim])` | rope=0 时产出空张量再喂 `npu_interleave_rope` / `npu_rotary_mul` → 逐处早退 |
| D4 | KV cache 从 `512+64=576` 变 `512` | `get_kv_buffer()` 返回的 `(ckv_cache, k_rope_cache)` 二元组语义要改 |
| D5 | `attention_registry.py:503-504` 无 NPU 分支 | GLM-5.3 分支目前落到 CUDA 的 `KDAAttnBackend`。照抄同文件 494-500 行 Kimi-K3 的写法即可 |
| D6 | `_mhc_pre_dispatch` / `_mhc_post_dispatch` 无 NPU 分支 | 算子有（A5），缺分支 |
| D7 | kpool 的非 CUDA 硬拦 | `kpool_fp8_index.py:588`、`dsa_indexer_kpool.py:1766` 直接 `raise NotImplementedError` |

### 2.8 与原评估文档的差异（工作量口径）

原文档 `GLM53_flash_ascend_support_assessment.html` 中，以下**工作量判断**经核实需要修正：

| 原文档 | 核实结论 |
|---|---|
| kpool 15–25 人日，"`kpool_fp8_index.py` 是可直接移植的普通 PyTorch" | 该文件含 **14 个 `@triton.jit`**，只有 `expand_pooled_groups_to_topk` 是纯 arange/gather/where。**但**打分（A3）与压缩（A4）昇腾已有，实际缺口是 B1+B3+C5 |
| mHC 10–18 人日 | **高估**。算子已有，只缺 dispatch 分支（D6） |
| "mla_preprocess 需无 RoPE 分支" 列为 P0 | **是 P6 性能项**（C3） |
| "qk/v_head_dim=256 突破现有 tiling 假设" | **不成立**。实测 `qk=256/v=256/N=64` 直接可跑（A1） |
| 量化推荐 msModelSlim | compressed-tensors W8A8-INT8 已在昇腾接好且是 **NPU 专属路径**（非 NPU 直接 raise）：`compressed_tensors.py:687-708, 797-799` |

---

## 3. 阶段计划

### P0 · 环境搭建 ✅ 基本完成（详细步骤见 [SETUP.md](./SETUP.md)）

- [x] **P0.1 PASS** Python 3.12.14 venv
- [x] **P0.2 PASS** torch **2.10.0+cpu** + torch_npu **2.10.0.post4**，`device_count=16`，`Ascend910_9362`，NPU bf16 matmul 通过
- [x] **P0.3 PASS** sgl-kernel-npu `20260826` 三件套（**cann9.1.0-a3-aarch64 / py312**）+ triton-ascend **3.2.2**
- [x] **P0.4 PASS** 未执行 `pip install -e python/`（会装 CUDA 变体顶掉 torch_npu），走 `PYTHONPATH`
- [x] **P0.5 PASS** 算子可见性（`probe/p0_5_ops.py`）：
      - `torch.ops.custom.*` **9/10 可见**：`compressor`、`npu_hc_pre`、`npu_hc_post`、`npu_quant_lightning_indexer(+_metadata)`、`npu_sparse_attn_sharedkv(+_metadata)`、`npu_moe_gating_top_k`、`inplace_partial_rotary_mul`
      - KDA/conv1d **6/6 可导入**：`causal_conv1d_fn_npu`、`causal_conv1d_update_npu`、`fused_kda_gate_npu`、`chunk_gla_fwd_o_gk_npu`、`chunk_gated_delta_rule_fwd_h_npu`、`kda_target_verify_npu`
      - 三项已确认不阻塞：`attentions`（只被 diffusion 用）、`deep_ep`（只被 DeepEP 分发用，**TP16+EP 阶段要回头解决**）、`torch.ops.custom.npu_mla_prolog_v3`（融合 prolog 默认关，且 `torch_npu.npu_mla_prolog_v3` 原生存在）
- [x] **P0.6 部分 PASS**（`probe/p0_6_shapes.py`、`probe/p0_6_rope0.py`）：见 §2.6
- [ ] **P0.7 出口判据（两级，未做）**：
  - [ ] P0.7a 冒烟：DeepSeek-V4-Flash W8A8 起服务 + 单条推理（验 mHC + compressor + compressed-tensors 三条路）
  - [ ] P0.7b 精度：GPQA-Diamond，**non-thinking**，**repeat 3**，**多 batch 并发**
  - 执行方式：派 agent，不占主上下文
- [ ] **P0.8 换机后需重做**：本节全部步骤照 SETUP.md 重跑；**Ubuntu 24.04 上可跳过附录 B 的 glibc 绕行**

> DSv4 W8A8 权重此前因 `-rw-r-----` 权限读不了，需 owner `chmod -R a+rX`。P0.7b 通过后可删（回收 275 GB）。

### P1 · 分支合流 ☐
- [ ] P1.1 把 18 个 NPU commit rebase 到 `0b9c38484e`（冲突面 10 个文件）
- [ ] P1.2 **出口判据**：新 base 上 DSv4-Flash 回归通过
> 注意：697-commit 窗口里 `ascend_dsv4_backend.py` 被改了 1187/2179 行，`dsv4_allocator/memory_pool/common_hooks` 全部重构，新增 `extra_ops_loader.py`。这些文件我们没动过，不会冲突，但行为要重验。

### P2 · BF16 权重 ☐
- [ ] P2.1 写逐 shard 反量化脚本（`weight_block_size=[128,128]`）
- [ ] P2.2 **第一个 shard 转完先不删，人工确认数值**
- [ ] P2.3 全量转换 + 逐 shard 校验 + 删源 shard
- [ ] P2.4 清洗 `config.json`（删 `quantization_config`）
- [ ] P2.5 **出口判据**：BF16 目录 tensor 数与 shape 全量比对通过
> 风险：删源不可逆。FP8 可从 hf-mirror 重下（306 GB 代价）。

### P3 · 逐模块对拍 ☐
按依赖顺序，CPU golden vs NPU。
- [ ] P3.1 **KDA 层**：`attention_registry.py:503-504` 加 NPU 分支路由到 `AscendKDAAttnBackend`（照抄 Kimi-K3 的 494-500）
- [ ] P3.2 **mHC**：`_mhc_pre_dispatch`/`_mhc_post_dispatch` 加 NPU 分支。核对 `post_mult_value=2.0` 与 kernel 内部一致性、`(post_mix, comb_mix)` ↔ `(post, comb)` 映射
- [ ] P3.3 **NoPE MLA**：拆 `npu_kv_rmsnorm_rope_cache` + 20 处 split 早退 + KV buffer 二元组语义 + `trans_rope_weight` assert
- [ ] P3.4 **kpool indexer**：解掉 `kpool_fp8_index.py:588` 与 `dsa_indexer_kpool.py:1766` 的非 CUDA 硬拦，用 `torch.topk` 打通 `group_topk=512`
- [ ] P3.5 **出口判据**：四个模块逐层 golden 对齐

> ⚠ P3.4 的 golden 问题：GPU 分支的 kpool 在 CPU 上也跑不起来（`dsa_indexer_kpool.py:1766` 直接 raise）。得先写 torch 参考路径，它既是被测对象又是基准 → 用小规模手算 + 与非 kpool 普通 DSA indexer 交叉验证来锚定。

### P4 · BF16 端到端 ☐
- [ ] P4.1 TP16 / 32K / 纯文本 / 关 NPU Graph / 关 MTP / 关 CP 启动
- [ ] P4.2 **出口判据**：GSM8K 对齐 CPU golden 与 GPU 分支公开口径

### P5 · W8A8 compressed-tensors ☐
- [ ] P5.1 llm-compressor recipe：weight per-channel + act per-token **dynamic**（静态会被 raise）
- [ ] P5.2 ignore list 照搬 checkpoint 的 `modules_to_not_convert`（已导出 79 条模式）：
      KDA 34 层全部权重（含 `A_log`/`dt_bias`/`*_conv1d`）、indexer 全套、`hc_*` 全部、所有 norm/embed/router。ViT 本期不涉及
- [ ] P5.3 288 专家校准（覆盖度是主要风险）
- [ ] P5.4 **出口判据**：精度回归到 BF16 基线 1% 以内

### P6 · 算子与性能 ☐（可与 P3–P5 并行开工）
按 §2.5 核实后的**必须自研最小集**排序：

- [ ] P6.1 **K=4 stateful causal Conv1d**（prefill + decode/MTP）—— 确定缺口（G5）。先看 `sgl_kernel_npu` 内是否已解决（P0.5）
- [ ] P6.2 **featurewise-gate KDA**：recurrent（decode/MTP）+ chunk（prefill）—— 确定缺口（G4）
- [ ] P6.3 **Compressor 的 LayerNorm 变体** —— 只是给 vendor `compressor` 扩一个 norm 类型，不是整套 KPool kernel（G3）
- [ ] P6.4 **mHC pre/post 的 NPU dispatch 接线** —— 不缺算子，缺分支（已在 P3.2）
- [ ] P6.5 **GLM 版 clipped SwiGLU（非对称 clamp）** —— `npu_clipped_swiglu` 是 gpt-oss 语义不可复用（G6）
- [ ] P6.6 **`aclnnLightningIndexerV2` 的 torch 绑定** —— aclnn C API 与 A3 二进制都在，只是 torch_npu 没绑（G8）。绑定工作，非 kernel 工作
- [ ] P6.7 分组 top-k（`group_topk=512`）—— 若 P6.6 的 V2 能覆盖，这条可能可以取消
- [ ] P6.8 kpool 剩余 Triton kernel（先试 triton-ascend）
- [ ] P6.9 NoPE MLA prolog 融合 —— 取决于 P0.6 的 `MlaPreprocess rope_dim=0` 探测结果；`MlaPrologV3` 已确认走不通
- [ ] P6.10 NPU Graph

**已确认可以不做的**：
- ~~零 RoPE64 适配~~（G1：`query_rope`/`key_rope` 本就是 Optional）
- ~~Hadamard-128 旋转~~（G2：正交归一，走 bf16 indexer 时数学等价）
- ~~`aclnnMixedQuantSparseFlashMla` 路线~~（G7：`rope_head_dim` 只能是 64，且 arch35-only，A3 无二进制）

---

## 4. 待补 / 待决

- [x] ~~算子缺口核实~~ → 已完成，见 §2.5，已并入 P6
- [ ] DSv4 W8A8 权重格式确认（compressed-tensors？）
- [ ] P5 之后磁盘怎么排：BF16(643) + W8A8(333) = 976 GB / 984 GB，只剩 8 GB。可能要把 W8A8 写到 `/usr/.devenv`

---

## 5. 变更日志

| 日期 | 变更 |
|---|---|
| 2026-08-27 | 初版落盘。含 P0–P6、5 条文档修正、NoPE shape 探测静态结论、CANN 实为 9.1.0 的发现 |
| 2026-08-27 | **重要修正：SoC 是 A3（`Ascend910_9362`）不是 A2/910B** → 所有 sgl-kernel-npu 包改选 a3 档 |
| 2026-08-27 | 修正：系统 py3.11 已有 torch_npu 2.7.1.post4（需 `source set_env.sh`），可立即做运行时探测 |
| 2026-08-27 | 新增 §2.5 算子缺口核实结论（G1–G8）；P6 按核实后的最小集重排；P0 加 P0.5/P0.6 探测项 |
| 2026-08-27 | P0.1 PASS（Python 3.12.14 venv 建好）；**P0.2 PASS**（torch 2.10.0 + torch_npu 2.10.0.post4 跑通 NPU） |
| 2026-08-27 | **P0.6 部分 PASS：`head_dim=256` 与 `rope=0` 在 A3 上实测可跑** → 最高风险项解除，NoPE 剩余工作全在 SGLang 侧 |
| 2026-08-27 | **第二章统一重写**为"算子现状与缺口"：A1–A7 已确认可用 / B1–B3 确认要开发 / C1–C5 明确标不确定 / 2.6 决策表（torch 绕过） / 2.7 SGLang 侧改造点 / 2.8 工作量口径差异 |
| 2026-08-27 | 修正 agent 的 G4/G5：其证据来自 `torch_npu` 原生算子，但昇腾 KDA 路径走 `sgl_kernel_npu`，**结论降级为❓不确定**（C1/C2），P0.5 消解 |
| 2026-08-27 | 确认 vendor `npu_quant_lightning_indexer` 已在 DSv4 生产路径用 `cmp_ratio=4` → pooled-key 打分能力已有（A3） |
| 2026-08-27 | **C1/C2 消解为"不是缺口"**（读 sgl_kernel_npu 的 triton 源码）；**C4 消解为"不支持"**（实跑）；C3/C5 因 GLIBC 待解 |
| 2026-08-27 | P0.3 曾因 GLIBC 2.32/2.34 阻塞（Ubuntu 20.04 / glibc 2.31）。LD_PRELOAD shim 验证无效（verneed 指名 libc.so.6）；**用独立解包的 glibc 2.35 loader 解决**，vendor `.so` 全部加载成功 |
| 2026-08-27 | **P0.3 / P0.4 / P0.5 PASS**。9/10 custom 算子 + 6/6 KDA kernel 可用。triton-ascend 必须 3.2.2 且 `--no-deps` |
| 2026-08-27 | 新增 [SETUP.md](./SETUP.md) 复现文档 + `probe/` 探测脚本 + `env.sh.example`。**下一步换 Ubuntu 24.04 镜像重建**（可跳过 glibc 绕行） |
