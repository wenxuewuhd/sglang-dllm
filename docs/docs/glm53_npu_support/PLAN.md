# GLM-5.3-Flash 昇腾 (Atlas A3 / Ascend910_9362) 适配计划

> 活文档。每完成一步就在对应条目标 `[x] PASS` 或 `[!] FAIL`，FAIL 的直接在本文件改计划。
> 最后更新：2026-08-28

---

## 0. 目标与已定决策

| 项 | 决定 | 决定时间 |
|---|---|---|
| 一期目标 | GLM-5.3-Flash **纯文本** 在 A2 上跑通并闭环精度 | 2026-08-27 |
| BF16 部署形态 | **单节点 TP16**（16 die × 64 GB = 1024 GB） | 2026-08-27 |
| 量化格式 | **compressed-tensors W8A8-INT8**（weight per-channel + act per-token dynamic） | 2026-08-27 |
| 磁盘策略 | FP8 → BF16 **逐 shard 转换、转完即删源 shard** | 2026-08-27 |
| 精度基准 | ~~CPU 上跑 GPU 分支参考实现~~ → **HF `transformers==5.16.1` 的 `glm5_next`**（见 §3 P3） | 2026-08-28 |
| 多模态 (ViT/video) | 一期**不做** | 2026-08-27 |
| MTP / NextN | 一期**不做** | 2026-08-27 |
| 长上下文 CP | 一期**不做**，只承诺 32K | 2026-08-27 |
| DSv4 权重何时删 | ~~保留到 P1.2 做完再删~~ → **P1.2 完成后已删**（见 §4） | 2026-08-28 |

---

## 1. 已核实的环境事实

### 硬件
- **SoC = `Ascend910_9362`（910_93 = Atlas A3）**，不是 A2/910B。
  - 证据：`torch.npu.get_device_name(0)` → `Ascend910_9362`；`npu-smi info -t board -i 0` → `Product Name: IT22HMDA_4_S`, `PCI Device ID 0xD803`；`/dev/davinci0..15`
  - ⚠ 2026-08-27 修正：初版按 A2/910B 判断，错误。所有 sgl-kernel-npu 包必须选 **a3** 档
- **16 × Ascend910 die**（npu-smi: 8 NPU × 2 chip），每 die **64 GB HBM** → 合计 1024 GB
- CPU **320 核**，内存 **1.8 TB**
- **OS：Ubuntu 24.04.3 / glibc 2.39 / libstdc++ 13**（2026-08-28 换镜像，此前是 20.04 / glibc 2.31）
- 磁盘 `/mnt/workspace` 984 GB 总量 / **402 GB 可用**（GLM FP8 306 GB + DSv4 W8A8 275 GB 已占）
  - P0.7b 通过删掉 DSv4 后回收到 **677 GB**，够 P2 的 BF16（643 GB）
  - 其他挂载：`/usr/.devenv` 466 GB 可用、`/home` 182 GB 可用

### CANN（关键：标称与实际不一致）
- `ascend_toolkit_install.info` 写 `version=9.2.0`，路径 `/home/developer/Ascend/cann-9.2.0`
- **但所有组件的 `version.info` 实际都是 `9.1.0`**（timestamp 20260715）：
  `compiler=9.1.0`、`opp=9.1.0`、`bisheng-compiler=9.1.0`、`hccl=9.1.0`、`runtime/ge/metadef=9.1.0`
- → **本机等价于 CANN 9.1.0。"9.2 还是 9.1" 这个问题不存在，我们就在 9.1.0 上。**
- driver 在 `/usr/local/Ascend/driver`；toolkit 在 `/home/developer/Ascend/ascend-toolkit/`
- **`opp/vendors/` 初始为空** → `torch.ops.custom.*` 全部缺失。已通过 `--install-path` 装到独立目录 `opp_custom/` 解决（不污染共享 toolkit）
- ✅ **glibc 门槛已消失**：sgl-kernel-npu 的预编译 `.so` 需 **GLIBC ≥ 2.34 + GLIBCXX ≥ 3.4.29**（CI 用 Ubuntu 22.04 编）。
  - 20.04（glibc 2.31）上需 SETUP.md 附录 B 的独立 loader 绕行（已验证可行）
  - **2026-08-28 实测：24.04（glibc 2.39）上整段不需要** —— 两个 `.run` 直接 SUCCESS（无 `--force`），
    `.so` 全部正常 dlopen，`import attentions` 也由 FAIL 变 OK
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

### 权重（2026-08-28 已重新下载完毕，两个模型 shard 数与 index.json 全对得上）
`/mnt/workspace/models/GLM-5.3-Flash`，62 个 safetensors，306 GiB
`/mnt/workspace/models/DeepSeek-V4-Flash-W8A8`，46 个 safetensors，275 GiB（P0.7 冒烟/精度用，通过后删）

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
> ✅ 2026-08-28：`probe/p0_5_ops.py` / `p0_6_shapes.py` / `p0_6_rope0.py` 已在目标环境
> **torch_npu 2.10.0.post4 + Ubuntu 24.04** 上复跑，A1 / C4 结论与 2.7.1 一致，无变化。

### 2.1 一条方法论前提（影响很多结论）

昇腾侧算子有**三个来源**，混淆它们会得出错误结论：

| 来源 | 命名空间 | 本机状态 |
|---|---|---|
| torch_npu 原生 | `torch_npu.npu_*` | ✅ 已装（**2.10.0.post4**） |
| CANN vendor 自定义算子包 | `torch.ops.custom.*` | ✅ 已装到 `$ROOT/opp_custom/vendors/`（P0.3 PASS） |
| sgl-kernel-npu python 包 | `sgl_kernel_npu.*` | ✅ 已装（2026.6.1，P0.3 PASS） |

**本仓库的昇腾 KDA / DSA 路径主要走后两者，不走 torch_npu 原生。**
所以"某个 `torch_npu.npu_xxx` 不支持某特性"**不能**直接推出"这是缺口"。

### 2.2 已确认可用（不需要开发）

| # | 算子 / 能力 | 结论 | 证据 |
|---|---|---|---|
| A1 | **NoPE MLA 的 attention core** | ✅ **实测可跑**。`npu_fused_infer_attention_score` 接受 `qk=256/v=256/N=64`（prefill 形态，→`[1,64,256,256]`）与 `kv_lora=512 + rope=0/N=64`（MLA-absorbed decode，→`[1,64,1,512]`） | 本机实跑，`probe/p0_6_shapes.py` |
| A2 | **稀疏 attention 的 rope 可选** | ✅ 签名确认：`npu_sparse_flash_attention(..., Tensor? query_rope=None, Tensor? key_rope=None, ...)`。**rope=0 直接不传，不需要"传 64 维全零"的 workaround** | torch_npu 运行时 doc。⚠ **仍未实跑**——vendor 包现已可加载（24.04），实跑这一条已无障碍，待补 probe |
| A3 | **pooled-key 打分（KPool 的打分环节）** | ✅ **vendor 包已提供且我们已在用**：`torch.ops.custom.npu_quant_lightning_indexer(..., cmp_ratio=4, sparse_count=2048, sparse_mode=3)` 就在 DSv4 生产路径里 | `ascend_dsv4_backend.py:1002-1021` |
| A4 | **KPool 的压缩写 cache** | ✅ vendor `torch.ops.custom.compressor` 已带 **fused norm + RoPE**，且 AscendC 实现（`AddApeToScore → ColumnSoftMax(逐 head-dim 列) → Mul → ColumnSum`）与 GLM 的 kernel **同构**。差异只有 norm 类型 → 见 B1 | `ascend_dsv4_backend.py:411-430`；AscendC `compressor/arch22/` |
| A5 | **mHC pre / post** | ✅ 算子存在：`torch.ops.custom.npu_hc_pre` / `npu_hc_post`，DSv4 在用。GLM-5.3 侧只缺 dispatch 分支（**代码工作，非算子**） | `kernels/ops/layernorm/mhc.py:1605`；`models/deepseek_v4.py:1713` |
| A6 | **MoE router / GMM** | ✅ `npu_moe_gating_top_k`、`npu_grouped_matmul` 已绑定，A2/A3 支持 | torch_npu doc |
| A7 | **Hadamard-128 可省略** | ✅ 它是正交归一阵（`* 0.08838834764831845` = 1/√128），q 与 k 同时旋转 → 点积严格不变。走 bf16 indexer 时**数学等价，可整体删掉** | `kpool_fp8_index.py:862-870`；`dsa_indexer_kpool.py:656` |

### 2.3 确认要新开发

| # | 算子 | 结论 | 证据 |
|---|---|---|---|
| B1 | **compressor 的 LayerNorm 变体** | 需要开发，但**范围很小**：vendor `compressor` 的 fused norm 是 **RMSNorm**（DSv4 用），GLM 的 index-K norm 是**真 LayerNorm（减均值 + bias）**。是"给已有算子扩一个 norm 类型" | `layers/layernorm.py:974-1014` GLM 走 `F.layer_norm(bias=True)`；`ascend_dsv4_backend.py:411` 传 `_fused_norm_weight_fp32` |
| B2 | ~~**GLM 版 clipped SwiGLU（非对称 clamp）**~~ | ❌ **不需要开发**。而且比第一次撤销时以为的还简单：**`torch_npu.npu_clipped_swiglu` 直接可用**——`alpha=1.0, limit=10.0, bias=0.0, interleaved=False` 下与 GLM 公式**逐位相同**（两次独立上机验证；用默认参数则差 156，这正是当初误判「gpt-oss 语义不可复用」的来源：**默认值是 gpt-oss 的，但每个都是参数**）。bf16 进 bf16 出，A3 支持。详见 §2.3.1 | 本机实测；`glm5_next.py:139-144` |
| B3 | **KPool 的 pool→raw 展开 + 尾部追加** | 需要开发（或用 torch 实现）。这部分是 GLM 特有的索引后处理，`compressor` / `lightning_indexer` 都不负责 | `kpool_fp8_index.py:379-401 expand_pooled_groups_to_topk`、`:421+ append_kpool_tail_to_topk` |

#### 2.3.1 B2 撤销的经过（2026-08-28 上机核实）

初版判断「`npu_clipped_swiglu` 是 gpt-oss 语义不可复用 → 要新开发」，**方向错了**：真正该看的不是
`npu_clipped_swiglu`，而是 DSv4 昇腾路径实际在用的东西。核实结果：

- **DSv4 现在用的根本不是融合 clamp 算子**，而是 `torch.clamp_` + 现成的
  `npu_swiglu`（bf16）/ `npu_dequant_swiglu_quant`（int8）。这个组合**与模型无关**，GLM 直接可用
- **两条仓库里写着的说法被实测推翻**：
  1. 注释说 `npu_dequant_swiglu_clamp_quant` 的 `clamp_limit` 无效 —— **只在默认 `swiglu_mode=0` 下成立**。
     `swiglu_mode=1` 时 clamp 是生效的：`glu_alpha=1.0, glu_bias=0.0, activate_left=True` 下
     与 GLM 公式 **relerr 0.00000**（排除性对照：交换半边 0.993、up 不做下界 0.800 → 确认是 chunk 切分、非对称 clamp）
  2. 注释说 `swiglu_clip_quant` 的 `group_alpha` 是 GLU alpha —— **不是**。该算子算完 `silu*up` 后
     把**输出**裁到 `±group_alpha × rowmax(|y|)`，是逐 token 的**量化离群点裁剪**，
     `group_alpha` 是行最大值的比例。**拿它做输入 clamp 是错的**
- **唯一可能的算子需求（优化，非阻塞）**：`npu_dequant_swiglu_clamp_quant` 的 `dst_type` 被静默忽略，
  **输出恒为 int8**。所以 W8A8 路径可以用它替掉「clamp + dequant_swiglu_quant」两趟白赚一次融合；
  **BF16 路径用不了**。若将来要 BF16 融合，正当需求是「该算子的 BF16 输出变体」，**按优化排期**
- ⚠ 未消解：`group_index` 的约定（逐组计数 vs 前缀和）没能区分开，集成时要确认
- 已修正 `npu/moe/activation.py` 里那两处会误导人的注释

**二次更正（同日，独立复核）**：上面「撤销」的方向仍然对，但**落点错了**——
不该转向 `npu_dequant_swiglu_clamp_quant`（它输出恒为 int8，bf16 用不了），
而应该转回 **`torch_npu.npu_clipped_swiglu`**。实测：
`npu_clipped_swiglu(x, alpha=1.0, limit=10.0, bias=0.0, interleaved=False)`
与 GLM 公式**逐位相同**，bf16 进 bf16 出，A3 支持。
初版判「gpt-oss 语义不可复用」是把**默认参数**当成了**能力上限**——
默认 `alpha=1.702, limit=7.0, bias=1.0, interleaved=True` 确实是 gpt-oss 的，
但**每一个都是可传参数**。→ 交接包里的 OP-4 已标为 WITHDRAWN。

### 2.4 不确定项的消解结果（2026-08-27 二次核实）

**消解手段**：C1/C2 靠**读 wheel 里的 Python/Triton 源码**（不需要执行）；C4 靠**运行时实跑**（torch_npu 2.10 已可用）；
C3/C5 依赖 vendor 包，**2026-08-28 起 GLIBC 障碍已消失、vendor 包可加载 → 两条都已具备实跑条件**（尚未跑）。

| # | 问题 | **结论** | 证据 |
|---|---|---|---|
| C1 | K=4 stateful causal conv1d 缺不缺 | ✅ **不缺，解决**。`sgl_kernel_npu.mamba.causal_conv1d` 是 **triton-ascend** kernel，`KERNEL_WIDTH` 是 `tl.constexpr`，**1/2/3/4/5/6 全有显式分支**（`==4` 在 363-400 与 444-458 两处），`width, _ = weight.shape` 从权重读，非硬编码 | `sgl_kernel_npu/mamba/causal_conv1d.py:66,387,449,550` |
| C2 | featurewise-gate KDA 缺不缺 | ✅ **不缺，解决**。`fused_kda_gate_npu(gate, A_log, head_dim, gate_bias, lower_bound)`：`gate` 是 `[tokens, heads*head_dim]` 的**逐通道** gate（`heads=A_log.numel()`，`heads*head_dim==hidden` 有断言），`gate_bias.numel()==hidden` 也是逐通道，且**带 `lower_bound` 参数**。与 GLM 的 `f_b_proj→[T,64*128]` + `dt_bias[8192]` + `gate_lower_bound=-5.0` 完全对得上。prefill 侧 `chunk_gla_fwd_o_gk_npu` 的 `gk` 也是逐通道命名 | `sgl_kernel_npu/fla/kda_gate.py:74-94`、`fla/kda_prefill.py:295` |
| C4 | `npu_kv_rmsnorm_rope_cache_v2` 支持 rope=0 吗 | ❌ **不支持，解决**。**实跑验证**：`rope=64` 时 v1/v2 都 [OK]；`rope=0`（cos/sin/k_cache 传 0 宽）时 v1/v2 **都 RuntimeError**。→ D2 拆 `rmsnorm + reshape_and_cache` 是确定要做的活 | `probe`，`aclnnKvRmsNormRopeCache*` 报错 |
| C3 | `MlaPreprocess` 的 `rope_dim=0` 是否合法 | ⏸ **仍不确定，但已降级**。`mla_preprocess` **不在 `torch.ops.npu` 里**（torch_npu 2.10 无此绑定），它来自 vendor 包 → 被 GLIBC 挡住，现在探不了。**但它是性能项**（`SGLANG_NPU_USE_MLAPO` 默认关），不阻塞 BF16 打通。
⚠ 2026-08-28：GLIBC 障碍已消失，vendor 包可加载，这条**现在可以探了** | `torch.ops.npu` 无 `mla_preprocess`；`mla_preprocess.py:386` 调 `torch.ops.npu.mla_preprocess` |
| C5 | 分组 top-k 是否要自研 | ✅ **已消解：不用自研**。`cmp_ratio=4` 时 `npu_quant_lightning_indexer` 返回的是 **pool 级下标 + `-1` 填充**——依据是它自己的 bf16 参考路径 `ascend_dsv4_backend.py:618-682`，在 `seq//ratio` 个 pooled 条目上算分并 `.topk(min(index_topk, seq//ratio))`，再 `F.pad(value=-1)`。**即 pooled 打分与分组 top-k 在昇腾侧都已存在且在 DSv4 生产中**；真正缺的只是 §2.3 的 B3（展开+尾部+页表映射）。原判断： |`aclnnQuantLightningIndexer` 有 `cmpRatio` + `sparseCount` + `sparseIndicesOut/sparseValuesOut`，且我们 DSv4 路径已用 `cmp_ratio=4`。**未确证的一点**：`cmp_ratio=4` 时 `sparseIndicesOut` 返回的是 **pool 下标**还是 **raw token 下标**。DSv4 是 compress-then-attend（拿到就直接喂 `cmp_ratio=4` 的 sparse attn，不展开），GLM 是 pool-score-then-expand（要展开回 raw token） | `aclnn_quant_lightning_indexer.h`；`ascend_dsv4_backend.py:1013-1021` |

### 2.4.1 vendor 包的算子分布（2026-08-27 实测解包）

装包时才发现算子分散在两个 vendor 里，且**只有部分编译了 `ascend910_93`（我们的 SoC）**：

| 包 | 提供的算子 | 编进 `ascend910_93` 的 |
|---|---|---|
| **custom-ops** | `dequant_swiglu_clamp_quant, gather_selection_kv_cache, **hc_post, hc_pre, hc_pre_inv_rms, hc_pre_sinkhorn**, indexer_compress_epilog, inplace_partial_rotary_mul, kv_compress_epilog, moe_gating_top_k_hash, moe_init_routing_group_quant, partial_rotary_mul_quant, rms_norm_dynamic_quant, scatter_nd_update_asc, swiglu_clip_quant, swiglu_group_quant` | `dequant_swiglu_clamp_quant, gather_selection_kv_cache, **hc_post, hc_pre, hc_pre_inv_rms, hc_pre_sinkhorn**, inplace_partial_rotary_mul, moe_gating_top_k_hash, rms_norm_dynamic_quant, scatter_nd_update_asc, swiglu_clip_quant` |
| **ops-transformer** | `compressor, quant_lightning_indexer(+metadata), sparse_attn_sharedkv(+metadata)` | 全部有 |
| **attentions**(wheel) | `ada_block_sparse_attention, laser_attention, sparse_block_estimate` | 全部有 |

→ **mHC 的 `hc_pre`/`hc_post` 确认为 A3 编译**（A5 结论坐实）。
→ **A5-only（有头文件但无 a3 kernel）**：`indexer_compress_epilog`、`kv_compress_epilog`、`moe_init_routing_group_quant`、`partial_rotary_mul_quant`、`swiglu_group_quant`。
→ ✅ **2026-08-28 起三个包的 `.so` 在 24.04 上全部正常 dlopen**（此前 20.04 因 GLIBC 全部失败）。

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

### P0 · 环境搭建 ✅ 已在 Ubuntu 24.04 上完整复现（详细步骤见 [SETUP.md](./SETUP.md)）

> 下列条目均为 **2026-08-28 在 24.04 / torch_npu 2.10.0.post4 上重跑通过**的结果。
> `ROOT=/mnt/workspace/y00359136/work/glm53_dev/env`（与代码仓库同级）。

- [x] **P0.1 PASS** Python 3.12.9 venv（24.04 系统自带，不再需要 uv）
- [x] **P0.2 PASS** torch **2.10.0+cpu** + torch_npu **2.10.0.post4**，`device_count=16`，`Ascend910_9362`，NPU bf16 matmul 通过
- [x] **P0.3 PASS** sgl-kernel-npu `20260826` 三件套（**cann9.1.0-a3-aarch64 / py312**）+ triton-ascend **3.2.2**
- [x] **P0.4 PASS** 未执行 `pip install -e python/`（会装 CUDA 变体顶掉 torch_npu），走 `PYTHONPATH`
- [x] **P0.5 PASS** 算子可见性（`probe/p0_5_ops.py`）：
      - `torch.ops.custom.*` **9/10 可见**：`compressor`、`npu_hc_pre`、`npu_hc_post`、`npu_quant_lightning_indexer(+_metadata)`、`npu_sparse_attn_sharedkv(+_metadata)`、`npu_moe_gating_top_k`、`inplace_partial_rotary_mul`
      - KDA/conv1d **6/6 可导入**：`causal_conv1d_fn_npu`、`causal_conv1d_update_npu`、`fused_kda_gate_npu`、`chunk_gla_fwd_o_gk_npu`、`chunk_gated_delta_rule_fwd_h_npu`、`kda_target_verify_npu`
      - 三项已确认不阻塞：`attentions`（只被 diffusion 用）、`deep_ep`（只被 DeepEP 分发用，**TP16+EP 阶段要回头解决**）、`torch.ops.custom.npu_mla_prolog_v3`（融合 prolog 默认关，且 `torch_npu.npu_mla_prolog_v3` 原生存在）
- [x] **P0.6 部分 PASS**（`probe/p0_6_shapes.py`、`probe/p0_6_rope0.py`）：见 §2.6
      - **2026-08-28 在 torch_npu 2.10 上复跑，结论与 2.7.1 一致**：
        `FIA GLM 512+0/N=64` → `[1,64,1,512]` OK；`FIA GLM 256/256 N=64` → `[1,64,256,256]` OK（A1 坐实）；
        `rope=0` 的 `npu_kv_rmsnorm_rope_cache` v1/v2 **仍双双 RuntimeError**（C4 坐实 → D2 必做）
      - ⚠ 新观察（**未定性**）：`FIA DSv3 512+64 / N=128` 在 2.10 上 **FAIL**（`aclnnFusedInferAttentionScoreV3` err 561002）。
        2.7.1 上是否通过**没有留下记录**，所以**不能断言是回归**。GLM 走的是 `N=64 / rope=0`，
        且 TP16 下每 die 只有 8 头，**不在关键路径**；等 P0.7a DSv4 冒烟时自然会被覆盖到
- [x] **P0.8 PASS** 换机重做：Ubuntu 24.04 上 P0.1–P0.6 全部重跑通过，**附录 B 的 glibc 绕行整段跳过**
      （glibc 2.39 / GLIBCXX_3.4.29 已确认；两个 `.run` 无需 `--force`）。24.04 差异见 SETUP.md 附录 D
- [x] **P0.7a PASS** 冒烟：DeepSeek-V4-Flash W8A8 在 **A3 单节点 TP16/DP16 + DeepEP** 起服务成功
      （`launch_dsv4_a3.sh.example`）。`/health_generate` 200，`/generate` 与 `/v1/chat/completions`
      都返回非空且连贯的文本。权重 **28.08 GB/die**（16 个 rank 一致），NPU graph 捕获通过。
      > DP-attention 下每个 rank 各存一份 attention/dense 权重，所以比纯 TP16 不开 DP 时
      > （18.45 GB/die）高。两者不可直接比。
      - 配方来自上游 PR [sgl-project/sglang#25144](https://github.com/sgl-project/sglang/pull/25144)
        （`[NPU] Add Ascend NPU support for DeepSeek-V4`，已合入 main），按本机路径改写
      - ⚠ **与 PR 的唯一实质差异：`--quantization compressed-tensors`，不是 PR 的 `modelslim`**。
        我们这份 modelscope 权重的 `config.json` 自述 `quant_method=compressed-tensors`，
        照抄 PR 的 `modelslim` 会被 SGLang 直接拒绝启动（"Quantization method specified in the
        model config (compressed-tensors) does not match ... (modelslim)"）
      - ⚠ 这套配置**不是** A2 那份单卡 KT CPU offload 教程（`deepseek_v4_flash.mdx`）。
        A3 有 16×64 GB HBM，275 GB 权重直接放得下，**不需要 KT CPU 卸载**
- [x] **P0.7b PASS** 精度：GPQA-Diamond，**non-thinking**，**3 轮**，并发 16（DP16 → 每 rank 1）

      | 轮 | 分数 | 用时 |
      |---|---|---|
      | 1 | **74.24%**（147/198） | 709 s |
      | 2 | **75.25%**（149/198） | 703 s |
      | 3 | **71.72%**（142/198） | ~700 s |
      | **均值** | **73.74%**（样本 SD 1.82 pp） | |

      - 对标口径 **73.23%**（PR#25144 与 `deepseek_v4_flash.mdx` 一致）→ **差 +0.51 pp，完全在噪声内**
      - 198 题在 temperature=1 下单轮二项标准误约 ±3.2pp，所以**只看均值**；三轮之间统计上不可区分
      - 工具：EvalScope（`gpqa_diamond`），`temperature=1 / top_p=1 / max_tokens=32768`，
        thinking 通过 `extra_body.chat_template_kwargs={"thinking": False}` 关闭；0 个请求错误
      - 脚本在 `$ROOT/eval/`（`run_gpqa.py` + `run_all.sh`），用**独立的 `.venv-eval`**，
        没有污染 `.venv-glm53`

> **✅ P0 全部完成。** 环境、算子、DSv4 冒烟与精度闭环都通过，可以进 P1。

> DSv4 W8A8 权重 2026-08-28 已重新下齐（46 shard，275 GiB，owner 即本用户，无权限问题）。P0.7b 通过后可删（回收 275 GB）。

### P1 · 分支合流 ✅ 完成

**参考实现已更新（2026-08-28 重新 fetch）**：上游 PR [#36507 `GLM-5.3-Flash support`](https://github.com/sgl-project/sglang/pull/36507)
**仍未合入 main**（open，非 draft），分支 `xinyuan/glm-5.3-flash-support` 头已从我们快照的
`0b9c38484e`(08-26) 前进到 **`033446bb05`(08-27)**，145 文件 / +16702 −836。
本地已 fetch 并打 tag **`glm53-gpu-ref-033446bb`**。

- [x] **P1.1 试跑通过**（2026-08-28，在隔离 worktree `../rebase-trial` / 分支 `p1-rebase-trial`，
      **`glm53_dev` 未动**）：`git rebase --onto 033446bb05 eea2e5d6e5`，19 个 commit 全部重放成功。
      - **只有 2 个冲突**（远少于预期）：
        1. `layers/moe/hash_topk.py` —— 上游新增了 `_is_xpu` 分支，我们加的是 `not _is_npu` 护栏。
           **两边都要**，解成 `if _is_xpu: ... elif ...FUSED_HASH_TOPK.get() and not _is_npu:`。
           ⚠ 核实过：新 base 里 **`fused_hash_topk_npu` 并不存在**（PR#25144 的描述提到它，但没落地），
           所以我们的 `not _is_npu` 护栏**仍然必要**
        2. `models/deepseek_v4.py` —— 上游把 `self.rope_scaling` 就地改写换成了
           `active_rope_scaling` 副本，且只在 `compress_ratio in (4,128)` 时才设 `deepseek_yarn`；
           我们的改动是 NPU 上把 `rotary_emb` 置 None 省 ~1.25 GiB HBM。
           解法：**采用上游的 `active_rope_scaling` 逻辑 + 保留我们的 NPU None 护栏**
      - `server_args.py` **自动合并且结果正确**：我们的 `kt_expert_placement_strategy` /
        `kt_activation_freq_path` 落在了新的声明式风格里，`NS("exec.moe")` 命名空间与上游自带的
        `kt_num_gpu_experts` 一致。原先担心的 #36255 重构**没有造成实际返工**
      - 验证：18 个改动文件 **0 语法错误、0 残留冲突标记**；`import sglang` +
        `import http_server` 通过；`ServerArgs` 构造出来 kt-* 三个参数取值正确
      - ⚠ **尚未验证运行时行为** —— 那是 P1.2
- [x] **P1.1b PASS** 已 promote 到 `glm53_dev`（2026-08-28，P1.2 回归通过之后才做）：
      `glm53_dev` 现在是 **`033446bb05` + 19 个 NPU commit**。
      - 回退用：本地 tag **`glm53_dev-pre-p1-rebase`** = 旧 head `a6be0fe83b`，
        且 `origin/glm53_dev` **尚未 push**，仍指向旧 head → 两条回退路都在
      - GPU 参考实现 tag：`glm53-gpu-ref-033446bb`
- [x] **P1.2 PASS** 出口判据：新 base 上 DSv4-Flash 回归通过

      | | round1 | round2 | round3 | 均值 | SD |
      |---|---|---|---|---|---|
      | P0.7b 基线（旧 base） | 74.24% | 75.25% | 71.72% | **73.74%** | 1.82pp |
      | P1.2（rebase 到 `033446bb05`） | 75.76% | 73.74% | 70.20% | **73.23%** | 2.81pp |

      **差 −0.50 pp，在噪声内**（单轮二项 SE ≈ ±3.2pp），均值正好等于公开口径 73.23%。
      服务启动、`/generate`、`/v1/chat/completions` 均正常；权重 **28.17 GB/die**
      （旧 base 28.08，差 0.09 = 噪声，**无内存回归**）
      - 回归基线就是 P0.7b 这三轮：**均值 73.74%**（74.24 / 75.25 / 71.72，SD 1.82pp）。
        新 base 上重跑 3 轮，均值落在 **约 70.5–77.0%**（±1.8pp 左右）即算不回归；
        ⚠ 别拿单轮下结论
      - **DSv4 权重要留到这一步做完才能删**（§0 决策）

**实测冲突面：只有 8 个代码文件重叠**（其余 7 个 NPU 文件上游没碰，应当干净）：

| 文件 | 我们 | 上游 | 风险 |
|---|---|---|---|
| `server_args.py` | 22+/2− | **2256+/1189−** | **高**（见下） |
| `models/deepseek_v4.py` | 52+/10− | 769+/106− | 中 |
| `models/deepseek_v2.py` | 13+/1− | 100+/48− | 低 |
| `moe/hash_topk.py` | 6+/1− | 37+/2− | 低 |
| `moe/kt_ep_wrapper.py` | 470+/42− | 5+/2− | 低（几乎全是我们的） |
| `moe/fused_moe_triton/layer.py` | 27+/1− | 7+/4− | 低 |
| `utils/hf_transformers/common.py` | 20+/1− | 45+/3− | 低 |
| `test/registered/moe/test_hash_topk.py` | 27+/0− | 1+/1− | 低 |

> ⚠ **`server_args.py` 的高churn 是有原因的**：`0b9c38484e..033446bb05` 的增量里合了
> main 的 **[#36255 `config: ServerArgs holds the raw input`](https://github.com/sgl-project/sglang/pull/36255)** ——
> ServerArgs 改成了「声明式 resolution」模型。我们那 22 行 kt-* 参数**不能直接 rebase 过去，要按新模型重写**。
> 动手前先读 `sglang-runtime-context` skill（仓库里有）。

### P2 · BF16 权重 ✅ 完成

**checkpoint 结构（2026-08-28 实测）**：76108 tensor / 62 shard / 305.78 GiB

| 命名空间 | 内容 | 体积 | fp8? |
|---|---|---|---|
| `model.language_model.layers.0..44` | 45 层正文 | 297.75 GiB | **是**（37338 个 fp8 + 同分片 scale） |
| `model.language_model.layers.45` | **MTP / NextN 层**（`eh_proj`/`enorm`/`hnorm`） | 6.98 GiB | 是 → 转换后 13.84 GiB |
| `model.visual` | ViT，347 个 tensor | 1.05 GiB | **否，本来就是 BF16** |
| `lm_head.weight` | | | 否，BF16 |

- ✅ **ViT 不需要"转"** —— `model.visual` 里 **0 个 fp8、0 个 `weight_scale_inv`**，脚本原样拷过去
- MTP 层一期不用，但空间够，**一起转**，免得以后缺
- 转换后总计 **598.51 GiB（643 GB）**，与初版估算一致

- [x] **P2.1 PASS** 逐 shard 反量化脚本落盘：[`tools/fp8_to_bf16.py`](./tools/fp8_to_bf16.py)
      （`weight_block_size=[128,128]`，`bf16 = fp8 * scale_inv` 按 128×128 块广播；
      丢弃 F32 scale；BF16 张量原样拷；`--delete-source` 默认关；带 `--min-free-gib` 空间闸门）

      本次实际用的命令（换权重版本要重转时照抄）：

      ```bash
      source $ROOT/env.sh
      SRC=/mnt/workspace/models/GLM-5.3-Flash
      DST=/mnt/workspace/models/GLM-5.3-Flash-BF16

      # 1) 先只转第一个 shard，什么都不删，人工核数值
      npy $REPO/docs/docs/glm53_npu_support/tools/fp8_to_bf16.py \
          --src $SRC --dst $DST --only model-00001-of-00062.safetensors

      # 2) 核过之后再全量。--min-free-gib 是安全闸门，空间不够会中止而不是写坏
      npy $REPO/docs/docs/glm53_npu_support/tools/fp8_to_bf16.py \
          --src $SRC --dst $DST --min-free-gib 20
      ```

      > 全量那一步会在最后写 `model.safetensors.index.json`（剔掉 scale 条目）、
      > 清洗 `config.json`、拷 tokenizer 等文件；`--only` / `--limit` 模式**不写**这些，
      > 所以单 shard 试转不会产生半成品目录。输出已存在的 shard 会跳过，**中断可续跑**。
- [x] **P2.2 PASS** 第一个 shard 转完人工核数值（**未删任何源**）：
      - 抽查：名称集合 = 源 − scale ✓、形状全等 ✓、480 个元素独立重算最大偏差恰好 0 ✓
      - **全 shard 逐位比对**：**2,759,852,032 个元素（27.6 亿）全部逐位一致**，
        重算用的是**与脚本不同的实现**（按 128×128 块 reshape 后逐块乘，
        而非脚本的 `repeat_interleave` 展开 scale）→ 排除了"用同一个 bug 验自己"
      - 直拷张量逐位相同 ✓；**全 shard 无 NaN/Inf** ✓
      - 单 shard 约 20 s → 全量约 21 min
- [x] **P2.3 PASS** 全量转换：62/62 shard，约 21 min，输出 `/mnt/workspace/models/GLM-5.3-Flash-BF16`
      **599 GB**（估算 598.51 GiB，吻合）。数目对账：本次 dequant 37009 + 首 shard 329 = **37338**，
      正好等于索引里 `weight_scale_inv` 总数
- [x] **P2.4 PASS** `config.json` 已删 `quantization_config`；tokenizer / chat_template /
      processor_config / generation_config / LICENSE 一并拷贝
- [x] **P2.5 PASS 出口判据**：BF16 目录全量比对
      - 索引名称集合 = 源 − scale：**76108 − 37338 = 38770**，实际 38770 ✓
      - 62 个 shard 逐个打开：索引声明缺失 0 / 文件多余 0 / **形状与源全等** / **无 scale 泄漏** ✓
      - 输出 dtype：**BF16 38479 + F32 291**。291 个 F32 是**原生 F32**（抽查 50/50 源侧同为 F32），
        构成正好对上模型结构：`hc_attn_base/scale`+`hc_ffn_base/scale` 各 45（mHC）、
        `mlp.gate.e_score_correction_bias` 43（MoE 层数）、`self_attn.A_log`+`dt_bias` 各 34（KDA 层数）


#### 哪些权重真的被转换了（2026-08-28 实测）

**按层**（45 层正文 + 1 层 MTP = 46）

| 层类型 | 层号 | 层数 | 转 BF16 | 直通 |
|---|---|---:|---:|---:|
| `linear_attention` + dense FFN | 0–2 | 3 | 9 | 69 |
| `linear_attention` + MoE | 4,5,6,8,… | 31 | 26877 | 775 |
| `deepseek_sparse_attention` + MoE | 3,7,11,…,43 | 11 | 9581 | 220 |
| MTP / NextN | 45 | 1 | 871 | 18 |
| `model.visual`（ViT） | — | — | **0** | 347 |
| `lm_head` / embed / 顶层 norm | — | — | **0** | 3 |
| **合计** | | **46** | **37338** | **1432** |

**层内部**

| 模块 | fp8 → BF16？ |
|---|---|
| MoE `experts.N.{gate,up,down}_proj`（每层 288×3） | ✅ 转 |
| `shared_experts.{gate,up,down}_proj`（每层 3） | ✅ 转 |
| dense FFN `mlp.{gate,up,down}_proj`（仅 0–2 层） | ✅ 转 |
| DSA `q_a_proj` / `q_b_proj` / `kv_a_proj_with_mqa` / `o_proj` | ✅ 转 |
| DSA `kv_b_proj` | ❌ 本来就是 BF16 |
| **KDA 全部 `self_attn.*`**（q/k/v_proj、`*_conv1d`、f/g_proj、`A_log`、`dt_bias`、`o_norm`、`o_proj`） | ❌ 本来就是 BF16/F32 |
| **indexer 全套**（`wq_b`/`wk`/`weights_proj`/`k_norm.{weight,bias}`/`index_kpool_compress_{ape,gate}`） | ❌ 本来就是 BF16 |
| **mHC 全部 `hc_*`**（`_base`/`_fn`/`_scale`） | ❌ 本来就是 BF16/F32 |
| 所有 norm、`mlp.gate.weight`、`e_score_correction_bias` | ❌ 本来就是 BF16/F32 |

> **对 P3 的推论（重要）**：官方 FP8 量化排除掉的，**恰好就是我们要移植的三块** ——
> KDA、indexer、mHC **全程没被量化过**，BF16 转换对它们是逐位直通。
> 所以 P3 若在这三块上测出数值偏差，**可以直接排除"权重转换引入的"**，
> 只可能来自我们的实现或算子。唯一被转换动过的是 MoE 专家 + dense FFN + DSA 的四个投影，
> 而那部分有首个 shard 的 27.6 亿元素逐位验证兜底。

> **计划变更（2026-08-28）**：原方案是"转完即删源 shard"，因为当时磁盘只剩 66 GB。
> 删掉 DSv4 后可用 668 GB，BF16 需 643 GB → **改为全程保留 FP8 源**，结束时余 ~25 GB。
> 这让整个 P2 变成**可逆**的：转换若有隐蔽 bug，不必重下 306 GB。
> **FP8 源等 P4 端到端验过再删**（P5 的 W8A8 333 GB 那时本来也容不下它）。

### P3 · 逐模块对拍 ☐
按依赖顺序，CPU golden vs NPU。
- [x] **P3.1a 已改**：`attention_registry.py` 的 `glm5_next_config` 分支加了 `_is_npu` 判断，
      路由到 `AscendKDAAttnBackend` / `AscendKDAHybridLinearAttnBackend`（照抄同文件 Kimi-K3 的写法）。
      **改动生效，服务能起来**，前向已经走到 KDA
- [x] **P3.1b PASS 已修**（`_flat_kda_gate`，`ascend_kda_backend.py:122`）：按「最后一维是否已等于
      `A_log.numel() * head_k_dim`」决定要不要 flatten。**不改算子、不增加运算量**
      —— flatten 在 python 封装层，算子内部本来就会 `reshape(-1, shape[-1])`；且连续张量上
      flatten/reshape 是纯视图操作。**实测生效**：KDA 的 gate 报错消失，前向推进到了 MLA
- [x] ~~P3.1b 布局契约不匹配（gate 张量）~~ 详情：
      `ascend_kda_backend.py:365` 传的是 `g.flatten(-2)`，那是 Kimi 的 4-D 布局
      `[..., heads, head_dim]`；但 **GLM 传进来的 `forget_gate` 已经是 flat 的**
      —— prefill 下是 `[1, T, heads*head_dim]`（`glm5_next.py:577` 的 `unsqueeze(0)`），
      decode 下是 `[T, heads*head_dim]`。对 3-D 输入做 `flatten(-2)` 会把 `T` 和特征维压在一起，
      于是 `shape[-1]` 变成 `T*512`，`fused_kda_gate_npu` 的
      `heads*head_dim != hidden` 校验必然失败（`sgl_kernel_npu/fla/kda_gate.py:89`）
      - **不是算子能力问题**：`A_log` 是 `(1,1,local_num_heads,1)`，TP16 下 numel=4，切分正确；
        `head_k_dim=128` → 期望最后一维 512，GLM 给的正是 512，只是被 `flatten(-2)` 破坏了
      - 修法：让 Ascend 后端按最后一维是否已等于 `heads*head_dim` 决定要不要 flatten
- [x] **P3.2 接线完成**：`_mhc_pre_dispatch`/`_mhc_post_dispatch` 已加 NPU 分支。核对 `post_mult_value=2.0` 与 kernel 内部一致性、`(post_mix, comb_mix)` ↔ `(post, comb)` 映射
      - **实测确认了 D6**：不改的话第一次前向就炸在
        `mhc.py:1676 _mhc_pre_dispatch` → `mhc.py:850 mhc_pre` →
        `deep_gemm_wrapper/entrypoint.py:245` → `NameError: name 'deep_gemm' is not defined`
      - ~~临时绕过 `SGLANG_OPT_USE_TILELANG_MHC_*=False`~~ → **已不需要，启动脚本里已删掉**
      - **落地形态**：分支加在 `_mhc_pre_dispatch` / `_mhc_post_dispatch` 两个 dispatch 点，
        不动 `glm5_next.py`。pre 分支带三个前置条件（`hc_post_mult_value == 2.0`、
        `hc_pre_eps == hc_sinkhorn_eps`、非空 batch），不满足就退回原路径 —— kernel 只在这些
        设定下等价于参考实现
      - `npu_hc_post` 的 `post` **必须是 2-D `[s, n]`**；调用方一路携带的 `[s, n, 1]` 会被直接拒绝
        （实测报 `post's dim num should be 2`）→ 分支里 `squeeze(-1)`
      - **实测**：服务起得来，前向越过 mHC（报错前移到 `deepseek_v2_attention_mla_npu.py:457`
        的 indexer，即 P3.4）。⚠ 这只证明**接线没崩**；
        算子本身的数值正确性是**单独验的**（见下），端到端数值还要等 P3.4 通了才能验
      - **好消息**：`npu_hc_pre` 的封装**已经存在**（`mhc.py:1780`）且 **DSv4 已在用**
        （`deepseek_v4.py:1906` / `:2029`），P3.2 是把 GLM 接上，不是从零写
      - ⚠ **不能直接照搬**：GLM 的 `_hc_pre_fn` 多了 `post_mult_value=2.0` 和
        `out_norm_weight` 折叠，而 `npu_hc_pre` 返回 `norm_fused=False`（norm 要调用方自己做）
      - **golden 已就绪**：[`tools/golden_mhc.py`](./tools/golden_mhc.py)，参考实现是
        HF `Glm5NextTextHyperConnection`。已解掉一个疑问：**`post_mult_value=2.0` 是写死在公式里的**
        （参考实现就是 `post = 2 * sigmoid(post_w * post_scale + post_b)`），
        所以 GLM 传 2.0 是对的；**还需确认的是 NPU kernel 内部是否也乘了 2**，别乘两次
      - ✅ **已用 golden 实测过 DSv4 的融合算子对 GLM 是正确的**（2026-08-28）：
        直接拿 `torch.ops.custom.npu_hc_pre` 跑 GLM 第 0 层真实权重，对 HF golden ——
        `post` 8.92e-5 / `comb` 6.14e-6 / `y` 4.65e-3，**三个都在噪声地板内**；
        `comb` 双随机自检也过（row 0.996–1.004，col 恰为 1.0）

      **接线时的四个坑（全部实测撞出来的）**：

        | # | 坑 |
        |---|---|
        | 1 | **kernel 内部已经乘了 2** —— `post vs golden=8.9e-5`，`post vs golden/2=1.0`。GLM 的 `post_mult_value=2.0` **不能在外面再乘一次**，否则 post 翻倍，服务照跑但分数悄悄掉 |
        | 2 | **`norm_eps` 要传 GLM 的 1e-5**，不是 DSv4 的 1e-6。两模型其余 mHC 配置（`hc_mult=4`/`sinkhorn=20`/`hc_eps=1e-6`）完全相同 |
        | 3 | **输入必须 4-D** `[b, s, hc_mult, hidden]`，传 2-D 报 `dim num should be 4` |
        | 4 | **`hc_fn`/`scale`/`base` 必须 fp32**，传 bf16 直接报错（checkpoint 里本来就是 F32） |

        另：`npu_hc_pre` 返回 `norm_fused=False`，**不折叠 input_layernorm**，
        调用方要自己做 —— 与 GLM 现在 `_hc_pre_fn` 传 `out_norm_weight` 的做法不同，
        接线时要把 norm 挪到外面。

      - **判据（第 0 层 / attn / 64 token 实测噪声地板）**：

        | 输出 | 相对噪声地板 |
        |---|---|
        | `post` | 6.6e-4 |
        | `comb` | **5.7e-6** |
        | `collapsed` | 4.6e-3 |

        `comb` 的地板极小是因为 Sinkhorn 双随机归一化把 dtype 误差压掉了。
        这顺带给了一个免费自检：**`comb` 必须行列和都≈1，不满足就是接线错了**
        （实测 row-sum 0.996–1.004、col-sum 恰为 1.0）
- [ ] P3.3 **NoPE MLA**：拆 `npu_kv_rmsnorm_rope_cache` + 20 处 split 早退 + KV buffer 二元组语义 + `trans_rope_weight` assert
      - **实测已撞到 D3 的第一处**：`npu/modules/deepseek_v2_attention_mla_npu.py:368`
        无条件访问 `m.rotary_emb.is_neox_style`，而 GLM 的 `qk_rope_head_dim=0` 根本没建 rope 模块
        → `AttributeError: 'NoneType' object has no attribute 'is_neox_style'`
      - [x] **已修第一批（4 处）**，`deepseek_v2_attention_mla_npu.py`：
        引入 `has_rope = m.qk_rope_head_dim > 0`；① 分流不再问 `rotary_emb.is_neox_style`；
        ② `fused_split_qk_norm` 没有 rope=0 形态，rope=0 时走未融合的普通 split；
        ③ `sin_cos_cache` 预取与 rope 应用整体跳过；
        ④ 下游 `attn_mqa` 的 `q_rope`/`k_rope` **传 `None` 而不是零宽张量**
        （依据 A2：这两个参数本就是 Optional）
      - **实测生效**：AttributeError 消失，前向推进到 indexer
      - ⚠ 仅证明「不崩」，**数值尚未对拍**（要配 HF `Glm5NextTextAttention` golden）
- [ ] P3.4 **kpool indexer**：解掉 `dsa/kpool_fp8_index.py:583/589` 与 `dsa/dsa_indexer_kpool.py:1766`
      的非 CUDA 硬拦，用 `torch.topk` 打通 `group_topk=512`
      （⚠ 路径已随 rebase 移入 `layers/attention/dsa/` 子目录，行号仍对得上）
      - **这是当前 GLM BF16 前向的最前沿阻塞点**：`deepseek_v2_attention_mla_npu.py:457`
        调 `m.indexer(...)` → `NotImplementedError`
      - 做法见 §3 的 P3.4 段（路由到拆解路径 + 自写 torch ragged top-k + 补 `page_table_row_index`）
- [ ] P3.5 **出口判据**：四个模块逐层 golden 对齐

#### P3 的 golden 来源：**HuggingFace `transformers==5.16.1`**（2026-08-28 确定）

> **这一条推翻了 §0 里"精度基准 = CPU 上跑 GPU 分支参考实现"的原方案。**
> 原方案对 kpool **物理上做不到**：我们的配置 `index_topk=2048, index_kpool=4`
> → `group_topk=512` → 走 `fast_kpool_topk_transform_fused`，它背后是
> `kernels/jit/csrc/dsa/kpool_topk_transform.cuh` —— **JIT CUDA kernel**，
> NPU 跑不了、CPU 也跑不了、triton-ascend 也救不了（它不是 triton）。
> 且本机**没有任何 CUDA 卡**，拿不到 GPU 参考输出。

**transformers 5.16.1 带了完整的 `glm5_next` 纯 PyTorch 实现**（5.16.0 还没有，是 5.16.1 才加的），
四个模块**全覆盖**，且**零 cuda / triton / flash 引用**，CPU 直接可跑：

| P3 模块 | HF 对应实现 |
|---|---|
| P3.1 KDA | `Glm5NextTextLinearAttention`、`recurrent_kimi_delta_attention`、`chunk_kimi_delta_attention`、`causal_conv1d_fn/update`、`Glm5NextTextForgetGate`、`Glm5NextTextRMSNormGated` |
| P3.2 mHC | `Glm5NextTextHyperConnection`、`Glm5NextTextHyperHead`、`Glm5NextTextUnweightedRMSNorm` |
| P3.3 NoPE MLA | `Glm5NextTextAttention`、`eager_attention_forward` |
| **P3.4 kpool indexer** | **`Glm5NextTextIndexer`**（`modeling_glm5_next.py:736-1027`，纯 torch） |

- 环境：**独立的 `$ROOT/.venv-ref`**（transformers 5.16.1 + torch 2.10.0 CPU）。
  **绝不能装进 `.venv-glm53`** —— sglang 的 `pyproject_npu.toml` 钉死 `transformers==5.12.1`
- 已实测：`AutoConfig.from_pretrained` 直接读我们的 BF16 目录成功，字段与 §1 逐条吻合
  （`qk_nope=256 / qk_rope=0 / v=256 / kv_lora=512 / index_topk=2048 / index_kpool=4`）；
  `Glm5NextTextIndexer(cfg, layer_idx=0)` 在 CPU 上构造成功
- ⚠ 缩小配置做单元测试时，`num_hidden_layers` 必须与 `layer_types` **和** `mlp_layer_types` 同时截断，否则 config 校验直接 raise

**口径说明（重要）**：HF 实现是**参考实现，不是 sglang GPU 实现的逐位复制**。
两者可能有融合差异。所以 golden 用 HF，sglang GPU 分支的代码当**第三方佐证**（读源码交叉核对），
而不是要求三方逐位相同。

> **P3.4 的做法（已与用户确认）**：不移植那个 fused CUDA kernel，而是
> **在 NPU 上把 `group_topk=512` 路由到 `kpool_fp8_index.py` 里已存在的拆解路径**：
> ragged top-k（自写 torch，替 `fast_topk_v2`——它硬断言 `topk==2048`）
> → `expand_pooled_groups_to_topk` → `append_kpool_tail_to_topk`。
> ⚠ **更正**：只有 `expand_pooled_groups_to_topk` 是纯 torch。
> **`append_kpool_tail_to_topk` 是 Triton kernel**（`kpool_fp8_index.py:466` 启动
> `_append_kpool_tail_to_topk_kernel`，定义在 `:490`）。我先前判它「纯 torch」是因为
> 用 awk 取函数体时结束模式把起始行本身匹配掉了，等于没扫。
> 好消息是我们有 triton-ascend 3.2.2（带 `cann` 后端），**它未必不能跑** —— 待验。
> HF 的 `Glm5NextTextIndexer` 做的正是同一套（`topk(select_k)` → `flatten(-2)` 展开 → `append_visible_tail`），
> 是对这条路线的独立佐证。
>
> 两个必须填的缺口：
> 1. 拆解分支有 `assert page_table_row_index is None`，而 `kpool_plan.py:417` 在分页 decode 时一定会设它
>    → 要给 torch 版 `expand` 加按行的 page_table gather。**已决定一次做全，不先砍分页路径**
> 2. 拆解路径在上游**从没在 `group_topk=512` 上跑过**（`fast_topk_v2` 只支持 2048）→ 我们是第一个走这条组合的
>
> **判据（已与用户确认）**：用「选出的集合 == 真实 top-k 集合 / 所选分数之和一致」，
> **不用「下标逐位相等」**。原因：并列分数的 tie-break 在 `torch.topk` 与 CUDA kernel 之间本就可能不同，
> 且我们**没有 CUDA 卡，任何环节都拿不到 CUDA 参考下标** —— 下标逐位比对在本项目**整体不可行**，
> 端到端只能用精度分数兜底，不能指望回头做下标定位。

### P4 · BF16 端到端 ☐
- [ ] P4.1 TP16 / 32K / 纯文本 / 关 NPU Graph / 关 MTP / 关 CP 启动
- [ ] P4.2 **出口判据**：GSM8K 对齐 CPU golden 与 GPU 分支公开口径
      - **GPU 公开口径已找到**（PR#36507 分支里的 cookbook，`docs/src/snippets/configs/zai-org/glm-5.3-flash-benchmarks.jsx`）：
        **GSM8K 97.50%**，全 1319 题，stop rate 100%，4×GB300 TP4/EP4
      - ✅ **权重版本已核实对上**（2026-08-28）：该数字绑定 `zai-org/GLM-5.3-Flash` revision
        **`c5b82b63e37b`**，本地这份与之 **71/71 文件齐全、62/62 个 safetensors 分片 size 逐个一致**
        （合计 328.34 GB 双方相同）。本地多出的 `LICENSE`/`configuration.json` 是 modelscope 自己加的。
        证据：`hf-mirror.com` 的 `/api/models/.../revision/c5b82b63e37b` + `paths-info`。
        ⚠ 口径说明：这是**文件清单 + 逐分片 size** 比对，**不是 sha256 校验**
      - ⚠ **97.50% 是 thinking 打开测的**，别拿 non-thinking 去比。cookbook 里的原始命令：
        ```
        sgl-eval run gsm8k --base-url http://HOST:PORT/v1 --model zai-org/GLM-5.3-Flash \
          --num-threads 64 --max-tokens 32768 --temperature 1.0 --top-p 0.95 --thinking
        ```
        （`pip install git+https://github.com/sgl-project/sgl-eval`）
      - ⚠ cookbook 的 speed 数字是带 `SGLANG_SIMULATE_ACC_LEN=3` 跑的，**只能当吞吐口径，不能用来对精度**
        （cookbook 自己也这么写：“Never run accuracy against it”）

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
- [x] ~~DSv4 W8A8 权重格式确认~~ → **compressed-tensors**（config.json 自述，且 SGLang 会拒绝 `--quantization modelslim`）
- [x] ~~核实本地 GLM-5.3-Flash 权重版本~~ → **对上了** revision `c5b82b63e37b`（71/71 文件 + 62/62 分片 size 一致；非 sha256 校验）
- [x] ~~DSv4 权重何时删~~ → **P1.2 完成后已删 275 GB 的 safetensors**（2026-08-28）。
      **元数据全部保留**在 `/mnt/workspace/models/DeepSeek-V4-Flash-W8A8/`（约 12 MB）：
      `config.json` / `model.safetensors.index.json` / `tokenizer*` / `.msc` / `.mv` / `README.md`
      → 将来重下可以用 index.json 逐分片校验，确认拿到的是同一版本。
      - 来源：modelscope，`.mv` 记 `Revision:master, CreatedAt:1777033839`；
        `.msc` 里是逐文件 revision（`6f4a67e0…` / `b8b7336b…` / `d3b9201f…`）
      - ⚠ **modelscope 的 repo id 没有记录下来**（权重是用户下的，我没查到），
        真要重下需要先确认 repo id
      - 重下成本实测：275 GB **约 14 分钟**（原始下载 23:43→23:57）
      - 为什么可以删：P2–P6 全是 GLM 自己的事，DSv4 不再参与；且 P5 的 W8A8(333 GB)
        + BF16(643 GB) = 976/984 GB，**本来也容不下 DSv4**
      - 什么时候会想要它回来：(a) 再 rebase 一次时复跑同样的回归；
        (b) GLM 起不来时当"已知能跑"的对照，用来区分模型问题与环境问题
      ~~不删，保留到 P1.2 做完~~（旧决策）。
      删除后磁盘：**669 GB 可用**，P2 的 BF16 需要 643 GB → 余量从 66 GB 变成约 341 GB，宽松得多
- [ ] P5 之后磁盘怎么排：删掉 DSv4 后 BF16(643) + W8A8(333) = 976 GB / 984 GB，只剩 8 GB。
      可能要把 W8A8 写到 `/usr/.devenv`（466 GB 可用）

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
| 2026-08-28 | **换 Ubuntu 24.04 镜像，环境从零重建完毕**：P0.1–P0.6 全部重跑通过，**P0.8 PASS**。附录 B 的 glibc 绕行确认整段不需要 |
| 2026-08-28 | 24.04 上的新坑：**pip 26 不读 `~/.pip/pip.conf`**，不带 `-i` 会静默回落 pypi.org 挂死；**`pybind11` 必须显式装**否则 KDA/conv1d 6/6 import FAIL。已写入 SETUP.md §3 与附录 D |
| 2026-08-28 | **P0.5 在 2.10 上更好**：`import attentions` 由 FAIL 变 OK；仍 FAIL 的只剩 `deep_ep`（`deep_ep_cpp`，TP16+EP 阶段再解） |
| 2026-08-28 | **P0.6 在 torch_npu 2.10 上复跑，A1/C4 结论不变**。新观察：`FIA 512+64/N=128` 在 2.10 FAIL（err 561002），因 2.7.1 无记录**不定性为回归**，且不在 GLM 关键路径 |
| 2026-08-28 | GLIBC 障碍消失的连带影响：§2.4 里 **C3 / C5 两条"被 GLIBC 挡住"的探测现在都可以实跑了**（尚未跑）；A2 的实跑同理已无障碍 |
| 2026-08-28 | ROOT 迁到 `/mnt/workspace/y00359136/work/glm53_dev/env`（与代码仓库同级）；`env.sh.example` 换成 24.04 实跑版 |
| 2026-08-28 | **P0.7a PASS**：DSv4-Flash W8A8 在 A3 **TP16/DP16 + DeepEP** 起服务并推理成功。落盘 `launch_dsv4_a3.sh.example` |
| 2026-08-28 | **deep_ep 修好了**（此前一直标"TP16+EP 阶段再解"）：不是 GLIBC，是 wheel 的**打包 bug** —— `deep_ep/__init__.py` 写的是**顶层** `from deep_ep_cpp import Config`，而 `.so` 装在 `deep_ep/` 包目录里。往 site-packages 放一个指向该目录的 `.pth` 即可。**至此 P0.5 的 21 项只剩 `npu_mla_prolog_v3` 一项 MISS（已知非阻塞）** |
| 2026-08-28 | 发现上游 PR#25144 是 DSv4 昇腾支持的**权威配方**（含全套 env var 与精度口径）。其中 `INF_NAN_MODE_FORCE_DISABLE=1` 标注为**必须**，否则 W8A8 溢出产生 NaN —— 这条对 P5 的 GLM W8A8 同样适用，记在这里免得重踩 |
| 2026-08-28 | 装 SGLang 依赖时发现新的"顶掉 torch"路径：`timm -> torchvision -> torch==2.13.0`（CUDA）。已用 constraints 文件锁死，写入 SETUP.md §8.2 |
| 2026-08-28 | **P1 的 base 变了**：PR#36507 仍未合入 main，但分支头已从 `0b9c38484e` 前进到 **`033446bb05`**，已 fetch 并打 tag `glm53-gpu-ref-033446bb`。实测冲突面**只有 8 个代码文件**，其中只有 `server_args.py` 是高风险（增量里合入了 #36255 的 ServerArgs 声明式重构，我们的 kt-* 参数要按新模型重写） |
| 2026-08-28 | **P4 出口判据有了公开对标数**：GPU 分支 cookbook 报 **GSM8K 97.50%**（全 1319 题 / stop rate 100% / 4×GB300 TP4-EP4），绑定权重 revision `c5b82b63e37b`。**已核实本地权重就是这一版**（71/71 文件 + 62/62 分片 size 一致，非 sha256） |
| 2026-08-28 | **P0.7b PASS，P0 阶段全部完成**：GPQA-Diamond non-thinking 三轮 74.24 / 75.25 / 71.72，**均值 73.74%**（SD 1.82pp），对标 73.23% 差 +0.51pp，在噪声内。0 请求错误 |
| 2026-08-28 | ⚠ 发现计划内部冲突：P1.2 判据要用 DSv4 回归，但原计划 P0.7b 后就删 DSv4 权重。**已决策：保留到 P1.2 做完再删**。磁盘账 918/984 GB，余 66 GB，P2 期间要盯 `df` |
| 2026-08-28 | **P1.1 试跑通过**：19 commit rebase 到 `033446bb05` 只有 **2 个冲突**（`hash_topk.py` / `deepseek_v4.py`），都已解并说明理由；`server_args.py` 自动合并且正确，#36255 重构**没造成返工**。在隔离 worktree 做的，`glm53_dev` 未动 |
| 2026-08-28 | **P1.2 PASS**：rebase 到 `033446bb05` 后 GPQA 三轮 75.76 / 73.74 / 70.20，**均值 73.23%**，对基线 73.74% 差 −0.50pp（噪声内）。权重占用 28.17 vs 28.08 GB/die，无回归 |
| 2026-08-28 | ⚠ 更正一处早先记错的数：DP16 配方的权重占用是 **28.08 GB/die**（16 rank 一致），不是 18.45 —— 18.45 是**纯 TP16 不开 DP** 那次启动尝试的数，两者不可比（DP-attention 下每 rank 各存一份 attention/dense 权重） |
| 2026-08-28 | **P1 完成**：`glm53_dev` 已 promote 到 `033446bb05` + 19 个 NPU commit。回退路径：tag `glm53_dev-pre-p1-rebase`（旧 head `a6be0fe83b`）+ 未 push 的 `origin/glm53_dev` |
| 2026-08-28 | **纯 TP16 跑不通，是结构性约束不是 NPU bug**：`deepseek_v4.py:608` 的 `n_local_groups = n_groups // attn_tp_size`，DSv4-Flash `o_groups=8`，TP16 时 `8//16=0` → `o.view(T,0,-1)` 崩。**attention TP 上限是 8**，所以 PR#25144 必须用 DP16(attn_tp=1)+EP16。GLM-5.3 config 无 `o_groups`，**这条不会搬到 GLM** |
| 2026-08-28 | 权重占用三档实测（同一权重、同一代码树）：DP16+EP16 **28.08 GB/die** / 纯 TP16 **20.39** / 纯 TP16 + `--context-length 32768` **18.45**。三者配置不同，**不可互相比较** |
| 2026-08-28 | **DSv4 权重已删**（275 GB safetensors，元数据 12 MB 保留）。P1.2 是它最后一个用途；P5 的账本来也容不下它。删后 `/mnt/workspace` 可用 669 GB，P2 余量从 66 GB 变 341 GB |
| 2026-08-28 | **P2.1 / P2.2 PASS**：反量化脚本落盘，首个 shard 独立重算**偏差恰好 0**。查明 **ViT (`model.visual`) 本来就是 BF16，无需转换**；`layers.45` 是 MTP 层，一并转 |
| 2026-08-28 | **P2 改为不删源**（DSv4 腾出空间后 668 GB 够放 643 GB 的 BF16），整个 P2 变可逆；FP8 源留到 P4 验过再删 |
| 2026-08-28 | P2.2 加强验证：对首个 shard 做**全量逐位比对**（27.6 亿元素），用独立实现重算，**0 处不一致**。反量化数学本身已证死，剩下的风险只在 I/O 与索引拼装 |
| 2026-08-28 | **P2 完成**：FP8 → BF16 全量转换 62/62，输出 599 GB，出口判据全过（名称集合、形状、dtype、无 scale 泄漏、config 已清洗）。FP8 源**保留**，等 P4 端到端验过再删。磁盘余 70 GB |
| 2026-08-28 | **P3 的 golden 来源换成 HF `transformers==5.16.1` 的 `glm5_next`**（纯 torch、CPU 可跑、四模块全覆盖，含 `Glm5NextTextIndexer`）。原方案"CPU 跑 GPU 分支"对 kpool 物理上做不到：`group_topk=512` 走的是 JIT CUDA `.cuh`，且本机无 CUDA 卡。装在独立 `.venv-ref`，**不得污染 `.venv-glm53`**（sglang 钉 transformers 5.12.1） |
| 2026-08-28 | **BF16 产物通过真实加载路径验证**：16 rank 全部 `Load weight end`，52.03 GB/die，`type=Glm…`，**无 missing/unexpected key、无 shape mismatch**。失败点在其后的显存预算（照搬了 DSv4 的 DP16 配方，DP-attention 会复制 attention/dense），是配置问题不是权重问题 |
| 2026-08-28 | **GLM-5.3-Flash BF16 首次在 NPU 起服务成功**（纯 TP16，权重 **37.25 GB/die** = 599/16，无复制）。踩到的启动期问题：`--page-size` 必须是 **64**（DSA pool 硬断言），不是 DSv4 的 128 |
| 2026-08-28 | **P3.1a 完成**：`attention_registry.py` 的 GLM 分支加了 NPU 路由。**P3.1b 发现新问题**：`ascend_kda_backend.py:365` 的 `g.flatten(-2)` 是 Kimi 的 4-D 布局契约，GLM 的 gate 已经是 flat 的 → 校验失败。**布局问题，非算子能力问题** |
| 2026-08-28 | **P3.2 的 D6 实测坐实**：不加 NPU 分支，第一次前向就 `NameError: deep_gemm`。已用 `SGLANG_OPT_USE_TILELANG_MHC_*=False` 临时绕开。`npu_hc_pre` 封装已存在且 DSv4 在用，P3.2 是接线不是重写 |
| 2026-08-28 | **B2（GLM clipped SwiGLU）撤销**：上机核实与 DSv4 语义逐字相同，昇腾侧已有可用路径，**不需要算子开发**。同时推翻仓库里两处错误注释（`clamp_limit` 只在 `swiglu_mode=1` 生效；`swiglu_clip_quant` 是输出离群裁剪不是输入 clamp），已修正注释 |
| 2026-08-28 | **P3.1 完成并实测通过**：`_flat_kda_gate` 修掉 Kimi/GLM 的 gate 布局契约不匹配（不改算子、零额外运算）。GLM BF16 前向现已越过 KDA，**最前沿阻塞点前移到 P3.3 的 NoPE MLA**（`deepseek_v2_attention_mla_npu.py:368` 假设 rope 存在） |
| 2026-08-28 | **P3.3 第一批修完并实测**：NoPE（rope=0）在 NPU MLA 路径上的 4 处假设已早退，前向越过 MLA。**P3 的四个模块按 PLAN 预测的顺序逐个暴露：KDA → mHC → NoPE MLA → kpool**，现在停在 P3.4 |
| 2026-08-28 | 走到这里为止，**PLAN §2.6「BF16 打通不被任何算子硬卡住」仍然成立**：暴露的全部是框架接线问题（page_size、mHC dispatch、KDA gate 布局、rope=0 空指针），**没有一个是缺算子** |
| 2026-08-28 | **算子交接包完成并经独立 review**：`operator_handoff/`（4 份规格 + 纯 torch 参考实现 + 可执行 pytest + 环境 + 验收判据），`./run_tests.sh` 64 passed / 1 skipped。review 结论见 `operator_handoff/REVIEW.md` |
| 2026-08-28 | **两处我先前的错误结论被独立复核推翻**：① `torch_npu.npu_clipped_swiglu` **可用且逐位相同**（我把默认参数当成了能力上限）；② `append_kpool_tail_to_topk` **是 Triton kernel 不是纯 torch**（我的扫描方法有缺陷）。两条都已更正 |
| 2026-08-28 | **C5 消解**：`npu_quant_lightning_indexer` 在 `cmp_ratio=4` 下返回 pool 级下标 + `-1` 填充，**pooled 打分与分组 top-k 昇腾侧都已有**，缺的只是 B3 的展开/尾部/页表映射 → OP-1 的工作量比原估计小得多 |
| 2026-08-28 | **性能缺口第一名不是 kpool 而是 mHC**：`mhc.py` 的 flat 入口完全没有 `_is_npu` 分支，每次 forward **90 次调用 / 约 12,600 次 kernel launch**（含 19 轮 Sinkhorn 循环）。但 `npu_hc_pre`/`npu_hc_post` **已存在且 DSv4 已在用** —— **这是接线不是算子开发**，即 P3.2 |
| 2026-08-28 | ⚠ **发现两个会挡住 P3 对拍的精度 bug**：① DeepEP routed 专家路径**静默丢掉 `swiglu_limit=10.0`**（`moe_runner/ascend.py:114-118` 只读 `gemm1_clamp_limit`，GLM 是 None），同一层里 shared 专家 clamp 而 routed 不 clamp；② NPU router GEMM 走 bf16（`deepseek_v2.py:567-568`），而 GLM 配置是 `moe_router_dtype: float32` |
| 2026-08-28 | **P3.2 的可行性已用 golden 实测坐实**：DSv4 的 `npu_hc_pre` 对 GLM 真实权重与 HF golden 三个输出全部落在噪声地板内。**并判死了 ×2 的归属——kernel 内部已经乘了，外面不能再乘**。另记录 4 个接线坑（norm_eps 1e-5、输入必须 4-D、权重必须 fp32、norm 不折叠） |
| 2026-08-28 | **P3.2 接线完成**：`_mhc_pre_dispatch`/`_mhc_post_dispatch` 加 NPU 分支走 `npu_hc_pre`/`npu_hc_post`，启动脚本里那两个 `SGLANG_OPT_USE_TILELANG_MHC_*=False` 绕过已删。`npu_hc_post` 的 `post` 必须 2-D（实测 `[s,n,1]` 被拒）。前向已越过 mHC |
| 2026-08-28 | **profiling 的时机定了**：现在做不了 —— 端到端跑不通（卡 P3.4），采不到 trace。顺序是 **P3.2 接线 → P3.4 先用 torch 兜底跑通 → 再 profiling 重排序**。⚠ 目前性能清单里的数字（12,600 次 launch 等）**全是静态推算不是实测**，profiling 之后要回来核 |
