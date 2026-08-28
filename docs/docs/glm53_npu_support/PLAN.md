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
  torch 侧 `x.to(torch.float8_e4m3fn)` 直接触发 device 异常

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

### 2.3 确认要开发 / 要改

| # | 项 | 结论 |
|---|---|---|
| **1** | **kpool index-K cache: fp8 → int8** | A3 无 fp8，4 个 compress-write Triton kernel 因此无法编译。**int8 实测比 fp8 更准**（键误差 4.2× 更低；32k 下选择重合 99.18% vs 96.53%）。三个条件见 `operator_handoff/specs/op1_*.md` |
| **2** | **`npu_kv_rmsnorm_rope_cache` 支持 rope=0** | 实测（2.7.1 与 2.10 两版）v1/v2 在 rope=0 时**双双 RuntimeError**。这是唯一一个从头到尾站得住的算子需求 |
| **3** | **全零 rope 的 workaround** | `npu_sparse_flash_attention` 签名 Optional 但**实际不接受缺省**；只有 `attention_mode=2`（MLA，kv_head=1）可用且要求非空 rope。传全零 rope 数值正确（零 rope 贡献恰为 0） |

### 2.4 陷阱（能跑但算错 / 名实不符）

| 陷阱 | 表现 |
|---|---|
| **FIA 在 TND 布局省略 `num_key_value_heads`** | 错 **200×**，**不报错**。BSND 下同一默认值却是对的。已修 `ascend_backend.py` 的 prefill 调用点（全文件唯一漏传的） |
| **`npu_clipped_swiglu` 的默认参数** | 四个默认值（`alpha=1.702, limit=7.0, bias=1.0, interleaved=True`）**对 GLM 全错**，只传部分错 109× |
| **Hadamard-128 不能删** | A7 曾判它可删（正交归一、q/k 同旋转、点积不变）。**该结论只对 bf16 indexer 成立** —— 一旦量化，它正是 int8 优于 fp8 的原因（旋转后 kurtosis≈3.0） |
| **`ue8m0` scale 舍入** | 对浮点格式免费，对 int8 要付一个真实 bit（32k 重合 99.18% → 98.84%） |
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

### 2.6 仍未验证

- **`torch.ops.custom.compressor`** 与 **`npu_quant_lightning_indexer`** 的 kernel —— 都需要活的 pool/page-table，
  独立 harness 驱动不了，返回不透明错误
- **`npu_quant_lightning_indexer` 无法表达 GLM 的 indexer**：metadata 算子**只接受 `num_heads_q=64`**，GLM 是 32（实测）
- `npu_sparse_attn_sharedkv` 的 kernel（metadata 能建，kernel 需活 pool）
- **昇腾侧由谁计算 indexer logits** —— CUDA 侧是 `deep_gemm.fp8_paged_mqa_logits`（CUDA 非 Triton，不会随 Triton 路径可用）。
  **这是目前唯一真正开放的算子问题**

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
- [ ] **P3.3 NoPE MLA** —— 已修 4 处 rope=0 早退（分流、`fused_split_qk_norm`、cos/sin 预取、rope 应用；
      下游 `q_rope/k_rope` 传 `None`）。**剩余**：20+ 处 split 早退、KV buffer 二元组语义、
      `trans_rope_weight(w,0)` 的静默损坏加 assert。**数值未对拍**
- [ ] **P3.4 kpool indexer** —— **当前阻塞点**。整个子系统是 Triton/CUDA，纯 torch 兜底等于移植子系统。
      路线：int8 化 index cache（见 §2.3）+ 解掉 `dsa/kpool_fp8_index.py:583/589`、`dsa/dsa_indexer_kpool.py:1766` 的非 CUDA 硬拦
- [ ] **P3.5 出口判据** —— 四模块逐层 golden 对齐

### P4 · BF16 端到端 ☐
- [ ] P4.1 TP16 / 32K / 纯文本 / 关 NPU Graph 启动
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
- [ ] P6.6 NPU Graph
- ⚠ **所有性能数字目前都是静态推算**，端到端跑通后必须用 profiling 重排序

---

## 4. 待决与已知缺陷

- [ ] **P5 的磁盘**：BF16(643) + W8A8(333) = 976/984 GB → 必须先删 FP8 源
- [ ] **两个会挡住对拍的精度缺陷**（发现但未修）：
      ① DeepEP routed 专家路径**静默丢掉 `swiglu_limit=10.0`**（`moe_runner/ascend.py:114-118` 只读
      `gemm1_clamp_limit`，GLM 为 None）→ 同层内 shared 专家 clamp 而 routed 不 clamp；
      ② NPU router GEMM 走 bf16（`deepseek_v2.py:567-568`），而 GLM 配置是 `moe_router_dtype: float32`
- [ ] **DSv4 的一处潜在 bug**：bf16 fallback 读 int8 buffer 却**不施加 scale**（`ascend_dsv4_backend.py:637-641`），
      仅因 `:685` 无条件强制 int8 而不可达
- [ ] **triton-ascend 的 `_hadamard128` codegen 缺陷**（UB 越界，上下文相关）—— 值得上报。
      在三个 compress kernel 内部正常，独立跑必挂
- [ ] `deep_ep` 的打包 bug 已用 `.pth` 绕过，可向上游反馈
