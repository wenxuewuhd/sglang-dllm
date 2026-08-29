# GLM-5.3-Flash 昇腾（Atlas A3 / Ascend910_9362）适配计划

> 活文档。**只记当前事实与计划，不记怎么走到这里的** —— 过程看 `git log`。
> 每条结论都标了证据等级：**实测**（本机跑过）/ **源码**（读代码或头文件得出，未执行）/ **推断**。
> 最后更新：2026-08-29

---

## 现状（2026-08-29）

**算子开发需求：0 项。** 五条推断出来的缺口逐条上机核实，**五条全部证伪**，
`operator_handoff/` 已清空。详见 §2.5。

| | 状态 |
|---|---|
| 环境 / 分支合流 / BF16 权重转换 | ✅ P0–P2 |
| **五类层逐层对拍**（DSA / KDA / MoE / mHC / dense FFN） | ✅ **全部端到端已验**，真实 TP16 形状，回归脚本在 `layer_check/` |
| **NPU Graph 捕获** | ✅ 五类层各自 + **两个完整 decoder 层捕进同一个图**（走真实 `NPUGraphRunner`）+ 多 bs 共池 + 2 卡 HCCL |
| **整网** | ✅ **2026-08-29 09:20 跑通**。TP16 真实 HCCL、45 层、prefill + decode、并发 ragged 批。见 P4.1 |
| **eager 精度判定**（回归阶梯第 1 级） | ✅ **8/8 在测出来的地板内**，最差 0.91×。见 `REGRESSION.md` |
| **整网 NPU Graph** | ✅ **2026-08-29 11:06 跑通**。45 层 / 6 个 bs 桶 / 16 卡 HCCL 全在图内；同 batch 宽度下与 eager **逐位相同**；decode **约 8×**。见 P6.6b |
| **P4.2 出口判据 GSM8K** | ✅ **97.35%**（全 1319 题、stop rate 100%、图模式 TP16 128 并发），判据 97.50%，差 0.32 个 SE。见 P4.2 |

**eager 基线的 logits 判定：已通过**（2026-08-29）。地板测出来了 ——
fp32 与 bf16 跑同一件事，逐提示 mean|dlp| **9.6e-3 ~ 2.85e-1**；
eager 服务对 fp32 CPU 参考 **8/8 在地板 × 2.0 之内，最差 0.91×**。
原先说不清的那些数字（0.013–0.25）从来就不可疑，缺的只是地板。方法与数据见 `REGRESSION.md`。

**eager 基线已全部录进 `$ROOT/goldens/logits/`**，覆盖回归阶梯 1、2、4 三级
（短提示 prefill / 每条 100 token 贪心 decode / **3256 token 长提示**，
后者 `> index_topk=2048`，稀疏选择真的走了，段落里六个事实全答对）。
**这是刻意提前录的** —— graph 一起来 eager 服务就没了。

**graph 已开且已对齐**（2026-08-29 11:06）：同一 batch 宽度下 graph 与 eager **逐位相同**
—— 8 条短提示 prefill、1000 个 decode token、2 条 3255/3252 token 长提示 + 200 decode token，
`max|dlp|` 全部 `0.000e+00`。decode **33–35 token/s，对 eager 的 4.2–4.6 约 7.7×**。详见 P6.6b。

**P4 已闭环**：GSM8K 全量 **97.35%** / stop rate 100%（判据 97.50%，差 0.32 个 SE）。
一轮 1360 秒 —— eager 下同样的事要 11 小时以上。

**下一步的顺序**：在 graph 下重做 P6 的性能排序 → P5 量化。
⚠ 现有的性能条目全是 eager 时代量的，**排序会变**：host 开销类的（`TASK_QUEUE_ENABLE`、
减少 aten dispatch）已经被图吃掉了，device 时间类的（AI_CPU 回退、int64 算术、标量瓶颈 kernel）
才继续值钱。

⚠ 起服务**必须独占整机**：GLM BF16 只能 TP16（TP8 每卡要 74.9 GB，放不下），
而 `bootstrap.py:339` 要求每卡空闲显存 ≥ 90%。**停服务后显存不是立刻回收的。**
⚠ 现有的性能数字全是 eager 时代的，**开 graph 后要重测**：host 开销类的优化
（`TASK_QUEUE_ENABLE`、减少 aten dispatch 次数）会被图吃掉，
device 时间类的（AI_CPU 回退、int64 算术、标量瓶颈 kernel）才继续值钱。

**欠账**在 `SHARED_CHANGES.md`：改到共享路径的每一处、谁受影响、还欠什么回归。
其中 **DSv4 的 GPQA 回归**尚未跑（swiglu_limit 那条改动欠的）。

---

## 0. 目标与已定决策

| 项 | 决定 |
|---|---|
| 一期目标 | GLM-5.3-Flash **纯文本**在 A3 上跑通并闭环精度 |
| **性能目标** | **BF16 + NPU Graph 下的最优性能**（用户明确要求）。图模式是硬要求，不是选项 —— host 开销类的优化会被图吃掉，**device 时间类的才是长期值钱的** |
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
- **单 die 可达 HBM 带宽 ≈ 1.25 TB/s（读写流）/ 1.17 TB/s（纯读）**，四个 die 离散 <1.5%。
  ⚠ 之前文档里用的 1.6–1.7 TB/s 是**凭记忆给的、高了约 36%**，据此算出的所有倍数都已作废重算。
  **厂商标称值找不到**（`npu-smi` 与 CANN 配置都不给），只能报实测
- **L2 = 168 MB**（`Ascend910_9362.ini`）。32–64 MB 的工作集实测到 2.2–2.5 TB/s ——
  **小于约 168 MB 的工作集可能根本不碰 HBM**，拿 HBM 带宽当下界会低估
- **每 kernel 固定开销约 13.5 µs** → **流量小于约 16 MB 的 kernel 一律由 launch 开销主导**，与带宽无关
- **`TASK_QUEUE_ENABLE=2` 把 launch 开销从 13.4–15.7 µs 降到 7.7–8.6 µs**（`0` 是最慢的 17.3 µs）。
  实测 DSA decode **1.74×**（4.857 → 2.788 ms），而 device-bound 的 prefill/MoE 纹丝不动。
  ⚠ 仓库自己的 `test/registered/npu/performance/glm5_1/` 里 **decode 节点用的是最慢的 `0`**。
  **正确性/确定性影响未测，上生产前必须验**
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
- `/mnt/workspace` 984 GB，**已用 655 / 余 329 GB**（2026-08-29）
- **FP8 源已删**（2026-08-29，P4 闭环后）：62 个 shard / 306 GB，
  **元数据 28 MB 保留**（config / tokenizer / chat_template.jinja / index.json），
  revision 溯源不丢。删前核过 BF16 自足：62/62 shard、0 缺失、38770 张量、
  除 `README.md` 外元数据齐全
- GLM BF16 **599 GB**（`du` 报 642.7 GB）
- ⚠ **P5 的 W8A8 约 333 GB，而现在余 329 GB —— 余量非常薄**。
  接手量化的人要先规划好，中途撑不住的话只能再动 BF16

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
- DSv4 的 TP16/DP16+DeepEP 配方见 `launch_dsv4_a3.sh.example`；**GLM 的见 `launch_glm_bf16.sh.example`**（已进仓库）
- **起服务要求整卡近乎全空**：`distributed/bootstrap.py:339` 检查「空闲显存 ≥ 总量 90%」，
  不满足直接 raise。⚠ **停掉 agent 之后显存不是立刻回收的** —— 本项目因此失败过一次
  （停完不到一分钟就起，每卡只有 51% 空闲）。而且 GLM BF16 只能 TP16
  （TP8 每卡要 74.9 GB，放不下），所以**整网必须独占整机**
- **启动脚本里那四个 `SGLANG_OPT_*=False` 现在都走不到**（源码核查，未经一次成功启动确认）：
  它们是 mHC 接到昇腾融合算子（`1cc9eda1c5`）**之前**的绕行，修复把绕行变成了遗留。
  逐条依据写在 `launch_glm_bf16.sh.example` 的注释里。**暂不删** —— 整网基线还没跑通，
  不要在那次高成本启动里同时引入新变量。
  ⚠ 其中 `SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch` 是**靠巧合工作的**：`"torch"` 不是合法取值，
  它落进 `else` 到 `_linear_bf16_fp32_cublas`，而那个函数的非 CUDA 兜底恰好是
  `torch.mm(x.float(), ...)`。哪天有人加了真的 `"torch"` 分支或取值校验，行为就变
- **`server_args.py` 没有 `is_npu()` 平台块**（只有 `is_sm120_supported()` 和 `is_hip()`），
  所以每个 NPU 部署都得在启动脚本里手工设这些开关。这四个眼下多余，但下一个真需要的开关会重演

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

### 2.3 曾经的三条「要开发」—— 两条已实现，一条已撤销（**没有待办**）

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
| **多索引张量的高级索引会回落到 AI CPU** | `t[i0.unsqueeze(1), i1]` 这种两个索引张量的写法**没有 AI Core 实现**，落到 `aclnnIndex_IndexAiCpu_Index`。实测：搬 35 KB 花 **196.9 µs**，换成扁平化后的 `index_select`（`aclnnIndexSelect_GatherV3AiCore`）只要 **7.3 µs**，**27×**。在真实 DSA decode 里它每步跑 2 次、每次 293–308 µs，**占该层全部 device 时间的 37.5%**。⚠ **Python 层的 D2H 计数器看不见它** —— 它不是 stream sync，是一个跑在控制 CPU 上的 device 算子，只有 kernel 级 profiler 能发现 |
| **KDA prefill 把 raw `beta` 直接喂给 chunk kernel** | GLM 交出来的 gate 是扁平的（`[1,T,H*K]`）、beta 是 raw；**CUDA 路径正是拿这个当判据**（`kda_backend.py:684` 的 `gate_was_flat`）并在 `chunk_kda` 内部做 sigmoid，而 Ascend 的 extend 链没有这个钩子。后果：**out.prefill 70.55× budget、out.decode 142.48×、ssm state 943.61×**，全程不报错。已修（按 `g.ndim == 3` 判据，与 CUDA 同源）。**模块级 golden 抓不到这个**——它只在真实路径上暴露 |
| **Hadamard 在 bf16 里做** | CUDA kernel 把 bf16 读进 **fp32 寄存器**再变换（`hadamard_jit.cuh:150` 的 `float x_vals[..]`），Triton 的 `_hadamard128` 同样作用在 fp32 accumulator 上。在 bf16 里做要 round 7 次而它们 round 1 次，**不报错**，只是悄悄挪走一批 pool —— 32k 下选择重合掉 0.0006（实测）。这个 bug 我写过一次，被端到端对拍抓出来 |
| **MoE 的路由在 fp32 与 bf16 之间会翻，从 layer 3 就开始** | 实测：top-8 集合不同的 token 占比 layer 3 为 12.5%、layer 41 达 **63.3%**。后果不是"精度差一点"，而是**双参考法的地板从第一个 MoE 层起就由离散的路由差异主导，不再是舍入**——地板从 9.5e-3 涨到 1.8e-1。**深层的宽地板不能当成"宽误差可接受"的依据**；注入 5% 误差实测只有 layer 7–25 测得出来，layer 26+ 测不出。这是**验收方法本身的边界**，不是某个算子的问题 |
| **NPU 的 bf16 矩阵乘不是 batch-shape 不变的** | 同一份输入只改 M（4096 行 vs 4080 行），过 `wk`+`k_norm` 后 **5/4080 行**差 1 个 bf16 ulp，gate 差 6/4080（实测）。根因在 torch_npu 的 matmul tiling，不在业务代码。**后果：NPU 上任何 prefill-vs-decode 的逐位一致性断言都不成立**，包括 P4 打算用的 KL 一致性 —— 只能定阈值，不能要求 bit-exact |
| **KDA prefill 的 Triton autotune —— 确认会在真实路径上触发** | 原记「实际服务未触发，列为上线前须确认」，**现已确认：每次 prefill 都走**。`chunk_kda_scaled_dot_kkt_fwd` 由 `_AscendKDAExtendKernel.extend` 直接调用，24-config 的 `do_bench` 扫描在**第一个** config 上就把 AI core 打超时（507014 `aicore timeout`，dies 0/1/2 复现）。单独跑 14 个 config 里 13 个正常，**所以问题在 `do_bench` 本身，不是某个坏 config**。已修：在 NPU backend import 时钉死一个 config（triton 在 `len(configs)==1` 时完全跳过 benchmark），实测 `BK=64` 最快（T=8192/H=4 下 3.98 ms vs BK=32 的 4.68 ms）。**没有改共享的 `kda.py`** |

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
- ~~`npu_lightning_indexer` 在 **NPU Graph 捕获**下能否用~~ —— **已验，可用**（P6.6，实测）。
  `npu_sparse_flash_attention` 同样可用。**性能**仍未测

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

## 3. 阶段计划

| 阶段 | 状态 | 出口结果 |
|---|---|---|
| **P0 环境** | ✅ | 24.04 上重建；算子可见性 21 项仅 `npu_mla_prolog_v3` MISS（非阻塞）；DSv4 冒烟 + **GPQA 73.74%**（自跑三轮 74.24/75.25/71.72 的均值）。对标的 **73.23%** 是公开数字，出处是本仓库的昇腾教程页 `docs/.../tutorials/deepseek_v4_flash.mdx`（GPQA-Diamond 198 题、evalscope、**thinking 关闭**、均值 73.23%、样本 SD 2.20pp），不是 cookbook，也与 GLM 无关 |
| **P1 分支合流** | ✅ | rebase 到 GPU 参考实现 **`033446bb05`**（tag `glm53-gpu-ref-033446bb`），19 commit / 2 冲突；回归 **GPQA 73.23%**（同样是三轮均值 75.76/73.74/70.20，差 −0.50pp） |
| **P2 BF16 权重** | ✅ | 62/62 shard → **599 GB**；首 shard **27.6 亿元素逐位比对 0 处不一致**；名称/形状/dtype 全量核对通过 |

### P3 · 逐模块对拍 ✅（五类层全部端到端已验）
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
  - [x] ~~ACL/graph capture（§2.9 有 host sync，**推断**会失败）~~ —— **推断错了**：decode 整条链
        （indexer + 稀疏注意力 + 缓存写入）捕获成功且逐位可复现，见 P6.6。§2.9 的 host sync 全在 extend 侧
  - [x] **chunked prefill —— 已验，单序列跨 forward 是对的**（2026-08-29，`tools/check_chunked_prefill.py`）。
        ⚠ **此前所有「长上下文」测试都没碰到它**：最长的提示 3256 token 而
        `chunked_prefill_size=8192`，**单条序列从来没被切过**；日志里那些 `#pending-token > 0`
        是**批级**切分（多条序列共分 8192 预算），和「一条序列跨 forward」是两个机制。
        用 9958 token 的提示才真的切开（日志 `#new-seq: 1, #new-token: 8192, #pending-token: 1766`
        → `#new-token: 1792`）：

        | 检查 | 结果 |
        |---|---|
        | 边界**之前**埋的独特事实能否召回 | ✅ FOUND（34 层 KDA 的 conv/SSM 状态、kpool 增量写入都跨过了边界）|
        | 切 vs 不切（`--chunked-prefill-size 32768`）的 logprob | **mean\|dlp\| 7.767e-04**，比形状地板 2e-2 低 **25 倍** |
        | 贪心续写 | **token 序列完全相同** |

        提示 9958 > `index_topk=2048`，**稀疏选择是真的走的**。
  - [x] **「未对齐 chunk 起点」—— 构造不出来，不是没验**（源码 + 实测）。
        两条路径都被挡住：① `chunked_prefill_size % page_size == 0` 是启动期断言
        （`server_args.py:10051`，实测传 5000 直接 raise）；② radix 前缀复用把命中长度
        **向下取整到页边界**（`radix_cache.py:222,227` 的 `(matched // page_size) * page_size`）。
        **这条从待办里划掉。**
  - [ ] **仍未验证**：
        TP>1（单层 harness 意义上）；多 DSA 层共享 pool；只测了 layer 3、一条 prompt
  - [ ] **接 spec decode 前必须先解决**：`kpool_decode_update_index_cache` 假设每请求一行，
        MTP 一次多 draft token 会让同一 `req_pool_index` 的多行抢同一个 ring 槽。
        共享 CUDA kernel 有 `kpool_max_closed_pools` 那套多 token 逻辑，NPU 这条没有
  - [ ] DSA 注意力本体还缺全零 rope 的接线（§2.3 第 3 条），否则拿到 topk 也跑不出注意力

- [x] **Dense FFN** —— 端到端已验（真实 TP16 每卡形状，M=1/16/8192），最差 0.66× 预算。
      两处与预期不同：dense FFN **根本不调 `npu_clipped_swiglu`**（走 `chunk`+`clamp`+`npu_swiglu`），
      所以那个 109× 的默认参数陷阱在这条路上不适用；而且**真实输入下 clamp 从不触发**
      （max|gate_up| = 2.17，limit 是 10），要验 clamp 必须放大输入
- [ ] **P3.5 出口判据** —— 四模块逐层 golden 对齐

### P4 · BF16 端到端 ☐ ← **当前战线：eager 已判定通过，等开 graph**
- [x] **P4.1 TP16 / 32K / 纯文本 / 关 NPU Graph 启动** —— ✅ **2026-08-29 09:20 跑通**
      - 权重 37.25 GB/die，`max_total_num_tokens=1195072`，可用 7.66 GB
        （比 fp8 索引缓存那版少 1.53 GB —— **bf16 索引缓存的实测代价**，此前只有推算）
      - 短 prompt：`"The capital of France is"` → `" Paris. In French, Paris is spelled ..."`
      - **长 prompt 3652 token > `index_topk=2048`，稀疏选择真的走了**，答案正确。
        短 prompt 走的是 `skip_logits_computation`，**测不到 kpool**，别拿它当验证
      - 4 个并发 ragged 请求（131/371/1211/2711）全部答对，同一批里既有稀疏也有跳过
      - 日志无异常（只有启动时 `/freeze_gc` 的连接竞态，无害）
      - ⚠ 耗时约 **480 ms/token**（eager、无 graph、无预热），**不是性能结论**
      **拉通过程中修的三个 bug**（都是「一次没执行过」暴露的，见 git log）：
      `IndexerKPool.forward_npu` 缺失、`AscendHybridLinearAttnBackend.forward_metadata` 缺失、
      `HybridLinearKVPool` 不转发 `set_index_k_bf16`。后两个是**同一类**：
      GLM 的顶层对象是包装，而新方法加在了被包的那个上 —— **单层 harness 结构上发现不了**
- [x] **P4.2 出口判据 —— 已通过**（2026-08-29，图模式、TP16、128 并发）：

      | | 准确率 | stop rate | 抽取失败 |
      |---|---|---|---|
      | run 1 | 1280/1319 = **97.04%** | 100.00% | 9（抽取器缺陷，见下）|
      | run 2 | 1284/1319 = **97.35%** | 100.00% | **0** |
      | 判据 | 97.50% | 100% | — |

      run 2 距判据 **−0.15pp = 0.32 个 SE**；cookbook 自己的 FP8-KV 行就是 97.35%，
      它把这 0.15pp 写成 "inside sampling noise"。两轮差 +0.30pp，也在 1 个 SE 内。
      全量 1360–1490 秒（0.89–0.97 q/s），平均 232–242 completion token。
      **eager 下这需要 11 小时以上**。结果存在 `$ROOT/goldens/gsm8k/`（含响应原文）。

      ⚠ **run 1 的 97.04% 被工具低估了**：39 个错里 9 个是抽取失败而非答错 ——
      模型写 `\boxed{70\%}`，而 `float("70\%")` 抛异常后直接返回 None，没有回退。
      已修（并从 `</think>` 之后的作答段取数，避免命中思考里的中间数字），run 2 是 0 例。
      **响应原文现在会一并存进结果文件** —— 这次就是因为没存，重新打分只能重跑 25 分钟。

      ⚠ **不需要像 DSv4 GPQA 那样跑三轮**：那是 198 题、单轮噪声 ±6pp，一轮没意义。
      GSM8K 用的是**固定的全部 1319 题**、与 cookbook 同一套，题目抽样方差整个抵消，
      只剩解码随机性，上界 0.47pp。
      **真正需要多轮的是 P5**（判据「回归到 BF16 1% 以内」）：每侧 1 轮时 1pp 只有 1.5σ、
      2 轮时 2.1σ。**上面这两轮就是 P5 的 BF16 基线**，别丢。

      判据出处：**GSM8K 97.50%**（全 1319 题、stop rate 100%、4×GB300 TP4/EP4）。
      出处是仓库里的 cookbook 配置 `docs/src/snippets/configs/zai-org/glm-5.3-flash-benchmarks.jsx`
      的 `accuracy: { gsm8k_pct: 97.50 }`，权重 revision **c5b82b63e37b**（与本项目所用一致）。
      **cookbook 对 GLM 只有 GSM8K，没有 GPQA** —— 项目里那些 GPQA 数字全是 DSv4 的合流回归，别混
      - ⚠ **单次全量的噪声约 ±1pp**（1319 题、p≈0.97 的二项 SE 是 0.47pp）。cookbook 自己把
        FP8-KV 变体的 97.35% 对 BF16 的 97.50% 判为 "inside sampling noise"，可以作参照。
        ⚠ 不要拿 DSv4 GPQA 那 3.5pp 的跨度类比 —— **那是 198 题**，±6pp 才是它的噪声
      - ⚠ **该数字是 thinking 打开测的**（`temperature=1.0, top_p=0.95, max_tokens=32768`，`sgl-eval run gsm8k --thinking`）
      - ⚠ cookbook 的速度数字带 `SGLANG_SIMULATE_ACC_LEN=3`，**只能当吞吐口径**
- [x] **P4.3 迭代期的快信号 —— 已建立**（不是最终判据）：`tools/logit_check.py` 的
      teacher-forced logprob 对拍，**地板已实测**、判定已自动化（`--emit-floor` / `--floor`）。
      eager **8/8 通过**，最差 0.91×。工具这轮补了四件事：
      `--streaming`（fp32 参考唯一可行的做法，整模型 `from_pretrained` 要 1.2 TB）、
      `--prompt-set long`（3256 token，唯一能让稀疏选择生效的一档）、
      `--decode-tokens`（贪心续写，覆盖 prefill logprob 完全够不到的 decode 路径）、
      `--floor`（判定必须显式传进来，不给默认阈值）
  - [ ] prefill-vs-decode 的 KL 一致性 —— 还没做

### P5 · W8A8 compressed-tensors ☐
- [ ] P5.1 recipe：weight per-channel + act per-token **dynamic**（静态会被 raise）
- [ ] P5.2 ignore list 照搬 checkpoint 的 `modules_to_not_convert`：KDA 34 层全部、indexer 全套、`hc_*` 全部、所有 norm/embed/router
- [ ] P5.3 288 专家校准（覆盖度是主要风险）
- [ ] P5.4 **出口判据**：精度回归到 BF16 基线 1% 以内
- ⚠ `INF_NAN_MODE_FORCE_DISABLE=1` **必须设**，否则 W8A8 溢出产生 NaN

### P6 · 性能 ☐（可与 P3–P5 并行）

⚠ **服务级 profiling 这条路目前不通（实测，2026-08-29）**：对跑着的服务 POST
`/start_profile`（`profile_by_stage=True` + `record_shapes=True`、16 rank、128 并发）
**把 16 个 scheduler 全部段错误打挂**，而且采到的数据本身是废的 ——
analyse 报 `no such table: TASK` / `The collected data has been lost`。
采集期间吞吐直接冻住（90 秒窗口内 0 个请求完成），6 个 step 没跑完，profiler 没能收尾。
**要 profile 就走 `layer_check/kernel_profile.py` 那条单模块路线**（Level1 + PipeUtilization，
已验证可用）；服务级的先别碰，真要碰就降到低并发、去掉 `record_shapes`、
并且**准备好服务会挂**。
**P6.13 overlap scheduler：开，实测 1.23× 且数值不变**（2026-08-29，A/B，200 题 GSM8K、128 并发）：

| | 墙钟 | q/s | token/s | 准确率 |
|---|---|---|---|---|
| `--disable-overlap-schedule`（此前的配方）| 186 s | 1.07 | 248 | 199/200 |
| **去掉它** | **139 s** | 1.44 | **304** | 199/200 |

**按 token 吞吐比 1.23×**（q/s 会被两轮生成长度不同污染，token/s 才是公允的）。
**而且数值一步没动**：对 eager 基线仍然 `max|dlp| = 0.000e+00`（prefill + 200 个 decode token）。
PLAN 里「overlap 下 `seq_lens_cpu` 领先设备张量一步」那个担忧**在 GLM 上没有兑现**。
→ **overlap 应该常开**，`launch_glm_bf16.sh.example` 已改。
⚠ 它此前是关着的，只是因为要让 graph-vs-eager 的对拍只有一个变量。

**P6.14 prefill 图在 NPU 上不是一个开关能解决的（源码定论）**：
去掉 `--disable-prefill-cuda-graph` **没有任何作用** —— 实测 `disable_prefill_cuda_graph: False`
但 `'prefill': {'backend': 'disabled'}`。原因在 `server_args.py` 的
`_disable_tc_piecewise_cudagraph_if_incompatible()`：NPU 上 prefill 的默认后端是
`tc_piecewise`（`cuda_graph_config.py:110` 的 `default_prefill_backend()`），
而规则表里第一条就是 **「non-CUDA hardware (HIP/NPU/CPU/MPS/XPU)」，把 tc_piecewise 整个否掉**。
`full` 那条路是「opt-in per model architecture via the declarative registry」
（注释指向 `arg_groups/overrides.py` 的 `_inkling_overrides`）。

所以要让 prefill 进图，是**两件代码工作**，不是调参：
① 在那个声明式 registry 里为 GLM 注册 full prefill capture；
② **先把 extend 侧的 host 同步清掉** —— `visible_pool_runs` 里的 `int(...max())`、
`_kpool_extend_rows_npu` 的 host 侧构造。捕获期间 `.item()`/`.cpu()` 会抛 **107027**（已实测），
不清掉一定失败。
**收益上界很大**（875 个 prefill 批 / 每批仅 163 token），但代价是真代码。

⚠ 顺带：**问题「prefill 占多少」根本不需要 profiler** —— 服务日志每个 batch 都有时间戳，
不受 profiler 扰动。GSM8K 那 23 分钟窗口实测：**875 个 prefill 批 vs 约 6840 个 decode step**，
每批 prefill 平均只有 **163 token**（约 1–2 道题）。`--disable-overlap-schedule` 下
prefill 不能与 decode 重叠，**每次 prefill 都是 decode 流的一次完全停顿**。


**图模式下的第一份基线（2026-08-29，实测，TP16、`ignore_eos`、贪心、128 并发配方）。**
prefill 与 decode 分开量：先跑 `max_new_tokens=1` 拿 prefill 墙钟，再跑 129 相减 ——
3256 token × 128 并发的 prefill 是 41.7 万 token，混在一个端到端数字里会把 decode 完全盖掉。
脚本 `tools/bench_graph_decode.py`。

| 并发 | 短上下文 (13 tok) ms/token | 短 合计 token/s | 长上下文 (3256 tok) ms/token | 长 合计 token/s |
|---|---|---|---|---|
| 1 | **28.5** | 35.1 | **29.3** | 34.2 |
| 8 | 38.4 | 208.1 | 36.1 | 221.5 |
| 16 | 42.9 | 373.3 | 44.8 | 356.9 |
| 32 | 52.1 | 614.2 | 53.5 | 598.5 |
| 64 | 63.8 | 1003.0 | 61.3 | **1044.3** |
| 128 | 76.2 | **1679.9** | 113.3 | 1130.0 |

**对 eager 的 220–238 ms/token（4.2–4.6 token/s），bs=1 是约 8×。**

**两条结论**：

① **短上下文到 128 并发还没饱和**（1003 → 1680），可以继续往上提。
**卡并发的是 KDA 的 mamba state pool，不是 KV** —— 日志里
`max_mamba_cache_size: 128, conv_state 0.04GB, ssm_state 1.07GB`，
即**每槽 8.75 MB/die**；16 槽换 128 槽只多花 1.11 GB，KV 从 1195072 token
只降到 1113600。按这个单价 256 槽也只再多约 1.1 GB。

② **长上下文在 64 并发就拐了** —— 64 → 128 吞吐只多 8%（1044 → 1130），
而单请求从 61.3 涨到 113.3 ms/token。此时 KV 用量和 mamba 用量都远没满，
所以拐点是 **kpool 的 device 时间**，正是 P6.7 / P6.10 / P6.11 那几条。
**它们是 device 时间类的，图吃不掉** —— 这是开图之后最值钱的一批。

⚠ 相减法有噪声；⚠ 这台机器共用，别人起训练时的数字不可用。

排序按实测/静态估算的影响。**前三项都不是算子开发，算子已存在。**
- [x] P6.1 **mHC** —— **收益已实测**。torch 的 pre 是 **155 次 aten 调用**，融合后 **8 次**；
      155 × 2 站点 × 45 层 = 13,950，和原先估的 ~12,600 次 launch 对得上。
      加速比 **decode 13.8× / prefill 5.1×**，D2H 同步 0 次。
      单站点 p50：decode(M=16) **0.211 ms**、prefill(M=8192) **2.784 ms**。
      **sinkhorn 是个只在 prefill 有意义的旋钮**：44.8 µs/轮，占 prefill pre 的 **46%**，
      decode 下占 **0%**（那里是 launch-bound，20 轮白送）。
      roofline：post 高于带宽下界 2.6×、pre 9.5×（去掉 sinkhorn 是 5×）——
      **prefill 下不是 host 开销主导**，和 DSA 的 100× 不是一回事
- [ ] P6.2 **KDA prefill conv1d** —— **已量化**：Ascend 的 extend 把深度卷积拆成 **3 次调用**，
      而共享 CUDA 路径对整个 qkv 宽度只做 **1 次打包调用**。实测 **6.5 ms / 单层 14.3 ms = 45%**，
      而且这 3 次调用**各带 3 次 host 等待**（prefill 全部 9 次 host 往返都在这里）。原条目： —— `causal_conv1d_fn_npu` 内部退回 `F.conv1d`（`sgl_kernel_npu` 上游的实现选择）；
      且 Ascend 后端拆成 3 次调用而共享后端只做 1 次
- [ ] P6.3 **MoE SwiGLU clamp** —— 现在是 2×clamp + `cat` + `npu_swiglu` 四个 kernel，可换成一个 `npu_clipped_swiglu`
- [ ] P6.4 DeepEP-normal 的 D2H 同步（`moe_runner/ascend.py:270-274`，prefill 每 forward 42 次）—— 修在第三方 wheel 里，先 profiling
- [ ] P6.5 NoPE 未融合的 split+RMSNorm（与 P3.3 同源，一起做）；顺带删掉那个看起来是死代码的 `q.clone()`
- [ ] **P6.7 kpool indexer 的 expand+tail**（实测，单层单 4096-chunk 的最大单项）——
      `expand_pooled_groups_to_topk` 中间物化了 `[4096, 512, 4]` 的 int64（67 MB）再 reshape，
      占 6.3 ms / 单层。11 个 DSA 层 → 每个 4096-token chunk 约 69 ms。**在共享代码里**
      （`kpool_fp8_index.py:379`），改动会影响 CUDA 路径，先 profiling 再决定
- [x] **P6.6 NPU Graph —— 四类层单层捕获全部跑通并验过数值**（实测，die 14/15）。
      回归脚本 `layer_check/graph_capture/`，判据是三问：能不能捕获 / replay 还跟不跟设备输入走 /
      从图里读出来的数还对不对（双参考法）。**逐位**是这里的常态，不是巧合 ——
      replay 与 eager 在同一批输入上逐位相同，所以 `check_*.py` 已经给过的 golden 结论直接迁移。

      | 层 | 捕获 | 换输入后 replay vs eager | 图内输出对 golden |
      |---|---|---|---|
      | DSA（layer 3，TP16 rank0，16 条 ragged 到 32k，真实 decode 形状）| ✅ | 逐位（换 x、换 seq_lens 两种）| 16/16 在预算内，受控比 0.97–1.05×，重合 0.992–1.000 |
      | KDA（TP16 rank0，bs=16，走 runner 的图 metadata 契约）| ✅ | 逐位（out + conv + ssm）| 迁移自 `check_kda.py` 的 6/6（最差 0.31×）|
      | MoE（288 专家 top-8，`--moe-a2a-backend none` 即部署配方）| ✅ | 逐位 | 4/4 在预算内（最差 0.28×），**16 个 rank 的和全部由同一个图 replay 出来** |
      | mHC（layer 20 两个站点）| ✅ | 逐位 | 4/4 在预算内（最差 0.33×）|
      | dense FFN（layer 2，TP16 rank0）| ✅ | 逐位 | 3/3 在预算内（最差 0.20×），TP16 求和同样全部来自 replay |

      顺带证伪/确认的机制事实（实测）：捕获期间 `.item()`/`.cpu()`/`.tolist()` **会抛** 107027，
      不会静默通过；`nonzero`/`unique_consecutive` 因动态输出形状被拒；`cumsum` 可捕获；
      **AI CPU 回落的算子可以捕获**（`aclnnIndex` 在图里正常跑）—— 这条原先被推断成阻塞项，现已证伪。
      `npu_lightning_indexer` 与 `npu_sparse_flash_attention` 在捕获下均可用（§2.6 的未决项之一，已关闭）。

- [x] **P6.6b 整网图模式 —— 已跑通并逐位对齐**（2026-08-29 11:06，实测，TP16 真机）。
      原①②⑤三条已关闭：

      | 问 | 结果 |
      |---|---|
      | 45 层整网能不能捕获 | ✅ 6 个 bs 桶 `[1,2,4,8,12,16]`，**15 秒捕完，图池 0.8 GB** |
      | 16 卡 HCCL 在图里 | ✅ 45 层 × 每层一次全部在图内重放（原先只有 2 卡实测） |
      | KV pool 是否受影响 | ✅ `max_total_num_tokens=1195072`，与 eager **一模一样** |
      | replay 与 eager 数值 | ✅ **逐位相同**，见下 |

      **对拍：在同一个 batch 宽度下，graph 与 eager 逐位相同（`max|dlp| = 0.000e+00`）**
      —— 8 条短提示的 prefill、**1000 个 decode token**、
      以及 **2 条 3255/3252 token 的长提示**（`> index_topk=2048`，稀疏选择真的走了）
      加 200 个 decode token，无一例外。参考是开图前专门录的 eager 基线
      （`$ROOT/goldens/logits/`）。

      **decode 性能：约 33–35 token/s，eager 是 4.2–4.6 → 约 7.7×**（bs=1 单流，
      稳态；紧跟 prefill 的第一个 decode batch 不算）。**GSM8K 因此变得可行了。**

- [x] **P6.6c batch 宽度会挪动 decode 结果，但不是 padding 的锅**（实测）。
      raw_bs 从 1 换到 3–16，同一条请求的 decode **不再逐位相同**：

      | raw_bs | 桶 | padding 行 | 逐位相同 | max\|dlp\| | mean\|dlp\| |
      |---|---|---|---|---|---|
      | 1 | 1 | 0 | 1/1 | **0.000e+00** | **0.000e+00** |
      | 3 | 4 | 1 | 2/3 | 1.174e-01 | 7.401e-03 |
      | 5 | 8 | 3 | 2/5 | 1.060e-01 | 6.329e-03 |
      | 7 | 8 | 1 | 5/7 | 2.826e-01 | 2.049e-02 |
      | 8 | 8 | **0** | 4/8 | 3.879e-01 | 2.162e-02 |
      | 13 | 16 | 3 | 7/13 | 4.792e-01 | 2.170e-02 |
      | 16 | 16 | **0** | 9/16 | 3.163e-01 | 2.065e-02 |

      **padding 行数和误差没有关系** —— `bs=8` 与 `bs=16` **一行 padding 都没有**，
      误差却和有 3 行 padding 的 `bs=13` 一样大；只有 1 行 padding 的 `bs=3`
      反而最小。变量是 **batch 宽度**，不是 padding。P6.6a 修的那条（padding 踩坏
      KDA 状态）在整网上**站得住**。
      **量级对得上独立测出来的形状地板**：CPU 上「同为 bf16、同样的数学、只改 GEMM 形状」
      实测 mean|dlp| ≤ 2.6e-2（`REGRESSION.md`），这里是 2.1e-2。
      **同源**：bf16 GEMM 不是 batch-shape 不变的，ulp 级扰动被 MoE 路由放大成离散翻转。
      分叉处的续写**两边都是通顺且正确的文本**（人工看过中英各一条），不是状态被写坏。

  - [ ] **P6.6 仍未覆盖**：
        ① `enable_torch_compile` + `npugraph_ex` 那条路（`patch_model_npu`）没碰
        ② MTP / spec decode 下的捕获（`kpool_decode_update_index_cache` 每请求一行的假设在那里不成立，
        见 P3.4）；DeepEP-normal 的 MoE（部署配方是 `--moe-a2a-backend none`，走的是已验的 TP dispatcher）
        ③ **prefill 图没开**（`--disable-prefill-cuda-graph`）—— extend 侧的 host 同步还在
        ④ overlap scheduler 仍然关着；开了之后 `seq_lens_cpu` 领先设备张量一步的场景没验

- [x] **P6.6a padding batch 踩坏 KDA 状态 —— 已修**（`ascend_kda_backend.py` 的 `_causal_conv1d_decode`）。
      图宽度固定，raw_bs 小于捕获 bs 时 runner 补齐尾部行并传 `num_padding`
      （`decode_cuda_graph_runner.py:188`），`_replay_metadata` 于是把这些行的 mamba 下标置成
      **-1**（`PAD_SLOT_ID`）。GLM 的 conv 权重是 fp32、conv cache 是 bf16，所以走的是
      fp32 工作集那条分支 —— 它用 `index_select` / `index_copy_` 直接吃 `cache_indices`，
      而这两个算子**不接受负下标，在昇腾上也不报错，是把 AI core 打挂**（507011 aivec error）。
      另一条分支没事，因为 `causal_conv1d_update_npu` 自己带 `pad_slot_id`。
      **34 个 KDA 层，只要有一次 padding decode 就必挂**，等于图模式对 GLM 整个不可用。
      修法：把负下标 clamp 到 mamba 槽 0 —— `MambaSlotAllocator` 正是为此保留了它
      （`mem_cache/allocator/mamba.py`，`free_slots = arange(1, size+1)`）。
      验证 `layer_check/graph_capture/cap_kda.py`：bs=15 / 3 真实行 / 12 padding 行，
      真实行的 out 与 slots 1.. 的 conv+ssm 对未 padding 的 eager **逐位相同**，
      **只有保留槽 0 被动过**；`check_kda.py` 全量回归仍是 6/6、数字不变（clamp 对非负下标是恒等）。
      ⚠ 这条测试**必须把真实请求从 0 号槽挪开**，否则请求 0 坐在保留槽上，测不出任何东西。

- [ ] P6.6 原条目 —— **decode 路径已扫清 host 同步**（实测 `timing.count_syncs`：
      kpool 的 decode 缓存更新 **0 次/调用**）。做法：decode 跳过 `visible_pool_runs`
      （每行本来就自成一段），并把缓存更新改成无分支——**不过滤行，而是给被屏蔽的行
      一个 scratch 目标**。⚠ 这里有个真陷阱：屏蔽行的 `req_pool_indices` 会被 clamp，
      padding 行通常带 0，于是和真实请求 0 撞同一个槽位，**重复下标的写入顺序未定义，
      真实行的写入可能被覆盖**——而这恰恰只在图捕获（padding batch）下发生。
      所以索引缓存多分配一页、tail ring 多分配一行，专供屏蔽行落地。
      **extend 仍有同步**（`visible_pool_runs` 里的 `int(...max())`、
      `_kpool_extend_rows_npu` 的 host 侧构造），但 prefill 本来就不捕获，不需要动
- ⚠ 原「所有性能数字都是静态推算」**已作废**：下面是 kernel 级 profiler（`torch_npu.profiler`
  Level1 + PipeUtilization）实测出的排序。**墙钟看不见其中任何一条**。

**诊断结论：DSA 慢不是因为注意力慢，是因为注意力周围的簿记。** 注意力本身很好 ——
`SparseFlashAttention` decode cube 利用率 **79.7%**、prefill **85.9%**，`LightningIndexer` **88.1%**。
对照 MoE 的 `GroupedMatmul`：搬 346 MB 用 330 µs = **1051 GB/s = 实测 roofline 的 84%**。
MoE 全程只有厂商融合算子，**没有 AI_CPU 回退、没有 Triton、没有 int64 索引算术**；
DSA 每步 170 次 aten dispatch，MoE 只要 25 次。

按实测收益排序：
- [x] **P6.8 消掉 decode 的 AI_CPU gather** —— 已修（§2.4）。占 DSA decode device 时间 37.5%，27× 提升
- [ ] **P6.9 `TASK_QUEUE_ENABLE=2`** —— DSA decode 立得 **1.74×，零代码改动**。先验正确性
- [ ] **P6.10 `expand_pooled_groups_to_topk` 改 int32** —— prefill 的 `aclnnAdd` 花 **5.73 ms**
      产出 `[8192,512,4]` 的 int64（134 MB），高于下界 **43×**。token id 最大约 32768，
      **int32 完全够**，既减半流量又避开 Ascend 上被模拟的 int64 向量运算。**在共享代码里**
- [ ] **P6.11 重写 `_append_kpool_tail_to_topk_kernel`** —— ⚠ **中断时已有关键进展，别从头做**：
      实测**那 4.73 ms 全部来自被 clamp 的 gather load**，单把它去掉就是 **5.557 → 0.282 ms（约 20×）**。
      而当初设计的「三处 store」重写方案**既不必要、本身也是错的**（三处 store 地址区间重叠，
      同一 CTA 内跨线程竞态）—— **那条路不要再走**。未验证的半成品在 `git stash` 第一条。
      原始分析：**4.73 ms**，
      `aiv_vec_ratio=0.027`、`aiv_mte2_ratio=0.0`，**既不算也不搬，是纯标量瓶颈**
      （8192 个 program 打在 40 个向量核上）。**在共享代码里**
- [ ] **P6.12 降低 DSA 的 170 次 host 调用** —— 每省一次约 13.5 µs（开 TQE=2 后约 8 µs）

---

## 4. 待决与已知缺陷

- [ ] **P5 的磁盘**：BF16(643) + W8A8(333) = 976/984 GB → 必须先删 FP8 源
- [ ] **索引缓存改 bf16 后显存翻倍**：每槽 256 B（打包 fp8 是 128+4=132 B）。
      按 11 个 DSA 层折算约 704 vs 363 B/token，相对 MLA KV 的约 11.3 KB/token 是 +3% 量级。
      **这是从 `mem_cache/index_key_cache.py:33-38` 的 buffer 形状推算的，未实测**；
      P4 起服务时量一次。真顶不住的退路见 §2.7
- [x] **两个精度缺陷，都已量化**（原「发现但未修」）：
      ① **DeepEP routed 专家丢 `swiglu_limit=10.0`** —— 缺陷是真的，但 **GLM 的出厂配方走不到**：
      `--moe-a2a-backend none` → `AscendTPDispatcher` → `NPUSwiglu(swiglu_limit=10.0)`，
      clamp 本来就在（实测构造过对象）。只有 DSv4 的 `--moe-a2a-backend deepep` 才命中。
      已修（`npu/moe/activation.py` + `moe_runner/ascend.py`）：`swiglu_quant` kernel
      **一直有 `do_limit`/`limit`，只是没人传**。真实 gmm1 输出上修前 2.85× budget 判失败、
      修后 0.35×，与出厂 `NPUSwiglu` 同数。**影响分层**：layer 3 有 912/2.7e8 个预激活越界
      （19% 的 token 受影响、最差偏 32%），**layer 40 一个都不越界**——只测一层会得出错误结论。
      ⚠ **这个改动会改变 DSv4 的数值**（给它的 routed 专家加上本就该有的 clamp），
      **合入前需要单跑一次 DSv4 GPQA 回归**，本轮没跑
      ② **router GEMM 走 bf16** —— **已修**。看清楚了：`deepseek_v2.py` 里 CUDA 的**每一条**分支
      都特意把 logits 累加到 fp32（最后一条还专门写注释解释），**只有 `elif not _is_cuda`
      停在 bf16**。所以这不是改共享行为，是让非 CUDA 分支跟上。
      代价是**离散的选专家错误，rel-L2 抓不到**：layer 3@8192 的 top-8 集合重合率 0.99291
      （463/8192 = 5.65% 的 token 选错，最差那个输出偏 34%），**同一块 die 上换 fp32
      精确恢复到地板**。耗时 decode +36 µs、prefill 8192 +383 µs（整层的 8%）。
      顺带一条：**`moe_router_dtype` 这个配置项在 sglang 全仓没有任何消费者**
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
- [ ] **预热**：冷:稳比高达 **971–1022×**（dense FFN 实测）。成因已查明：**是「进程内首次使用某个算子」
      的代价，不是编译、不是 tiling 搜索、不是按 shape**。证据：`npu_swiglu` 首用 212–251 ms 而
      **换新 shape 一分钱不花**；三个全新进程复现同一数字（不是 page cache）；`kernel_meta/` 全程为空
      （不是 TBE JIT）。交叉印证：`check_moe.py` 报 MoE decode 首次 257.65 ms，而 `npu_swiglu`
      单算子首用就是 212–251 ms。
      **好消息是缓解极便宜**：预热只需让每个算子在**任意一个** shape 上跑一次，不必扫 shape。
      真实启动脚本带 `--skip-server-warmup`，等于把这笔钱推给第一个真实请求 —— **P4 起服务前处理**
- [x] ~~能不能让 qk_rope=0 走非 yarn 分支~~ —— **问题是空的**：那条分支 GLM 根本进不去，
      不需要改条件。OP-3 已撤销（§2.5），**工单包清空**。已在
      `deepseek_v2_attention_mla_npu.py` 的分支上方留了一行说明这个跨文件不变量
