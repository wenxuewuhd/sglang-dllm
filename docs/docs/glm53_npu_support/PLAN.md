# GLM-5.3-Flash 昇腾（Atlas A3 / Ascend910_9362）适配计划

> 活文档。**只记当前事实与计划，不记怎么走到这里的** —— 过程看 `git log`。
> 每条结论都标了证据等级：**实测**（本机跑过）/ **源码**（读代码或头文件得出，未执行）/ **推断**。
> 最后更新：2026-08-30

---

## 现状（2026-08-29）

**算子开发需求：0 项。** 五条推断出来的缺口逐条上机核实，**五条全部证伪**，
逐条的证伪理由见 §2.5。（原先那个给算子团队的工单包 `operator_handoff/` 已随之删除，
它的验收方法学保留为 `layer_check/ACCEPTANCE.md`。）

| | 状态 |
|---|---|
| 环境 / 分支合流 / BF16 权重转换 | ✅ P0–P2 |
| **五类层逐层对拍**（DSA / KDA / MoE / mHC / dense FFN） | ✅ **全部端到端已验**，真实 TP16 形状，回归脚本在 `layer_check/` |
| **NPU Graph 捕获** | ✅ 五类层各自 + **两个完整 decoder 层捕进同一个图**（走真实 `NPUGraphRunner`）+ 多 bs 共池 + 2 卡 HCCL |
| **整网** | ✅ **2026-08-29 09:20 跑通**。TP16 真实 HCCL、45 层、prefill + decode、并发 ragged 批。见 P4.1 |
| **eager 精度判定**（回归阶梯第 1 级） | ✅ **8/8 在测出来的地板内**，最差 0.91×。见 `REGRESSION.md` |
| **整网 NPU Graph** | ✅ **2026-08-29 11:06 跑通**。45 层 / 6 个 bs 桶 / 16 卡 HCCL 全在图内；同 batch 宽度下与 eager **逐位相同**；decode **约 8×**。见 P6.6b |
| **P4.2 出口判据 GSM8K** | ✅ **97.35%**（全 1319 题、stop rate 100%、图模式 TP16 128 并发），判据 97.50%，差 0.32 个 SE。见 P4.2 |
| **长上下文** | ✅ **2026-08-30 跑通 1,048,576**，**两个构型都确认**：INT8 TP8（8 张 die，不用整机）与 **BF16 TP16 交付构型**。五深度召回 5/5、前缀对拍 `max\|dlp\| = 0.000e+00`；TP16 上 TTFT 292.1 s、decode 25.5 ms/token。见 **P7** |

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

⚠ **BF16 起服务要独占整机**（不是模型属性，是 BF16 的显存约束 —— 详见下面 §「必须独占整机」
那条：**INT8 TP8 只要 38.20 GB/die，8 张卡就够**）。
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
- **每 kernel 固定开销不是一个常数，是一个跟构型走的区间**（2026-08-30 更正）：
  eager + `TASK_QUEUE_ENABLE=1` 下一次 8 KB 的 `ConcatD` 实测 **13.5 µs**；
  **图模式下本项目自己实测 2.8 µs/kernel**（见 P6.2 那一段）；
  `glm53_int8_1card` 的 profile 里最小的那几行是 **1.3–1.5 µs**。
  ⚠ **原文把那个 13.5 µs 当成机器常数写在这里，并据此推出「流量小于约 16 MB 的 kernel
  一律由 launch 开销主导」—— 那条推论作废。** 16 MB 就是 13.5 µs × 1.25 TB/s 反推的；
  用 1.5 µs 反推是约 1.8 MB，**差了 9 倍，而且差在会让人放弃优化小 kernel 的方向上**。
  同一份文档里 P6.2 的 2.8 µs 早就和它打架了，一句除法就能看出来
  （单卡线的加总校验：launch 主导的 10.179 ms ÷ 2177 个 kernel = 4.7 µs，
  而若真是 13.5，2177 × 13.5 = 29.4 ms 已经超过整步 33.348 ms 的大半）。
  **要用这个数就先说清楚是哪个构型下的**；跨构型照抄是这条错误的成因。
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
- ⚠ **「必须独占整机」是 BF16 的约束，不是模型的**（2026-08-30 实测推翻）。
  BF16 TP8 每 die 要 74.9 GB，塞不进 64 GB —— 所以 BF16 只能 TP16。
  但 **INT8 TP8 每 die 只要 38.20 GB**（实测，与 306.1/8 = 38.26 GiB 对上），剩 22.88 GB，
  整除性全过（64/64/32/288 都能被 8 整除）。
  **`ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15` + `--tp-size 8` 直接可用**，
  启动脚本 `$ROOT/run/launch_glm_w8a8_tp8.sh`。
  **意义**：日常验证不用再抢整机，8 张卡就够，可以和别的任务并行。
  ⚠ **但换了配置就不能沿用 TP16 BF16 的基线** —— TP 宽度变了规约顺序就变，精度也不同。
  要在同一配置上自己造 before/after（做法：`git worktree add --detach <改动前的提交>`，
  用它录一份基线，再用改动后的对）。
- **起服务要求整卡近乎全空**：`distributed/bootstrap.py:339` 检查「空闲显存 ≥ 总量 90%」，
  不满足直接 raise。⚠ **停掉 agent 之后显存不是立刻回收的** —— 本项目因此失败过一次
  （停完不到一分钟就起，每卡只有 51% 空闲）。而且 GLM BF16 只能 TP16
  （TP8 每卡要 74.9 GB，放不下），所以 **BF16 整网必须独占整机**。
  ⚠ **INT8 不受此限**（TP8 实测 38.20 GB/die）—— 见上面那条。
  这里曾经把它写成模型属性，于是「要 8 张卡验一下 INT8」被当成不可能的事。
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

⛔ **`tensor.is_cuda` 在这台机器上是 True，所以每一处 `if x.is_cuda` 的门 NPU 都会通过。**
`hardware_backend/npu/utils.py:152` 导入 `torch_npu.contrib.transfer_to_npu`，之后：

| 判据 | NPU 上的值 | |
|---|---|---|
| `tensor.is_cuda` | **True** | ❌ 撒谎 |
| `tensor.device.type` | `"npu"` | ✅ |
| `sglang.is_cuda()` | False | ✅ |
| `torch.cuda.is_available()` | False | ✅ **因为 utils.py:155 手工补回来了** |

⚠ **最后一行是这条的形状**：那行补丁的注释写着「Re-mock torch.cuda.is_available cuz
transfer_to_npu mocks it True」—— **别名早就被知道，补的是撞上的那一个实例，不是这一类。**

**实际后果（2026-08-30 实测）**：`logprob_processor.py:528` 的
`if ... and pruned_states.is_cuda` 放 NPU 进了一个 CUDA 专用的融合 Triton kernel，
而它的编译让 **`bishengir-compile` SIGSEGV，八个 rank 全死** ——
**任何客户端传一个合法的 `top_logprobs_num` 就能打掉服务**。已修（改判 `device.type`），
验过 k=1/5/8/20 全部正常返回。

⚠ **`srt/` 下还有 88 处张量级 `.is_cuda` 门没查。** 每一处背后的 CUDA 快路径
在 NPU 上都是**静默启用**的。这一处靠打死进程暴露了自己；
**其余的更可能是「安静地算错」，那才是这类问题的常见形态。**
=> **新写代码不要用 `tensor.is_cuda` 判平台**，用 `device.type` 或 `sglang.is_cuda()`。


⚠ **`top_logprobs_num` 会打死整个服务**（2026-08-30 实测）。请求里带
`{"return_logprob": true, "top_logprobs_num": 5}` 触发一个此前没编译过的 Triton kernel
的 JIT，**`bishengir-compile` 自身 SIGSEGV**（LLVM 栈回溯让人去提 llvm-project 的 bug），
八个 rank 全部 `Scheduler hit an exception`，服务当场死 —— **不是返回错误，是进程没了**。
与下面 `_hadamard128` 的 codegen UB 同族，**建议一起上报上游**。
生产含义：任何客户端都能用一个合法的 OpenAI 兼容参数把服务打掉。


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

**AOT `causal_conv1d` 那一族的四个陷阱**（2026-08-30，`glm53_int8_1card` 单卡实测；
本线未复现，但值得任何碰 conv 路径的人先看）：

| 陷阱 | 表现 |
|---|---|
| **`torch.ops.npu.causal_conv1d_update` 按自己文档的 layout 算是错的，且不报错** | 脉冲响应（weight 第 k 抽头设 `10^k`，输出即抽头编号）测出它把 4 抽头窗口读成 `[S1, S2, S2, x]` —— **只有两个不同的历史值**，任何对 3 行 state 的置换都救不回来。真实约定是 `conv_state` 要 **`[cache_len, WIDTH, dim]`（WIDTH 行，不是 WIDTH−1）**，且调用方每次后要 `torch.roll(+1)`。**未文档化，疑似 kernel bug**。⚠ 而且它比它本该替换的 torch 回退**还慢 2.5×**（b=1：689 vs 270 µs，约 0.3 GB/s） |
| **`conv_state_indices` 传 int64 被接受但静默算错** | 相对误差 **7.5e-1**。**必须 int32**（在 update 算子上实测，varlen 上未测）|
| **padding lane 的输出是垃圾不是零** | 一次有限、一次非有限。**调用方必须自己丢弃** |
| **`pad_slot_id` 这个参数根本不被查** | 跳过是硬编码在「下标为负」上的。传 `pad_slot_id=99` 配下标 `99` 会直接 device assert `Index 99 out of range[0 17)` 然后 507035 |

✅ **该用的是另一个**：`torch.ops.npu.causal_conv1d(..., run_mode=1)` —— GDN 路径已经在用
（`ascend_gdn_backend.py:125`）。**60 µs/call、对 batch 平坦、比 torch 回退快 4.5×、一个 kernel 顶九个**；
数值在双参考地板内、state 回写逐位精确、`cache_indices` 带 `-1` 时正确跳过。
⚠ 约束：weight/state/x 的 dtype **必须三者一致且 ∈ {bf16, fp16}**；
fp32 是**干净的 host 侧拒绝**（不是静默算错 —— 这点是好事）。
**所以 KDA 要用它，conv 权重必须从 fp32 降到 bf16，这不是逐位不变的改动，要走双参考法。**

### 2.5 已排除（不要再做）

| 项 | 为什么 |
|---|---|
| compressor 的 LayerNorm 变体 | **GLM 从不调 vendor `compressor`**（全仓两处引用均在 DSv4 路径）；其 index-K LayerNorm 是独立模块、压缩前施加、从未融合 |
| K=4 causal conv1d | Triton kernel `KERNEL_WIDTH` 1–6 全有。⚠ **「decode 已在用」是错的，2026-08-30 源码核实并已改**：`causal_conv1d_update_npu` 只在 `cache_seqlens` 或 `num_accepted_tokens` 非 None 时才进 Triton，而我们的调用点（`ascend_kda_backend.py:393`）两个都不传 → 走 `torch_causal_conv1d_update_npu`。实测代价**每个 KDA 层 16 个 kernel、4.04 ms/step = 9.4%**（`Slice+Mul+ReduceSum+ConcatD` 就是被拆开的 depthwise conv）。**这条不该留在「已排除」里** —— 算子存在，但我们没用上，见 P6.2 |
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
**先要解决消费者**（`npu_quant_lightning_indexer` 只接受 `num_heads_q=64`，GLM 是 32，见 §2.5）。

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

  - [x] **并发下的 chunking 与 3 个 chunk —— 都已验，全绿**（2026-08-30）。
        这两项单请求测试**结构上碰不到**：chunk 之间该请求被 `stash_chunked_request`
        **移出 running batch**（`scheduler.py:3140`），而其它请求继续 decode、
        继续写各自的 mamba 槽 —— 批里没有别人的时候，这一幕根本不发生。

        | 场景 | 切分（日志坐实）| 边界前的针 | 对无并发那次 | 后台请求 |
        |---|---|---|---|---|
        | 13918 token + 8 路并发 | 8192 + 5760 | FOUND | **逐位相同** | 8 完成 / 0 降级 |
        | 19858 token + 8 路并发 | **8192 + 8192 + 3520** | FOUND | **逐位相同** | 8 完成 / 0 降级 |

        `max|dlp| = 0.000e+00` —— 不是"在地板内"，是**逐位相同**。
        8 路后台请求全程 decode 对被切的长请求**一点扰动都没有**，反向也没有。
        stash 走 `maybe_cache_unfinished_req(..., chunked=True)` → tree cache，
        本配方下是 `ChunkCache`（日志 `impl=ChunkCache hybrid_ssm=True`，它知道 SSM 状态）。
  - [x] **「未对齐 chunk 起点」—— 构造不出来，不是没验**（源码 + 实测）。
        两条路径都被挡住：① `chunked_prefill_size % page_size == 0` 是启动期断言
        （`server_args.py:10051`，实测传 5000 直接 raise）；② radix 前缀复用把命中长度
        **向下取整到页边界**（`radix_cache.py:222,227` 的 `(matched // page_size) * page_size`）。
        **这条从待办里划掉。**
  - [ ] **仍未验证**：
        TP>1（单层 harness 意义上）；多 DSA 层共享 pool；只测了 layer 3、一条 prompt
  - [ ] ⚠ **KDA conv 池若改成 window-major，`ascend_kda_backend.py:700-780` 的
        speculative 快照路径同样吃这个布局，但未验证**（`glm53_int8_1card` 正在做这个改动，
        由该线声明、请本线代记 —— commit message 会被 `git log` 埋掉，
        而下一个碰这块的人是从这里进来的）。
        **不是「没时间」，是这个部署不跑 MTP**，没有能触发那条路径的负载。
        **能验的条件**：起一个带 MTP / spec decode 的配置，
        或者构造一个直接驱动快照路径的单层 harness。
  - [ ] **接 spec decode 前必须先解决**：`kpool_decode_update_index_cache` 假设每请求一行，
        MTP 一次多 draft token 会让同一 `req_pool_index` 的多行抢同一个 ring 槽。
        共享 CUDA kernel 有 `kpool_max_closed_pools` 那套多 token 逻辑，NPU 这条没有
  - [x] ~~DSA 注意力本体还缺全零 rope 的接线~~ —— 早已接好并在跑；2026-08-30 还把它
        从每层 expand 改成一次性分配（见 P6.16）

- [x] **Dense FFN** —— 端到端已验（真实 TP16 每卡形状，M=1/16/8192），最差 0.66× 预算。
      两处与预期不同：dense FFN **根本不调 `npu_clipped_swiglu`**（走 `chunk`+`clamp`+`npu_swiglu`），
      所以那个 109× 的默认参数陷阱在这条路上不适用；而且**真实输入下 clamp 从不触发**
      （max|gate_up| = 2.17，limit 是 10），要验 clamp 必须放大输入
- [x] ~~**P3.5 出口判据** —— 四模块逐层 golden 对齐~~ —— 五类层全部端到端已验，见上

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

### P5 · W8A8 compressed-tensors ✅（已闭环：转换 → 加载 → 精度判据全过）
- [x] **P5.0 权重已转换**（2026-08-29）：`/mnt/workspace/models/GLM-5.3-Flash-W8A8`，
      **62 shard / 306.1 GiB / 564 秒**，转换器 `tools/bf16_to_int8_ct.py`。
      **不需要机器** —— 激活是 per-token 动态量化，没有要校准的东西、没有前向要跑，
      纯 CPU + 磁盘的离线变换。
      ⚠ **源是 BF16 不是 FP8**（FP8 分片已删）。但**「哪些权重当初是 FP8」是精确恢复的**：
      保留下来的 FP8 `index.json` 里每个被量化的权重都带一个 `weight_scale_inv`，
      读回来正好 **37338 个**，在 BF16 里一个不缺。**这不是按名字模式重新推的** ——
      按模式推会变成第二个真相来源，早晚和第一个漂移。当初留那 28 MB 元数据就是为了这个。
      校验：输出 index **76108 条 == FP8 原 index 的 76108 条**；37338 个权重缺失 0、缺 scale 0；
      抽查 40 个量化张量最差 rel-L2 **9.1e-03**；抽查 40 个未量化张量 **40/40 逐位不变**。
- [x] **P5.1 recipe** —— weight per-channel symmetric 静态 + act per-token **dynamic**。
      这是 `compressed_tensors.py:489` 的 `_is_dynamic_token_w8a8` 唯一匹配的组合（静态激活会被拒）。
      张量契约：`weight` int8 `[out,in]`、`weight_scale` fp32 **`[out,1]`**，动态激活**不需要 `input_scale``。
      ⚠ **NPU 走专用分支** `NPUCompressedTensorsW8A8Int8`（`:762`），不是通用那个。
      量化式：`scale = absmax(W,dim=1)/127`，`q = round(W/scale).clamp(-127,127)` ——
      用 127 而不是 128 保持对称，`q*scale` 复原时没有偏移项；scale 在 fp32 里算，
      先在 bf16 里取 max 等于把 scale 自己也量化了。
- [x] **P5.2 ignore list** —— 直接抄 checkpoint 的 `modules_to_not_convert` 原文（1509 条）。
      **注意由此产生的一个后果**（2026-08-30 从 checkpoint header 逐张量核实，
      由 `glm53_int8_1card` 那条线量出流量占比）：**KDA 34 层的 q/k/v/o_proj 全是 BF16**
      （每种 2.12 GiB，合计约 8.7 GiB），DSA 的 `kv_b_proj` 也是 BF16。
      厂商的 FP8 checkpoint 就没量化它们，我们照搬了。

      ⚠ **两个百分比别混**：未量化权重只占**常驻**的 4.2%（12.9 / 305.7 GiB），
      但 **bs=1 每 token 的流量**里 MoE 只读 top-8/288，于是 KDA 反而是最大单项 ——
      单卡 TP1 实测每 token 必读 **20.9 GiB** 里 KDA 占 **8932.5 MiB = 41.8%**，
      与 MoE 三项合计（42.9%）基本打平。
      （这两个数是对方发现自己第一版算错后订正的：原先每专家字节除以 43 层却乘 42 层，
      而第 43 个 MoE 层是不建的 MTP 层；分桶又用 `'mlp.gate' in rest` 把 dense 层的
      `mlp.gate_proj` 算进了 MoE。连带 roofline 合计 17.759 → **17.941 ms**、
      实测/地板 2.41× → **2.39×**。方向没变，但既然要进文档就用对的数；
      现在由 `tools/kernel_roofline.py` 自动算，不再手输。）
      **所以「INT8 在小 batch 只小赢」有一半是因为流量的 42% 根本没被量化。**
      TP16 下 KDA 被 16 分（每 die 约 558 MiB/token ≈ 0.45 ms），所以在我们这条线不显眼。
      **量化 KDA 是这份 checkpoint 剩下最大的一个杠杆**，但**用户决定先不做**（2026-08-30）。

      查清楚了两件事（源码 + 保留下来的 FP8 index）：
      ① **KDA 在原始权重里根本不是 fp8，是 BF16** —— 它的 q/k/v/o_proj
      一个 `weight_scale_inv` 都没有；② `modules_to_not_convert` 是**逐层精确点名**的，
      `o_proj` 出现在 **34/34 个 KDA 层、0/11 个 DSA 层**。
      **所以这不是笼统的模式匹配，是厂商刻意逐层挑出来排除的，我们的转换没漏东西。**

      不做的理由主要不是收益（单卡实测 1.14×，TP16 下 KDA 被 16 分、收益小得多），
      **而是「测不出它坏」**：KDA 是线性注意力的递归路径，权重误差会沿 SSM 状态
      随序列累积，而不像标准注意力每步从 KV 重新读；`modules_to_not_convert` 同时排除了
      `A_log` / `dt_bias` / `conv1d`——正是 HF 强制留 fp32 的那批递归参数
      （**投影本身没有那个约束，所以这是怀疑不是解释**）。
      如果怀疑成立，**症状是长序列上的缓慢漂移，而 GSM8K 那种几百 token 的题目看不见**。
      真要做的顺序：量化 → GSM8K 每侧 2 轮 → **再加一个长上下文判据**。
      ⚠ 还有个接线坑：ignore 列表里直接写着融合名
      `model.layers.N.self_attn.qkv_proj`（和 `fused_qkvbfg_a_proj`），
      只删 q/k/v_proj 不够 —— `should_ignore_layer` 命中融合名就不展开，整层仍是 bf16。


- [x] ~~P5.3 288 专家校准~~ —— **不需要**。激活是动态的，没有静态激活 scale 要标定
- [x] **P5.4 加载 —— 通过**（2026-08-29 21:56）：`quant=compressed-tensors`，
      权重 **19.57 GB/die**（BF16 是 37.25，约一半，这是格式正确的独立证据），无报错。
      省下的 17.7 GB 全给了 KV：pool 从 111 万涨到 **260 万 token**。
      冒烟：短提示正确；**3256 token 长提示（稀疏路径生效）段落六个事实全对**
- [x] **P5.5 出口判据 —— 通过**（2026-08-30，每侧 2 轮）：

      | | run 1 | run 2 | 均值 |
      |---|---|---|---|
      | BF16 | 97.04% | 97.35% | **97.19%** |
      | **INT8 W8A8** | 97.80% | 97.19% | **97.50%** |

      差 **+0.30pp**（差值 SE 0.46pp），判据「1% 以内」**通过**。
      结果都在 `$ROOT/goldens/gsm8k/`。
      旁证：INT8 错的 29 题里 **23 题 BF16 也错** —— 错的是同一批难题，
      不是量化打坏了某一类
- [x] **P5.6 kpool 三连之后的复测 —— 通过**（2026-08-30，TP8）：
      **97.42%**（1285/1319、stop rate 100%、0 题打到 max_tokens），
      `$ROOT/goldens/gsm8k/int8_tp8_kpool_20260830.json`。

      ⚠ **这一轮不是 kpool 改动的精度证据**，别这么读。kpool 三连（P6.7/6.10/6.11）
      的证据是**同构型前后逐位相同**（`0.000e+00`）—— 那是个比任何评测都强的判据，
      因为它排除的是"有差异但被采样噪声盖住"这种情况，而 GSM8K 排除不了。
      这一轮回答的是另一个问题：**改完之后整条链路还是好的**（服务能起、
      图能捕获、1319 题全部正常停下来）。

      ⚠ **TP8，不能和上面 TP16 那两轮直接并列**：TP 宽度变了，all-reduce 的
      归约顺序就变了，浮点加法不结合，这本身就会产生差异。97.42% 落在
      BF16 97.19% / INT8 97.50% 中间、单轮噪声 ±1pp 之内，**结论是"没有异常"，
      不是"比 97.50% 低了 0.08pp"** —— 后者是在读噪声。
  - [ ] logprob 对拍：INT8 对 fp32 参考 7/8 在 **BF16 时代**的地板内，1 个到 1.32×。
        ⚠ 那个地板是给 BF16 量的，INT8 多了真实量化误差，超出是预期内的；
        真正的判据是上面的 GSM8K。要给 INT8 单独测地板需要一份 int8 的 CPU 参考
- ⚠ **磁盘现在只剩 23 GB**（BF16 599 + W8A8 306）。再要腾空间只能动 BF16
- [x] ~~P5.4 旧编号的出口判据~~ —— 见上面的 P5.5，已通过（+0.30pp）
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
**P6.15 INT8 与 BF16 的吞吐：基本持平，局部 INT8 更快**（2026-08-30，空闲机器实测）。

⚠ **先纠正一处错误结论**：此前根据 GSM8K 的「aggregate tok/s」报过
「INT8 比 BF16 慢 1.47×（159 vs 234）」，**那是错的**。
**aggregate tok/s 不是吞吐指标** —— 它把 128 路满批（约 2000 tok/s）和
单请求长尾（约 30 tok/s）平均在一起，谁碰上一条不收敛的生成谁就难看。
那一轮 INT8 有 2–3 条请求跑到了 32768 的上下文上限，独占机器约 1210 秒；
两轮的**满批阶段长度几乎一样**（INT8 1200 s vs BF16 1230 s），
985 秒的差全在尾巴上。

**按 `#running-req ≥ 120` 开窗口比较（这才是可比的部分）**：

| | 满批 gen throughput 均值 | 中位 |
|---|---|---|
| INT8 | **263** | 251 |
| BF16 run1 / run2 | 245 / 249 | 246 / 248 |

**INT8 反而快约 6%。** 干净重跑的 e2e 也落在 BF16 的轮间散布内
（INT8 1512 s / 209 tok/s，对 BF16 的 1490 s / 206 与 1360 s / 234）。

分阶段受控 A/B（`bench_graph_decode.py`）：

| decode ms/token | bs=1 | bs=8 | bs=128 |
|---|---|---|---|
| BF16 | 25.4 | 35.0 | 71.7 |
| INT8 | **28.9（慢 14%）** | 36.6 | **66.0（快 8%）** |

prefill 按调度器自己的每批计时：BF16 4849 tok/s、INT8 4906 tok/s，**差 1% 以内**。

**bs=1 那 14%（+3.5 ms/token）的现象成立，但我给的成因是错的，在此撤回。**
原先写的是：量化线性每次 `npu_quant_matmul` 前单发一次 `npu_dynamic_quant`，
每 forward 多约 140 个 kernel，3.5 ms / 140 ≈ 25 µs 一个「量级对得上」。
那是**源码 + 计数推出来的，从没 kernel profile 过**。

**`glm53_int8_1card` 在单卡上实测推翻了它**：图模式下这些 kernel 合计
**382 µs/step（0.9%）、每个 2.8 µs**，不是 3.5 ms —— 图把 launch 开销吃掉了。
⚠ 被推翻的是**机制**，不是「TP16 bs=1 上 INT8 比 BF16 慢 3.5 ms」这个**观测**。

**2026-08-30 有了一个候选机制，而且是源码确认的**（由 `glm53_int8_1card` 发现）：
`glm5_next.py:366` 的
```python
self.do_fuse_qkvbfg = quant_config is None and head_shard_size == self.tp_size
```
**这个条件问错了问题** —— 它问「有没有 quant_config」，该问「这几层是不是被量化了」。
本线核实：KDA 的 `q/k/v/b/f_a/g_a/f_b/g_b_proj` **每一个都在 W8A8 的 ignore 列表里**
（抽查 5 个 KDA 层全部 5/5），**连融合模块名 `fused_qkvbfg_a_proj` 自己也在** ——
厂商显然预期过这条路在量化下被走。

**后果：BF16 走融合路径（`quant_config is None`），INT8 不走。**
所以 INT8 相对 BF16 白白多出每个 KDA 层 4 个小矩阵乘。
对方在 TP1 bs=1 实测这笔是 **约 1.68 ms/step、每步 170 次启动**，全部 launch 主导
（没有一个搬超过 2 MiB）。

⚠ **机制是源码确认的，量级在 TP16 上未测**。可证伪的判据（对方给的）：
在 TP16 上分别用 BF16 和 INT8 采一份 bs=1 profile，数 `MatMulV2 [1,4096;128,4096]`
这一组 —— **BF16 应当没有，INT8 应当有 68/step**。
⚠ 本线的服务级 profiling 在 16 rank 上会段错误，所以这个判据要么等一个能用的采法，
要么改成 A/B 墙钟（对方修好之后，INT8 打不打这个补丁各测一次）。
**修在 `glm5_next.py`，由 `glm53_int8_1card` 那条线做**，本线不重复。

**教训**：一个数量级对得上的算术不构成机制证据。当时那句「量级对得上」正是让它
读起来像结论的东西。

**「MoE 会不会偷偷反量化回 bf16」这个假设是错的**（源码核实）：MoE 路径
**一个额外 kernel都没多** —— 激活量化被融进了 `npu_moe_init_routing_v2(quant_mode=1)`
（gmm1 的输入）和 `npu_dequant_swiglu_quant`（gmm2 的输入）。
`process_weights_after_loading` 转置成 `[E,K,N]` 并转 `FRACTAL_NZ`，对齐前置检查通过，
两份日志里**都没有 "Skipping FRACTAL_NZ format cast" 警告** —— 快路径在用。

**结论：INT8 不需要性能修复。** 真要动 bs=1 那 140 个 kernel，得把 dynamic quant
融进前面的 RMSNorm（`npu_add_rms_norm_quant` 那类），是逐层接线且有数值风险，
而且对真实服务负载（大 batch）没有收益。

⚠ **给后来者的度量建议**：别再用 GSM8K 的 aggregate tok/s 判吞吐。
用 `#running-req` 对齐后的满批 `gen throughput`，或者直接用 `bench_graph_decode.py`。
要让墙钟可复现，把 `--max-tokens` 压到 4096（实测 p99 是 891、最大 11891），不损精度。

**P6.16 `_nope_zero_rope` 的 stride-0 expand 每步都被物化**（`glm53_int8_1card` 单卡实测，
本线已核对源码，**TP16 上未测**）。`ascend_backend.py:1089` 用一个全零页 `expand` 出
`query_rope`/`key_rope`，函数自己就写着 ⚠「算子文档说不支持非连续输入，所以这个 aliasing
是观察到的行为，不是承诺的行为」。**⚠ 兑现了，但不是以算错的方式，而是以静默变慢的方式**：
torch_npu 在调 aclnn 前把输入变连续，于是那个 expand 每层每步真的写一遍。

单卡 kernel profile：`BroadcastTo`，`[1,64,1,64]` → `[19403,64,1,64]` bf16，
`aiv_mte3_ratio=0.945`（纯 store），**63.8 µs × 11 层 = 0.70 ms/step**，每步写 1.75 GiB 的零。

**最坏的性质：它是 O(KV pool) 而不是 O(batch) 或 O(seq_len)** —— pool 从 124 万涨到
152 万 token，这一项就 0.70 → 1.34 ms/step；而 bs=1 与 bs=16 几乎一样贵（0.70 → 1.03），
**大 batch 摊不掉**。TP16 每 die `max_total_num_tokens=1195072`，页数同量级，
估计每步约 0.6–0.7 ms，在 28.9 ms 的 step 里约 **2.4%** —— 比在单卡上占比还高，
而**墙钟一直看不见它**。

改法：不 expand，按 (device, dtype, 完整 shape) 缓存一次全零张量，所有 DSA 层共用。
数值上恒等（零就是零），代价是一次性约 159 MiB HBM。
**已在 `int8_singlecard` 分支上改好并实测（单 die，bs=1）：42.823 → 39.741 ms/step，−7.2%。**

⚠ **收益比那两个 kernel 本身大得多，而多出来的部分是推断**：BroadcastTo 自己只值
0.80 ms，剩下 2.3 ms 对方归因为 **L2 冲刷**（159 MiB 几乎正好是 L2 的 168 MB，
每个 DSA 层写一遍就冲干净；34 个 KDA 层与 11 个 DSA 层交错，挨着的 KDA 层替它付钱）。
**支持它的是分布形状而不是均值**：KDA qkv matmul 的中位数只降 2.4%
（182.1 → 177.8 µs）而每步总和降 23%（7823 → 6022 µs），修复前的长尾在修复后消失。
这个论证方式值得学 —— 均值降了可以有很多解释，长尾消失基本只有一种。

**本线已 cherry-pick 并上机验完**（`9f2a43aaad`，2026-08-30，TP16 图模式）：

| | 修复前 | 修复后（3 遍） | |
|---|---|---|---|
| 短上下文 bs=1 | 25.4 | **23.2 / 23.3 / 23.2** | **−8.7%** |
| 短上下文 bs=16 | 38.1 | 37.8 / 38.1 / 37.7 | −0.8%（噪声内）|
| 长上下文 bs=1 | 26.3 | 24.4 / 24.3 | −7.7% |

数值**逐位相同**（零就是零）。**bs=1 的收益复现得很紧**，与单卡那边的 −7.2% 一致。

⚠ **两件要诚实说的**：

① **第一次量到短上下文 bs=16 是 26.3（−31%），三遍复现都是 37.7–38.1，那是离群值。**
去复跑是因为「短上下文 bs=16 涨 31% 而长上下文 bs=16 一动不动」自相矛盾 ——
**不一致本身就是要求复现的信号**。按第一个数报就会报出一个假收益。

② **bs=16 几乎没收益，这一点我解释不了。** 按对方的机制（开销是 O(KV pool 页数)、
与 batch 无关），每步省下的绝对时间应该一样，即 38.1 → 约 35.9（−5.8%），
实测却是 37.8。**要么修复前那个 38.1 本身偏低**（同一构建早前的扫描量到过 42.9，
说明 bs=16 的轮间散布不小），**要么这个开销不是 batch 无关的**。
两边我都没有足够的重复测量来判定 —— 要判得给修复前也跑三遍，那需要 revert 再测。
**在此之前不要引用「bs=16 无收益」当结论。**

⚠ 没有用 profile 验「BroadcastTo 消失」：16 rank 的服务级 profiling 在本线踩过全 rank
段错误，而同机同构建、相隔几分钟的墙钟 A/B 在这里够用且风险小得多。

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
② 清掉 extend 侧挡住捕获的东西。**⚠ 原先「extend 仍有 host 同步」这个说法是错的，已作废**
（2026-08-29 逐行审计）：`kpool_indexer_npu.py` 里 **`.item()` / `.cpu()` / `.tolist()`
一个都没有**，`visible_pool_runs` 的 `int(pool_lens.max())` 早就删了，
而 `_kpool_extend_rows_npu` 读的是 `*_cpu` 字段 —— 那本来就在 host 上，不是 D2H 等待。
（搜到的 `.item()` 在 `ascend_torch_native_backend.py` 和 `ascend_dsv4_backend.py`，
**都不在 GLM 的路径上**。）

**真正挡住捕获的是两件别的事**：

  - [x] **`_kpool_extend_rows_npu` 的 host 侧构造 —— 已改成设备侧、静态形状**。
        原来每次 forward 都在 Python 里循环 batch、每请求一个 `arange`、`cat`、再拷到设备；
        **捕获会把某一次 forward 的值永久烘进图里**。新形式是纯张量算术，
        形状由行数固定：`ends = extend_seq_lens.cumsum(0)`，
        `req_index = (pos[:,None] >= ends[None,:]).sum(1)`。
        **对 `ends` 而不是 starts 比较是有原因的** —— 零长度请求会让 starts 落在空请求上
        而不是下一个，而旧代码是 `q_len == 0: continue` 跳过它的。
        用 `ge`+`sum` 而不是 `searchsorted`，因为前者确定是普通 AI Core 算子。
        **回归 `layer_check/check_extend_rows.py`：3006 个用例逐元素相等**
        （含零长度请求、部署形状、全空），而且测的是**从模块里抽出来的仓库源文本**，不是抄件。
        ⚠ **只证明了算术等价，没证明它在设备上可捕获或更快** —— 那需要机器。
  - [x] **`visible_pool_runs` 的 `nonzero()` —— 已改成静态形状**（2026-08-30）。
        原先它的输出长度就是 run 个数，随数据变，捕获直接拒。

        **关键的未知先上机问掉了**（`probe/p6_a1_padded_runs.py`）：消费者是
        `npu_lightning_indexer`，`actual_seq_lengths_query` 是**前缀和** ——
        补齐前缀和 = 重复末值 = 多出来的 run **跨零个查询行**。
        算子认不认这种空 run，文档没写、源码读不出来，**只能问**。
        答案是干脆的：把 128 个 run 的分段补上 1 / 8 / 128 个空 run，
        输出**三次都逐位相同**。

        实现：不再用 `nonzero` 压缩，而是把每个边界 **scatter 到它的名次所指的槽**，
        非边界送到末尾的 scratch 槽再切掉；空槽填 `n`，于是它「起点就是终点」——
        正好是空 run 该有的样子。上界由 `max_visible_pool_runs()` 给：
        一个请求内 key 数每行加一，所以 `pool_lens = key//kpool` 每 `kpool` 行才变一次，
        每请求最多 `ceil(q_len/kpool)+1` 个 run，合计 `ceil(n_rows/kpool)+batch`。
        回归 `layer_check/check_pool_runs.py`：**2006 个配置**，补齐版在真实 run 上与
        精确版逐元素相同、填充部分确实是空 run、上界从不被突破（余量 1..13）。
        **2026-08-30 上机验完**：调用点改成**总是走补齐路径**（而不是把它做成可选、
        默认走老路），因为那样捕获路径和 eager 路径才是同一条路 —— 两条不同的代码路径
        碰巧给出相同结果，和同一条路径给出相同结果，是不同强度的证据。
        TP16 实测对改动前的 eager 基线**逐位相同**（`0.000e+00`），
        而且**没有代价**：prefill 3256×16 是 11.77 s 对 11.72 s（0.4%，噪声内），
        decode bs=1 不变（25.4 ms/token）。空 run 是彻底的 no-op。

        ⚠ **差点漏掉的一步**：`max_runs` 一开始是可选参数、调用点没传，
        所以第一轮上机跑的还是老的 `nonzero` 路径 —— **全绿证明的是「没改坏老路」，
        不是「新路对」**。和「短 prompt 测不到 kpool」「单请求测不到并发」
        「单层测不到整网」是同一类：**测试通过了，但它测的不是你以为的那件事。**
  - [x] **`_kpool_compress_write_extend_npu` 也改成批量静态形状了**（2026-08-30，
        **⚠ 仅离线验证，未上机**）。原来每请求一趟 Python 循环，切片、`pool_ids`、
        `write_locs`、散写目标**全是数据相关的尺寸**。现在一次散写：
        pool 数上界 `n_rows//kpool + batch`，tail 上界每请求 `kpool-1`，
        无效项**送 scratch 而不是过滤掉** —— 过滤正是那个要不到静态形状的动作。
        新增 `NPUDSATokenToKVPool.set_compress_tail_batched`，并给包装类
        `HybridLinearKVPool` 补上 `scratch_loc` 与它的转发
        （**包装类没转发**是这个项目栽过的同一类坑）。

        **回归 `layer_check/check_compress_write_plan.py`：1418 组，
        比的是最终缓冲区内容而不是索引** —— 把索引缓存和 tail ring 用普通张量建模，
        新旧两版写完之后逐元素相同，所以写错位置会以「某行不同」暴露，而不是溜过去。
        测试里显式断言了**不同请求的物理页互不相交**（分配器的不变量），
        否则两个 pool 撞同一行、散写顺序未定义，那个比较就是空的。

        **2026-08-30 上机验完**：TP16 对改动前的 eager 基线**逐位相同**
        —— 3256 token 长提示（extend + 稀疏）、短提示 + 100 decode、
        以及 19858 token 切三刀 + 8 路并发的 chunked prefill（针召回、后台 0 降级）。
        ⚠ **仍未测**：扁平 `index_select` 是否真的把页表 gather 挡在 AI CPU 之外
        —— 那要 kernel profile 才看得到，数值正确不构成证据。
        ⚠ 写的时候差点自己引进两个坑，都靠读 decode 路径的既有注释躲开：
        ① `block_tables[req, col]` 这种两张量高级索引**没有 AI Core 实现**，
        会回落到 AI CPU 的 `aclnnIndex`（decode 那边实测占该层 device 时间的 37.5%），
        必须用扁平 `index_select`；② 对齐检查一开始写成对设备张量取 `bool(...any())`，
        那是 D2H 等待，捕获下还会抛 107027 —— 改成读 host 侧的 `*_cpu` 字段。
  - [ ] **prefill 进图的真正阻塞：Ascend 后端没实现 prefill 侧的图 metadata 契约**
        （2026-08-30 上机定位，**推翻了本条原先的说法**）。

        ⚠ **原先写的「差最后一步：在 registry 里注册」是错的。** 注册根本不是阻塞 ——
        `--cuda-graph-backend-prefill full` 这个现成的 CLI 参数就能开，实测
        `prefill.backend` 成功解析成 `full`（`_disable_full_prefill_cudagraph_if_incompatible`
        的规则表是**空的**，FULL prefill 没有任何兼容性门槛）。
        ⚠ 顺带：`cuda_graph_config.py:114` 那条注释说 full prefill 是
        「opt-in via the declarative registry（见 `_inkling_overrides`）」**也不准** ——
        `_inkling_overrides` 自己的 docstring 写着 full-graph prefill **不在 registry 里设**，
        因为 cuda-graph 配置在 `__post_init__` 里先于声明被解析。真正的机制是 inline 的
        `_apply_inkling_prefill_cuda_graph_default()`。

        **真正炸在哪**：捕获跑到 `capture_one_shape` 抛
        `KeyError: 'block_tables'`（`ascend_backend.py:638`）。原因是
        `graph_metadata` 只在 `init_cuda_graph_state()` 里分配（`:585`），
        而**那个函数只被 `decode_cuda_graph_runner.py` 调用**（`:369`、`:539`）；
        prefill 运行器走的是另一条契约 —— 它只调
        `init_forward_metadata_out_graph(forward_batch, in_capture=True)`
        （`prefill_cuda_graph_runner.py:1407`），**从不调 `init_cuda_graph_state`**。
        Ascend 的 `init_forward_metadata_out_graph` 却假设静态 buffer 已经按 decode 的
        bs 预分配好了，于是读一个空 dict。

        **决定：记账，暂不做**（用户 2026-08-30）。理由不是"太难"，是**深度未知而收益未量化**：

        - **只知道第 1 个阻塞，不知道一共有几个。** 捕获停在第一个 KeyError，
          后面还有多少层没接上的线只能一次崩溃一次地发现。这个项目里同类的
          「接线缺口」有先例 —— 整网拉通时连撞三个，**每修一个才露出下一个**
        - **在共享文件里，而我们刚决定不跑 DSv4 回归** —— 等于在共享的捕获路径上动手
          却主动放弃了安全网。而捕获路径的错误特别阴：**值被烘进图里不报错、不崩、
          数字看着也正常**
        - **收益从没量化过。** 「prefill 是收益上界最大的一项」的依据只是
          「GSM8K 一轮有 875 个 prefill 批」，**没有人量出 prefill 占真实负载一步的百分之几** ——
          想量的那次服务级 profiling 把 16 个 rank 全段错误打挂了

        **什么条件下值得重启这件事**（按顺序，前两步不占卡或只占一次）：
        1. **先把收益量出来** —— 在一个真实负载下拆出 prefill 与 decode 各占多少。
           不需要 profiler：`--max-running-requests 1` 跑固定负载，
           对比 `--chunked-prefill-size` 大小不同的两档，或者直接用调度器日志的
           per-batch 计时求和（`glm53_int8_1card` 的 `tools/attribute_kernels.py`
           在低并发下是活的，可以借）
        2. 收益够大再动手，**并且先补 DSv4 的回归安全网**
        3. 动手时按「一次崩溃一次」的节奏走，每接上一层就记一层，别攒着

        **所以这是后端工作，不是配置工作**：要给 `AscendAttnBackend` 加一条 prefill 分支，
        直接从 `forward_batch` 建 metadata（而不是从 decode 的预分配 buffer 取），
        或者把 buffer 也按 prefill 形状分配。⚠ **在共享文件 `ascend_backend.py` 里**，
        DSv4 那类非 hybrid 的昇腾模型也走它。
        ⚠ 设计上还有个未解问题：捕获的 prefill 图里 `block_tables` 也必须是静态的，
        而它的宽度取决于序列占多少页 —— 这个上界怎么定，需要先想清楚。

        **好消息**：我这三处改写（`_extend_rows` / `visible_pool_runs` 补齐 /
        compress-write 批量化）**都不是这次失败的原因** —— 捕获在碰到它们之前就炸了。
        它们已各自上机验过逐位相同。
  - [ ] `_kpool_compress_write_extend_npu` 仍是 host 侧循环（每请求形状不同的散写），难度更高

捕获期间 `.item()`/`.cpu()` 会抛 **107027**（已实测）。
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
      ⚠ **图模式下复核过，结论不变**（`glm53_int8_1card` 实测）：`hc_sinkhorn_iters`
      20 → 1 只值整步的 **0.7%**，而且**确实生效了**（`HcPre` 中位 30.34 → 26.42 µs，
      不是「改了没反应」）—— 19 次迭代共 347 µs/step，摊到 90 个站点是每站点每次 0.04 µs。
      **所以 `HcPre` 那 2.4 ms 不是 sinkhorn**，是算子在 M=1 下的固定开销
      （26 µs 搬 1.5 MiB = 带宽地板的 20 倍）。**这个旋钮改数值不改性能，别动。**
      roofline：post 高于带宽下界 2.6×、pre 9.5×（去掉 sinkhorn 是 5×）——
      **prefill 下不是 host 开销主导**，和 DSA 的 100× 不是一回事
- [ ] P6.2 **KDA 的 conv1d，prefill 与 decode 是同一个上游问题的两面**（2026-08-30，
      根因由 `glm53_int8_1card` 找到；**该线正在做，本线不要重复**）。
      **根因**：`memory_pool_npu.py:38-48` 把 KDA 的 conv 池建成 **channel-major**
      `[17, 24576, 3]`，而 GDN 建成 **window-major**；wheel 里那个 AOT Ascend C 算子
      `torch.ops.npu.causal_conv1d_update` 要的正是 window-major，
      **所以 GDN 用得上、KDA 用不上**。把 KDA 池翻面可能一次解决 decode（4.04 ms/step）
      和 prefill 两侧。⚠ 约束：该算子要求 weight/state 是 bf16，而 GLM 的 conv 权重是 fp32，
      **有真实的精度问题要验**。
      ⚠⚠ **这个函数还有一个已知的正确性缺陷，不只是慢**（2026-08-30，
      `glm53_int8_1card` 实测）：`causal_conv1d_fn_npu` 在**一个 extend 批里混合
      `has_initial_state`** 时会**写坏 conv state**（输出是对的，只有 state 回写坏）。

      | 批内 `has_initial_state` | state err（`\|state\|` ≈ 4.2）|
      |---|---|
      | 全 True / 全 False | **0** |
      | `F,T,F,T` / `T,F,T,F` | **5.84 / 5.98** |

      触发条件是**一个 extend 批里同时有 prefix-cache 命中和冷请求**，生产上完全可达。
      ⚠ **`check_kda` 抓不到**（基线 6/6 全绿）—— 它只比对 golden 那个槽位。

      **本线记录在案的所有精度数字都不受影响，但那是碰巧**：我们全部 8 个启动脚本都带
      `--disable-radix-cache`（服务端 `disable_radix_cache=True` 已核实），
      没有 prefix-cache 命中，于是 `has_initial_state` 每批一致为 False，
      触发形状从未出现。**那个 flag 在那儿是为了让测量可复现，不是为了躲这个 bug。**
      ⛔ **所以：在这条修好之前，不要打开 radix cache 去跑精度评测** ——
      而"上生产先把 prefix cache 打开"恰恰是最自然的下一步动作。

      原 prefill 侧的量化：Ascend 的 extend 把深度卷积拆成 **3 次调用**，
      而共享 CUDA 路径对整个 qkv 宽度只做 **1 次打包调用**。实测 **6.5 ms / 单层 14.3 ms = 45%**，
      而且这 3 次调用**各带 3 次 host 等待**（prefill 全部 9 次 host 往返都在这里）。原条目： —— `causal_conv1d_fn_npu` 内部退回 `F.conv1d`（`sgl_kernel_npu` 上游的实现选择）；
      且 Ascend 后端拆成 3 次调用而共享后端只做 1 次
- [x] **P6.3 SwiGLU clamp —— 关闭**（2026-08-30，由 `glm53_int8_1card` 实测收尾）。
      ⚠ **只对 shared expert 成立，别按字面读成 routed**：「2×clamp + `cat` + `npu_swiglu`
      四个 kernel」按形状对应的是 **shared expert**（`deepseek_v2.py:463-471`）；
      **routed 那条早已不是这个形态** —— 1× 向量界 clamp + `npu_dequant_swiglu_quant`，
      两个 kernel、没有 `cat`。

      **`torch_npu.npu_clipped_swiglu` 是好的**：参数传对
      （`alpha=1.0, limit=10.0, bias=0.0, interleaved=False`）时与两步式**四种情况全部逐位相同**，
      包括把输入放大 48× 让 limit 真正咬到的两种。两个反向对照都有牙：
      全默认参数 `max|Δ|=156`（就是 §2.4 那个 109× 陷阱），
      对「完全不 clamp」`max|Δ|=2.9e4`（证明它真的在 clamp）。

      ⚠⚠ **和同族的假算子形成直接对照，这条比收益值钱**：
      `custom::npu_dequant_swiglu_clamp_quant` **静默忽略 `clamp_limit`**（与不 clamp 逐位相同），
      而 `torch_npu.npu_clipped_swiglu` 参数传对就精确。
      **同一族、名字都带 clamp、一个真一个假。逐个验，不能类推。**

      **收益很小**：可融合的只有 shared expert 那段，约 **108 µs（0.3%）**，低于单次 profile 的噪声底，
      所以判据用 **kernel 计数**（`ClipByValueV2 [1,4096]` 42→0）而不是墙钟。
      做它的理由是逐位相同、验证近乎零成本、顺带去掉 45 个 kernel
- [x] ~~P6.4 DeepEP-normal 的 D2H 同步~~ —— **对当前部署配方不成立，关闭**（2026-08-30）。
      那 42 次 D2H 在 `pre_permute_deepep_normal_to_ascend` 里，而 `--moe-a2a-backend none`
      走的是 `pre_permute_ascend_tp_to_ascend`（`moe_runner/ascend.py:249`）——
      **对 GLM 是死代码**。与 §4 早先记的「DeepEP 那条 GLM 走不到」一致。
      只有 DSv4 的 `--moe-a2a-backend deepep` 才命中
- [ ] P6.5 NoPE 未融合的 split+RMSNorm（与 P3.3 同源，一起做）；顺带删掉那个看起来是死代码的 `q.clone()`
- [x] **P6.7 kpool indexer 的 expand+tail —— 已修，15.9×**（2026-08-30）。
      ⚠ **它不是第三条待办，它就是 P6.10 和 P6.11 加在一起的那个观测。**
      原记录「`expand_pooled_groups_to_topk` 物化 `[4096,512,4]` 的 int64 再 reshape，
      占 **6.3 ms/单层**」—— 那 6.3 ms 里，expand 的 int64 中间物化是 P6.10，
      tail 的寻址退化是 P6.11。修完这两条，这一条自动没了。

      实测（把改动前后的两份 `kpool_fp8_index.py` 各自 import 进来，跑完整的 expand+tail）：

      | rows | 改前 | 改后 | |
      |---|---|---|---|
      | **4096** | **6.257 ms** | 0.393 ms | **15.9×** |
      | 8192 | 12.482 ms | 0.845 ms | 14.8× |

      **6.257 与原记录的 6.3 ms 对得上**，说明这两个观测确实是同一件事。
      按 11 个 DSA 层折算，每个 4096-token chunk 从约 **69 ms → 4.3 ms**。输出逐位相同。

      **教训**：同一段代码被从两个角度量过两次（一次按函数、一次按「阶段」），
      就会在待办清单上变成两条，让人以为还有活没干。
      **合并的判据是数字对得上，不是名字像。**
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
      ⚠ **「extend 仍有同步」这句已作废**（2026-08-30 逐行审计）：整个
      `kpool_indexer_npu.py` 里 `.item()`/`.cpu()`/`.tolist()` 一个都没有。
      extend 侧的 host 侧构造与动态形状也已在 2026-08-30 清掉（见 P3.4），
      但 prefill 进图另有阻塞，已记账
- ⚠ 原「所有性能数字都是静态推算」**已作废**：下面是 kernel 级 profiler（`torch_npu.profiler`
  Level1 + PipeUtilization）实测出的排序。**墙钟看不见其中任何一条**。

**诊断结论：DSA 慢不是因为注意力慢，是因为注意力周围的簿记。** 注意力本身很好 ——
`SparseFlashAttention` decode cube 利用率 **79.7%**、prefill **85.9%**，`LightningIndexer` **88.1%**。
对照 MoE 的 `GroupedMatmul`：搬 346 MB 用 330 µs = **1051 GB/s = 实测 roofline 的 84%**。
MoE 全程只有厂商融合算子，**没有 AI_CPU 回退、没有 Triton、没有 int64 索引算术**；
DSA 每步 170 次 aten dispatch，MoE 只要 25 次。

按实测收益排序：
- [x] **P6.8 消掉 decode 的 AI_CPU gather** —— 已修（§2.4）。占 DSA decode device 时间 37.5%，27× 提升
- [x] ~~**P6.9 `TASK_QUEUE_ENABLE=2`**~~ —— **图模式下用不了，这条关闭**
      （`glm53_int8_1card` 实测）。不是「被图吃掉」，是 torch_npu 的硬约束，
      第一个 bs 桶刚开始捕获就抛：
      `RuntimeError: Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture,
      please export TASK_QUEUE_ENABLE=1/0`（`torch_npu/npu/graphs.py:625` 的
      `capture_begin()`）。**响亮地失败，不会静默降级** —— 这是好事。
      那 1.74× 是 eager 时代的数，图模式下 host 侧本来就没有气泡了。
- [x] **P6.10 `expand_pooled_groups_to_topk` 改 int32 —— 已修，实测 10.7×，逐位相同**（2026-08-30，`d2da5cce93`）
      原条目： —— prefill 的 `aclnnAdd` 花 **5.73 ms**
      产出 `[8192,512,4]` 的 int64（134 MB），高于下界 **43×**。token id 最大约 32768，
      **int32 完全够**，既减半流量又避开 Ascend 上被模拟的 int64 向量运算。**在共享代码里**
- [x] **P6.11 tail kernel —— 已修，实测 32.6×，逐位相同**（2026-08-30）。
      **一行改动**：`tl.load(... safe_history_cols ...)` → `tl.load(... cols ...)`。
      那个 clamp **改不了任何被 mask 保留的通道**（`is_history` 是 `cols < history_len`
      而 `history_len <= N_COLS`，所以取值处 `cols` 本来就在范围内），它只改了**地址表达式** ——
      对 `cols` 非仿射，于是连续向量 load 退化成逐元素寻址。**这就是这个 kernel 的全部开销**，
      与 profile 的 `aiv_vec_ratio=0.027`、`aiv_mte2_ratio=0.0`（既不算也不搬）完全对上。

      实测（`probe/p6_11_tail_clamp.py`，单卡，两个变体并排跑）：

      | rows | 带 clamp | 去掉 | |
      |---|---|---|---|
      | **8192（部署形状）** | 5.313 ms | **0.163 ms** | **32.6×** |
      | 4096 | 2.603 | 0.092 | 28.3× |
      | 1024 | 0.576 | 0.088 | 6.6× |
      | 16 | 0.087 | 0.087 | 1.0× |

      四种形状**输出全部逐位相同** —— 这同时证明了 **triton-ascend 遵守「被屏蔽的通道不访问内存」
      这个契约**，而那正是去掉 clamp 的前提。收益随行数增长、小行数归零，符合「每 program 的
      寻址开销」这个解释。
      **这个模式在别处扫过一遍了**（`glm53_int8_1card` 按同样的判据查了自己的热路径，
      结论是**否定的**，值得记下来免得有人重扫）：
      - 它那 25 个小向量算子**根本不是 Triton** —— 是 `torch_causal_conv1d_update_npu`
        拼出来的 aten 算子（`Slice+Mul+ReduceSum+ConcatD`）。同样是「时间不在字节里」，
        但**病因不同**：这条是**寻址退化**（改一行表达式），那条是**算子个数**（要换算子）
      - 全包 80 处 `tl.minimum/maximum` 里绝大多数是 softmax 累加器
        （`new_e_max = tl.maximum(...)`），**不是地址表达式**
      - 唯二同形状的两处在 `flash_block_score_decode.py:911/919-920`，
        但 **GLM 走不到**（我们的 NPU/DSA 代码一处都不 import，GLM 用厂商的
        `npu_lightning_indexer`）。⚠ **而且它在 wheel 里不在本仓库**
        （`sgl_kernel_npu/indexer/`）—— 即使哪天可达，也不能在树内改，得打补丁或走上游

      **这条修复的价值两边差 32 倍，而且是量化过的**：prefill 侧（8192 行）值 **32.6×**；
      而在对方 bs=1 decode 的 profile 里，这个 kernel 是 **50.1 µs/step、11 次、4.6 µs/次** ——
      **已经贴在启动开销上，可省的是 0**。与本表 `rows=16 → 1.0×` 是同一件事的两个独立观测。
      **「同一个优化在两边价值差很远」这次有了数。**

      **整网已验**（2026-08-30，TP8 INT8 / die 8-15）：与 P6.10 一起，对**同一配置下
      改动前的基线**逐位相同（`0.000e+00`，3256 token 长提示 + 短提示各 60 decode token）。
      单卡探针只证明单个函数等价，这一步证明接线没被改坏。
      ⚠ **在共享代码里**（`kpool_fp8_index.py`），CUDA 路径同样走这个 kernel。
      掩码 load 的契约在 CUDA 上同样成立，且 clamp 本就来自上游原始提交
      （`0b9c38484e`，CUDA 上开发）而非为昇腾加的防御 —— 但**本线没有 CUDA 机器可验**。

      ⚠ **原记录里那条「4.73 ms 全部来自被 clamp 的 gather load」是对的，但我一开始猜错了是哪一个**：
      kernel 里有两处 clamp 的 load，我先怀疑页表那个，查调用点才发现
      **NPU 路径 `page_table` 和 `topk_offsets` 两个都不传**（`kpool_indexer_npu.py:600-606`），
      `HAS_PAGE_TABLE=False`，那段根本没编进来。是历史值那个 load。
      **「哪一个」这件事必须查调用点，不能从 kernel 里看。**

      原记录（保留）：当初设计的「三处 store」重写方案**既不必要、本身也是错的**
      （三处 store 地址区间重叠、同一 CTA 内跨线程竞态）—— **那条路不要再走**，
      `git stash` 第一条是它未验证的半成品。原始分析：
      实测**那 4.73 ms 全部来自被 clamp 的 gather load**，单把它去掉就是 **5.557 → 0.282 ms（约 20×）**。
      而当初设计的「三处 store」重写方案**既不必要、本身也是错的**（三处 store 地址区间重叠，
      同一 CTA 内跨线程竞态）—— **那条路不要再走**。未验证的半成品在 `git stash` 第一条。
      原始分析：**4.73 ms**，
      `aiv_vec_ratio=0.027`、`aiv_mte2_ratio=0.0`，**既不算也不搬，是纯标量瓶颈**
      （8192 个 program 打在 40 个向量核上）。**在共享代码里**
- [ ] **P6.12 降低 DSA 的 170 次 host 调用** —— ⚠ 原文写「每省一次约 13.5 µs」，
      **那个单价已作废**（见 §1，它是 eager 下一次 8 KB ConcatD 的观测，不是机器常数）。
      图下的单价是 **2.8 µs** 量级，收益要按那个重算。
      ⚠ 原文的「开 TQE=2 后约 8 µs」**已作废**：TQE=2 在图模式下根本起不来（见 P6.9）。
      而且图模式本身就把 launch 开销吃掉了，**这条在图下的收益需要重新量**

### P7 · 长上下文 ✅（2026-08-30，INT8 W8A8 TP8 上闭环；BF16 TP16 的交付构型确认待做）

**结论：GLM-5.3-Flash 在 8 张 die 上跑通了 checkpoint 声称的完整 1,048,576 上下文**，
五个深度的针全部召回。**不需要整机** —— 这一条推翻了「长上下文得先做容量测算才知道能不能跑」
的预期：容量这一关在 TP8 上就过了。

构型：`launch_glm_longctx.sh.example`（W8A8 TP8 / die 8-15 / 端口 30023 /
`--mem-fraction-static 0.90` / `--max-running-requests 8` / chunk 8192 /
graph 开 / overlap 开 / **radix 关**）。KV 池 **1,363,904 token / 17.89 GB**。

工具：`tools/check_long_context.py`（三个探针，见下）。

#### 容量：每 token 的 KV 是 14080 B，且**每个 die 存一份**

| 部署 | KV 池 | #tokens | B/token |
|---|---|---|---|
| BF16 TP16（`glm_bf16_graph_1104.log`）| 15.67 GB | 1,195,072 | 14080 |
| INT8 TP8 mrr=128（`tp8_int8_gsm8k_1343.log`）| 12.32 GB | 939,712 | 14080 |
| INT8 TP8 mrr=8 本轮 | 17.97 GB | 1,370,624 | 14080 |

拆开正好是 **11 个 DSA 层 × (kv_lora 512 + index_head_dim 128) × 2 B**。
MLA latent 不按 TP 切，所以这是**每 die** 的值。三份独立部署整除到同一个常数 ——
**这个数可以直接拿去做容量估算**。

- 一条 1M token 的请求要 **13.75 GiB/die**。TP8 池子 17.89 GiB → 装得下，**并发上限 1**。
- **34 个 KDA 层的状态与上下文长度无关**（TP8 实测约 20 MiB/槽）。
  这正是混合架构值钱的地方：长上下文的边际成本只在那 11 个 DSA 层。
- **卡并发的是谁，随上下文长度换人**：mamba 每槽 ÷ KV 每 token 的交叉点，
  TP8 约 **1490 token**、TP1 约 **11150 token**（单卡线实测 149.8 MiB/槽）。
  交叉点随 TP 度数下降 —— mamba 每槽被切分，而 KV 每 die 每 token 不变。
  ⚠ **同一句「并发被 X 卡住」在两个工况下一句对一句错，写结论必须带工况。**

#### ⚠ 1M 下必须显式传 `--max-running-requests`（新发现，此前没人记过）

`kv_cache_configurator.resolve_max_num_reqs` 在不传时算
`estimated = token_capacity / context_len * 512`，然后 `max(min(estimated, 4096), 2048)`
—— **那个 2048 的下界与 `context_len` 无关**。1M 上下文下它会把
`req_to_token`（`(mrr+1) × context_len × 4 B`）顶到 **8.6 GB/die**。

而 `req_to_token` 是在 KV 池定容**之后**分配的（`_resolve_memory_pool_config`：
先 `_profile_available_bytes` → `config_from_budget` → `resolve_max_num_reqs` → 建池），
**不从 KV 预算里扣**，吃的是 `mem_fraction_static` 留的 slack。
所以它**不会表现成 OOM，会表现成「莫名其妙少了几个 GB」**。传 8 只要 34 MB。

同一类的还有一个：`_zero_rope` 的零 rope 缓存是 **O(KV 池)**（1.24M token 时 159 MiB），
也是定容之后惰性分配。**提高 mem-fraction 会同时放大它。**

#### 三个探针，缺一不可

| 探针 | 判据 | 抓什么 / 抓不到什么 |
|---|---|---|
| **召回** | 五个深度（0/25/50/75/99%）的针全中 | 唯一能说「稀疏选择选对了区域」的。**与构型无关**，TP8 INT8 上的结论对 TP16 BF16 也成立 |
| **前缀不变性** | **逐位相同**，不是「在地板内」 | 抓「长尾往回够到了前缀」—— 那种情况**针照样找得到而 logits 已经漂了** |
| **时序曲线** | 机制指得出来 + 预测成立 | 抓「有东西在扫全长」 |

前缀不变性为什么该是逐位相同：模型全程因果，且前缀长度取 `chunked_prefill_size` 的整数倍时，
两次跑切成同样的 chunk、同样的 GEMM 形状。**取非整数倍就不成立**（末 chunk 宽度不同 →
形状地板回来），工具会警告。

#### 阶梯（全部在同一台 `--context-length 1048576` 的服务上跑）

⚠ **下表的 decode 数字是 `ba2a6372fa` 合并 `int8_singlecard` 之前量的，已偏慢。**
合并后同一条 32640 提示在同构型上是 **22.3 ms/token**，表里是 27.6 —— 差 19%。
**召回与前缀不变性不受影响**（那是功能与逐位判据，不是计时）；
要引用性能数字请重测，或标明「合并前」。

| 提示 token | 召回 | TTFT (s) | 二次模型预测 | 误差 | decode ms/token | 前缀不变性 |
|---|---|---|---|---|---|---|
| 32,640 | **5/5** | 6.7（热态）| — | — | 27.6 | — |
| 130,944 | **5/5** | 28.3 | — | — | 28.2 | **0.000e+00 / 32767 位** |
| 261,952 | **5/5** | 60.8 | 61.5 | −1.1% | 28.9 | — |
| 523,904 | **5/5** | 139.4 | 142.3 | −2.0% | 30.2 | — |
| **1,048,448** | **5/5** | **347.6** | 363 | +4.2% | **32.9** | **0.000e+00 / 32767 位** |

⭐ **`--context-length` 不影响给定长度请求的成本**，实测两次独立印证：
130,944 的提示在 ctx=131072 与 ctx=1048576 两台服务上 TTFT 都是 **28.3 s**；
32,640 的提示在 ctx=32768 上 6.6 s、在 ctx=1048576 上热态 6.7 s
（冷启 8.7 s 的差全是首请求热身，`--skip-server-warmup`）。
**所以整条阶梯一台服务就能跑完** —— 省 4 次重启，每次约 4 分钟。

#### prefill 有真实的二次项，而且二次的是「选」不是「算」

只用 32k / 128k **两个点**拟合 `T(n) = a·n + b·n²`
（a = 1.976e-4 s/token，b = 1.415e-10 s/token²），在 256k / 512k / 1M 上误差
**−1.1% / −2.0% / +4.2%**。1M 处二次项占 **43%**。

机制指得出来：kpool indexer 给第 r 行 query 打分时要扫 `r/index_kpool` 个 pool，
整条 prefill 是 **n²/8 次 pool 打分**。稀疏注意力本体是线性的（每行只看 `topk=2048`）。

**这一项在服务日志里不用 profile 就能看见**：`Prefill batch` 那行的
`input throughput (token/s)` 是**逐 chunk** 打的。同样 8192 token 的 chunk，
512k 请求里头部 3740 → 尾部 3052；1M 请求里头部 3740 → 尾部 2237。
**判断长上下文 prefill 是否健康，先看这条曲线是不是线性上升；超线性就是另一个 bug。**

#### decode 近似平坦，但「近似」的含义要说准

27.6 (32k) → 28.2 → 28.9 → 30.2 → **32.9 (1M)**：**32× 上下文只涨 19.6%**，
每翻一倍稳定 +1.1 ms，拟合 `decode(n) ≈ 27.3 + 5.4e-6·n` ms（1M 处线性项 5.7 ms）。

**封顶的是「读多少 KV」（`index_topk=2048` 固定），涨的是「在多少候选里选」
（`n/4` 个 pool，32k 是 8192 个、1M 是 262144 个）。这是两件事，别混。**
decode 之所以没像 prefill 那样二次，是因为它每步只选一次，而 prefill 要为 n 行各选一次。

#### BF16 TP16 交付构型 —— ✅ 已确认（2026-08-30 19:02，整机）

`--context-length 1048576 --max-running-requests 8 --mem-fraction-static 0.85`，
其余同 `launch_glm_bf16.sh.example`。KV 池 **1,193,728 token / 15.65 GB**
（= 14077 B/token，**第四份独立部署印证 14080**），1M 需 1,048,576 -> 富余 145,152。

| 提示 token | 召回 | TTFT (s) | decode ms/token | 前缀不变性 |
|---|---|---|---|---|
| 32,640 | **5/5** | 6.3 | **20.2** | — |
| **1,048,448** | **5/5** | **292.1** | **25.5** | **0.000e+00 / 32767 位** |

**两个构型并排（同一条提示、同一套判据）**：

| | TP8 INT8（8 die）| TP16 BF16（16 die）| 比 |
|---|---|---|---|
| 32,640 decode | 27.6 ms/tok | 20.2 | 1.37x |
| 1,048,448 TTFT | 347.6 s | 292.1 s | 1.19x |
| 1,048,448 decode | 32.9 ms/tok | 25.5 | 1.29x |

⚠ **die 数翻倍只换来 1.19x / 1.29x，原因可以指名**：1M 的 prefill 里约 43% 是
kpool 的「选」，而 `index_n_heads = 32` 在 TP16 下**每 die 只剩 2 个头**（TP8 是 4，
TP1 是全部 32）。**主导项恰好不是被 TP 切得最干净的那部分。**
=> 推论一：**日常长上下文验证走 TP8 是划算的** —— 慢 19%，省下整台机器。
=> 推论二：**任何在窄 TP 上量出来的 indexer / DSA 比值，都是整个 TP 谱系里最有利的一端，
   不能往多卡外推。**

#### 还没做的

- [ ] **长上下文 + 并发**。本轮全是单请求（1M 下并发上限本来就是 1），
      但 256k 以下可以并发，而「批里有别人时才会发生的事」单请求结构上碰不到（RESUME 教训 7）。
- [ ] **开 radix cache 之后重跑**。本轮全程 `--disable-radix-cache`：
      `causal_conv1d_fn_npu` 混合 `has_initial_state` 会写坏冷请求的 conv state（P6.2），
      **开着它出来的召回数字不可信**。
- [ ] **1M 的性能优化**。二次项占 43% 是可攻击的，但先量清楚 prefill 在真实负载里的占比
      （与 P3.4 的重启条件同理）。

### P5.1 · shared expert 并进 GroupedMatmul —— 精度回归通过（2026-08-30，TP8 288 专家）

单卡线的 `3f7db2fece`（放开 `glm5_next.py` 的 `not _is_cuda` 门 + `npu/moe/topk.py`
的 `_append_fused_shared_slot`）在**它自己的 16 专家部署上验不了精度**：teacher-forced
`mean|dlp| = 1.243e-01` 对重测地板 `1.233e-01` = 1.01x，**地板本身高到 1.2e-01**，
判据没有分辨率。**这里用 288 专家的真 checkpoint 回归。**

| 轮次 | 构型 | GSM8K | stop rate |
|---|---|---|---|
| 原基线（`d279764c1b`）| INT8 TP8 | 97.42% | — |
| **B′** | 合并树 + 本线全部改动，**无融合** | **97.65%**（1288/1319）| 100.00% |
| **C** | **+ 融合**（日志确认 `Shared experts fusion optimization enabled`）| **97.42%**（1285/1319）| 100.00% |

**差 3 题 = −0.23pp。** 两轮独立二项之差的 SE 是 `sqrt(2) x 0.47 = 0.66pp`，
实测是 **0.35 个 SE** —— 一致。

⚠ **说准它答了什么**：排除的是「明显掉精度」，**排除不了小于约 1.3pp（2 SE）的真实退化**。

⚠ **为什么要 B′ 这一轮**：单卡线原本只要「融合前后对比」，但那 18 个只在 TP1 验过的
commit（含 KDA 投影融合、conv 池翻 window-major 两条**非逐位**改动）当时已经在树上。
不先隔离，C 与原基线的差就是「18 个 commit + 融合」的混合，**掉了也说不清该回退谁**。
B′ 顺带回答了那 18 个：在 288 专家上干净。

⚠ **两轮的构型可比性核过**：KV 池 938,176 vs 938,048（差 128 token），
并发/mem-frac/ctx/chunk 全同。**并且先确认了融合真的生效** —— 否则「无差异」
测的是同一个东西两遍，是假阴性。

### P8 · MTP / 投机解码 ☐（bring-up 通了，**并发 >= 32 有未解决的越界**）

**跑起来了，但只在单请求或长度一致的批上。** 这一节最重要的是那个未解决的 bug。

#### 已经通的

| | |
|---|---|
| draft 模型 | `Glm5NextForConditionalGenerationNextN`，**1.27 GB/die**，走 compressed-tensors 量化 |
| 池 | target + draft 两个 KV 池都建起来 |
| 图捕获 | decode / TARGET_VERIFY / draft-extend **三种全过** |
| 投机确实在发生 | 接受长度 **1.71–2.00**（`num_draft_tokens=2`，两条打满 2.000）|
| 长提示召回 | MTP 开着，32576 token 五深度 **5/5** |

W8A8 checkpoint 的 layer 45 **是量化的**（871 个 scale），`quantization_config.ignore`
只列了本就不量化的 norm/gate、没有通配符，所以 `_resolve_nextn_quant_config` 走量化路径是对的。

#### ⛔ 未解决：**批内长度不一致（ragged）时** AI Core 越界

```
errorStr: MTE accesses an invalid GM address or the cross-device memory access times out
```

⚠ **触发条件是 ragged，不是并发数** —— 这一条我先判错过一次，记下来免得重蹈：
- 长度**几乎相同**的短提示：并发 1/2/4/8/16 全过，32 崩
- **GSM8K（提示长度差异大）**：`#running-req: 16` 就崩
=> 先看到的「32 崩」让我写成了「并发 >= 32」，实际是**批越大越容易 ragged**。
   **「阈值」形式的结论要先问一句「我扫的那个维度是真正的自变量吗」。**
- 单请求（bs=1）六次跑全绿 —— 这个失败模式**结构上需要批里有别人**，
  与 RESUME 教训 7 同形
- 异步错误，宿主端在 `prepare_for_draft_extend -> ForwardBatch.init_new`
  或 `overlap_utils.resolve_forward_inputs` 处撞到它撞上的下一个 copy
- **已排除三个嫌疑**：① draft-extend 的行布局（`prepare_for_draft_extend` 的
  `extend_num_tokens = bs * (num_draft_tokens + num_front_tokens)` 可能不等于
  `bs * num_draft_tokens`）—— 加了响亮断言，**没触发**；
  ② `kpool_spec_update_index_cache` 里的每个索引 —— 逐个算过边界；
  ③ `verify_intermediate_state_indices[:batch_size]` —— 按
  `max(get_eager_max_batch_size, pool_size)` 分配，128 够用
- ⭐ **已定位到图内（2026-08-30 收尾时测出）**。两条独立证据：
  ① 错误自己报在 **`NPUGraph.cpp:284, replay`** —— 崩的是**图重放**，不是 eager 路径
     （这也是 `ASCEND_LAUNCH_BLOCKING=1` 定位不到具体算子的原因：
     **一次图重放是一个不透明提交**，同步模式管不到它内部）
  ② **加 `--disable-cuda-graph` 之后，同一个 ragged 16 批连跑两次全过**
     （接受长度 1.978 / 1.982，几乎全接受）
  => **不是算子错，是图内某个「捕获时定尺寸」的静态界被 ragged 运行时超过。**
- **下一步（起点很具体）**：查投机路径上所有在**捕获期**定尺寸、运行期由 ragged 数据填充的
  缓冲与界。**本轮已经在这一类里修过一个** —— `max_visible_pool_runs`
  （ceil 的和 ≠ 和的 ceil，界不够大 -> scatter 跑出 `max_runs+1` 的缓冲区，
  同样是 AI CPU/AI Core 越界）。**很可能还有第二个同类的。**
  ⚠ 排查顺序建议：先列出 spec 路径上每一个 `[:bs]` / `[:batch_size]` 的静态缓冲切片，
  逐个问「它的分配尺寸覆盖得住 ragged 运行时的最坏取值吗」。
- ⚠ **两次通过不是证明**：组批本身不确定（见 RESUME「精度是怎么判的」），
  这个失败是**间歇性**的。**先测崩溃率再谈修好**，否则「修好了」和「运气好」分不开。
  上面那两次只是把嫌疑从「算子」移到「图」，不是「关图就没问题」的结论。
- 已排除（别重走）：快照方向错误（wrapper 形状校验会 ValueError 不会 MTE，**演绎**）、
  conv layout 静默错（contiguous 校验响亮，**演绎**）、
  索引/槽位错误（`layer_check/check_kda_spec_snapshot.py` 在 ragged 2-32 上过，
  带置换等变性 + 只读 + 负对照）、draft-extend 行布局（响亮断言没触发）、
  投机路径里的 `is_cuda` 门（扫过，只有 `memory_pool.py:943` 且前有 `not _is_npu`）

#### 性能：bs=1 下净加速为零，而且要吃掉一半 KV 池

| | 每步 | 每步产出 | 每 token |
|---|---|---|---|
| 无 MTP | 23.3 ms | 1 | **22.3 ms**（三次重复一致）|
| MTP | 43.2 ms | 1.97 | **21.9 ms** |

**verify 步是 decode 步的 1.86x，只换来 1.95 个 token —— 打平。**
一个 verify 周期要跑 draft-extend + draft decode + target verify **三次前向**，
而 draft 虽然只有一层，**每次前向的固定开销（launch、HCCL、图重放）不按层数缩放**。

⚠ **容量代价更值得看**：`mrr=128` 下投机的中间缓冲吃掉 **4.57 GB/die**
（`intermediate_ssm_state_cache` 4.41 + conv window 0.16），它随
`max_running_requests x num_draft_tokens` 线性增长，且在 KV 池定容**之后**从同一份预算里扣：

| | 无 MTP | MTP |
|---|---|---|
| KV 池（mrr=128, mem-frac 0.85）| 938,176 token | **418,496 token** |

**MTP 把 KV 池砍掉 55%**，而 KV 池决定能同时装多少长请求。
=> **在把那个 OOB 修掉、并证明高并发下有实际吞吐收益之前，不建议开 MTP。**

#### ⚠ 判据上的一个坑（花了这一轮很多时间才想明白）

「贪心下 MTP 必须与非 MTP 输出逐位相同」**在这台机器上不是合法判据** ——
基线自己跨 batch 宽度就不可复现（见 RESUME「精度是怎么判的」一节）。
能用的是：**GSM8K 统计** + **接受长度**（后者不需要参考：verify 算错就配不上
target 的 argmax，接受长度会塌向 1.0，**算错的 verify 产不出「每个 draft 都被接受」**）。

---

## 4. 待决与已知缺陷

- [ ] ⚠ **KDA conv 池改成 window-major 之后，两条路径本部署验证不了**（`int8_singlecard` 线，
      2026-08-30，见 `int8_singlecard/REPORT.md` §8.1）。**这是「验不了」不是「没时间」** ——
      这个部署没有能触发它们的负载。
      | 路径 | 为什么验不了 | 什么条件下能验 |
      |---|---|---|
      | MTP / speculative 快照路径（`ascend_kda_backend.py` 约 700–780） | 不跑 MTP / spec decode。两处改动在代码里都标了 `UNVERIFIED` | 起带 MTP / spec decode 的配置；或构造直接驱动快照路径的单层 harness |
      | mask-track 散写分支（`has_mamba_track_mask`） | `check_kda` 用 `enable_mamba_extra_buffer=False` 建池且从不设 `mamba_track_mask`，**这条分支从未执行过** | 让 harness 构造带 track mask 的池 |

      ⚠ mask-track 那条**转而验了它依赖的不变量**（算子 state 回写与 `x[L-3:L]` 在
      L=64/256/8192 下逐位相同，那正是散写要写的东西）。**构造不出那条路径时，
      验它必须成立的性质比不验强，也比假装验过诚实** —— 但它不是端到端测试，别当端到端引用。

      ⚠ 与 P3.4 的 MTP 待办相邻：那里已记着「接 spec decode 前必须先解决
      `kpool_decode_update_index_cache` 每请求一行的假设」。**这两条要一起做** ——
      同一个配置能同时验掉。


- [x] ~~**P5 的磁盘**~~ —— FP8 源已删，W8A8 已转出，实际占 306.1 GiB（不是估的 333）。
      **现在余 23 GB**，没有再腾挪的余地了
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
- [ ] **MoE 的 `weight_scale` 被降到 bf16，linear 侧保持 fp32**（源码 + 实测，**未处理**）：
      `NPUW8A8Int8MoEMethod.process_weights_after_loading`（`moe_methods.py:528`）
      把每通道 scale 从 fp32 降成 **bf16**，而 `linear_method_npu.py:129` 保持 fp32。
      在本 checkpoint 上实测：这在权重本身约 9.1e-3 的量化误差之上，
      再叠一层系统性的每通道增益误差，**均值 1.7e-3、最大 2.5e-3**。
      torch_npu 文档说 `npu_grouped_matmul` 的 `scale` 在 `group_list` 是张量时
      **fp32 也合法**，所以改回 fp32 是可行的 —— 但没测它会不会选到同一个 kernel，
      而且 97.19% 的结果不构成动它的理由。**记作线索，不是 bug。**
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
