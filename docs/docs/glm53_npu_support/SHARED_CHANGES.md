# 共享路径改动台账

这个仓库是 sglang 的 fork。**改到「非 NPU 专属」文件的每一处，都要在这里记一笔** ——
因为它可能影响 CUDA、影响 DSv4、或者影响将来往上游合。

NPU 专属目录（`srt/hardware_backend/npu/**`）的改动**不需要**记在这里。

每条记：**改了什么 / 为什么 / 谁会受影响 / 要跑什么回归才算闭环**。

---

## 1. `models/deepseek_v2.py` —— `MoEGate.forward` 的非 CUDA 分支改用 fp32

**改动**：`elif not _is_cuda: logits = F.linear(hidden_states, self.weight, None)`
→ `F.linear(hidden_states.float(), self.weight.float(), None)`

**为什么**：这一条是唯一停在 bf16 的分支。CUDA 的三条分支**每一条都特意拿 fp32**
（`dsv3_router_gemm(out_dtype=float32)`、`aiter_dsv3_router_gemm`、`linear_bf16_fp32`，
最后那条还专门写了注释解释为什么绕路）。所以这是**让非 CUDA 跟上，不是改行为**。

损失落在 top-k 上，表现为**离散地选错专家**而非小误差：layer 3@8192 的 top-8 集合重合率
0.99291（5.65% 的 token 选错，最差那个输出偏 34%），而同一用例的 rel-L2 **在预算之内**。

**谁受影响**：所有非 CUDA 平台（NPU / XPU / HIP-无-aiter / CPU）。**CUDA 一行不受影响**。

**代价**：decode +36 µs，prefill 8192 +383 µs（整层的 8%）。

**回归**：CUDA 不需要（走不到这条分支）。NPU 已验（`layer_check/check_moe.py`）。
**其他非 CUDA 平台没验。**

---

## 2. `layers/moe/moe_runner/ascend.py` + `hardware_backend/npu/moe/activation.py`
—— DeepEP 分支补上 `swiglu_limit`

**改动**：DeepEP 分支多传 `swiglu_limit=config.swiglu_limit`；
`NPUSwigluDeepEPKernel` 接上 `swiglu_quant` **本来就有**的 `do_limit`/`limit` 参数。

**为什么**：DeepEP 路径静默丢掉了 `swiglu_limit=10.0`，同一层里 shared 专家 clamp、
routed 专家不 clamp。真实 gmm1 输出上修前 **2.85× budget 判失败**，修后 0.35×。

**谁受影响**：⚠ **DSv4**。它的出厂配方就是 `--moe-a2a-backend deepep`，所以这个改动
**会改变 DSv4 的数值**（给它的 routed 专家加上本就该有的 clamp）。
GLM 不受影响 —— 它的配方是 `none`，走不到这条分支。

**方向是对的**（与 CUDA 路径一致、与它自己的 shared 专家一致），但这不是「逐字节不变」的改动，
**目的就是改变输出**。

**回归**：❌ **未做。合入前需要单跑一次 DSv4 的 GPQA 对账**（基线 73.23–73.74%）。

---

## 3. `layers/attention/dsa/kpool_fp8_index.py` —— 性能改动（进行中）

**计划改动**：`expand_pooled_groups_to_topk` 的 int64 索引算术改 int32；
重写 `_append_kpool_tail_to_topk_kernel`。

**为什么**（kernel 级 profiler 实测）：前者的 `aclnnAdd` 花 **5.73 ms** 产出
`[8192,512,4]` 的 int64（134 MB），高于带宽下界 **43×**；后者 **4.73 ms**，
`aiv_vec_ratio=0.027`、`aiv_mte2_ratio=0.0`，**既不算也不搬，是纯标量瓶颈**
（后续实测：那 4.73 ms **全部来自被 clamp 的 gather load**，去掉即 **5.557 → 0.282 ms**）。

---

### CUDA 侧 dtype 契约核查（2026-08-28，纯源码，我们没有 CUDA 卡）

**结论：int32 化不是放宽，是回到 CUDA 参考实现本来的做法。** 五条依据：

**① 返回值 dtype 本来就是 int32，不变。**
`expand_pooled_groups_to_topk` 的三个分支**全部**以 `.to(torch.int32)` 收尾，
最后的 `torch.full_like(output, -1)` 也是 int32。**int64 纯粹是内部中间量**，
从来没有出现在这个函数的对外契约里。

**② CUDA 对同一个变换的融合实现，全程 int32。**
`kernels/jit/csrc/dsa/kpool_topk_transform.cuh`：
- `:277` / `:296` —— `const auto raw_token = group_id * pool_size + slot;`，
  `group_id` 来自 int32 的 indices 数组，`pool_size` 是 `const int32_t`
- `:230-241` —— `__device__ int32_t transform_kpool_token(int32_t raw_token,
  const int32_t* page_table_entry, const int32_t* topk_indices_offset, int32_t offset)`，
  三条分支（查页表 / 加偏移 / 原样）**输入输出都是 int32**
- `:280` / `:299` —— 尾部拼接 `raw_token = length * pool_size + (col - history_len)`
  同样是 int32

**所以 torch 兜底用 int64 才是那个异类。**

**③ 影响面比原先估计的窄得多。**
`expand_pooled_groups_to_topk` 全仓只有两个调用点：
- `kpool_fp8_index.py:638` —— `topk_from_pooled_history_logits` 的**非融合**分支
- NPU 侧的 `hardware_backend/npu/attention/kpool_indexer_npu.py:199`

而 `:593` 的分派是：`group_topk ∈ (128,160,192,224,256,512)` 走**融合 CUDA kernel**，
只有 `group_topk == 2048` 才落到 `:638` 的 torch 路径。
**GLM 的 `group_topk = 2048/4 = 512`，所以 CUDA 上的 GLM 走融合 kernel，根本到不了这个函数。**
CUDA 侧会走到它的，只有 `index_topk / index_kpool == 2048` 的配置
（即 `topk=4096, pool=2` 或 `topk=8192, pool=4` 这类）。

**④ 数值范围安全。**
`token_ids` 的最大值 = 最大 pool id × pool_size + (pool_size-1) ≈ **上下文长度**；
`topk_offsets` 的量级 ≤ 一个 batch 的总 token 数（本机 `max_total_num_tokens=1195392`）。
int32 上限 2.1e9，**要到 20 亿的上下文才会溢出**。

**⑤ 逐位相同的论证，以及它的边界。**
对 `+` 与 `*`，int32 回绕 ≡ int64 后截断；而结果**本来就要截断成 int32**，
所以在不溢出的前提下**逐位相同**。
⚠ **唯一的例外**：查页表那条分支现在是「int64 算 → clamp → 截断」，
改成「int32 算 → clamp」之后，**若中间值溢出 int32，两者的 clamp 结果不同**
（int64 会 clamp 到上界，int32 回绕后可能 clamp 到别处）。由 ④ 的界，
上下文 < 2^31 时不可能发生。

### 一个实现约束

`torch.gather` 的 `index` **必须是 int64**，所以查页表那条分支仍要 `.to(torch.int64)`，
**且必须放在 clamp 之后**（clamp 前转宽就白省了）。
⚠ 也就是说**那条分支省不掉 int64 的物化**。
好消息是 **NPU 路径不走它**：`topk_from_pooled_selection` 传的是
`page_table=None, topk_offsets=None`，落在最后那条 `else`（纯 int32），
而实测的 5.73 ms 正是在这条路上花掉的 —— **改动完整覆盖了被测到的开销**。

### 有 CUDA 卡的人要跑什么才算闭环

1. 找一个 `index_kpool > 1` 且 `index_topk / index_kpool == 2048` 的配置
   （GLM 不满足，它走融合 kernel）
2. 三条分支**各跑一次**：传 `page_table` / 传 `topk_offsets` / 两者都不传
3. 比对改动前后的输出张量：**应当逐位相同**
4. 顺带确认 `_append_kpool_tail_to_topk_kernel` 的改动（它是 Triton，CUDA/NPU 同一份代码）

**回归状态**：❌ 未做（本机无 CUDA 卡）。

---

## 待决：三处只报告、没改的问题（图捕获那一轮发现）

这三处**没有改动**，记在这里是因为**都会影响 GLM 之外的东西**，改不改要先定。

**①「图 padding 的填充值两边不一致」** —— `hybrid_linear_attn_backend.py:88` / `:750`
对 `ascend_backend.py:792`。runner 按顶层 backend 报的填充值去填 padded `seq_lens`，
而 `HybridLinearAttnBackend` 把这个问题**委托给全注意力那一半**（昇腾上是 **0**），
`MambaAttnBackendBase` 自己却缓存成 **1**。CUDA 上两边碰巧都是 1，所以没暴露。
**今天不炸**（decode runner 显式传 `num_padding`，那个比较是死代码），但任何不传
`num_padding` 的路径会**静默把 padding 当真实行**。
修法 A：`AscendKDAAttnBackend` 覆写返回 0（NPU 树内，2 行）；
修法 B：改共享 `_replay_metadata` 去问顶层 backend（**动共享 CUDA 路径**）。

**②「`seq_lens_cpu_list` 在捕获时被永久烘死」** —— `ascend_backend.py:645` 算一次，
`_apply_cuda_graph_metadata` 从不刷新，然后 `forward_decode_graph` 把它当
`actual_seq_lengths_kv` 喂给 FIA。**GLM 逃过一劫**：DSA 在 `:2607` 就短路进
`forward_sparse`，到不了 `:2620`。但**对任何走 FIA 的非 DSA 昇腾模型是活的 bug**，
而且是那种「不报错、数值悄悄错」的。在 NPU 树内可改，但会影响其他昇腾模型。

**③「MoE 的 `group_list` 会被烘死」** —— `layers/moe/moe_runner/ascend.py:277` 把每步都变的
host list 物化成设备张量喂 `npu_grouped_matmul`。**GLM 的部署配方
（`--moe-a2a-backend none`）走不到**，`--deepep-mode normal` 就活了。**共享代码。**

---

## 关于 DSv4 回归

多条改动都指向「需要跑一次 DSv4 GPQA」。基线在 PLAN §3：**73.74%**（P0，三轮 74.24/75.25/71.72），
P1 rebase 后 **73.23%**。起服务脚本见 `launch_dsv4_a3.sh.example`。
**攒够了一起跑一次，不要每条都跑。**
