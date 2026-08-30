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

## 4. `layers/quantization/mxfp4_flashinfer_trtllm_moe.py` —— 只报告，**没改**

**问题**：`maybe_fuse_routed_scale_and_shared_add()` 的**第一句**是

```python
from sglang.srt.layers.quantization.expert_pack import ExpertPackMoEMethod
```

而 `expert_pack.py` 的模块头部是 `from sgl_kernel.quantization import ggml_moe_a8_vec`。
`sgl_kernel` 是 CUDA 扩展。于是在任何 quant-method 判断之前就
`ModuleNotFoundError: No module named 'sgl_kernel'`。

**谁会碰到**：`DeepseekV2MoE.forward_normal` **每次 forward 的结尾**都调它，
而 GLM 的 MoE 就是它（`glm5_next.py:106`：`from ...deepseek_v2 import DeepseekV2MoE as Glm5NextMoE`）。
所以 **GLM 在昇腾上的每一次 MoE decode 都会撞上**；DSv4 / 任何非 CUDA 平台的
DeepseekV2 系 MoE 同理。

**实测**（2026-08-29，`layer_check/graph_capture/cap_runner_layers.py`，A3 单卡）：
GLM layer 3 的 MoE 在图捕获的第一次 forward 就抛这个异常。
之前没暴露，是因为 `check_moe.py` / `cap_moe.py` 都是**手搭 moe runner**、
不经过 `DeepseekV2MoE.forward`。

**为什么没改**：共享路径（`layers/quantization/`），按本文件的规矩先报告。

**建议的修法**（一行位置）：把那个 import 挪进 `isinstance` 判断里，或者包一层
`try: ... except ImportError: ExpertPackMoEMethod = ()`。函数体里另外三个
Mxfp4* import 也是 CUDA-only 但它们**没有**模块级的 `sgl_kernel` 依赖，所以只有
`expert_pack` 这一条是致命的。

**绕过（测试用，不进产品路径）**：
`layer_check/graph_capture/runner_fixture.py:patch_shared_path_gaps()`
把这个函数替成它自己的 `fused=False` 分支（昇腾必然走的那条）。

**回归**：CUDA 侧不受影响（import 本来就成功）。修完需要在 CUDA 上跑一次
DeepseekV2/V4 的 MoE 前向确认 `fused` 判定没变。

---

## 4. `layers/quantization/mxfp4_flashinfer_trtllm_moe.py` —— 把一个 CUDA-only 的 import 包起来

**改动**：`maybe_fuse_routed_scale_and_shared_add()` 里对 `ExpertPackMoEMethod` 的 import
改成 `try/except ImportError`，导不进来就不放进 `isinstance` 的元组。

**为什么**：`expert_pack.py` 模块头部有 `from sgl_kernel.quantization import ggml_moe_a8_vec`，
而 `sgl_kernel` 在昇腾上**不存在**（实测：`ModuleNotFoundError: No module named 'sgl_kernel'`）。
这个函数被 `DeepseekV2MoE.forward_normal` 在**每次 forward 的结尾**调用
（`deepseek_v2.py:1071` 与 `:1211`），GLM 的 MoE 就是它。
**所以昇腾上每一次 MoE forward 都会 ModuleNotFoundError** —— 整网跑不起来。

（这个 bug 之前没暴露，是因为整网启动更早死在 kpool 的 `NotImplementedError` 上，
根本没走到 layer 3 的 MLP；而单层的 `check_moe.py` 直接驱动 MoE 模块，不走 `forward_normal`。）

**谁受影响**：所有**没有** `sgl_kernel` 的平台。**CUDA 上 import 成功，元组完全相同，行为不变。**

**为什么这样改是安全的**：那个类只出现在一个 `isinstance` 判断里。
**模块导不进来，就不可能有对象是它的实例**，所以从元组里省掉它在语义上恰好等价。
另外三个 mxfp4 方法在昇腾上 import 都是好的（实测），只有 `expert_pack` 一个坏。

**回归**：CUDA 不需要（import 成功时代码路径逐字相同）。昇腾上实测函数两条分支都正确。

---

## 5. `mem_cache/memory_pool.py` —— `HybridLinearKVPool` 转发 `set_index_k_bf16`

**改动**：给 `HybridLinearKVPool` 加一个 `set_index_k_bf16` 转发，和它已有的
`set_index_k_scale_buffer` 完全同形（含 `_transfer_full_attention_id` 的层号翻译）。

**为什么**：GLM 的 pool 是 `HybridLinearKVPool` **包着** `NPUDSATokenToKVPool`，
而 `get_token_to_kv_pool()` 返回的是**外层**。bf16 索引缓存的写入方法只加在了内层，
于是整网第一次 prefill 到 layer 3 就 `AttributeError`（16 个 rank 同时报）。

**为什么单层验证没发现**：`layer_check` 里直接构造 `NPUDSATokenToKVPool` 并直接调它，
**绕过了包装**。这和图捕获那轮发现的 `AscendHybridLinearAttnBackend.forward_metadata`
是**同一类** —— 「GLM 的顶层对象是个包装，而我们的新方法加在被包的那个上」。

**为什么不在 NPU 侧绕过包装**：外层转发时会做 `_transfer_full_attention_id`
（全局层号 → 11 个 DSA 层里的下标）。绕过包装直接调内层就得自己复制这个翻译，很脆。

**谁受影响**：**没有人**。它委托的 `set_index_k_bf16` 只存在于 `NPUDSATokenToKVPool`，
CUDA 上没有任何调用者能走到这里。纯增量。

**回归**：CUDA 不需要。昇腾上由整网启动本身验证。

**顺带做的排查**：把 indexer 用到的 9 个 pool 成员、注意力本体用到的 4 个、
backend 的 2 个，逐个对 `HybridLinearKVPool` / `AscendHybridLinearAttnBackend` 核过，
**同类缺口没有别的了**。（`page_size` / `slots_per_page` 是实例属性，
类级 `hasattr` 看不到，已单独确认 `:1696` 与 `:4825` 会设。）

---

## 待决：三处只报告、没改的问题（图捕获那一轮发现）

这三处**没有改动**，记在这里是因为**都会影响 GLM 之外的东西**，改不改要先定。

**①「图 padding 的填充值两边不一致」—— ✅ 已按修法 A 修**（`ascend_kda_backend.py` 覆写 `get_cuda_graph_seq_len_fill_value` 返回 0，与 runner 实际填的值、以及 `AscendMambaAttnBackendBase:212` 的先例一致；NPU 树内，不动共享路径）。原始分析： —— `hybrid_linear_attn_backend.py:88` / `:750`
对 `ascend_backend.py:792`。runner 按顶层 backend 报的填充值去填 padded `seq_lens`，
而 `HybridLinearAttnBackend` 把这个问题**委托给全注意力那一半**（昇腾上是 **0**），
`MambaAttnBackendBase` 自己却缓存成 **1**。CUDA 上两边碰巧都是 1，所以没暴露。

**2026-08-29 实测**（走真实 `NPUGraphRunner`，`cap_runner_layers.py`，部署形状）：

- **不一致是真的**：`runner.seq_len_fill_value = 0`（顶层 `_AscendKDAHybrid` 报的），
  `AscendKDAAttnBackend.get_cuda_graph_seq_len_fill_value() = 1`。
  GLM 的线性半边解析到的是 **`MambaAttnBackendBase._replay_metadata`**
  （`AscendKDAAttnBackend → KDAAttnBackend → MambaAttnBackendBase`；
  昇腾自己那份 `AscendMambaAttnBackendBase._replay_metadata`（fill=0）**GLM 走不到**）。
- **今天确实是死代码**：`build_replay_fb_view` 无条件设
  `num_padding=bs-raw_bs`（`decode_cuda_graph_runner.py:188`），所以
  `if num_padding is None` 那个 fallback 在 decode 图路径上永远不进。
- **如果进了会怎样（实测，不是推断）**：把同一个 padded batch
  （bucket 16，3 行 padding，`seq_lens=[...,0,0,0]`）用 `num_padding=None`
  重新 plan 一次再 replay ——
  fallback 数出 `num_padding = 0`（它拿 1 去比 0），真值是 3。
  后果是 `mamba_indices[bs-0:] = -1` 这一步被跳过，**padding 行被当成真实行**
  写进 mamba slot 0，而不是被 `-1`（PAD_SLOT_ID）屏蔽掉。
  **真实行的输出仍然逐位相同**，被弄脏的只有 mamba slot 0 ——
  也就是说**爆炸半径是被池子的布局兜住的**（`MambaSlotAllocator` 的
  `free_slots = arange(1, size+1)` 把 slot 0 留作 padding 的 dummy），
  **不是被 metadata 代码兜住的**。哪天有池子把 slot 0 发给真实请求，
  那个请求的状态就没了。
- 修法不变：A（`AscendKDAAttnBackend` 覆写返回 0，NPU 树内 2 行）
  或 B（改共享 `_replay_metadata` 去问顶层 backend）。**仍未改。**

**②「`seq_lens_cpu_list` 在捕获时被永久烘死」** —— `ascend_backend.py:645` 算一次，
`_apply_cuda_graph_metadata` 从不刷新。

**2026-08-29 实测**：确认**永久失效**。捕获时填的是静态 buffer 的初值
（`[0]*bs`），之后不管 replay 用什么 `seq_lens`，
`graph_metadata[bs].seq_lens_cpu_list` 一直是 `[0, 0, ...]`；
同一个 metadata 的**设备** `seq_lens` 则被就地刷新成真值。
另外 `seq_lens_cpu_int` 在图模式下**根本是 None**（只有 `init_forward_metadata`
这条 eager 路径会设它）。

**GLM 确实逃过**：把 `seq_lens` 换掉再 replay，输出与 eager **逐位相同**
（`cap_runner_layers.py` 的场景 C，部署形状下 bs 桶 1..16 全绿）。
因为 DSA 在 `:2607` 就短路进 `forward_sparse`，那里用的是
`forward_metadata.seq_lens`（设备张量），到不了 `:2620` 的 `forward_decode_graph`。
**对任何走 FIA 的非 DSA 昇腾模型仍然是活的、静默的 bug。** 仍未改。

**③「MoE 的 `group_list` 会被烘死」** —— `layers/moe/moe_runner/ascend.py:277` 把每步都变的
host list 物化成设备张量喂 `npu_grouped_matmul`。

**2026-08-29 实测**：GLM 的部署配方下 `get_moe_a2a_backend() = NONE`、
`is_deepep() = False`，quant method 是 `UnquantizedFusedMoEMethod`，
**这条分支根本不进**。`--deepep-mode normal` 就活了。**共享代码，仍未改。**

## 关于 DSv4 回归 —— **本项目决定不跑（2026-08-29 用户拍板）**

⚠ **这是一条被接受的风险，不是一条待办。** swiglu_limit 那条改动**确实会改变 DSv4 的数值**
（给它的 routed 专家加上本就该有的 clamp；在真实 gmm1 输出上实测：修前 2.85× budget 判失败、
修后 0.35×，与出厂 `NPUSwiglu` 同数）。本项目不验证它，**合入前需要下游自己确认**。
另外 DSv4 权重在 P1.2 后已删（只留元数据），真要跑得重新下载约 275 GB。

下面是原本的回归口径，留给要跑的人：



多条改动都指向「需要跑一次 DSv4 GPQA」。基线在 PLAN §3：**73.74%**（P0，三轮 74.24/75.25/71.72），
P1 rebase 后 **73.23%**。起服务脚本见 `launch_dsv4_a3.sh.example`。
**攒够了一起跑一次，不要每条都跑。**
