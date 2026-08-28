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
`aiv_vec_ratio=0.027`、`aiv_mte2_ratio=0.0`，**既不算也不搬，是纯标量瓶颈**。
两者合计占 DSA prefill 单层 50.8 ms 的 **21%**。

**谁受影响**：⚠ **CUDA 直接受影响** —— 这是共享的 kpool 实现，不是 NPU 分支。
token id 最大约 32768，int32 在数值上完全够；但 CUDA 侧的 kernel 与下游对 dtype 的假设要逐一核。

**回归**：待定。**这是本台账里 CUDA 风险最高的一条。**

---

## 关于 DSv4 回归

多条改动都指向「需要跑一次 DSv4 GPQA」。基线在 PLAN §3：**73.74%**（P0，三轮 74.24/75.25/71.72），
P1 rebase 后 **73.23%**。起服务脚本见 `launch_dsv4_a3.sh.example`。
**攒够了一起跑一次，不要每条都跑。**
