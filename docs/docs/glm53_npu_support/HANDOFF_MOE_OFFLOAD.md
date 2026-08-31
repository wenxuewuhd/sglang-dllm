# 交接：MoE offload（单卡 A3）—— 从性能分析这条线交出去的背景

**这份不设计方案。** 它只交代三件事：**MoE 在哪里接线**、**在这台机器上怎么把单卡跑起来**、
**MoE 这块已知的坑**。方案（dsv4 + KTransformers + sglang 的融合）是你的事。

⚠ **已有文档不重复，缺的才写在这里**：

| 要什么 | 看哪 |
|---|---|
| 环境搭建、CANN / torch_npu 版本 | `SETUP.md` |
| 从零拉代码到跑通、权重转换 | `REPRODUCE.md` |
| 改完跑什么回归 | `REGRESSION.md` |
| 共享路径改动台账（**改 MoE 前必看**） | `SHARED_CHANGES.md` |
| 整体适配状态、待办编号 | `PLAN.md` / `RESUME.md` |
| 单卡 INT8 的完整性能分析 | `int8_singlecard/REPORT.md` |
| 单层算子性能用例（KDA / DSA） | `oplab/README.md` |

---

## 1. 一句话背景：为什么 offload 是容量问题

**routed 专家权重 283.5 GiB（int8，288 专家 × 42 层），HBM 一张 die 64 GiB。**
这是 offload 的全部理由 —— **不是为了更快**。

**但 bs=1 每 token 只读 8 个专家**：每层 192 MiB（gmm1 的 w13 128 MiB + gmm2 的 w2 64 MiB），
42 层合计 **7.88 GiB/token**。

⚠ **这个数是 offload 设计的中心约束**，而且它**和专家总数无关** —— top-k=8 固定，
288 专家和 16 专家逐字节相同。已实测证实：`GroupedMatmul` gmm1 中位 **107.3 µs**，
正好是 8 个专家 128 MiB ÷ 1.25 TB/s = 107.4 µs，**1.00× 带宽地板**。
**这同时证明了 gmm 只读被选中的 8 个，不是全部。**

推论：**HBM 侧已经没有性能可捡，offload 的代价全在「7.88 GiB/token 从哪来」。**
按 CPU 侧链路带宽反推每 token 时间，这是第一个要算的数。

---

## 2. MoE 在哪里接线

### 2.1 类的落点

```
models/glm5_next.py:107        from ...deepseek_v2 import DeepseekV2MoE as Glm5NextMoE
models/deepseek_v2.py:644      class DeepseekV2MoE          ← MoE 层本体
                    :736       self.experts = get_moe_impl_class(quant_config)(...)
                    :830       self.shared_experts = DeepseekV2MLP(...)
                    forward_normal()                        ← decode 走这条
```

**GLM-5.3 的 MoE 就是 DeepSeek 的 `DeepseekV2MoE`，别去 `glm5_next.py` 里找。**

### 2.2 权重实际住在哪 —— offload 要搬的就是这几个张量

```
layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8_moe.py
  class NPUCompressedTensorsW8A8Int8DynamicMoE
    :57   create_weights()                 ← w13_weight / w2_weight / *_weight_scale 在这里分配
    :127  process_weights_after_loading()  ← 布局改写（NZ 等）发生在这之后
    :143  apply_weights()                  ← 走 MoeRunner → AscendQuantInfo
```

**`create_weights` 是 offload 最自然的拦截点**：张量在这里被 `register_parameter`，
之后 `process_weights_after_loading` 才改布局。
⚠ **注意顺序**：任何按「加载后布局」做的搬运，都必须在 `process_weights_after_loading` 之后。

### 2.3 NPU 侧的 MoE 算子

```
hardware_backend/npu/moe/
  topk.py            ← 路由 top-k（本线改过，见 §4.3）
  init_routing.py    ← MoeInitRoutingV3
  matmul.py          ← GroupedMatmul
  activation.py      ← swiglu / clamp（**有坑，见 §4.1 §4.2**）
  finalize_routing.py
  quant.py
  fuseep.py
```

### 2.4 ⚠ 仓库里已经有 KTransformers 的接入点，别重造

```
layers/moe/kt_expert_masks.py    "GPU expert placement for the KT hybrid CPU/accelerator MoE"
layers/moe/kt_stream_prefill.py  "stream ALL routed experts per layer DDR->HBM"
layers/moe/kt_ep_wrapper.py
```

server args 也已经有：

```
--cpu-offload-gb            预留多少 GB 主机内存
--offload-group-size        每组多少层
--offload-num-in-group      组内 offload 几层
--offload-prefetch-step     预取几步
--kt-weight-path            KT 的 amx 量化专家权重目录
--kt-method                 CPU 侧量化格式
--kt-cpuinfer               CPUInfer 线程数
--kt-threadpool-count       和 NUMA 节点一一对应
```

**先读这几个文件再动手** —— 你要融合的两套方案，其中一套的骨架可能已经在了。
⚠ 但**它们在 NPU 上跑没跑过，我没验过**，别当成能用。

---

## 3. 这条线上怎么把单卡跑起来

### 3.1 起服务

```bash
source /mnt/workspace/y00359136/work/glm53_dev/env/env.sh
cd /mnt/workspace/y00359136/work/glm53_dev/env/run
DIE=0 PORT=30013 ./launch_glm_w8a8_tp1.sh          # 约 45 秒起来
```

⚠ `launch_glm_w8a8_tp1.sh:31` 有 `WT=${WT:-.../wt-int8-singlecard}`，可以用 `WT=` 覆盖指向你的 worktree；
但 **`ab_tp1.sh:75` 的路径是写死的**，换 worktree 要一起改。

**默认用的是裁剪到 16 专家的 checkpoint**（`/var/tmp/glm53/GLM-5.3-Flash-W8A8-e16`，31.5 GiB），
为了让单卡放得下。**做 offload 你多半要换回 288 专家的原版**
（`/mnt/workspace/models/GLM-5.3-Flash-W8A8`），用 `MODEL=` 覆盖。
裁剪脚本在 `tools/prune_experts_int8.py`，39 秒能再裁一份别的专家数。

### 3.2 量性能（A/B 一个变量）

```bash
NAME=myopt ./ab_tp1.sh        # 自动等 die 空闲 → 起服务 → warmup → bench
```

这个脚本里有几条**踩过坑才对的东西，别简化掉**：

- **只按端口杀进程**（`pkill -f -- "[-]-port 30013"`）。这台机器多人共用一个 OS 账号，
  全局 `pkill -f sglang.launch_server` 打断过别人跑到一半的评测
- **die 空闲判据的 `\s*` 是 load-bearing**：`npu-smi` 对忙的 die 打印 `33861/ 65536`（无空格），
  写成 `(?= / 65536)` 会**恰好漏掉忙的卡**，然后你在别人的作业上采数
- **bench 前先发 warmup**：`bench_graph_decode` 用相减法，一次冷 prefill 会整个进被减项，
  让「1.39× 加速」这种假结论活一整轮（真值 1.14×）
- **并发用 `1,3,13,16`**：1/4/16 全是图的桶值，测不到 padded replay
- **撞车 fail-fast**：起服务时检查 `Load weight begin. avail mem=`，第三方起训练时会说
  「这是撞车不是回归」

### 3.3 采 profile 和归因

```bash
cd <worktree>/docs/docs/glm53_npu_support
python tools/profile_server_decode.py --port 30013 --out /var/tmp/glm53/prof/xxx --steps 20 --concurrency 1
python tools/attribute_kernels.py --profile /var/tmp/glm53/prof/xxx --steps 20
python tools/kernel_roofline.py   --profile /var/tmp/glm53/prof/xxx --steps 20
```

⚠ profile 写 `/var/tmp/glm53/`（在 `/` 盘）。**`/mnt/workspace` 是 99% 满的，别往那写。**

---

## 4. MoE 这块的坑（都是实测踩过的）

### 4.1 ⚠ `custom::npu_dequant_swiglu_clamp_quant` 的 `clamp_limit` 被静默忽略

实测：`swiglu_mode` 取 **0/1/2/3 结果完全一样**，而且与「**完全不 clamp**」逐位相同
（`max|Δint8| = 0`）。仓库 docstring 写着「mode 1 下精确复现参考实现」——**那句话是错的**。

**要验这个算子必须放大输入让 clamp 真的咬到**，否则「不 clamp」和「clamp」是同一个结果，
你会验出一个假的通过。（REPORT §7.2）

### 4.2 ⚠ 融合 ≠ 更快，这个形态在 MoE 上出现了三次

| 试的 | 结果 |
|---|---|
| `npu_clipped_swiglu` 融合 shared expert 的 clamp+swiglu | **逐位相同、少 45 个 kernel、净慢 65.6 µs** |
| `npu_dequant_swiglu_clamp_quant` 省掉 routed 预 clamp | 它根本不 clamp，见 §4.1 |
| 追加列写成 `F.pad` | 同样 168 次 launch（降解成 `MemSet`+`PadV3`）、**3.3× device 时间** |

**「减少 kernel 个数」是代理不是目标。** 杠杆是减少**总固定成本**；当换进来的 kernel
单次做的活不等价时，代理就失效。（REPORT §7.7 / §7.2 / §6.7）

### 4.3 shared expert 已经融进 GroupedMatmul 了（本线改的）

`glm5_next.py` 原来在非 CUDA 上直接拒绝，那**从来不是测量结论**。现在 NPU 放行了：

```
commit 3f7db2fece
  glm5_next.py                       门放开 NPU
  hardware_backend/npu/moe/topk.py   _append_fused_shared_slot（NPU 缺的是 top-k 那一步）
```

**shared expert 现在是 experts 数组的第 N 号槽**（N = `n_routed_experts`），
不再是独立的两个 `QuantBatchMatmulV3`。**做 offload 时它跟着 routed 专家一起走 gmm，
别按「独立的 shared expert」去设计搬运。**

⚠ **上游有一条判据会误伤**：`deepseek_v2.py:3171` 的 allow-list
`n_routed_experts in (256, 384)`，注释警告 compressed-tensors 若把 shared expert **loose** 存着
会被「静默装错」。**GLM 确实是 loose 存的**（288 专家原版和裁剪版都是，读 weight_map 实测），
**但实测没装错** —— gate/up 的合并就是 routed 专家本来走的 `w1`/`w3` 分片加载。
**那条判据用「专家数」当代理，对 GLM 是误报。** 如果你要补护栏，正确的判据是
**判 checkpoint 里 shared expert 是不是 pre-fused**（从 weight_map 直接读得出来），不是判专家数。

### 4.4 ⚠ 16 专家构型下，哪些 MoE 结论不能外推

本线的性能数字大多在裁剪到 **16 专家**的 checkpoint 上采的：

- ❌ **专家负载分布**：top-8 of 16 和 top-8 of 288 是完全不同的分布
- ❌ **`group_list` 形状分布**：group_list 长度就是专家数，gmm 性能对它敏感。
  **bs>1 的 MoE 数字全带这个偏差** —— 16 专家时 token 很快填满每个专家，288 专家时每组小得多
- ✅ **bs=1 的权重流量可以外推**：top-k=8 与专家总数无关（已由 1.00× 地板实测证实）
- ❌ **精度**：强制路由到 16 个专家，输出本来就是坏的

**offload 的工作点是小 batch，正好落在能外推的那一半** —— 但你一旦量 bs>1，就得换回 288 专家。

### 4.5 MoE 路由簿记：1.88 ms/step，但真正可动的只有约 108 µs

拆开之后：`MoeInitRoutingV3` 594 + `DequantSwigluQuant` 255 + `MoeFinalizeRoutingV2` 247 +
`MoeGatingTopK` 188 + `Cast` 59 = **1343 µs（62%）是本质的厂商算子**；
routed 预 clamp 278 **没有融合路径**（§4.1）；shared clamp+swiglu 282 **融了更慢**（§4.2）；
`Add`+`Muls` 160 **低于噪声**。

**「X 占了 N 毫秒」是观测，「X 里有 M 毫秒不必发生」才是待办。** 别把 1.88 ms 当成目标。

### 4.6 ⚠ `tensor.is_cuda` 在这台机器上返回 `True`

`torch_npu.contrib.transfer_to_npu` 补了属性本身，**连 import 之前建的张量也变**。
`device.type` 和 `sglang.is_cuda()` 是对的。

**`srt/` 下有约 88 处张量级 `.is_cuda` 门**，每一处背后的 CUDA 快路径在 NPU 上都是
**静默启用**的。MoE 路径上已知一处：`layers/quantization/expert_pack.py:36` 的
`_clamped_swiglu`，`if gate.is_cuda:` 会让 NPU 滑进 CUDA 的 `silu_and_mul_clamp`。
**本构型（compressed-tensors）走不到它，但你换量化格式就会。**

**新写代码不要用 `tensor.is_cuda` 判平台。** （REPORT §7b.17）

### 4.7 两条会咬人的既有缺陷

- **`causal_conv1d_fn_npu` 在一批内混合 `has_initial_state` 时写坏 conv state**（KDA 那侧，
  不是 MoE，但会污染你的精度评测）。**PLAN P6.2 记了禁令：修好前不要打开 radix cache 跑精度评测。**
- **`{"return_logprob":true, "top_logprobs_num":N}` 会打死整个服务**
  （触发未编译 Triton kernel 的 JIT，`bishengir-compile` SIGSEGV，所有 rank 全死）。
  **教师强制对拍只用 `input_ids` + `return_logprob`，别加 `top_logprobs_num`。**

---

## 5. 精度怎么验（MoE 相关的部分）

⚠ **这台机器上「输出逐位相同」只在 batch 宽度 1 下是合法判据。**
实测：同一进程、同一份代码、同样的提示、**同样的宽度**，重复跑结果都可能不同 ——
不确定的是调度器的**组批**。**batch > 1 上不存在可复现基线。**（REPORT §7b.15 / §7b.16）

**所以判据要按这个顺序选**：

1. **静态张量的 `torch.equal`** —— 权重装载类的改动（offload 搬运正是这类）用这个，
   **和 batch 无关，最硬**。示范见 REPORT §6.7 的权重检查：六个张量含**三个 `weight_scale`**，
   带 slot0 正对照和 slot16≠slot0 负对照
2. **有解析答案的性质** —— 比如扫一个权重看极小值是否落在推导值上
3. **batch=1 的逐 token 对拍**
4. **真 checkpoint 上的 GSM8K**（TP8，另一条线做过，`tools/run_gsm8k.py`）——
   ⚠ 它排除的是「明显掉精度」，**排除不了小于约 1.3pp 的真实退化**

⚠ **`weight_scale` 必须和权重一起对。** W8A8 per-channel，scale 装错了权重对了也没用，
而且这种错只表现成「精度差一点」。本项目有一条既有教训是同形态的：
`check_kda` 6/6 全绿 + logprob 对拍 0.000e+00，而算子正在写坏一块**当时没人读**的内存。
**验收只覆盖你去读的东西。**

---

## 6. 机器共用的硬规矩

- **绝对不要全局 `pkill -f "sglang.launch_server"`**。多人共用一个 OS 账号，只按端口杀
- **用卡前 `npu-smi info`，并跟同机的其他 session 打招呼**
- **磁盘 `/mnt/workspace` 99% 满**。临时文件、profile、venv 都写 `/var/tmp/glm53/`
- ⚠ **CANN 崩溃时会往 cwd 落几 GB 的 `extra-info/data-dump/`**（权限 `-r--------`，容易漏看），
  正常跑也会落 `fusion_result.json`。定期 `du -sh extra-info` 看一眼

---

## 7. 起点

分支 `glm53_dev`。单卡的跑法钉在 `wt-int8-singlecard` 这个 worktree 上
（`env/run/ab_tp1.sh:75` 写死了路径），**建议直接在它上面开新分支，别新建 worktree**：

```bash
cd /mnt/workspace/y00359136/work/glm53_dev/wt-int8-singlecard
git checkout -b moe_offload glm53_dev
```
