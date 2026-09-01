# 逐层 NPU-vs-CPU 对拍

**这里的东西是为了「以后整网出问题能回来定位」而落盘的**，不是一次性脚本。
整网输出不对的时候，你需要的第一个答案是「**哪一层开始偏的**」，而不是「模型是错的」。

## 两个阶段

| 阶段 | 环境 | 脚本 | 做什么 |
|---|---|---|---|
| A | CPU，`$ROOT/.venv-ref`（transformers 5.16.1） | `dump_reference.py` | 用 HF 参考实现跑到待测层，存下该层的**输入**，以及**同一套输入分别用 fp32 和 bf16 跑出的两份输出** |
| B | NPU，`$ROOT/.venv-glm53` | `check_<模块>.py` | 把同一份输入喂给**真实的 sglang 模块**，比对 |

两个 venv 不能共存（sglang 钉 transformers 5.12.1，而 5.12.1 不认识 `glm5_next`），
所以必须分两个进程、用文件传递。`harness.py` 只依赖 torch 和标准库，两边都能 import。

## 判据

用 `layer_check/ACCEPTANCE.md` 的**双参考法**，实现在 `tolerance.py`，
`harness.py` 直接复用它，**不要另写一份**：

> fp32 参考与 bf16 参考之间的距离，**就是**这个用例的预算。

固定阈值对这个模型是错的——KDA layer-0 在 seq=64 下 fp32-vs-bf16 的相对误差就有
**1.06e-2**，定 1e-3 会把一个逐位正确的实现判失败。

## 跑

```bash
source $ROOT/env.sh

# 阶段 A（CPU）。golden 很大，存在仓库外。
# --ref-prefill-chunk 在 8k 以上是必须的：torch 的 CPU depthwise conv1d
# JIT 不了那么长的窗口（illegal immediate parameter）。
$ROOT/.venv-ref/bin/python dump_reference.py --module kda --layer 0 \
    --prefill 32768 --decode 8 --ref-prefill-chunk 8192 \
    --out $ROOT/goldens/kda_layer00_s32k.pt

# 阶段 B（NPU）。PYTHONPATH 必须**追加**，不能覆盖 —— CANN 的 tbe 在里面，
# 冲掉它 F.conv1d 会以 "No module named 'tbe'" 失败。
PYTHONPATH=$REPO/python:$PYTHONPATH $VENV/bin/python check_kda.py \
    --case $ROOT/goldens/kda_layer00_s32k.pt --ranks all --batch 16 --prefill-chunk 8192

# 同一个脚本量本层稳态耗时（形状照部署，不打分）
PYTHONPATH=$REPO/python:$PYTHONPATH $VENV/bin/python check_kda.py \
    --case $ROOT/goldens/kda_layer00.pt --bench
```

**形状要照部署来。** `$ROOT/run/launch_glm_bf16.sh` 是 `--tp-size 16 --page-size 64
--context-length 32768 --max-running-requests 16`，所以 KDA **每卡只有 4 个头**，
不是 64。`check_kda.py` 在一张卡上顺序跑 16 个 rank 再把 `o_proj` 的部分和加起来：
每卡形状是真的，同时整层还能跟未切分的 CPU 参考对上。玩具形状测不出东西 ——
`kda.py:214` 的 autotune 挂死就是短序列也照样触发、但只有真跑真实路径才看得见。

**golden 不进仓库**（KDA 的 32k 用例 1.6 GB）。放 `$ROOT/goldens/`，用脚本重新生成。

### KDA 这一轮抓到的三个真 bug（都在 `ascend_kda_backend.py`，已修）

对拍不是走过场：这三个都是「跑得通但算错 / 直接崩」，模块级 golden 一个也看不见。

1. **prefill 的 beta 没过 sigmoid。** GLM 把 gate 摊平（`[B,T,H*K]`）交出去，同时把
   beta 交原始值 —— CUDA 路径正是用「gate 是不是摊平的」当 `beta_is_raw`，在
   `chunk_kda` 里补 sigmoid。Ascend 的 extend 链没这个钩子。修之前 out.prefill 差
   **70.55×** 预算、ssm 状态差 **943.61×**；修完 0.31× / 0.22×。
2. **decode 的 conv state 写回 dtype 不对。** GLM 的 conv 权重是 fp32、conv cache 是
   bf16，昇腾 update 算子按权重 dtype 写回，`aclnnIndexPut` 直接报
   `expected DT_FLOAT but found DT_BFLOAT16`。**第一次 decode 就崩**。extend 路径早就
   有这个 fp32 工作集的绕法，decode 路径漏了。
3. **`kda.py:214` 的 autotune 会挂死 AI core。** 真实路径必经（`chunk_kda_scaled_dot_kkt_fwd`
   就在 `_AscendKDAExtendKernel.extend` 里），24 个 config 的 `do_bench` 扫描在**第一个**
   config 上就 `aicore timeout`（507014）。单钉一个 config 时 triton 根本不 bench，就没事。

## 整网 tracing

`harness.py` 的 `save_trace` / `first_divergence` 是给这个用的：阶段 A 存下**每一层**
的 hidden state（fp32 与 bf16 两份），NPU 跑同一条 prompt 并抓每层输出，
`first_divergence` 逐层报误差与该层自己的地板，指出**第一个越界的层**。

逐层都打印，不只打印第一个失败——**缓慢漂移和某一层突变，只看第一个失败是分不清的**。

tracing 用短 prompt（128 token 量级）就够，不要用 32k。

## 当前覆盖

| 模块 | 层 | 状态 |
|---|---|---|
| DSA indexer（kpool） | 11 层（3,7,…,43） | ✅ **端到端已验**，见 `../tools/check_kpool_indexer_e2e_npu.py`。32k 下 pool overlap 0.99616，正好落在地板上 |
| DSA 整层（含稀疏注意力） | 同上 | 进行中 |
| KDA 线性注意力 | 34 层 | ✅ **端到端已验**，`dump_reference.py --module kda` + `check_kda.py`。真实部署形状（tp16 → 每卡 4 头、bs=16 ragged、prefill 32768 分 8192 一块）下 6/6 在预算内，最差 0.31× |
| mHC（pre + post） | 每层 2 个站点 × 45 层；已验 1/5/20/40 层 | ✅ **端到端已验**，见 `check_mhc.py`。走真实 `hc_pre`/`hc_post` → `npu_hc_pre`/`npu_hc_post`，**NPU 分支用调用计数证明走到了，不是假设**。97/97 用例、485 个张量全在预算内，最差 0.50×。部署形状 M=1…16384 全覆盖（TP 不切 mHC，源码确认）。sinkhorn 20 轮误差**不累积而是收缩**（1.9e-6 → 1.6e-7），迭代数经辨识确为 20（次近迭代远 267–16892 倍）|
| MoE（288 专家） | 42 层 | ❌ **未验**，且 PLAN §4 记着**两个已知未修的精度缺陷** |
| Dense FFN | 前 3 层 | ✅ **端到端已验**（layer 2，真实 TP16 形状：每卡 gate_up `[M,1536]`、down `[M,768]`；M=1/16/8192），最差 0.66× 预算。反向对照 `KT_DISABLE_SWIGLU_CLAMP=1` 时 `act` 失败 52–95×，说明检查有牙。⚠ **真实输入下 clamp 从不触发**（max\|gate_up\|=2.17 vs limit 10），要验 clamp 必须 `--scale-input 48` |
| 整网逐层 trace | 45 层 | ⚠ **stage A 已跑通**（`trace_reference.py`，CPU 流式建模，128 token 3.5 min / 峰值 RSS 646 GB / 输出 755 MB）。stage B 只有**格式与 hook 定义**，真实抓取等 P4.1。**只在浅层有牙**，见下一行 |
| ⚠ 整网 fp32-vs-bf16 的**路由翻转** | layer 3 起 | 📌 **实测**：MoE 的 top-8 集合在两种精度下会不同 —— layer 3 就有 12.5% 的 token，最深处 **63.3%**。**从第一个 MoE 层起，噪声地板就由离散的路由差异主导，而不是舍入**。地板从 layer 0 的 9.5e-3 涨到 layer 26+ 的 **1.8e-1**。后果：注入 5% 误差实测**只有 layer 7–25 能被检出**，layer 26 起 err/floor≈1.03–1.09 落在预算内、测不出来。**深层的宽地板不能当成「宽误差可接受」的依据** |
| 计时口径 | — | `timing.py`：冷调用单独报 / 稳态 p50 / Event 分段 / **host 同步计数** / 强制免责声明。dense FFN 实测冷:稳 = **971–1022×**（M=1/16），**每次调用 0 次 host 同步** |

⚠ **不要写逐位一致性断言。** NPU 的 bf16 矩阵乘不是 batch-shape 不变的：同一输入把 M
从 4096 改成 4080，5/4080 行会差 1 个 bf16 ulp（实测，见 PLAN §2.4）。
