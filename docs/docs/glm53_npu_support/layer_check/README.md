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

用 `operator_handoff/ACCEPTANCE.md` 的**双参考法**，实现在 `reference/tolerance.py`，
`harness.py` 直接复用它，**不要另写一份**：

> fp32 参考与 bf16 参考之间的距离，**就是**这个用例的预算。

固定阈值对这个模型是错的——KDA layer-0 在 seq=64 下 fp32-vs-bf16 的相对误差就有
**1.06e-2**，定 1e-3 会把一个逐位正确的实现判失败。

## 跑

```bash
source $ROOT/env.sh

# 阶段 A（CPU）。golden 很大，存在仓库外
$ROOT/.venv-ref/bin/python dump_reference.py --layer 3 --out $ROOT/goldens/layer03.pt

# 阶段 B（NPU）
PYTHONPATH=$REPO/python $VENV/bin/python check_dsa.py --case $ROOT/goldens/layer03.pt
```

**golden 不进仓库**（一个 32k 的用例约 776 MB）。放 `$ROOT/goldens/`，用脚本重新生成。

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
| KDA 线性注意力 | 34 层 | ⚠ 只有**模块级** golden（`../tools/golden_kda.py`），**没走 Ascend backend 的真实路径** |
| mHC | 每层 | ⚠ 同上，模块级验过（`../tools/golden_mhc.py`） |
| MoE（288 专家） | 42 层 | ❌ **未验**，且 PLAN §4 记着**两个已知未修的精度缺陷** |
| Dense FFN | 前 3 层 | ❌ 未验 |
| 整网逐层 trace | — | 框架就位，stage A/B 未写 |

⚠ **不要写逐位一致性断言。** NPU 的 bf16 矩阵乘不是 batch-shape 不变的：同一输入把 M
从 4096 改成 4080，5/4080 行会差 1 个 bf16 ulp（实测，见 PLAN §2.4）。
