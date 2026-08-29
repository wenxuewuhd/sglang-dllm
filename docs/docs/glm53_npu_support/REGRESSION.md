# 回归阶梯：改完跑什么

按代价从低到高。**日常改动跑 1+2；动了 kpool 或注意力跑到 3+4；上线前才 5+6。**

每一级都有它抓不到的东西，**下一级不是"更全面的同一件事"，是补上一级的洞**。

| # | 用例 | 代价 | 覆盖 | ⚠ 抓不到 |
|---|---|---|---|---|
| **1** | `tools/logit_check.py compare` | **秒** | 45 层 prefill 的数值，无采样噪声，第一个偏离的 token 指向具体层 | **稀疏选择**（提示都 < `index_topk=2048`，走 `skip_logits_computation`）；**decode 路径**；批处理 |
| **2** | 同一 prompt 生成 100+ token，比对 token 序列 | 十几秒 | **decode 路径**：KDA 递归更新、kpool decode 缓存、稀疏注意力的 decode 分支 | 长上下文；并发 |
| **3** | `layer_check/check_*.py` | 分钟 | 单层，**含 32k 的稀疏选择**，对 CPU fp32 双参考法 | **层与层的交接**、整网误差累积、调度 |
| **4** | 长 prompt 端到端，`logit_check.py --ref-source server` | 分钟 | **整网 + 长上下文**，即 1 和 3 之间那个洞 | 需要一个已知正确的服务当参考 |
| **5** | GSM8K 子集（~200 题） | 小时 | 真实任务、口径是否正确 | 统计功效不足以判小幅回退 |
| **6** | GSM8K 全量 1319 题 | 数小时 | **出口判据 97.50%** | — |

## 为什么第 4 级要拿服务当参考

整网 + 长上下文的 CPU 参考**做不出来**：DSA 注意力在 CPU 上是 O(S²)，实测 512 token 跑了 75 分钟被杀。

所以用 `logit_check.py --ref-source server`：**拿一个已知正确的服务去对另一个**。
这也是 **graph vs eager** 要用的机制 —— 先用 eager 服务录参考，开 graph 后再对。

⚠ 判据在这里可以定得很硬：单层实测 **graph replay 与 eager 逐位相同**，
所以整网也该逐位相同。**不等就是有东西被烘进图里**，不是「精度差一点」。

**2026-08-29 实测：整网确实逐位相同 —— 但要在同一个 batch 宽度下比。**
`raw_bs=1` 对 `raw_bs=1`：短提示 prefill、1000 个 decode token、
两条 3255/3252 token 长提示 + 200 decode token，`max|dlp|` 全部 `0.000e+00`。
**换了 batch 宽度就不再逐位相同**（mean|dlp| ~2.1e-2），而且**与 padding 无关** ——
`bs=8`/`bs=16` 一行 padding 都没有，误差和有 3 行 padding 的 `bs=13` 一样大。
那是上面那个**形状地板**，不是图的问题。所以：**对拍要固定并发度**，
`logit_check.py` 是一条一条打的（`raw_bs=1`），正合适。

## 几条容易踩的

- **短 prompt 测不到 kpool。** `seq_len < 2048` 时 indexer 直接全选。
  「Paris 答对了」只证明 45 层能串起来，**不证明我们做的索引缓存和稀疏选择对**
- **单层全绿不等于整网对。** 整网拉通时修的三个 bug 里有两个是
  「顶层对象是包装，新方法加在了被包的那个上」—— 单层 harness 直接构造内层，
  **结构上就发现不了这一类**
- **不要用耗时判对错**，也不要拿 eager 的耗时数字下性能结论

---

## 第 1 级的地板：已测出来了（2026-08-29）

之前卡在「有数字但没有基准」。现在有了。

### 怎么测的

`tools/logit_check.py` 的 `--streaming`：fp32 整模型 `from_pretrained` 要 1.2 TB
（实测加载到 26% 就开始换页），所以复用 `layer_check/trace_reference.py` 的
**逐层物化**模型，后面接上 `lm_head`。峰值 RSS 几十 GB。

⚠ **8 条提示必须一次前向跑完**，不能一条一条来：物化一层的代价是把 599 GB
checkpoint 读一遍加转一次 dtype（实测 bf16 205 MB/s、fp32 77 MB/s），
一条一条跑就是每个 dtype 8 遍 checkpoint，fp32 那边约 **17 小时**。
批一次是 **470 秒**。右填充在这个模型上是**精确**的而不是近似的 ——
每条路径要么因果（注意力掩码、KDA 递归、深度卷积），要么逐 token
（MoE 路由、mHC 的 sinkhorn 只在 hc_mult 四路之间做），t 之后的 token 到不了 t。

```bash
$ROOT/.venv-ref/bin/python tools/logit_check.py reference --streaming --dtype fp32 --out ref32.json
$ROOT/.venv-ref/bin/python tools/logit_check.py reference --streaming --dtype bf16 --out ref16.json
python tools/logit_check.py compare --ref ref32.json --against ref16.json --emit-floor floor.json
python tools/logit_check.py compare --ref ref32.json --port 30003 --floor floor.json
```

### 三个数量级不同的「地板」，别混

| 地板 | 是什么 | 实测 mean\|dlp\| |
|---|---|---|
| **精度地板** | fp32 与 bf16 跑同一件事 | **9.6e-3 ~ 2.85e-1**（逐提示） |
| **形状地板** | 同为 bf16、同样的数学，只是 GEMM 形状不同（批 vs 不批） | 0 ~ 2.6e-2，8 条里 **3 条逐位相同** |
| 判据 | 候选 ≤ 精度地板 × SLACK(2.0)，逐提示 | — |

**形状地板是这轮新测到的。** 它说明「同样的数学、不同的规约形状」本身就能把 logprob
挪动 1e-2 量级 —— 因为 bf16 的 ulp 级扰动会翻 MoE 路由。它比精度地板小一个量级，
所以「参考是批出来的、服务是不批的」这件事**不主导判定**；但它是 GPU/NPU 之间、
甚至同一台机器不同 batch 之间不可能逐位相同的原因。

### 判定结果：eager 基线**通过**

TP16 / eager / 45 层，对 fp32 CPU 参考：

```
prompt   tokens   max|dlp|  mean|dlp|      dNLL     floor  x floor
#0           12  1.371e+00  2.703e-01 -2.411e-01 1.519e-01   0.89x
#1           20  6.744e-02  1.018e-02 +3.298e-03 9.607e-03   0.53x
#2           16  7.313e-02  2.382e-02 -4.866e-03 4.809e-02   0.25x
#3           22  1.812e+00  1.855e-01 +7.040e-02 1.015e-01   0.91x
#4           24  2.213e-01  3.482e-02 +1.404e-02 5.096e-02   0.34x
#5           23  1.558e-01  3.359e-02 +1.877e-02 2.367e-01   0.07x
#6           14  2.500e+00  4.111e-01 -4.416e-02 2.849e-01   0.72x
#7           11  2.729e-01  5.957e-02 +1.013e-02 5.502e-02   0.54x
-> 8/8 在测出来的地板 x slack 2.0 之内
```

**原先那些「既不能说通过也不能说失败」的数字（0.013–0.25）从来就不可疑** ——
地板本身就是同一量级（0.0096–0.285）。缺的一直只是地板。

⚠ **判定用 mean 不用 max。** 两边都会在某些 token 上翻专家，翻一次就把那个 token 的
logprob 挪很远，max 报的是「谁碰巧翻了最差的那个 token」，是噪声。地板的 max 到 3.02。

## eager 基线已录，开 graph 前不要丢

在 `$ROOT/goldens/logits/`。**graph 起来之后 eager 服务就没了**，这些是唯一的对照：

| 文件 | 是什么 | 覆盖到第几级 |
|---|---|---|
| `ref_cpu_fp32.json` / `ref_cpu_bf16_batched.json` | CPU 双参考（批） | 地板 |
| `ref_cpu_bf16_unpadded.json` | CPU bf16，一条一条跑 | 形状地板 |
| `floor_precision.json` / `floor_shape.json` | 上面两个地板 | — |
| `ref_server_eager_short.json` | eager 服务，8 条短提示 | 1 |
| `ref_server_eager_short_d100.json` | 同上 + 每条贪心 100 token | 1 + **2** |
| `ref_server_eager_long_d100.json` | **3256 / 3253 token** 提示 + 100 token | **4**（稀疏选择真的走了） |

长提示那两条 eager 下答对了段落里全部六个事实（11 / 42 / 8 / 34 / 3 / 288），
**这是第一次有「稀疏选择生效」的端到端正确性证据** —— 不是「Paris 答对了」。

## 一个会静默吃掉请求的坑

这台机器 `HTTP_PROXY=http://127.0.0.1:1056`，`requests` 连 `127.0.0.1:30003`
也会走它，代理回 **503**。`env.sh` 里 unset 了，但直接跑工具时不会经过它 ——
`logit_check.py` 已经自己带 `proxies={"http": None}`。别的脚本要连本机服务的，
**先 unset**。
