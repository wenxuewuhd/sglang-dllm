# 接手指南（2026-08-30，P4 与 P5 都已闭环）

新 session 从这里开始，然后读 `PLAN.md`。**上一份交接的内容已经全部消化，本文件是新的。**
第一次接触这个项目、想从零把环境和精度跑通的，看 [`REPRODUCE.md`](./REPRODUCE.md)。

---

## 一句话现状

**BF16 和 INT8 两条线都闭环了，NPU Graph 也开着。算子开发需求 0 项。**

| | |
|---|---|
| 整网 | TP16 / 45 层 / 真实 HCCL / graph |
| 精度（BF16） | GSM8K 两轮 97.04 / 97.35%，判据 97.50%，在噪声内 |
| **精度（INT8 W8A8）** | 两轮 97.80 / 97.19%，**对 BF16 +0.30pp**，判据「1% 以内」**通过** |
| graph vs eager | 同 batch 宽度下**逐位相同**；decode 约 **8×** |
| overlap scheduler | 开着，实测 **1.23×**，数值不变 |
| chunked prefill | 单请求 / 并发 / 2 chunk / 3 chunk 全绿，**逐位相同** |

**下一步是 P6 性能**：图下的排序要重做，主线是让 prefill 也进图。

## 机器现在的分工（2026-08-30 起）

另一个 session **`glm53_int8_1card`** 在同一台机器上做 INT8 单卡性能分析
（TP1，固定用 **die 0**，worktree `../wt-int8-singlecard`，分支 `int8_singlecard`）。

- **我们这条线的单卡活用 die 14/15**（`layer_check/`、`probe/` 一直是
  `ASCEND_RT_VISIBLE_DEVICES=14`），和它不冲突
- **但 TP16 要全部 16 个 die**，所以起整网前必须先跟它说，它停服 + 等释放约 3–5 分钟
- 用 `SendMessage` 直接发给 `glm53_int8_1card`

⚠ **消息不是可靠的互斥，`npu-smi` 才是真相** —— 这台机器上还有别的用户。
今天出过两次：放卡后不到一小时被第三方占走；以及在 `npu-smi` 看着空的情况下起服务，
**别人在加载权重那一两分钟里插进来**，于是在 MoE 的 `create_weights` 处 OOM
（看起来像显存不够，其实是被插队）。**起服务前和开始加载前各看一眼。**

## 精度是怎么判的（方法比数字重要）

**判据是测出来的地板，不是拍的阈值。** 这个模型的地板是**离散的 MoE 路由差异**，
随深度从 1.2e-2 涨到 1.8e-1；工具里曾经写死过 `<1e-2`，比实际低一个数量级还多。
现在 `logit_check.py` **不给默认阈值**，判定必须显式 `--floor` 传进来。

三个量级不同的地板，别混（全部实测，方法见 `REGRESSION.md`）：

| 地板 | 是什么 | mean\|dlp\| |
|---|---|---|
| 精度地板 | fp32 与 bf16 跑同一件事 | 9.6e-3 ~ 2.85e-1 |
| 形状地板 | 同为 bf16、同样的数学，只是 GEMM 形状不同 | 0 ~ 2.6e-2 |
| 判据 | 候选 ≤ 精度地板 × SLACK(2.0)，逐提示 | — |

**所有基线都在 `$ROOT/goldens/`**：`logits/` 是 CPU 双参考 + 两个地板 + eager 服务基线
（覆盖回归阶梯 1、2、4 级），`gsm8k/` 是 BF16 和 INT8 各两轮的全量结果（含响应原文）。
**开 graph 前专门录 eager 基线是刻意的** —— graph 一起来 eager 服务就没了。

## graph 这一轮验了什么（细节在 `PLAN.md` P6.6b / P6.6c）

- 45 层整网捕获：6 个 bs 桶 `[1,2,4,8,12,16]`，**15 秒捕完、图池只花 0.8 GB**。
  显存完全不是问题（原先担心的那条落空），KV pool 与 eager 一模一样
- 16 卡 HCCL 全在图里重放（此前只有 2 卡实测）
- **同一 batch 宽度下 graph 与 eager 逐位相同**：短提示 prefill、1000 个 decode token、
  两条 3255/3252 token 的长提示 + 200 decode token，`max|dlp|` 全是 `0.000e+00`
- **换 batch 宽度就不再逐位相同了，但这不是 padding 的锅**：`bs=8` 和 `bs=16`
  **一行 padding 都没有**，误差却和有 3 行 padding 的 `bs=13` 一样大（mean|dlp| ~2.1e-2）。
  变量是 batch 宽度，量级正好等于独立测出来的**形状地板**。P6.6a 修的那条站得住

## 图模式的性能基线（第一份，细节在 `PLAN.md` P6 开头）

bs=1 **27.5 ms/token**（eager 是 220–238），16 并发短上下文 **449 token/s**。
prefill 约 4000–4500 token/s。
**长上下文的 decode 代价是真的**：同样 16 并发，13 token 上下文 35.6 ms/token，
3256 token 上下文 **73.1 ms/token**，翻倍 —— 这就是 P6.7 / P6.10 / P6.11 那几条
kpool 开销该去的地方，而它们**全是 device 时间类的，图吃不掉**。

⚠ 量的时候 prefill 和 decode 要分开（先 `max_new_tokens=1` 再 129 相减），
否则 3256 token × 16 并发的 5.2 万 prefill token 会把 decode 完全盖掉。
脚本在 `tools/bench_graph_decode.py`（padding 那条是 `tools/check_graph_padding.py`）。

## GSM8K：P4 与 P5 都已闭环

**97.35%**（1284/1319）、stop rate **100.00%**、抽取失败 0 —— 判据 97.50%，差 0.32 个 SE。
run 1 是 97.04%，但那轮有 9 例是**抽取器**把 `\boxed{70\%}` 判成无答案，不是模型错，已修。
一轮 1360 秒（128 并发）；**eager 下同样的事要 11 小时以上**。
结果连同响应原文存在 `$ROOT/goldens/gsm8k/`。

**不用再跑第三轮**：GSM8K 是固定的全部 1319 题、与 cookbook 同一套，题目抽样方差抵消，
只剩解码随机性（上界 0.47pp）。
DSv4 那个跑三轮是因为 GPQA 只有 **198 题**（单轮 ±6pp），量级完全不同。

**P5（INT8）也已经用这两轮做基线判完了**：

| | run 1 | run 2 | 均值 |
|---|---|---|---|
| BF16 | 97.04% | 97.35% | 97.19% |
| **INT8 W8A8** | 97.80% | 97.19% | **97.50%** |

差 **+0.30pp**，判据「1% 以内」**通过**（每侧 2 轮，差值 SE 0.46pp）。
INT8 错的 29 题里 **23 题 BF16 也错** —— 错的是同一批难题。

## 这一轮性能上已经落地的（细节见 PLAN P6.13 / P6.14）

- **overlap scheduler 开了，1.23×，数值一步没动**（248 → 304 token/s，对 eager 仍逐位相同）。
  它此前关着只是为了让 graph-vs-eager 的对拍只有一个变量。**已改成默认配方**
- **prefill 图别去试开关**：去掉 `--disable-prefill-cuda-graph` 完全没用 ——
  NPU 上 prefill 默认后端是 `tc_piecewise`，而 `server_args.py` 的兼容性规则第一条
  「non-CUDA hardware (HIP/NPU/CPU/MPS/XPU)」把它整个否掉了。要开是**两件代码工作**：
  在声明式 registry 里给 GLM 注册 full prefill capture，**并且先清掉 extend 侧的 host 同步**
  （否则捕获期间必抛 107027）。收益上界大，但代价是真代码
- ⚠ **服务级 profiling 会把服务打挂**（16 rank 全段错误），采到的数据也是废的。
  要 profile 走 `layer_check/kernel_profile.py` 那条单模块路线
- **INT8 与 BF16 吞吐基本持平**，满批下 INT8 反而快约 6%。
  ⚠ 此前「INT8 慢 1.47×」的说法**已证伪** —— 那是拿 GSM8K 的 aggregate tok/s 当吞吐，
  而它把 128 路满批（约 2000 tok/s）和单请求长尾（约 30 tok/s）平均在一起。
  **别再用那个指标**，用 `#running-req` 对齐后的 `gen throughput` 或 `bench_graph_decode.py`。
  bs=1 时 INT8 慢 14%（量化线性每次 `npu_quant_matmul` 前要单发一次 `npu_dynamic_quant`，
  每 forward 多 140 个 kernel），bs=128 时摊薄后反超 8%。见 PLAN P6.15

## chunked prefill：已完整验完（2026-08-30）

单请求、并发、2 chunk、3 chunk 四种组合全绿。最强的一条：**19858 token 切成
8192+8192+3520，同时 8 路后台请求全程 decode，结果对无并发那次逐位相同**
（`max|dlp| = 0.000e+00`），后台请求 0 降级。细节见 PLAN P3.4。

## 下一步（已定顺序）

**主线：让 prefill 也进图。** 这是目前收益上界最大的一项 —— GSM8K 那种短输出高翻台的负载
实测 **875 个 prefill 批、每批仅 163 token**，而 prefill 全程在 eager 跑。三步里已完成一步：

1. [x] `_extend_rows` 与 `visible_pool_runs` 都已改成设备侧 + 静态形状（CPU 等价 + 上机复核都过）
2. [ ] `_kpool_compress_write_extend_npu` 仍是 host 侧循环（每请求形状不同的散写）——
       **比前两个难一档**：写的是 KV 索引缓存，改错是静默数据损坏而不是报错，
       核心部分离线验不了
3. [ ] 在声明式 registry（`arg_groups/overrides.py` 的 `_inkling_overrides` 那套）
       里给 GLM 注册 full prefill capture

**并行可做**：图下重做 P6 排序。已有的两个指向 ——
① **长上下文 64 并发就拐**（1044 → 1130 只涨 8%，而 KV 用量才 0.04、mamba 满 1.00），
指向 kpool 的 device 时间（P6.7 / P6.10 / P6.11），**图吃不掉**；
② P6.9 `TASK_QUEUE_ENABLE=2` 在 eager 下测得 1.74×，**图下可能已被吃掉，要重测**。
其中 **P6.11 有实测线索别浪费**：那 4.73 ms 全来自被 clamp 的 gather load，
去掉是 5.557 → 0.282 ms；而原设计的「三处 store」重写方案作者已判定是错的（地址区间重叠），
**那条路不要再走**。

## 还没碰的（别当成已验）

① **prefill 图没开** —— 见上面主线的第 2、3 步。⚠ 注意「extend 侧还有 host 同步」
   这个说法**是错的、已作废**：逐行审计过，`kpool_indexer_npu.py` 里
   `.item()`/`.cpu()`/`.tolist()` 一个都没有。挡住捕获的是**被烘进图的 host 侧构造**
   和**动态输出形状**，两件不同的事
② `enable_torch_compile` + `npugraph_ex`（`patch_model_npu`）那条路
③ MTP / spec decode 下的捕获
④ DeepEP-normal 的 MoE（出厂配方走 `--moe-a2a-backend none`，是已验的 TP dispatcher）

## 对外页面（别新建，要更新那一个）

算子清单的对外呈现页：https://claude.ai/code/artifact/54dbfb20-667f-465d-84c1-ea7d0cc1a827

⚠ **更新时必须把这个 URL 传给 Artifact 工具**，否则会新建一页而不是更新它。
**以仓库为准**（算子结论在 `PLAN.md` §2）。

**现在页脚和 banner 写着「端到端精度尚未跑通」，已经严重过期**：
整网跑通、eager 与 graph 的 logits 判定通过、**GSM8K 97.35%（BF16）与 97.50%（INT8）都过了**。
P4 和 P5 都闭环了。**这一页需要重写一次。**

## 欠账

`SHARED_CHANGES.md` 有 5 条已改 + 3 条待决。其中：
- **DSv4 的 GPQA 回归没跑**（swiglu_limit 那条改动欠的，公开对标值 73.23%，198 题）。
  ⚠ **本项目决定不跑**（用户 2026-08-29 拍板）。但那条改动**确实会改变 DSv4 的数值**
  （给它的 routed 专家加上本就该有的 clamp，实测修前 2.85× budget 判失败、修后 0.35×），
  所以这是一条**已知且被接受的风险，合入前需要下游确认**，不是一条会有人认领的待办。
  DSv4 权重在 P1.2 后已删（只留元数据），真要跑得重新下载约 275 GB
- **待决 ②** `seq_lens_cpu_list` 在捕获时被烘死 —— GLM 逃过，
  但**对任何走 FIA 的非 DSA 昇腾模型是活的静默 bug**
- `git stash` 里有 kpool 共享路径的半成品（tail kernel，作者已判定原方案是错的；
  但**实测那 4.73 ms 全部来自被 clamp 的 gather load，去掉就是 5.557 → 0.282 ms**）

## 六条最贵的教训

1. **短 prompt 测不到 kpool** —— `seq_len < index_topk=2048` 时 indexer 直接全选。
   「Paris 答对了」只证明 45 层能串起来。现在有 3256 token 的长提示基线了
2. **单层全绿不等于整网对** —— 整网拉通修的三个 bug 里两个是
   「顶层对象是包装，新方法加在了被包的那个上」，单层 harness 直接构造内层，**结构上发现不了**
3. **工具里拍脑袋的阈值比没有阈值更危险** —— 它让人对着错的基准下判断。
   现在阈值是**测出来的**，而且要显式 `--floor` 传进去才会给判定
4. **这台机器的 `HTTP_PROXY` 会劫持 127.0.0.1**，代理回 503。连本机服务前先 unset

5. **指标选错比没有指标更糟。** 我拿 GSM8K 的 aggregate tok/s 报过「INT8 慢 1.47×」，
   实际满批下 INT8 快 6% —— 那个数把 128 路满批和单请求长尾平均在了一起，
   谁碰上一条不收敛的生成谁就难看。**先问这个指标在测什么，再看它的值**
6. **「单请求全绿」和「单层全绿」是同一类错觉。** chunked prefill 的并发场景、
   graph 的 padding batch，都是**批里有别人时才会发生的事**，单请求/单层测试
   结构上碰不到。要问的是「这个失败模式需要什么条件才会出现」，
   而不是「我的测试通过了吗」
