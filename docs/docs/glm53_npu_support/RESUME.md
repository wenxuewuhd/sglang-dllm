# 接手指南（2026-08-29 中午，图模式已跑通并逐位对齐）

新 session 从这里开始，然后读 `PLAN.md`。**上一份交接的内容已经全部消化，本文件是新的。**

---

## 一句话现状

**GLM-5.3-Flash 在 A3 上整网跑通、精度判定通过、NPU Graph 也开起来了。**
eager 对 fp32 CPU 参考 8/8 在实测地板内（最差 0.91×）；
**graph 在同一 batch 宽度下与 eager 逐位相同**；decode **约 7.7×**（33–35 vs 4.2–4.6 token/s）。
**算子开发需求 0 项。**
下一步是 **GSM8K**（eager 下不可行，现在才够快）。

## 服务现在开着的是 graph 模式

端口 **30003**，日志 `$ROOT/run/glm_bf16_graph_1104.log`，
启动脚本 `$ROOT/run/launch_glm_bf16_graph.sh`（仓库里的 `launch_glm_bf16.sh.example`
已同步成这份配方）。要回 eager 就把那两行图参数换成 `--disable-cuda-graph`。
要停就 `pkill -f "[s]glang.launch_server"`（**注意方括号**，否则会把自己的 shell 一起打掉）。

⚠ **停服务后显存不是立刻回收的**，而且 `bootstrap.py:339` 的「每卡空闲 ≥ 90%」检查
**挡不住这件事**（2026-08-29 实测）：kill 后 3 秒就重起，那个检查放行了，
进程一路跑到加载权重才炸 —— `NPU out of memory ... 22.15 GiB already allocated;
525.39 MiB free`，看起来像显存不够，其实是上一个进程还没退干净。

**正确做法是等到 `npu-smi` 自己说话**，别看秒表：
```bash
until [ "$(npu-smi info | grep -oP '\d+(?=\s*/ 65536)' | awk '$1>6553{c++}END{print c+0}')" = 0 ]; do sleep 15; done
```
16 个 die 全部回到约 3 GB（95% 空闲）实测要 **2–3 分钟**。

## 卡是共用的

另一个用户（`l00960396`）会不定时起 16-die 的 DSv4 训练。
**GLM BF16 只能 TP16**（TP8 每卡要 74.9 GB，放不下），所以**整网必须独占整机**。
单卡的验证任务可以并行，但要先 `npu-smi info` 看一眼，并且**跑完让进程干净退出**。

---

## 第 1 级回归：地板测出来了，eager 通过

细节全在 `REGRESSION.md` 的新章节，这里只留结论：

- **地板 = fp32 与 bf16 跑同一件事**，逐提示 mean|dlp| **9.6e-3 ~ 2.85e-1**
- **eager 服务对 fp32 参考：8/8 在地板 × 2.0 之内，最差 0.91×** → **通过**
- 原先那些「说不清」的数字（0.013–0.25）**从来就不可疑**，缺的只是地板
- fp32 参考是用 `logit_check.py --streaming` 出的（逐层物化 + `lm_head`），
  **8 条提示一次前向**：一条一条跑要 8 遍 599 GB checkpoint，fp32 那边约 17 小时；
  批一次 470 秒。右填充在这个因果 + 逐 token 的模型上是精确的
- 顺带测到**第三个地板**：同为 bf16、只是 GEMM 形状不同（批 vs 不批），
  mean|dlp| 也能到 2.6e-2，8 条里 3 条逐位相同。比精度地板小一个量级

## eager 基线已经全部录下来了（开 graph 前的唯一对照）

在 `$ROOT/goldens/logits/`，清单见 `REGRESSION.md`。覆盖回归阶梯 **1、2、4 三级**：
短提示 prefill、每条 100 token 的贪心 decode、以及 **3256 token 的长提示**
（> `index_topk=2048`，**稀疏选择真的走了**，段落里六个事实全答对）。

**这一步是刻意提前做的**：graph 一起来 eager 服务就没了，而重启一次很贵。

---

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

## GSM8K：已过（P4 闭环）

**97.35%**（1284/1319）、stop rate **100.00%**、抽取失败 0 —— 判据 97.50%，差 0.32 个 SE。
run 1 是 97.04%，但那轮有 9 例是**抽取器**把 `\boxed{70\%}` 判成无答案，不是模型错，已修。
一轮 1360 秒（128 并发）；**eager 下同样的事要 11 小时以上**。
结果连同响应原文存在 `$ROOT/goldens/gsm8k/`。

**不用再跑第三轮**：GSM8K 是固定的全部 1319 题、与 cookbook 同一套，题目抽样方差抵消，
只剩解码随机性（上界 0.47pp），两轮差 0.30pp 已经一致。
DSv4 那个跑三轮是因为 GPQA 只有 **198 题**（单轮 ±6pp），量级完全不同。
**真正需要多轮的是 P5**（判据「回归到 BF16 1% 以内」）：每侧 1 轮时 1pp 只有 1.5σ，
2 轮时 2.1σ —— **上面这两轮就是 P5 的 BF16 基线，别丢**。

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

## 下一步（已定顺序）

1. **在 graph 下重做 P6 的性能排序** —— 现在所有条目都是 eager 时代量的，**排序会变**：
   host 开销类的（`TASK_QUEUE_ENABLE=2`、减少 aten dispatch）已经被图吃掉，
   device 时间类的（AI_CPU 回退、int64 算术、标量瓶颈 kernel）才继续值钱。
   **已有的两个指向**：① 长上下文 64 并发就拐（见性能基线）；
   ② GSM8K 这种短输出高翻台的负载实测只跑到 206–234 token/s，
   而同并发的纯 decode benchmark 是 1680 —— 差在**每答完一题就要插一次 prefill，
   而 prefill 没有捕获图**（`--disable-prefill-cuda-graph`），overlap 也还关着
2. **P5 W8A8 量化**（磁盘只剩 23 GB，必须先删 FP8 源）

## 还没碰的（别当成已验）

① prefill 图没开（`--disable-prefill-cuda-graph`）—— extend 侧的 host 同步还在
② overlap scheduler 仍然关着
③ `enable_torch_compile` + `npugraph_ex`（`patch_model_npu`）那条路
④ MTP / spec decode 下的捕获

## 对外页面（别新建，要更新那一个）

算子清单的对外呈现页：https://claude.ai/code/artifact/54dbfb20-667f-465d-84c1-ea7d0cc1a827

⚠ **更新时必须把这个 URL 传给 Artifact 工具**，否则会新建一页而不是更新它。
内容来自 `operator_handoff/`，**以仓库为准**。

**现在有一句已经过期**：页脚和 banner 写着「端到端精度尚未跑通」——
整网 2026-08-29 已跑通，**eager 基线的 logits 判定也已通过**，只剩 GSM8K 未跑。

## 欠账

`SHARED_CHANGES.md` 有 5 条已改 + 3 条待决。其中：
- **DSv4 的 GPQA 回归没跑**（swiglu_limit 那条改动欠的，基线 73.23–73.74%）
- **待决 ②** `seq_lens_cpu_list` 在捕获时被烘死 —— GLM 逃过，
  但**对任何走 FIA 的非 DSA 昇腾模型是活的静默 bug**
- `git stash` 里有 kpool 共享路径的半成品（tail kernel，作者已判定原方案是错的；
  但**实测那 4.73 ms 全部来自被 clamp 的 gather load，去掉就是 5.557 → 0.282 ms**）

## 四条最贵的教训

1. **短 prompt 测不到 kpool** —— `seq_len < index_topk=2048` 时 indexer 直接全选。
   「Paris 答对了」只证明 45 层能串起来。现在有 3256 token 的长提示基线了
2. **单层全绿不等于整网对** —— 整网拉通修的三个 bug 里两个是
   「顶层对象是包装，新方法加在了被包的那个上」，单层 harness 直接构造内层，**结构上发现不了**
3. **工具里拍脑袋的阈值比没有阈值更危险** —— 它让人对着错的基准下判断。
   现在阈值是**测出来的**，而且要显式 `--floor` 传进去才会给判定
4. **这台机器的 `HTTP_PROXY` 会劫持 127.0.0.1**，代理回 503。连本机服务前先 unset
