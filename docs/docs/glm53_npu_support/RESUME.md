# 接手指南（2026-08-30）

**本线已按用户指示暂停。** 新 session 从这里开始，然后读 `PLAN.md`。
第一次接触这个项目、想从零跑通的，看 [`REPRODUCE.md`](./REPRODUCE.md)。

**状态只写在这一份文件里。** `README.md` 曾经也有一张状态表，于是有两张、其中一张是旧的
—— 一个事实写两遍，迟早只更新一遍。

---

## 一句话现状

**BF16 和 INT8 两条线都闭环，NPU Graph 开着，算子开发需求 0 项。**

| | |
|---|---|
| 整网 | TP16（也验过 TP8）/ 45 层 / 真实 HCCL / graph |
| 精度（BF16） | GSM8K 两轮均值 **97.19%** |
| 精度（INT8 W8A8） | 两轮均值 **97.50%**，对 BF16 **+0.30pp**，判据「1% 以内」通过 |
| graph vs eager | 同 batch 宽度下**逐位相同**；decode 约 **8×** |
| overlap scheduler | 开着，实测 **1.23×**，数值不变 |
| chunked prefill | 单请求 / 并发 / 2 chunk / 3 chunk 全绿，**逐位相同** |
| kpool 三连 | P6.7 / P6.10 / P6.11 已修，**逐位相同**，最大单项 32.6× |

**TP8 也能跑 INT8**（38.20 GB/die）。「GLM 必须独占整机」是 **BF16 的约束**
（TP8 下 74.9 GB/die），不是模型属性 —— 这条曾经被当成模型属性写进计划。

## 机器（两个 session 共用一台 A3，同一个 OS 用户）

### 你需要自己去协调的那条线

另一个 session **`glm53_int8_1card`** 在做 **INT8 单卡（TP1）性能分析**，
是「单卡 CPU MoE offload」项目的前置调研。

| | |
|---|---|
| worktree | `../wt-int8-singlecard`，分支 `int8_singlecard` |
| 端口 / die | **30013**，只用 **die 0**（外加偶尔 die 1 跑子 agent） |
| 怎么联系 | `ListAgents` 找到它，然后 `SendMessage` 直接发；它会回 |
| 进度（2026-08-30） | bs=1 每 token **42.8 → 34.0 ms（−20.5%）**，kernel 3578 → 2710 |

**起来之后先给它发一条**，说明你是谁、要用哪些 die、大概占多久。
**它要用卡时也会来找你。**这条约定是用户定的，两边都执行。

⚠ **它做的改动有一部分在共享代码里**（`glm5_next.py`、`kpool_fp8_index.py`、
KDA conv 池布局），所以**它的改动会影响你**，反之亦然。
它那条线的结论已经进了对外页面第十节和第十三节。

- ⛔ **不要用全局 pkill。** 同一个 OS 用户，`pkill -f "sglang.launch_server"` 会连对方一起杀。
  只杀自己端口的：`pkill -f -- "[-]-port 30003"`（方括号防自匹配）。
  2026-08-30 因为这个丢过一次 GSM8K 全量跑。

  **真正的风险不是被打断，是被打断得不明显**：杀在评测中间会响亮地报错，
  杀在**两次测量之间**，你拿到的是一半旧构型一半新构型的数字而毫无察觉。
- ⚠ **消息不是可靠的互斥，`npu-smi` 才是真相** —— 机器上还有第三方。
  出过两次：放卡后不到一小时被占走；以及在 `npu-smi` 看着空的情况下起服务，
  **别人在加载权重那一两分钟里插进来**，于是在 MoE `create_weights` 处 OOM
  （看起来像显存不够，其实是被插队）。**起服务前和开始加载前各看一眼。**
- 服务停掉后 HBM 要 2–3 分钟才回落。**轮询 `npu-smi` 到每个 die 约 3 GB 再起下一个**，
  不然会在权重加载处 OOM。等待循环**只看自己那几个 die**（对方的永远不会归零）。

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

⚠ **改动不是逐位相同时（比如融合了几个 GEMM），必须 teacher forcing 量，不能自由生成。**
贪心路径一旦分叉，后面比的就是**不同 token 的 logprob**，那个数字没有意义 ——
实测同一个改动两种量法差 **15 倍**（2.849e-01 vs 1.911e-02），差别全在方法上，
差点把一条好改动判死。`logit_check.py` 的 decode 侧此前**也有这个 bug**
（`max|dlp|` 算的是全长不是分叉前的前缀），2026-08-30 才修。细节见 `REGRESSION.md` 末节。

⚠ **验收只覆盖「被读的东西」，看不见「被写的东西」。** 2026-08-30 吃过一次：
`check_kda` 6/6 全绿、kpool 的 logprob 对拍 `0.000e+00`，而 `causal_conv1d_fn_npu`
正在写坏 conv state。两套信号都亮着，bug 在下面 —— 因为 `check_kda` 只比对一个槽，
而 logprob 对拍只看被读的量。**逐位相同排除不了「写坏了一块暂时没人读的内存」。**
补法见 `check_kda.py --check-mixed-state`，原理与实测见 `REGRESSION.md` 末节。

## 剩下的活（按性质分，不是按优先级）

### A. 正确性欠账 —— 7 条，共同形状是「都不报错，都在等一个开关被打开」

| 条目 | 现在为什么碰不到 |
|---|---|
| `do_cp_balance_attn` 有与已修 `forward_sparse` **完全相同**的 3-D/PA_BSND 缺陷 | 一期不开 prefill CP。**开 CP 前必修**，会撞 `561002` |
| `causal_conv1d_fn_npu` 混合 `has_initial_state` 时写坏 conv state | ⛔ **我们全部启动脚本带 `--disable-radix-cache`，所以没走到**。见 PLAN P6.2 |
| `get_kv_buffer()` 二元组语义随 pool 类而变，`forward_sparse` 只按一种解读 | 走的那条恰好是对的解读 |
| `set_kv_buffer:679` 在 `cache_v is None` 检查**之前**就 `cache_v.to(...)` | GLM 走共享实现 |
| MoE `weight_scale` 被降 bf16、linear 侧保持 fp32 | 实测叠加误差 均值 1.7e-3 / 最大 2.5e-3。**记作线索，不是 bug** |
| DSv4 bf16 fallback 读 int8 buffer 不施加 scale | `:685` 无条件强制 int8，不可达 |
| triton-ascend `_hadamard128` codegen UB 越界 / `deep_ep` 打包 bug | 已绕过，**值得上报上游** |

⚠ **conv 那条是这批里唯一「开关随时会被打开」的**：打开 prefix cache 是上生产最自然的
第一个动作。**在它修好之前，不要打开 radix cache 去跑精度评测。**

### B. 性能续做 —— ⚠ 与单卡那条线**重叠**，开第二条线做等于安排撞车

P6.3 SwiGLU clamp（只对 shared expert 成立）、P6.5 NoPE split+RMSNorm、
P6.12 DSA host 调用（图下要重测）、P6.6 剩余 host 同步、
`_kpool_compress_write_extend_npu` 的 host 循环、预热 971–1022×。

### C. 验证缺口 —— 要机器

prefill-vs-decode 的 KL 一致性；INT8 对 **int8 CPU 参考**的 logprob 地板
（现在只能借 BF16 时代的地板，超出属预期内）；
spec decode 前置（`kpool_decode_update_index_cache` 假设每请求一行）。

### D. 从来没打开过的开关 —— ⚠ 这一节最容易被漏掉

上面 A/B/C 列的都是**已知问题**。这一节是**从来没试过的东西**，
清单上没有，所以没人会去找。全部按 2026-08-30 的启动脚本与 checkpoint 核实：

| 开关 | 我们的现状 | 模型/权重其实支持 |
|---|---|---|
| **上下文长度** | `--context-length 32768` | **1048576（1M）** |
| **MTP / 投机解码** | 那一层不建 | `num_nextn_predict_layers = 1`，**权重里带着** |
| **prefix cache** | `--disable-radix-cache`（8 个脚本全带）| 标准生产特性 |
| **DeepEP** | `--moe-a2a-backend none`（TP dispatcher）| 上规模/开 DP 才有意义 |

**① 长上下文是这个模型的卖点，而我们从来没测过。**
DSA 稀疏注意力、kpool、`index_topk=2048` 这一整套**就是为长上下文做的**，
而我们最长跑到 32k，验精度的长提示是 3256 token。**这套架构最贵的部分基本没让它干活。**

⚠ **不是改个数字就完事**：TP8 那次服务 `max_total_num_tokens=939712`，
**一条 1M token 的请求根本放不下**。得先做容量测算（KV 池 / page 数 / TP 宽度怎么配），
再谈精度与性能。**这是一个完整的题目，够一个 session。**

**② MTP：权重里就带着那一层，我们没建。**
前置阻塞已记在案：**`kpool_decode_update_index_cache` 假设每请求一行**，
而投机解码一步要验证多个 token，这条得先改。
另外单卡那条线有两处 `UNVERIFIED` 的 MTP 快照路径 ——
**没有 MTP 配置就验不了，正好一起解决。**

**③ prefix cache** 相对小，但**卡在 `causal_conv1d_fn_npu` 那个 conv state bug 上**（见 A 类）。
单卡线翻 conv 池时应已顺手修掉，确认后打开、跑一轮 GSM8K 即可。

**⚠ 别把这四条和「记账不做的」混为一谈**：
记账那两条是**已知收益、暂时不做**；这四条是**从来没试过、连收益都不知道**。
后者才是真正的未知，也是下一个 session 最值得接的。

### 记账不做的

- **prefill 进图** —— 真正的阻塞是 `KeyError: 'block_tables'`：
  **Ascend 后端只实现了 decode 的图 metadata 契约**（原先记的「差 registry 注册」是错的，
  `--cuda-graph-backend-prefill full` 现成参数就能开）。深度未知、收益未量化、
  且在共享文件里而 DSv4 回归又不跑。**重启条件写在 PLAN P3.4：先把 prefill 占真实负载的比例量出来。**
- **量化 KDA** —— 用户 2026-08-30 决定先不做。但它是这份 checkpoint 剩下**最大的杠杆**：
  厂商 FP8 checkpoint 就没量化 KDA 的 q/k/v/o_proj，而 bs=1 时它占每 token 流量的 **41.8%**。
  TP16 下被 16 分（约 0.45 ms/token）所以在多卡这条线不显眼。

## 还没碰的（别当成已验）

① prefill 图没开。⚠ 「extend 侧还有 host 同步」这个说法**是错的、已作废**：
   逐行审计过，`kpool_indexer_npu.py` 里 `.item()`/`.cpu()`/`.tolist()` 一个都没有。
   挡住捕获的是**被烘进图的 host 侧构造**和**动态输出形状**，两件不同的事
② `enable_torch_compile` + `npugraph_ex`（`patch_model_npu`）那条路
③ MTP / spec decode 下的捕获
④ DeepEP-normal 的 MoE（出厂配方走 `--moe-a2a-backend none`，是已验的 TP dispatcher）

## 对外页面（别新建，要更新那一个）

https://claude.ai/code/artifact/54dbfb20-667f-465d-84c1-ea7d0cc1a827

⚠ **更新时必须把这个 URL 传给 Artifact 工具**，否则会新建一页而不是更新它。
**以仓库为准**（算子结论在 `PLAN.md` §2）。

**2026-08-30 已刷新到最新**：十三节，第二部分是单卡性能分析（单卡那条线在写），
第十二节是共享 kpool 代码那三条。

⚠ **两条线都在写这一页，所以发布前必须先 `action: "read"` 把线上版本读回来再合。**
本次就撞上了：编辑期间对方发布了 E/F 两条，直接发布会把它们覆盖掉（工具挡住了）。

⚠ 上一版 RESUME 在这里写着「页脚和 banner 还写着端到端精度尚未跑通，严重过期」——
**那条待办本身才是过期的**，页面早就改正了。**照抄旧文档而不核实，就是这么产生的。**

## 欠账台账

`SHARED_CHANGES.md`：5 条已改 + 3 条待决。其中 **DSv4 的 GPQA 回归没跑**
（swiglu_limit 那条改动欠的）—— ⚠ **本项目决定不跑**（用户 2026-08-29 拍板）。
但那条改动**确实会改变 DSv4 的数值**（给 routed 专家加上本就该有的 clamp，
实测修前 2.85× budget 判失败、修后 0.35×），所以这是一条
**已知且被接受的风险，合入前需要下游确认**，不是一条会有人认领的待办。
DSv4 权重在 P1.2 后已删（只留元数据），真要跑得重新下载约 275 GB。

## 七条最贵的教训

1. **短 prompt 测不到 kpool** —— `seq_len < index_topk=2048` 时 indexer 直接全选。
   「Paris 答对了」只证明 45 层能串起来。现在有 3256 token 的长提示基线了
2. **包装类缺转发时，最坏的情况不是 `AttributeError` 而是静默的默认值。**
   `HybridLinearKVPool.slots_per_page` 是
   `getattr(self.full_kv_pool, "slots_per_page", self.page_size)` ——
   被包对象没有这个属性时它**不报错，返回包装类自己的 `page_size`**。
   本项目被「方法加在被包的那个上」咬过三次（`forward_metadata`、`set_index_k_bf16`，
   以及 `scratch_loc` 在 2026-08-30 前一直不可达而无人发现 —— 因为没有代码路径走到它）。
   ⚠ **而且包装层不止一处**：KV cache 走 `HybridLinearKVPool`，但 KDA 的 conv state
   走的是 `req_to_token_pool`（`memory_pool.py:1417` 同样是一层转发）。
   **加访问器前先问「这东西挂在哪个池上」，别默认是 KV 池。**
3. **单层全绿不等于整网对** —— 整网拉通修的三个 bug 里两个是
   「顶层对象是包装，新方法加在了被包的那个上」，单层 harness 直接构造内层，**结构上发现不了**
4. **工具里拍脑袋的阈值比没有阈值更危险** —— 它让人对着错的基准下判断。
   现在阈值是**测出来的**，而且要显式 `--floor` 传进去才会给判定
5. **这台机器的 `HTTP_PROXY` 会劫持 127.0.0.1**，代理回 503。连本机服务前先 unset

6. **指标选错比没有指标更糟。** 我拿 GSM8K 的 aggregate tok/s 报过「INT8 慢 1.47×」，
   实际满批下 INT8 快 6% —— 那个数把 128 路满批和单请求长尾平均在了一起，
   谁碰上一条不收敛的生成谁就难看。**先问这个指标在测什么，再看它的值**
7. **「单请求全绿」和「单层全绿」是同一类错觉。** chunked prefill 的并发场景、
   graph 的 padding batch，都是**批里有别人时才会发生的事**，单请求/单层测试
   结构上碰不到。要问的是「这个失败模式需要什么条件才会出现」，
   而不是「我的测试通过了吗」

8. **共享机器上最贵的不是被打断，是被打断得不明显。** 见上面「机器」一节
9. **一次采样不是地板。** 做 `--check-mixed-state` 时按槽用它自己那一次 uniform 采样
   当地板，**把一个干净的槽判成了污染**（它是自己那次采样的 2.4 倍，却远低于另一个槽的采样）。
   和第 4 条是同一条，只是这次踩在「测了，但只测了一次」上
