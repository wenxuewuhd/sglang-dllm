# 临时工作文档:4K/1.5K 910B 逐算子自洽性(未定稿,勿并入报告)

状态:910B 单边已完成;H20 对照待抓。结论先落在这里,验证/对齐后再决定是否写回正式报告 §2.5/§4.2。

## 0. 采样口径(重要:采样方法本身踩过坑)

- **权威数据集:`scratch_profile/profiles/prof_910b_4k_bs72_diverse/`**
  - 命令:`--max-running-requests 72` + graph `max_bs=72` bs=`[1,8,16,32,48,56,64,72]` + `mem-fraction 0.78` + `SGLANG_ENABLE_DLLM_MIXED_BATCH=1` + `SGLANG_DLLM_FDFO_STEPS_PER_ROUND=2` + `SGLANG_NPU_PROFILER_LEVEL2=1`,attention-backend ascend。
  - capture:`capture_4k.py --conc 100 --steps 6 --warmup 150`,**每请求不同 prompt**。
  - 采样点:token usage 0.70 → Σ KV ≈ 305K tok ≈ 4.2K/req,12 个 forward 稳在 74–75 ms/forward。
- **踩坑记录(务必保留):`capture_4k.py` 原来给所有并发请求发同一条 prompt。当 `--conc == MRR`(=72)时,72 个请求同时起跑、锁步在同一去噪相位 → top-8 路由几乎相同 → MoE grouped-matmul 只碰 ~142/256 专家(EDR 实测)→ GMM 被人为压快到 329us(表观 3.3 TB/s,超规格=物理不可能,是采样假象)。**修复:每请求不同 prompt(已提交 `51b95ac9b2`)。**
  - ⚠ **但"合成 diverse prompt"仍不够**:算术序列生成的 token 模式路由仍偏集中(757us ≈ ~211/256 专家)。**只有真实语料(ShareGPT)才读满 256(913us,1.17 TB/s)**。完整反推见 §3g 末。
  - **→ 唯一可用于 MoE 归因的数据集是 `profiles/prof_npu_norad_bench/`(真实 ShareGPT)。** `prof_910b_4k_bs72_diverse`(合成)和更早的同 prompt 版本 MoE 数偏快,不可用于 roofline。

## 1. 逐算子自洽性(vs 910B3 规格 320 TFLOPS / 1.6 TB/s)

数据源 **`profiles/prof_npu_norad_bench/`(真实 ShareGPT,radix off,K=1,4 forwards)**——唯一 MoE 读满 256 专家的一版。per forward,bs=72,ctx≈3.9K,总 **76.37 ms**:

| 类别 | 实测 ms/fwd | 占比 | AI | 屋顶类型 | roofline 下限 ms/fwd | 效率(下限/实测) | 自洽? |
|---|---|---|---|---|---|---|---|
| MoE GMM | 24.93 | 32.6% | 72 | 带宽(30.6 GB / 1.6 TB/s) | **19.13** | **77%** | ✅ 自洽 |
| attention (FIA) | 22.67 | 29.7% | 128 | 带宽(11.5 GB / 1.6 TB/s) | **7.17** | **32%** | ❌ 不自洽 |
| dense matmul | 12.70 | 16.6% | ≫ridge | 算力(2.49 TFLOP / 320T) | **7.80** | 61%¹ | ✅ 自洽 |
| MoE glue(路由) | 6.44 | 8.4% | — | vector/scalar | — | — | — |
| 去噪+vector 杂项 | 5.28 | 6.9% | — | vector | — | — | — |
| norm/rope | 4.36 | 5.7% | — | vector | — | — | — |

¹ dense 下限只算了 LM head + QKV/O + router,漏了 shared-expert 等,真实 FLOP 更高、MFU 比 61% 更好(LM head 单核 82%,mac=0.91)。dense 实际更贴屋顶。

### 逐条

- **MoE GMM —— 自洽,带宽受限,正是 §1.2 预测。** AI = t/expert = 72 < ridge 200 → 读满 256 专家(30.6 GB/forward)。下限 19.1 ms,实测 24.9 ms = 规格带宽的 **77%**(有效 **1.23 TB/s**),mte2=0.98 载入管线饱和、mac=0.36 算力闲。硬地板:除非把 t/expert 推过 200(EP / 更大 batch),这 19 ms 不可压。
  - ⚠ 早前版本记的"21.99 ms / 87% / 1.39 TB/s"来自合成 prompt(专家未激活满),**已作废**,见 §0 与 §3g 末。
- **dense matmul —— 自洽,算力受限。** LM head 82% MFU,合计 ~62–80%。AI 远超 ridge,纯算力区。
- **FIA attention —— 唯一不自洽,长序列档最大可回收项。** AI=128 落带宽区,下限只要 7.2 ms,实测 22.7 ms = 屋顶的 **32%**,有效 0.51 TB/s(规格的 1/3)。mac=0.25 cube 闲 + mte2=0.93 载入忙但低效 → 卡在 online-softmax 的 QK→softmax→AV 串行气泡,两条 pipe 都没吃满。白扔 ~15.5 ms/forward = 整步的 20%。
  - 若修到吃满带宽(→7.2 ms 下限):整步 76 → ~61 ms,单 forward 提速 ≈ **+25%**。

## 2. 对 H20 PK 的指向【已由 §3g 解答:假设推翻】

~~attention 最可能是被 H20 拉开的位置~~ —— **实测否定**。H20 attention 24.12 ms 反而**比 910B 的 22.67 慢**;换算有效带宽 H20 ≈0.56 TB/s = **其 4 TB/s 规格的仅 14%**,比 910B 的 32% 还差。thin-M(每段 32 query 行)是**两家共同的**瓶颈,H20 的带宽优势在 attention 上完全没兑现。

真实答案见 §3g:同 bs 下两边打平,1.4× 是软件开销(radix + K=2),H20 剩余 ~9% 来自显存容量。

## 3. 交付物

- **`scratch_profile/fia_decode_bench.py`**(已提交 `8ee4f165e5`):独立复现 `forward_dllm` 的 FIA 调用,给算子团队做优化输入。shape `query[2304,16,128]` / `kv[13697,32,512]`,TND + paged。
  - 实测:默认 1092us / 36% roofline / 0.57 TB/s;`--contiguous` 1064us;`--uniform-ctx` 1069us —— 三者 <3%,**低效是 kernel 内在(online-softmax 串行),不是散页 gather 也不是变长**。优化目标 = QK→softmax→AV pipeline。

## 3b. FIA 低效的机理分析(从算子文档 + 实测)

算子 `npu_fused_infer_attention_score`(`forward_dllm`,`ascend_backend.py:1884`)。

- **输入口径(勿再写错)**:query `[2304,16,128]` = **72 段 × 每段 32 token 的 TND 打包**,总 2304 token。每段 32 个 query 只 attend 自己那条 ~4231 的 KV(靠 `actual_seq_lengths` 分段)。"扁"指**每段 M=32**(决定 cube tile 填充),不是总量小。
- **分发路径**:文档按 Q_S 分两支——Q_S=1→IncreFlashAttention(decode 专用),Q_S>1→PromptFlashAttention(prefill 路径)。dLLM 每段 Q_S=32>1 → 走 **PromptFlash**,但形状扁(32×4231)→ cube 的 M 维填不满 → mac=0.24。根因是 **thin-M(32)+ online-softmax 的 QK→softmax→AV 串行气泡**。
- **两个次优/不合规参数**:
  1. `block_size=32` < 文档要求的 128–512(步长 128)。实测 32→≥128 提速 ~11%(1088→960us),≥128 平台在 ~40%。
     - **page=32 在 sglang 侧无 fallback,一路真到 kernel 边界(agent 已核实,带 cite):** dLLM override `overrides.py:2182 _dllm_page_size` 把 page_size **钉到** `DllmConfig.block_size=32`(是往下钉到 32,不是往上到 128);KV pool 是 NHD 扁平 token 粒度(`memory_pool.py:1583/2150`,物理上没有 page 维),`[13697,32,512]` 是调用点 `k_cache.view(-1,32,512)` 现造的;block_table 按 32 粒度建(`ascend_backend.py:454/712`);FIA 实参 `block_size=self.page_size`=32 逐字传(`ascend_backend.py:1891`)。无任何"page 必须是 128 倍数"的 NPU assert。那些强制 128 的 override 都是 CUDA/XPU/MLA 专用,与 ascend 无关。
     - **但算子内部 compute 恒按 128-tile 算(探针实测坐实):** `fia_page_probe.py` 固定 block_size=32、扫 ctx,发现延迟跟 **ceil(ctx/128)** 走、不跟 ceil(ctx/32):同一 128-band 内 ctx 129→192→256(真实 KV +60%)延迟持平(226/220/230us),257 才跳(275us);exact-128 倍数最快、正下方 224/255/383 有部分-tile mask 惩罚。→ **给它 32 不会让 compute 更细,只是把 32-页在内部重拼成 128-tile,多付 4× block_table 遍历(133 vs 34 项/4.2K 请求)= 之前 bench 里 32→128 的那 11%。**
     - **修正结论**:KV 的 N-tile(128)没问题;`block_size=32` 的 11% 是纯 block_table 间接开销。给算子团队的措辞去掉"支持 block_size=32";我们侧把 KV page_size 从 dllm_block_size(32)解耦到 128 即可省这 11%。真正大头(~60%)仍是 **thin query M=32**(cube 填不满 mac=0.24)+ online-softmax 串行。
  2. `input_layout="TND"` + `block_table` —— 文档称 TND 不支持 page attention,但生产在跑。需算子团队确认没走降级 kernel。
- **替代算子(codebase 全查过,无更优可换)**:IncreFlashAttention 只吃 Q_S=1(用不了);`_npu_paged_attention` 假设每序列 1 query token(32-token 块映射不过去);`npu_fused_infer_attention_score_v2` 是同一 kernel 的新 API,换它不改性能。→ 解法只有:算子团队优化 thin-M PromptFlash tiling + QK/softmax/AV 流水(~60% 大头),或算法侧增大去噪块把 M 撑大(改调度,另评估)。
- **交付**:`fia_decode_bench.py`(整核 roofline,带 `--page-size` 扫描)+ `fia_page_probe.py`(内部 tile 粒度探针,commit `d007511dd2`),固定 shape 可复现。

### 给算子团队的最终一句话
> 算子 `npu_fused_infer_attention_score`,TND + paged,shape `q[2304,16,128]`(=72 段×32 query 打包)/ `kv[·,page,512]`,GQA 16/4,head_dim 128,72 段变长 KV(avg 4231)。现 ~40% of 1.6 TB/s roofline(mac=0.24 / mte2=0.93)。
> 1. **KV 内部已按 128-tile 算**(探针实测:延迟跟 ceil(ctx/128) 走,exact-128 最快、just-below 有 mask 惩罚)。所以**不用为 block_size=32 做事**——我们会把 KV page 解耦到 128 自行消掉那 ~11% 的 block_table 间接开销。
> 2. **真正要优化的是 thin query M=32**:每段只有 32 个 query 行,cube 的 M 维填不满(mac 才 0.24),叠加 online-softmax 的 QK→softmax→AV 串行气泡,两条 pipe 都空转。请针对 **M=32 × N=4231 的扁平 attention** 优化 tiling / 流水。复现:`fia_decode_bench.py`、`fia_page_probe.py`。

## 3c. H20 逐算子对照(gpu_trace/1784859185...bs72,6 forwards)【已被 §3g 取代】

> ⚠ **本节数据已过时,以 §3g 为准。** 这版是 radix on + K=2 + `with_stack=True` 抓的(bench 的 `--profile` 不传 with_stack,服务端默认 True),host 侧被放大;§3g 是两边 radix off + K=1 + `with_stack=False` 的干净对照。结论方向一致(910B kernel 工作量更少),数值以 §3g 为准。

✅ **graph mode 确认**(6 次 `cudaGraphLaunch` = 6 forward,用的就是 launch_h20 脚本)。

⚠ **能信什么、不能信什么(重要)**:
- **可信:GPU kernel 的 device 时长**(不受 host 插桩影响,kernel 照原速跑)→ **逐算子 kernel-sum 那张表可信**,是本节唯一可做归因的量。
- **不可信:任何 wall-clock / GPU 空闲 / CPU 开销**。profiler 开着时 host 侧被插桩(70378 个 cpu_op + CUDA API hook + sync),forward 间空档被人为放大。GPU 时间线上"6 forward 间隔 ~136ms、kernel 活动 ~88.8ms/forward、~47ms 闲置"——**这 47ms 是 profiling 伪值,不是生产真值,禁止用于归因**(NPU 侧开 profiler 同样虚高)。
- **H20 去掉了 `STEPS_PER_ROUND=2`** → 去噪 schedule / tokens-per-forward 可能不同,e2e 吞吐不能直接除。
- 真实 wall-clock/CPU 开销只能用**不开 profiler 的 e2e**(TPOT)去测,trace 给不了。

**逐类 kernel-sum,ms/forward(NPU=prof_910b_4k_bs72_diverse,H20=本 trace):**

| 类别 | NPU ms/fwd | H20 ms/fwd | 谁快 |
|---|---|---|---|
| attention | 24.64 | **27.87**(flashinfer BatchPrefill 24.58 + Prefill 2.51 + merge 0.79) | **打平**(H20 略慢) |
| dense matmul | 12.67 | **25.79**(含 LM head 10.4) | **NPU 快 2×**(算力) |
| MoE grouped gemm | 21.99 | **22.21**(fused_moe_kernel) | **打平** |
| MoE glue(路由) | 6.46 | 1.94 | H20 快 3× |
| 去噪/argmax | 4.89 | 4.89 | 平 |
| norm/rope | 4.39 | 0.70 | H20 快 6× |
| **合计 kernel-sum** | **~75** | **~88.8** | **NPU 少 ~15% kernel 工作量** |

**三个反直觉但重要的结论:**
1. **attention 打平,H20 的 4 TB/s 带宽被浪费。** H20 BatchPrefill 1228us/call ≈ NPU FIA 1232us/call。若 H20 context 也 ~4.2K(待确认),其有效带宽同样只有 ~0.5 TB/s = **H20 自己 roofline 的 13%**——flashinfer 在 **thin-M(32)** 这个形状上和 NPU 一样卡在 online-softmax,带宽优势用不上。**所以 attention 不是 H20 拉开差距的地方,先前假设推翻。**
2. **MoE 也打平**(22.0 vs 22.2)。H20 带宽优势在 MoE 同样没兑现。
3. **dense 是 NPU 的主场**(H20 慢 2×,算力受限),小算子(glue/norm)是 H20 主场。

**净结论:在这个 4K/1.5K decode 的 raw-op 层面,910B 并不输 H20,kernel-sum 反而少 ~15%。** e2e 那 1.4× 差 **不来自单算子硬件速度**(attention/MoE 打平、dense 还是 NPU 快)。剩下的可能:去噪 CPU/调度开销两边不同、`STEPS_PER_ROUND` 改了 tokens-per-forward、或 e2e 测量条件。**但这些都不能从 profiled trace 归因**(见上:wall-clock 被 profiling 污染)。

**下一步(不能再靠 trace)**:
1. **不开 profiler**,两边同 config(同 STEPS_PER_ROUND=2、同 graph)各跑纯 e2e,拿真 TPOT + 每 token 去噪步数 → 反推每 forward 真 wall-clock,减去可信的 kernel-sum,差值才是真去噪 CPU/调度开销。差距很可能在这,不在算子。
2. H20 补设 `STEPS_PER_ROUND=2`;确认 context 深度 ≈4.2K。

## 3d. 打点探针:device 与 kernel 自洽 + host 开销真值(NPU 实测)

埋点 `SGLANG_DEBUG_DLLM_STEP_TIMING=1`(commit `16684c9a63`),bench_serving 真实负载(random+ShareGPT,`--random-range-ratio 1` 定长 4096/1536,seed 42,bs=72)。

- **一个 scheduler step = `fdfo_steps_per_round`(=2)个 model forward**(`base.py:_run_fdfo`,bs≥64 无 early-exit)。
- 稳态实测(range-ratio 1,context ~3.8–4K):**device ≈ 156 ms/step,host ≈ 215 ms/step,step ≈ 375 ms,host 占 ~57%。**
- **自洽 ✓**:device 156 / 2 forward = **78 ms/forward ≈ trace 的 75 ms/forward kernel-sum**(context 略深故 78>75 合理)。→ **打点的 device 时间与 profiler kernel 时间对得上,探针可信。**
- **新增关键量:host ≈ 215 ms/step(~107 ms/forward-equiv),比 device 还大** → dLLM 一步里 **一半以上是 CPU**(调度 + 72-req 去噪结果循环 + 流式输出),不是 GPU。这才是 1.4× 最该查的方向。
- ⚠ **caveat**:host 是 step wall-clock − device,含 (a) forward 间 GPU 空闲被算进 device 窗口(略压低 host)、(b) bench client + tokenizer 同机争 CPU(略抬高 host)。device 无争用、可信;host 的绝对值有 ±,但"host≳device"这个量级结论稳。NPU vs H20 都同机跑 bench,争用大致对称。

**给 H20 的判据**:同 config 出 `[dllm-step-timing]`,比 **host/step**。device/fwd 两边应各自 ≈ 自己的 kernel-sum(NPU~75、H20~89);若 **NPU 的 host 明显大于 H20**,1.4× 就坐实在去噪 CPU/调度,不在算子。

## 3e. host 那 209ms 到底在干什么(py-spy 采样 scheduler,4763 样)

**不是去噪(去噪是 on-device kernel)。是 radix 前缀缓存的树操作。**

| | 占 scheduler self-time |
|---|---|
| `get_next_batch_to_run`(inclusive) | 84.6% |
| └ `cache_unfinished_req`(把这步去噪出的 block 提交进前缀树,inclusive) | 74.6% |
| **radix_cache.py 自身 self-time** | **72.1%** |
| ├ radix insert `_insert_helper` | 35.6% |
| └ radix match `match_prefix` | 26.3% |
| 去噪 `fdfo_batched_end` | ~3% |
| torch_npu getenv/transfer_to_npu patch | ~4% |

- 每步 `cache_unfinished_req` 把每个 req 的 4K+ token 序列 match+insert 进前缀树,O(seq_len) 的 Python 树遍历 × 72 req × 每步 → 吃掉 host 的 ~72%。
- **本 bench 是 random/ShareGPT 不同 prompt,前缀零共享 → radix cache 命中率 0、纯开销。** 对 dLLM 长序列 decode,radix 前缀树在做无用功。
- **`--disable-radix-cache` 实测:host 208.6 → 21.2 ms(−90%),step 362 → 174 ms,device 不变(76.5 → 77.3)。**

| NPU bs=72 4K/1.5K | host/step | device/fwd | step | host 占比 |
|---|---|---|---|---|
| 默认(radix on) | 208.6 ms | 76.5 ms | 362 ms | 56% |
| **`--disable-radix-cache`** | **21.2 ms** | 77.3 ms | **174 ms** | **12%** |

device 完全不变 → 砍掉的纯是 CPU 浪费。**"host≈device"不是 dLLM 固有成本,是 radix cache 在长序列 decode + 零前缀共享下的无用功。**

⚠ radix 是设备无关的 Python,**H20 也在付同样的税** → H20 也该加 `--disable-radix-cache` 再比,否则比的是"谁的 CPU 跑 radix 更快"。这也可能改写 1.4× 的账。

## 3f. 【结论级】去掉 radix 税后 910B vs H20 打平,1.4× 归零

同代码同 bench(bench_serving random,定长 4096/1536,1152 prompts,conc 72,seed 42):

| 配置 | 910B 输出吞吐 | H20 输出吞吐 | 对比 |
|---|---|---|---|
| radix **on**, K=2(原报告口径) | 1058 | 1476 | H20 **1.40×** |
| radix **off**, K=2 | **1811** | 1738 | **910B 快 4.2%** |
| radix **off**, K=1 | 2004 | **2032** | H20 快 1.4% |

- **原来的 1.40× 基本全部来自 radix 前缀缓存**(设备无关 Python,两边都付,但把 910B 压得更狠:910B +89%、H20 +38%)。去税后**两边打平(±4%)**。
- 与逐算子结论自洽:attention 打平、MoE 打平、dense 910B 快 2×、小算子 H20 快 → 算子层面本就不该有 1.4×。
- **K=1 在两边都赢**(910B +10.7% / H20 +16.9%,TTFT 双双 −40%/−42%,TPOT 仅 +7.6%/+2.4%)→ 坐实是 `_run_fdfo` 的调度行为(bs≥64 时 `check_early_exit=False` 无条件多跑一个 forward + 冻结批次推迟准入),与硬件无关。H20 的 K=1 收益更大,说明其 host 占比更高(同样 Python、device 更快)。

**报告 §2.5 需要的修订**:
1. **撤回"910B 在 4K/1.5K 落后 H20 1.4×"** —— 那是 radix 开销,不是硬件差距。
2. 4K/1.5K 的正确配置是 `--disable-radix-cache` + `STEPS_PER_ROUND=1`(两边都适用)。
3. 此前"推荐 K=2"要标前提(仅 radix on 时成立)。

caveat:910B 两版是否串行跑待确认(端口 31600/31500);H20 的 1476 是早先测的、条件未必完全一致。但 **radix off K=2 那组(1811 vs 1738)是严格同代码同参数**,单这组即足以推翻 1.4×。

## 3g. 【定稿对照】两边 radix off + K=1 的逐算子 + 吞吐总账

采样口径两边完全一致:`--disable-radix-cache`、`FDFO_STEPS=1`、bench_serving random 定长 4096/1536 seed 42 conc 72、**bench 不带 `--profile`**、用 `arm_profile.py`(`with_stack=False`/`record_shapes=False`)在稳态 arm、各抓 **4 个 forward**,shape 均为 2304 token = 72×32。
数据:`profiles/prof_npu_norad_bench/`(msprof)、`profiles/prof_h20_norad_bench/`(kineto)。

### 逐算子(ms/forward)

| 类别 | 910B | H20 | 谁快 |
|---|---|---|---|
| **dense matmul** | **12.70** | 26.02 | **910B 快 2.05×**(LM head 5.64 vs 10.46) |
| MoE grouped gemm | 24.93 | **23.42** | H20 快 6% |
| attention | **22.67** | 24.12 | **910B 快 6%** |
| MoE glue(路由) | 6.44 | **1.81** | H20 快 3.6× |
| norm/rope | 4.36 | **0.70** | H20 快 6.2× |
| 去噪/其他小算子 | 5.28 | 9.81 | 910B 快 |
| **kernel 合计** | **76.37** | **85.88** | **910B 少 11% 工作量** |

- **dense 是 910B 绝对主场**(2.05×),320 vs 148 TFLOPS 的算力优势完全兑现。
- **attention 910B 反而更快**,尽管 H20 有 2.5× 带宽。反推有效带宽:910B ≈0.51 TB/s(规格 32%),**H20 ≈0.56 TB/s(规格仅 14%)**。flashinfer 在 thin-M(每段 32 query 行)上比 FIA 更没把带宽用起来 → **H20 的带宽优势在 attention 上作废**。
- **小算子(glue+norm)是 H20 主场**:10.80 vs 2.51(4.3×),NPU 一贯弱项。

### 吞吐总账(含 req/s)

| 配置 | req/s | 输出 tok/s | 总 tok/s | TPOT | Mean E2E |
|---|---|---|---|---|---|
| 原报告 910B(radix on, K=2) | ~0.69 | 1058 | 3909 | 48.1 ms | — |
| 原报告 H20(radix on) | ~0.96 | 1476 | 5452 | 40.7 ms | — |
| 910B bs=72, K=2 | 1.18 | 1811 | 6641 | 23.25 ms | 58.9 s |
| **910B bs=72, K=1**(910B 最优) | **1.30** | **2004** | **7349** | 25.01 ms | 52.4 s |
| H20 bs=72, K=2 | 1.13 | 1738 | 6374 | 24.71 ms | 62.3 s |
| **H20 bs=72, K=1** | **1.32** | **2032** | **7450** | 25.30 ms | 53.0 s |
| **H20 bs=128, K=1**(H20 最优) | **1.42** | **2181** | **7999** | 41.21 ms | 86.5 s |

四组数据 tok/s ÷ 1536 ≈ req/s 处处成立 → 所有请求都跑满 1536 输出、无提前 EOS,口径一致可比。

### 三条结论

1. **同 bs=72 两边打平**(1.30 vs 1.32 req/s,差 1.5%),而且 910B 的原始 kernel 工作量还少 11%。**原报告"H20 快 1.40×"应撤回**——那 1.4× 里绝大部分是 radix 前缀缓存(设备无关 Python)+ K=2 调度补丁的软件开销,不是硬件差距。910B 从 0.69→1.30 req/s(+88%),H20 从 0.96→1.32(+38%),软件开销对 910B 伤害大得多。
2. **H20 最后 ~9% 的领先来自显存容量**(96 vs 64GB → bs=128 vs 72),不是单位算力/带宽。910B 64GB 放不下 128×5632=72 万 token。**且 bs=128 是拿延迟换吞吐**:TPOT 25.3→41.2ms(+63%)、E2E 53→86.5s。若按 TPOT ≤30ms 的 SLO 卡,bs=128 不可用,两边就是 1.30 vs 1.32 的平局。
3. **910B 那 11% 的 kernel 优势没转成吞吐** —— 说明还有 host/串行度的损失(radix 已砍掉最大项,剩下是调度 Python 其余部分 + kernel launch 串行度:H20 每 forward 1546 个小 kernel 但可异步重叠,NPU 230 个但串行度更高)。**这是当前最大的未解项,方向在 host 不在算子。**

### ✅ MoE GMM 的历次差异已闭环:prompt 多样性 = 激活专家数

早前几版 GMM 偏快**全是同 prompt 造成的专家未激活满**,不是频率/缓存玄学。以本版(真实 ShareGPT,满 256 专家)实测的 **1.17 TB/s** 为有效带宽基准,反推各版实际读取的权重量:

| capture(w13 核) | us | 表观 TB/s | 反推读取量 | ≈激活专家数 |
|---|---|---|---|---|
| 同 prompt,conc=72(完全锁步) | 329 | 3.26 ❌ 超规格 | 0.385 GB | **~92** |
| 同 prompt,conc=100(canonical) | 643 | 1.67 ❌ 超规格 | 0.752 GB | ~179 |
| 同 prompt,conc=100(复现) | 659 | 1.63 ❌ 超规格 | 0.771 GB | ~184 |
| 合成 diverse prompt,conc=100 | 757 | 1.42 | 0.886 GB | ~211 |
| **真实 ShareGPT diverse(本版)** | **913** | **1.18** | **1.068 GB** | **~255 = 满** |

- 单调关系一目了然:**prompt 越多样 → 激活专家越多 → 读的权重越多 → GMM 越慢**。
- 前三版表观带宽 **>1.6 TB/s 规格上限,物理不可能** → 直接证明它们没读满 256 专家。
- 与 expert-distribution recorder 的直接实测吻合:同 prompt conc=72 时实测**平均 142/256 激活**(层间 88–203),反推的 ~92 属同一量级(反推假设各层均匀,偏低正常)。
- **→ 913us / 1.17 TB/s(规格的 73%)是真实负载下的有效值**,§3g 与 §1 一律以它为准。早前 643/659 那版**不可用于任何 roofline 归因**。

推论:合成/重复 prompt 的 benchmark 会**系统性高估 MoE 性能**(这里高估 1.4×)。凡涉及 MoE 的测量必须用真实多样语料。

### 待验证

- (H20 bs=128 无需再抓 trace,吞吐已在上表对照。)

## 3h. 【已解】那 11% 丢在哪:每 step ~37ms 裸露 host,不是 kernel launch 串行度

**方法**(2026-07-27,radix off + K=1,复现命令同 handoff):bench_serving random 4096/1536 seed 42、400 prompts、conc 72;稳态窗口内 py-spy 40s @250Hz nonblocking 采 scheduler(9625 样,误差 1.5%);**round 速率直接数 server log**(`Prefill batch, #new-seq: 72` 每 round 一行):41s 内 351 round → **8.56 round/s → step = 116.8 ms**。本轮 bench 1.21 req/s / 1854 tok/s(400 prompts 含收尾拖尾,conc 均值 63.2;与 §3g 的 1152-prompt 1.30/2004 一致)。

### step 的 116.8 ms 解剖(py-spy 分段 × step 时长)

| 段 | ms/step | 说明 |
|---|---|---|
| **等 forward**(阻塞在 `joint_threshold.py:217` 的 H2D copy,stream-ordered,吸收全部等待) | **78.3** | ≈ kernel-sum 76.37 → **graph 内 kernel 空隙仅 ~2-4ms,"NPU launch 串行度"假设排除** |
| **裸露 host(device 空转)** | **~37** | 占 step **31%**,如下拆分 |
| ├ `get_next_batch_to_run`(每 round 重建 ScheduleBatch:`_create_dllm_batch` + `prepare_for_extend` + `_alloc_extend_loc_with_kv_reuse` 各处 Python) | 18.1 | 最大单项 |
| ├ forward 发射:torch_npu graph task update(`replay_with_input_update` **每次 replay 现场 spawn+join 一个线程**跑 `graph.update`;`update_capture_record` 在 helper 线程里 ~7ms Python) | 9.5 | `npu_cudagraph_backend.py:141-172` |
| ├ 去噪状态 gather/scatter(`_step_vectorized_fdfo`:每 round 把 72 个 per-req dict stack/tensor 上卡、`tolist` 回来、Python 循环写回) | 5.1 | gather 不依赖 forward 输出,却排在 forward 后 |
| ├ `process_batch_result_dllm`(72-req 循环;`base.py:237` 对全批 [72,32] `input_ids.tolist()`,但只有 done 行需要) | 2.8 | |
| └ 其余(run_batch 胶水、event loop) | ~3 | |

旁证:scheduler 进程 CPU 157%(主线程 ~90% busy + graph-update 线程);npu-smi AICore 计数器均值 86% 不可信(真实 device busy ≈ 8.56×76.37/1000 = 65-68%),**round 速率 × kernel-sum 的算术才是准的**。

### 与 H20 对账(推算,H20 未直接 py-spy)

H20 同负载 tok/round 相同 → round/s = 8.56×(2032/1854) = 9.38 → step ≈ 106.6 ms,kernel-sum 85.88 → **H20 裸露 host ≈ 21 ms/step**。对比:**910B 37ms vs H20 21ms,同一份 Python 910B 付 1.8×**(ARM 单核弱于 x86 + 全部 torch 调用过 `transfer_to_npu` decorated 包一层 + graph task update 线程机制是 NPU 独有)。§3d 旧账自洽:K=2 时裸露 host 21ms/step 摊 2 个 forward,K=1 后重建/发射每 forward 都付。

### 判据落定 + 兑现空间

**host 差得多 → 那 11% 在调度侧,可兑现。** 若把 37ms 压到 H20 的 21ms,step 116.8→100.8,round/s +16%(1.21→1.40 req/s 量级,反超 H20 bs=72)。优化排序(按 ms 和侵入度):

1. **round 间增量复用 ScheduleBatch(18.1ms)**:每 round 只有 ~7/72 行换新 block(1854 tok/s ÷ 32 tok/block ÷ 8.56 round/s ≈ 6.8 行/round),其余行的 alloc/fill_ids/metadata 原样重建了 72 遍。
2. **graph update 瘦身(9.5ms)**:done 行以外 seq_lens 不变 → 可跳过/稀疏 update;线程改常驻 worker 或直接同步做(spawn+join 每 step 一次)。
3. **去噪状态常驻批式 device 张量(5.1ms)**:FDFO state 按 slot 常驻,round 间免 gather/scatter;最少也应把 216-224 的 gather 挪到 forward 发射前(不依赖 logits),别让阻塞 H2D 在 forward 后才排队。
4. **`base.py:237` 只取 done 行的 tokens(~1-2ms)**:全批 tolist → 按 done gather。

## 4. 待办

已完成:
- [x] H20 同 config 抓一版 → §3g(两边 radix off + K=1 + `with_stack=False`,各 4 forwards)。
- [x] 两边逐类 ms/forward 对齐 → §3g。**1.4× 已证伪:软件开销(radix + K=2),不是硬件。**

已完成(续):
- [x] **查 910B 那 11% kernel 优势为何没转成吞吐** → §3h。每 step ~37ms 裸露 host(batch 重建 18 + graph update 9.5 + 去噪 gather/scatter 5 + 结果处理 3),不是 kernel launch 串行度(graph 内空隙 ~2-4ms)。

待做:
- [ ] 复核 MoE GMM 913us vs 早前 643us 的差异(路由分散度?频率?),确认哪个是真值。
- [ ] H20 bs=128 trace(若抓)→ 验证 t/expert 72→128 是否让 MoE 进算力区。
- [ ] **定稿后写回正式报告**(`LLaDA2_910B_vs_H20_report.md`,目前未改动):
  - §2.5:撤回"4K/1.5K H20 快 1.40×";改为"同 bs 打平(1.30 vs 1.32 req/s),H20 剩余 ~9% 来自显存容量允许 bs=128,且以 TPOT +63% 为代价"。补正确配置 `--disable-radix-cache` + `STEPS_PER_ROUND=1`。
  - §4.2:FIA 效率数字更新(0.51 TB/s = 规格 32%;H20 flashinfer 仅 14%,thin-M 是共性瓶颈),加 `fia_decode_bench.py` / `fia_page_probe.py` 指针。
  - §1.2:MoE "AI=t=72 深度带宽饥饿"的预测在 4K decode 上偏保守,实测 1.17–1.39 TB/s(73–87% 规格)。
  - 新增:radix 前缀缓存在长序列零前缀共享下是最大单项开销(72% 调度 CPU),两边都受影响。
