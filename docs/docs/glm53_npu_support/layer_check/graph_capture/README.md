# NPU Graph 捕获对拍

`../` 的 `check_*.py` 回答「这一层算得对不对」。这里回答**加上图捕获之后还对不对**，
以及**哪些东西挡住了捕获**。四类层（DSA / KDA / MoE / mHC + dense FFN）各一个脚本。

## 判据：三问，缺一不可

| 问 | 怎么答 |
|---|---|
| **cap** 能不能捕获 | `torch.npu.graph(...)` 不抛异常 |
| **bake** replay 还跟不跟它的**设备输入**走 | 原地改写同一批设备 buffer，再 replay，要求跟 eager 在**新输入**上逐位一致 |
| **gold** 从图里读出来的张量还对不对 | 用 `harness.py` 的双参考法打分（`gold.py`） |

**只做前两问是不够的。** replay-vs-eager 只说图自洽；一个把权重烘死的图也能自洽。
**只做第一问更不够** —— 图捕获最坏的失败方式是把 host 侧的值烘进去，之后每次都用那个旧值，
不报错、不崩、数字看着也正常。所以 bake 这一问必须**换一批输入**，而且要**同时换 seq_lens**
（`cap_dsa.py` 的场景 C），否则测的是个恒等式。

## 跑

```bash
source $ROOT/env.sh
export SCRATCH=<dump_reference.py / reference_dsa.py 写 golden 的目录>   # 只有 cap_dsa 需要
D=$REPO/docs/docs/glm53_npu_support/layer_check/graph_capture

# die 只用空的那几个；跑之前 npu-smi info 看一眼
run() { ASCEND_RT_VISIBLE_DEVICES=14 PYTHONPATH=$REPO/python:$PYTHONPATH $VENV/bin/python "$@"; }

run $D/cap_mhc.py -M 16
run $D/cap_ffn.py -M 16
run $D/cap_moe.py -M 16 --parts full --golden
run $D/cap_kda.py --batch 15
run $D/cap_dsa.py                    # 默认就是部署形状，16 条 ragged 请求到 32k
run $D/cap_dsa_pad.py --real 1       # padding 行会不会踩到真实请求

# 图里的 HCCL（两张卡就够回答，不需要 16 张）
ASCEND_RT_VISIBLE_DEVICES=14,15 PYTHONPATH=$REPO/python:$PYTHONPATH \
    $VENV/bin/torchrun --nproc_per_node=2 --master_port=29866 $D/cap_hccl.py
```

`ASCEND_RT_VISIBLE_DEVICES=14` 让逻辑 0 号 die 落在物理 die 14 —— 脚本里一律
`set_device(0)`，靠这个变量选卡。

## 为什么 padding 单独有脚本

**捕获出来的图宽度是固定的。** 真实 batch 比捕获的 bs 小时，runner 把尾部行补齐：
`seq_lens → fill value`、`req_pool_indices → 0`、`num_padding = bs - raw_bs`
（`model_executor/runner/decode_cuda_graph_runner.py:188`）。于是**每一个 padding 行都自称是请求 0**。
任何按 `req_pool_index` 散写而不排除它们的地方，都会把请求 0 的状态悄悄写坏。
这个失败模式**只在图模式下存在**，`check_*.py` 一个也看不见。

池子那边留了后路：`ReqToTokenPool` 把 req 槽 0、`MambaSlotAllocator` 把 mamba 槽 0
留作 padding 的 dummy 写入目标（`mem_cache/allocator/mamba.py`，
`free_slots = arange(1, size+1)`）。所以**测这件事的时候，真实请求不能占 0 号槽**
—— `cap_kda.py` 把 `req_index_to_mamba_index_mapping` 整体挪了一位就是为了这个。
不挪的话，请求 0 坐在保留槽上，这条测试问不出任何东西。

同理，判分要**跳过 0 号槽**：padding 行本来就该写在那儿，把它算进去会把正确实现判失败。

## 已知的事实（本机实测）

- 捕获期间 `.item()` / `.cpu()` / `.tolist()` **会抛** `107027`（"stream is captured"），不会静默通过
- `torch.nonzero` / `torch.unique_consecutive` 也抛 —— 输出形状是动态的
- `torch.cumsum` 可以捕获
- **HCCL 集合通信可以捕获**（2 卡实测）。而且**只改对端输入**时 replay 也跟着变 ——
  这一问比「replay==eager」强：它排除了把集合通信结果烘死的可能
- **AI CPU 回落的算子可以捕获** —— `aclnnIndex`（两个索引张量的高级索引）在图里跑得好好的。
  这条之前被推断为阻塞项，实测证伪

---

# 走真实 runner：整层 / 两层同一个图 / 多 bs 共池

上面那批脚本用 `torch.npu.graph(...)` **直接捕单个模块**，绕开了部署真正跑的那段代码。
`cap_runner_layers.py` + `runner_fixture.py` 把它补上：**真的 `NPUGraphRunner`**，
**两个完整 decoder 层进同一个图**，**真的 bs 桶列表共用一个 memory pool**。

## 为什么非要走 runner

`SHARED_CHANGES.md` 记的三处「只报告没改」的隐患，**有两处只长在被绕开的那段代码里**
（runner 的静态 buffer + 各 backend 的 `init_forward_metadata_out_graph` /
`_replay_metadata` / `_apply_cuda_graph_metadata`）。裸捕获永远问不到它们。

而「整层」也不是「四块分开都能捕」的推论：mHC 的四路残差要穿过注意力再回来，
cube 密集的注意力和向量密集的 MoE 要共处一图，DSA 与 KDA 要在一次 replay 里碰同一批池，
层与层之间还要交接 buffer。

## 这里什么是真的、什么是假的

| | |
|---|---|
| **真** | `Glm5NextDecoderLayer` 本体（模型自己的构造器 + 模型自己的 `load_weights` + checkpoint 权重） |
| **真** | `AscendAttnBackend` + `AscendKDAAttnBackend` 包在 `AscendKDAHybridLinearAttnBackend` 下 —— `attention_registry.py` 给 GLM 建的那一个 |
| **真** | `NPUDSATokenToKVPool` + `HybridReqToTokenPool`（含 mamba 侧） |
| **真** | `NPUGraphRunner` 自己：capture 循环、bs 桶、**一个** 共享 pool、`load_batch` / `execute` |
| 假 | `ModelRunner`：`ModelRunner.__new__` 空壳 + runner 真正读的那 ~25 个字段 |
| 假 | embedding：`hidden = h_table[input_ids]`。表是我们的，**但 `input_ids` 是 runner 自己的静态 buffer**，所以隐状态仍然从 runner 管的设备 buffer 进来 |
| 假 | TP：一个进程用 `get_parallel().override(...)` 假装是 `tp` 路里的 rank 0（`tp_fixture.py` 的老办法）。`--tp 16` 得到部署形状下**rank 0 的部分和**；`--tp 1` 得到完整结果，才能跟 CPU trace 对 |

## 跑

```bash
source $ROOT/env.sh
D=$REPO/docs/docs/glm53_npu_support/layer_check/graph_capture
run() { ASCEND_RT_VISIBLE_DEVICES=14 PYTHONPATH=$REPO/python:$PYTHONPATH $VENV/bin/python "$@"; }

# ① 部署形状：TP16 / page 64 / ctx 32768 / 16 并发 / bs 桶 1..16，layer 3(DSA)+4(KDA) 同一个图
run $D/cap_runner_layers.py --tp 16 --layers 3,4 --bs 1,2,4,8,12,16 \
    --nreq 17 --ctx 32768 --prefill 4095 --real-weights

# ② gold：只有 --tp 1 才跟 trace 可比（TP16 分片算的是 rank 0 的部分和）
#    需要一张**空卡**：TP1 的两层 MoE 要 ~30 GiB
run $D/cap_runner_layers.py --tp 1 --layers 3,4 --bs 1,2 --nreq 3 --ctx 512 \
    --real-weights --golden $ROOT/goldens/trace_128.pt

# ③ 反过来的顺序 + dense FFN 也在同一个图里（layer 2 = KDA+dense, layer 3 = DSA+MoE）
run $D/cap_runner_layers.py --tp 1 --layers 2,3 --bs 1,2 --nreq 3 --ctx 512 \
    --real-weights --golden $ROOT/goldens/trace_128.pt
```

## 三问怎么答的

- **cap** —— `NPUGraphRunner(model_runner)` 不抛异常，且 `backend._graphs` 里每个 bs 桶都在，
  `backend._pool` **只有一个** handle。
- **bake** —— 六个场景，每个都要求 replay 与 eager **逐位相同**，而且
  **连被改写的池张量也要逐位相同**：
  A 同输入 / B 换隐状态 / C **同时换 seq_lens** / D 打乱 `input_ids` /
  E raw_bs 不是桶（runner 补 padding）/ F 换一个桶（证明共池是安全的）。
- **gold** —— 从**被 replay 的图**里读回的张量，用 `harness.py` 的双参考法对
  `trace_128.pt`（HF 在 CPU 上 fp32 + bf16 跑出来的逐层隐状态）打分。
  `hidden_fp32[L-1]` 是第一层的输入，`hidden_fp32[L][127]` 是第 127 个 token 那一步
  decode 的参考输出（因果性：128-token prefill 的第 127 行 == 有 127 token 上下文时的 decode）。

## padding 判分的规则（重要）

padded replay 里，池子**本来就应该**动：每个 padding 行都自称是请求 0 / token 槽 0。
所以脚本不问「有没有动」，问「动的地方是不是**预留槽**」，而且预留槽的位置是从池对象
**读出来的**，不是写死的：

| 张量 | 允许动的下标 | 出处 |
|---|---|---|
| `kv_buffer` | token 槽 0 | `out_cache_loc` 的 padding policy 是 ZERO |
| `index_key_cache.buffer` | `kv.scratch_loc // page` | `NPUDSATokenToKVPool.scratch_loc` |
| `_compress_tail_{k,score}` | `kv._tail_scratch_row` | NPU 池多加的那一行 |
| `mamba_cache.{conv,temporal}` | slot 轴的 0 | `MambaSlotAllocator` 的 `free_slots = arange(1, size+1)` |

**真实请求因此不能占 0 号槽** —— 脚本把请求从 1 号开始排，理由和 `cap_kda.py` 一样。

## 实测结果（2026-08-29，本机）

**① 部署形状 TP16 / ctx 32768 / prefill 4095（> `index_topk=2048`，DSA 走稀疏）/ bs 桶 [1,2,4,8,12,16]：10/10 PASS**

- cap：6 个桶全部捕获，共用一个 pool handle
- bake：A–F **全部逐位相同**（输出 + 22 个可变池张量）
- padded replay（raw_bs 13 → 桶 16）：输出逐位相同；动了 5 个池张量，
  **每一个都只动在自己的预留槽**：
  `kv_buffer[0]`、`_compress_tail_k/score[18]`、`index_key_cache.buffer[8707]`、
  `mamba conv slot 0`

**② gold（TP1，layer 3+4，decode 第 127 个 token）：2/2 在预算内**

```
[ok] layer 3 output  err=3.082e-03  floor=8.208e-03  budget=1.642e-02  (0.19x)
[ok] layer 4 output  err=3.824e-03  floor=7.582e-03  budget=1.516e-02  (0.25x)
```

**③ gold（TP1，layer 2+3，KDA+dense 在前、DSA+MoE 在后）：2/2 在预算内**

```
[ok] layer 2 output  err=2.948e-03  floor=7.973e-03  budget=1.595e-02  (0.18x)
[ok] layer 3 output  err=3.302e-03  floor=8.208e-03  budget=1.642e-02  (0.20x)
```

⚠ **gold 只覆盖 dense 的 DSA**：trace 只有 128 个 token，`kv_len=128 < index_topk=2048`，
所以稀疏选择那一段没有被 gold 覆盖（bake 在 4096 上覆盖了它，但 bake 只说图与 eager 一致）。
要把 gold 推到稀疏区，需要一份更长的 CPU trace。

⚠ **所有耗时数字都不可作结论**：这台机器上另一个用户的 16 卡 DSv4 训练任务在反复起停。

## 顺带发现的两个真实缺口（都是走 runner 才碰得到的）

**（1）`get_attn_backend().forward_metadata` 在 hybrid 上不存在 —— 已修（NPU 树内）**

`kpool_indexer_npu.py:359` 和 `dsa_npu_indexer.py` 直接读
`get_attn_backend().forward_metadata.{block_tables,seq_lens,actual_seq_lengths_*}`。
非 hybrid 的昇腾模型（DSv3.2 / DSv4）里 `get_attn_backend()` **就是** `AscendAttnBackend`，
读得到；GLM 的顶层是 `AscendKDAHybridLinearAttnBackend`，**没有** `forward_metadata`，
于是第一个 DSA 层就 `AttributeError`。

修法：`AscendHybridLinearAttnBackend` 加一个 property（**带 setter**）代理到
`full_attn_backend`（`ascend_hybrid_linear_attn_backend.py`，NPU 专属目录）。

⚠ **自我纠正**：第一版写成了只读 property，理由是「indexer 只改 metadata 对象的字段、
从不重新绑定这个属性」。这个理由对 indexer 成立，但**漏了别的调用者** ——
`runner/eager_runner.py` 的 idle 分支会做 `attn_backend.forward_metadata = None`
来丢掉过期 metadata，只读 property 会把它变成 `AttributeError`。
补上 setter 后与 `ShortConvHybridAttnBackend.forward_metadata`
（`linear/inkling_sconv_backend.py`）**完全同形** —— 那里早就是这个写法，
说明这是既有先例，不是新发明。

**（2）`maybe_fuse_routed_scale_and_shared_add` 在非 CUDA 上 import 就炸 —— 只报告**

`DeepseekV2MoE.forward_normal`（GLM 的 MoE 就是它，`Glm5NextMoE is DeepseekV2MoE`）
每次 forward 结尾都调它，而它**第一句**就 `from ...quantization.expert_pack import
ExpertPackMoEMethod`，那个模块头部 `from sgl_kernel.quantization import ggml_moe_a8_vec`。
昇腾上没有 `sgl_kernel`，在任何 quant-method 判断之前就 `ModuleNotFoundError`。
共享路径，按仓库规矩**不改**，记在 `SHARED_CHANGES.md`；脚本里用
`runner_fixture.patch_shared_path_gaps()` 临时替成 `fused=False` 那条分支
（昇腾必然走的那条）绕过。

## 一个搭 fixture 的坑（不是 bug，但会静默毁掉 gold）

`install_shared_experts_fusion_decision(model_class, hf_config, quant_config)`
必须在建层**之前**调一次。不调的话 `is_shared_experts_fusion_disabled()` 退回
server_args 的意图（False），每个 MoE 层就会按「shared expert 融进第 288 号槽」来建，
而 weight loader 那边 `num_fused_shared_experts=0` 仍然保留 `mlp.shared_experts.*` 这个名字
—— 于是 shared expert 的权重**被静默丢掉，288 号槽是未初始化的内存**。
`Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason` 的注释
写的就是这件事。实测代价：gold 从 **0.19x 预算** 变成 **15x 预算**，而 cap/bake 全绿。
—— 这正是「图捕获最坏的失败方式」的同款：不报错、不崩、数字看着正常。
