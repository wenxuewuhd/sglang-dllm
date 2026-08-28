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
