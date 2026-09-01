# 从零复现：拉代码 → 建环境 → 转权重 → 起服务 → 验精度

给**第一次接触这个项目**的人。照着从上往下做，每一步都给了**期望看到什么**——
对不上就停在那一步，不要往下走。

环境搭建的细节不在这里，在 [`SETUP.md`](./SETUP.md)；本文只负责把整条链串起来，
并说清楚**每一步怎么算通过**。

**总代价**：磁盘约 950 GB、一台空闲的 Atlas A3（16 die）、首次约一天。

---

## 0. 先确认硬件对得上

```bash
python -c "import torch, torch_npu; print(torch.npu.get_device_name(0), torch.npu.device_count())"
# 期望：Ascend910_9362 16
```

⚠ **只能用这条认型号**。`npu-smi info` 对 A2 和 A3 都显示 `Ascend910`，
而所有 sgl-kernel-npu 包要选 **a3** 档，选错了会以难懂的方式失败。

⚠ **A3 没有 fp8**：连 `torch.zeros(4, dtype=torch.float8_e4m3fn, device="npu")` 都会
`aclnnInplaceZero failed, 161002`。任何要**实体化** fp8 张量的路径都走不通，
这不是 triton 的问题。索引缓存因此改存 bf16（PLAN §2.7）。

## 1. 拉代码

```bash
# ⚠ 私有 fork，地址按你自己的 remote 填。开源版本从 ktransformers-AK 的
#   third_party/sglang submodule 进，不需要单独 clone。
git clone <your-sglang-fork-url> sglang-dllm
cd sglang-dllm
git checkout glm53_dev
```

`glm53_dev` 是工作分支。GPU 参考实现在 tag `glm53-gpu-ref-033446bb`（本地 git 对象里已有，
不用联网），需要对照「GPU 上是怎么写的」时 `git show` 它。

## 2. 建环境

**照 [`SETUP.md`](./SETUP.md) 从 §0 做到 §7 的「算子可见性验收」通过。** 不要跳，
里面每一节都是踩出来的。三个最容易出事的点：

- **§1 网络规则**：代理只对 github 有效，`pip` / modelscope 必须直连。
  ⚠ 这台机器 `HTTP_PROXY=http://127.0.0.1:1056`，**连 `127.0.0.1:30003` 也会被它劫走**，
  代理回 503。连本机服务前先 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`
- **§8.2 装依赖必须带 constraints**：不带 `-c` 会给你拉来 torch 2.13.0 + 15 个 `nvidia-*`
- **§10 参考环境 `.venv-ref`** 是独立的第二个 venv（transformers 5.16.1 + CPU torch）。
  **绝不能装进 `.venv-glm53`**——sglang 钉的是 5.12.1。HF golden 只有 `.venv-ref` 出得来

做完之后 `source $ROOT/env.sh`，后面用 `npy` 代替 `python`。

## 3. 权重：FP8 → BF16

官方 checkpoint 是 FP8 blockwise，而 **A3 跑不了 fp8**，所以必须离线转成 BF16。

```bash
# 源：zai-org/GLM-5.3-Flash，revision c5b82b63e37b（已核实：71/71 文件 + 62/62 分片 size 一致）
npy docs/docs/glm53_npu_support/tools/fp8_to_bf16.py \
    --src /mnt/workspace/models/GLM-5.3-Flash \
    --dst /mnt/workspace/models/GLM-5.3-Flash-BF16
```

**期望**：62/62 shard，输出 **599 GB**。

⚠ **磁盘**：FP8 源 306 GB + BF16 599 GB = 905 GB。这一步之后盘会非常紧张。
源 shard 不会被自动删除；**确认 BF16 验过之后**再回收。

本项目的做法（2026-08-29 已执行）：**只删 62 个 `.safetensors`，保留那 28 MB 元数据**
（config / tokenizer / chat_template.jinja / index.json），这样 revision 溯源还在。
删之前核这三条：`ls *.safetensors | wc -l` 是 62；index 引用的 shard 一个不缺；
BF16 目录里 `config.json` / `tokenizer.json` / `chat_template.jinja` /
`generation_config.json` / `model.safetensors.index.json` 全都在 ——
**BF16 必须能脱离 FP8 目录独立存活**，删完就没有回头路。

## 4. 起服务

```bash
cp docs/docs/glm53_npu_support/launch_glm_bf16.sh.example $ROOT/run/launch_glm_bf16.sh
# 按需改 --model-path 与 PORT，然后：
$ROOT/run/launch_glm_bf16.sh
```

**期望**（日志里逐条对）：

| 看什么 | 期望值 |
|---|---|
| `Load weight end` | `mem usage=37.25 GB`（= 599/16，纯 TP16 无复制）|
| `Mamba Cache is allocated` | `max_mamba_cache_size` 等于 `--max-running-requests` |
| `KV Cache is allocated` | `#tokens: 1113600`（128 并发档）|
| `Capturing batches` | 桶数与 `--cuda-graph-max-bs-decode` 对应，几十秒内完成 |
| 最后一行 | `The server is fired up and ready to roll!` |

⚠ **启动时会有一条 `/freeze_gc` 的 Traceback，是无害的启动竞态**（服务还没监听端口就去
POST 自己）。除此之外日志里不该有 Traceback。

⚠ **BF16 必须独占整机**：TP8 每 die 要 74.9 GB，塞不进 64 GB，所以 BF16 只能 TP16。
⚠ **但 INT8 不用**：TP8 每 die 只要 38.20 GB（实测），
`ASCEND_RT_VISIBLE_DEVICES=8,...,15` + `--tp-size 8` 就能跑，剩下 8 张卡给别人。
日常验证走这条，别为了一次对拍去抢整机。

⚠ **抢机器的碰撞可能发生在你检查之后**（2026-08-30 实测撞到）：启动前
`npu-smi` 看着是空的，但另一个任务在你加载权重的这一两分钟里起来了，
于是你在 MoE 的 `create_weights` 处 OOM（16 个 rank 都报 `avail mem≈4.06 GB`）。
**看起来像显存不够，其实是有人插队。** 起服务前和真正开始加载权重前各看一眼。

⚠ **换配置重启前，等 `npu-smi` 自己说话，别看秒表**：kill 后 3 秒重起，
`bootstrap.py:339` 的「每卡空闲 ≥ 90%」检查**会放行**，然后在加载权重时炸 OOM
（`22.15 GiB already allocated; 525.39 MiB free`）——看起来像显存不够，其实是上一个进程没退干净。
16 个 die 全部回到约 3 GB 实测要 **2–3 分钟**：

```bash
until [ "$(npu-smi info | grep -oP '\d+(?=\s*/ 65536)' | awk '$1>6553{c++}END{print c+0}')" = 0 ]; do sleep 15; done
```

## 5. 验精度：三级，从秒到半小时

**完整的回归阶梯（六级、每级抓不到什么）在 [`REGRESSION.md`](./REGRESSION.md)。**
这里只给「新环境第一次验」要跑的三级。

### 5.1 冒烟（秒）

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -s http://127.0.0.1:30003/generate -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}'
```

**期望**：`"text"` 非空且接得上（`" Paris. ..."`）。200 但 `"text":""` 是失败。

⚠ **这一步只证明 45 层能串起来，不证明数值对。** 短 prompt 的
`seq_len < index_topk=2048`，DSA indexer **直接全选**，稀疏路径根本没走。
「Paris 答对了」是本项目最贵的教训之一。

### 5.2 logprob 对拍（秒，判据是**测出来的地板**）

```bash
G=$ROOT/goldens/logits
npy docs/docs/glm53_npu_support/tools/logit_check.py compare \
    --ref $G/ref_cpu_fp32.json --port 30003 --floor $G/floor_precision.json
```

**期望**：`8/8 在测出来的地板 x slack 2.0 之内`。参考基线最差是 0.91×。

**没有 `$ROOT/goldens/` 的话自己造**（fp32 那份约 8 分钟，只需一次）：

```bash
$ROOT/.venv-ref/bin/python tools/logit_check.py reference --streaming --dtype fp32 --out ref32.json
$ROOT/.venv-ref/bin/python tools/logit_check.py reference --streaming --dtype bf16 --out ref16.json
npy tools/logit_check.py compare --ref ref32.json --against ref16.json --emit-floor floor.json
npy tools/logit_check.py compare --ref ref32.json --port 30003 --floor floor.json
```

⚠ **`--streaming` 不是优化，是唯一可行的做法**：整模型 fp32 `from_pretrained` 在这台
1.8 TB 的机器上加载到 26% 就吃掉 1.26 TB 开始换页。逐层物化只留一层，峰值几十 GB。
⚠ **不要自己拍一个固定阈值。** 这个模型的地板是**离散的 MoE 路由差异**，随深度从
1.2e-2 涨到 1.8e-1；工具里曾经写死过 `<1e-2`，比实际低一个数量级还多。
**判定必须显式 `--floor` 传进来，工具不给默认阈值。**

### 5.3 GSM8K 全量（约 25 分钟，这是出口判据）

```bash
npy docs/docs/glm53_npu_support/tools/run_gsm8k.py --concurrency 128 --out gsm8k.json
```

**期望**：**97.5% 左右、stop rate 100.00%**。本项目实测 **97.35%**（1284/1319）。

⚠ **口径**：thinking 打开（这个 checkpoint 的 `chat_template.jinja` 结尾无条件接
`<|assistant|><think>`，没有开关可以传错）、temp 1.0 / top_p 0.95。
**不要用 sglang 自带的 `benchmark/gsm8k/bench_sglang.py`** ——那是 5-shot 贪心短输出，
和 97.50% 这个判据不可比。
⚠ **单轮噪声约 ±1pp**（1319 题、p≈0.97 的二项 SE 是 0.47pp）。96.5–98.4 都算一致；
掉到 95% 以下才是信号。**不需要跑三轮**，理由见 PLAN §P4.2。

---

## 6. 出问题时看哪里

| 症状 | 先看 |
|---|---|
| 起服务 OOM，但 `npu-smi` 说卡是空的 | 上一个进程没退干净，见 §4 的等待循环 |
| 连本机服务 503 | 代理劫持了 127.0.0.1，`unset http_proxy ...` |
| 精度数字「看着不对但说不清」 | **先测地板**（§5.2），没有地板的数字不能下判断 |
| 某个算子行为诡异 | PLAN §2.4「陷阱（能跑但算错 / 名实不符）」 |
| 改了共享路径不知道影响谁 | [`SHARED_CHANGES.md`](./SHARED_CHANGES.md) |
| 想知道某一层为什么这么写 | [`layer_check/`](./layer_check/) 里对应的 `check_*.py` |
