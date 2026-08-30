# 交接：在这台 A3 上从零跑到「精度对齐」

给**接着做 MXFP4 量化**的人。目标只有一个：**在隔离环境里，最快地把精度基线复现出来**，
这样你换了量化格式之后，有一个可信的东西可以对。

其余的（架构、算子结论、踩坑史）在 [`PLAN.md`](./PLAN.md) 和 [`RESUME.md`](./RESUME.md)，
**别先读它们**，先按本文跑通。

---

## 0. 先拉代码，建你自己的环境

**在你自己的目录下做**，不要在别人的 worktree 里改东西——这台机器是共用的。

下面用 `w12345` 当你的工号，**换成你自己的**。两个变量贯穿全文：

| 变量 | 是什么 | 例子 |
|---|---|---|
| `GLM_REPO` | 你 clone 出来的仓库 | `/mnt/workspace/w12345/glm53/sglang-dllm` |
| `GLM_ENV` | 你的环境目录（venv、算子包、启动脚本、golden 都在这）| `/mnt/workspace/w12345/glm53/env` |

⚠ **不是 `/root`，也不是任何系统目录** —— 就是你在 `/mnt/workspace/<你的工号>/` 下自己建的两个文件夹。

```bash
export GLM_BASE=/mnt/workspace/w12345/glm53     # <- 改成你的工号
mkdir -p $GLM_BASE && cd $GLM_BASE

git clone git@github.com:wenxuewuhd/sglang-dllm.git
cd sglang-dllm && git checkout glm53_dev

export GLM_REPO=$GLM_BASE/sglang-dllm
export GLM_ENV=$GLM_BASE/env
mkdir -p $GLM_ENV/run $GLM_ENV/goldens
```

`glm53_dev` 是工作分支。GPU 参考实现在 tag `glm53-gpu-ref-033446bb`，
需要对照「GPU 上是怎么写的」时 `git show` 它。

```bash
git log --oneline -3        # 最上面应该是 docs(npu): ... handoff for the MXFP4 work 一带
```

### 哪些东西该复用，哪些必须自己建

| | 怎么办 | 为什么 |
|---|---|---|
| **代码** | 自己 clone | 你要改它 |
| **venv / 算子包 / `$GLM_ENV`** | **自己建**（见 §1）| 环境变量、`kernel_meta` 缓存、启动脚本都会互相干扰 |
| **模型权重** | **只读复用**，别复制 | BF16 599 GB + W8A8 306 GB，盘上没有第二份的余量 |
| **golden 基线** | 复制一份或自己造 | 只读参考数据，复制最省事 |

```bash
# 权重：只读引用，路径直接写进启动脚本
#   /mnt/workspace/models/GLM-5.3-Flash-BF16     599 GB
#   /mnt/workspace/models/GLM-5.3-Flash-W8A8     306 GB
# golden：复制（约几十 MB）
cp -r /mnt/workspace/y00359136/work/glm53_dev/env/goldens $GLM_ENV/goldens
```

⚠ **那个源目录属于上一个 session，可能已经不在了。** 拿不到也没关系，
自己造一份，做法在 §3.2 末尾（要先建好环境，所以放在那里）。

⚠ **磁盘**：`/mnt/workspace` 余量很薄（交接时约 23 GB）。
**你的 MXFP4 权重要先算好占多少再动手**，中途撑不住只能去动 BF16 那 599 GB。

### 机器

Atlas A3，**16 个 die**，每个 64 GB HBM。⛔ **共用，而且有第三方。**

```bash
npu-smi info      # 起服务前看一眼，真正开始加载权重前再看一眼
```

- **`npu-smi` 才是真相，别人的口头承诺不是。** 交接当天就撞到：检查显示 1 张 die 忙，
  没停下来核实就起服务，结果是第三方的训练任务占了 8 张，服务起不来
- ⛔ **只杀自己端口的进程**：`pkill -f -- "[-]-port 30023"`（方括号防自匹配）。
  **绝不 `pkill -f sglang`** —— 同一个 OS 用户下会连别人一起杀
- **die 编号看 Phy-ID，不是 NPU 号**。一张 NPU 卡有 2 个 die，
  `ASCEND_RT_VISIBLE_DEVICES` 用的是 Phy-ID（0..15）。
  空闲的 die **不必连号**，凑够数就行（`0,9,10,11,12,13,14,15` 是可用的 TP8）
- 停服务后 HBM 要 **2–3 分钟**才回落，**等它回到约 3 GB 再起下一个**，否则在加载权重处 OOM

```bash
# 等自己那几个 die 回落（只看自己的，别人的永远不会归零）
until [ "$(npu-smi info | grep -oP '\d+(?=\s*/ 65536)' | awk '$1>6553{c++}END{print c+0}')" = 0 ]; do sleep 15; done
```

---

## 1. 建环境

照 [`SETUP.md`](./SETUP.md) 从 §0 做到 §7 的「算子可见性验收」通过。
`env.sh.example` 复制成 `$GLM_ENV/env.sh`，把里面的 `ROOT=` 改成你的 `GLM_ENV` 路径、
`REPO=` 改成你的 `GLM_REPO` 路径（脚本内部仍叫 `ROOT`/`REPO`，不用改这两个名字）。

```bash
source $GLM_ENV/env.sh          # 之后用 npy 代替 python
npy -c "import torch, torch_npu; print(torch.npu.get_device_name(0), torch.npu.device_count())"
# 期望：Ascend910_9362 16
```

⚠ **只能用这条认型号** —— `npu-smi` 对 A2 和 A3 都显示 `Ascend910`，
而所有 sgl-kernel-npu 包要选 **a3** 档，选错了会以难懂的方式失败。

两个最容易出事的点：

- 装依赖必须带 `-c constraints`，否则会拉来 torch 2.13 + 一堆 `nvidia-*`
- 参考环境 `$GLM_ENV/.venv-ref` 是**第二个独立 venv**（transformers 5.16.1 + CPU torch）。
  **绝不能装进 `.venv-glm53`**——sglang 钉的是 5.12.1。CPU golden 只有 `.venv-ref` 出得来

⚠ **A3 硬件不支持 fp8**（连分配一个 fp8 张量都会报错）。
**所以 MXFP4 大概率要写仿真算子** —— 用 bf16/int8 存储 + 反量化来模拟，
而不是指望硬件原生支持。开工前先确认这条路怎么走。

---

## 2. 起服务

两版都是现成脚本，`$GLM_ENV/run/` 下。**参数不要凭感觉改**，每一条都有理由（脚本注释里写了）。

### BF16（TP16，**要独占整机**）

```bash
cp $GLM_REPO/docs/docs/glm53_npu_support/launch_glm_bf16.sh.example $GLM_ENV/run/launch_glm_bf16.sh
# 改里面的 --model-path 和 PORT（用你自己的端口，别和别人撞）
$GLM_ENV/run/launch_glm_bf16.sh
```

BF16 每 die 要 37.25 GB，**TP8 塞不进 64 GB，所以只能 TP16**。

### INT8 W8A8（TP8，**只用 8 张卡，日常验证走这个**）

```bash
cp $GLM_REPO/docs/docs/glm53_npu_support/launch_glm_w8a8_tp8.sh.example $GLM_ENV/run/launch_glm_w8a8_tp8.sh
# 改三处：ASCEND_RT_VISIBLE_DEVICES（你抢到的 8 个 die 的 Phy-ID）、env.sh 路径、PORT
$GLM_ENV/run/launch_glm_w8a8_tp8.sh
```

每 die 38.20 GB，剩 22.88 GB。**这是日常迭代该用的构型**——不用抢整机，
而且和 TP16 的精度结论一致（长上下文那轮两个构型都验过）。

**起来的标志**（日志里逐条对）：

| 看什么 | 期望 |
|---|---|
| `Load weight end` | BF16 `mem usage=37.25 GB` / INT8 `38.20 GB` |
| `KV Cache is allocated` | 有这行就对了；#tokens 随并发和 mem-fraction 变 |
| 最后一行 | `The server is fired up and ready to roll!` |

⚠ 启动时有一条 `/freeze_gc` 的 Traceback 是**无害的启动竞态**。除此之外不该有 Traceback。

⚠ **`--page-size` 必须是 64**（DSA pool 有 `assert page_size == 64`）。
⚠ **`--disable-radix-cache` 现在必须留着**：`causal_conv1d_fn_npu` 在混合
`has_initial_state` 时会写坏 conv state（PLAN P6.2）。**开着它跑出来的精度数字不可信。**

---

## 3. 精度怎么对（三级，从秒到十几分钟）

### 3.1 冒烟（秒）——**它只证明 45 层能串起来**

```bash
export PORT=30023          # 你启动脚本里用的那个
curl -s http://127.0.0.1:$PORT/generate -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}'
```

期望 `" Paris. ..."`。⚠ **短 prompt 的 `seq_len < index_topk=2048`，DSA indexer 直接全选，
稀疏路径根本没走。**「Paris 答对了」是这个项目最贵的教训之一，**不要拿它当精度通过**。

### 3.2 logprob 对拍（秒）——**判据是测出来的地板，不是拍的阈值**

```bash
G=$GLM_ENV/goldens/logits
npy $GLM_REPO/docs/docs/glm53_npu_support/tools/logit_check.py compare \
    --ref $G/ref_cpu_fp32.json --port $PORT --floor $G/floor_precision.json
```

期望 **`8/8 在测出来的地板 x slack 2.0 之内`**（参考基线最差 0.91×）。

⚠ **不要自己拍一个固定阈值。** 这个模型的地板是**离散的 MoE 路由差异**，
随深度从 1.2e-2 涨到 1.8e-1。工具**不给默认阈值**，必须显式 `--floor` 传进来。

⚠ **地板是每个部署自己的，不能跨构型搬。** 你换成 MXFP4 之后，
`floor_precision.json` 里那份是 BF16 时代的，**只能当参考不能当判据** —— 要自己造一份。

**自己造 golden 和地板**（CPU fp32 参考约 8 分钟，只需一次；需要 §1 的 `.venv-ref`）：

```bash
cd $GLM_REPO/docs/docs/glm53_npu_support
mkdir -p $GLM_ENV/goldens/logits
$GLM_ENV/.venv-ref/bin/python tools/logit_check.py reference --streaming --dtype fp32 \
    --out $GLM_ENV/goldens/logits/ref_cpu_fp32.json
$GLM_ENV/.venv-ref/bin/python tools/logit_check.py reference --streaming --dtype bf16 \
    --out /tmp/ref16.json
npy tools/logit_check.py compare --ref $GLM_ENV/goldens/logits/ref_cpu_fp32.json \
    --against /tmp/ref16.json --emit-floor $GLM_ENV/goldens/logits/floor_precision.json
```

⚠ **`--streaming` 不是优化，是唯一可行的做法**：整模型 fp32 `from_pretrained`
在这台 1.8 TB 的机器上加载到 26% 就吃掉 1.26 TB 开始换页。逐层物化只留一层，峰值几十 GB。

### 3.3 GSM8K 全量（约 15 分钟）——**这是出口判据**

```bash
npy $GLM_REPO/docs/docs/glm53_npu_support/tools/run_gsm8k.py \
    --concurrency 128 --port $PORT --out gsm8k.json
```

**已知基线**（全部 1319 题、thinking 开、temp 1.0 / top_p 0.95）：

| 构型 | GSM8K |
|---|---|
| BF16 TP16 | 97.35% |
| **INT8 W8A8 TP8** | **97.42% / 97.65% / 97.42% / 97.35%**（四轮，本文写作时的最后一轮是 `e291e54dea` 那个节点）|

⚠ **单轮噪声约 ±0.47pp**（1319 题、p≈0.97 的二项 SE）。**96.5–98.4 都算一致，
掉到 95% 以下才是信号。** 不需要跑三轮 —— 上面那四轮跨越了三次代码改动，
最大差 0.30pp，**这就是这个判据在这个部署上的实际分辨率**。

⚠ **stop rate 也要看**，四轮都是 100.00%。如果它掉下来，说明有生成不收敛，
那和答对率是两个不同的问题。

⚠ **不要用 sglang 自带的 `benchmark/gsm8k/bench_sglang.py`**——那是 5-shot 贪心短输出，
和上面这些数字不可比。

---

## 4. 你做 MXFP4 时最该知道的四件事

**① 量化脚本有现成的，而且它的「量什么」不是猜的。**
`tools/bf16_to_int8_ct.py` 从**厂商 FP8 checkpoint 的 index** 反推该量化的张量集合
（37338 个），而不是用模块名模式去匹配。**照这个做**——模式匹配会变成第二个真值来源然后漂移。
⚠ 厂商**没有**量化 KDA 的 q/k/v/o_proj，也没量 indexer、norm、router、embedding、lm_head。

**② 判据的分辨率取决于你的构型。**
单卡线在 16 专家的裁剪 checkpoint 上量到的地板高达 `1.2e-01`，
**在那个部署上端到端数值判据基本没有分辨率**——他们报的是「判不出」，不是「通过」。
**288 专家的真 checkpoint 才有分辨率。** 你要判 MXFP4 的精度，用真 checkpoint。

**③ 「逐位相同」只在同 batch 宽度下才是合法判据。**
实测：**完全不开投机解码，只把 decode 的 batch 宽度从 1 改成 8，贪心输出就分叉**
（一条提示在第 16 个 token，另一条在**第 0 个**）。所以任何改变 batch 形状的东西
都不能用「输出相同」去判。**同宽度下要求逐位是对的**，项目里那些 `0.000e+00` 都是同宽度对拍。

**④ A/B 之前先拿同一个 build 对自己跑一次。**
本项目实测：8 条并发请求的 prefill 分组**跨运行不确定**，同一个 build 对自己跑
`max|dlp|` 能到 **1.867** —— 拿这种测试去判一个改动，信号从来没高过噪声。
**要确定性就绕开服务**：`probe/` 下的探针直接调算子。

---

## 5. 出问题看哪里

| 症状 | 先看 |
|---|---|
| 起服务 OOM 但 `npu-smi` 说卡是空的 | 上一个进程没退干净，等 HBM 回落到 3 GB |
| 连本机服务 503 | **你的终端里设了 `HTTP_PROXY`**（agent 环境常有，普通终端一般没有）—— 它连 `127.0.0.1` 也劫持。`unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY` 即可；`env.sh` 已经做了这件事 |
| 精度数字「看着不对但说不清」 | **先测地板**（§3.2），没有地板的数字不能下判断 |
| 某个算子行为诡异 | `PLAN.md` §2.4「陷阱（能跑但算错 / 名实不符）」 |
| 平台判断没生效 | ⛔ **`tensor.is_cuda` 在这台机器上是 True**（`transfer_to_npu` 的副作用），用 `device.type`。见 PLAN §2.4 |
| 改了共享路径不知道影响谁 | [`SHARED_CHANGES.md`](./SHARED_CHANGES.md) |

