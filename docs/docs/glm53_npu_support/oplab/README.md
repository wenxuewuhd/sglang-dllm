# oplab — 两个可独立运行的单层算子性能用例

给昇腾算子优化团队。覆盖 GLM-5.3-Flash 单卡 INT8 decode 里 **KDA** 和 **DSA**
两个层族的算子序列，**不起服务、不加载 checkpoint、不 import 模型代码**。

| 文件 | 内容 |
|---|---|
| `bench_kda_layer.py` | KDA（线性注意力）单层，入图，扫序列长度、扫并发槽位 |
| `bench_dsa_layer.py` | DSA（稀疏注意力）单层，入图，扫序列长度 |
| `baseline_kda.txt` / `baseline_dsa.txt` | 在本机一张空闲 A3 die 上的参考输出 |

---

## 00. 从零搭一个只跑这两个用例的环境

**写给不在这台机器上、也没有我们那套 `env/env.sh` 的团队。**
下面每一条命令都在一个**全新的 `python -m venv`** 里真敲过一遍（2026-08-31，
本机 die 2），失败过的步骤也照实写在 §00.7。已有环境的人直接看 §1。

### 00.1 硬性前提 vs「我们验过的版本」

这两件事不一样，分开写：

| | 是不是硬要求 | 说明 |
|---|---|---|
| **Ascend A3（`Ascend910_9362`）** | **硬**（对数字） | 用例本身在任何 torch_npu 能跑的卡上都能跑完，但 §0 的微秒数和 `baseline_*.txt` 是这张卡的。换卡数会变，见 §00.6 |
| **CANN toolkit** | **硬** | 必须装好并 `source .../set_env.sh`。`npu_format_cast`（DSA 用例）会**懒加载 CANN 的 tbe**，缺了当场报 `error code 500001` |
| **驱动 / HDK** | **硬**（下限） | 要能带起你装的 torch_npu |
| **Python 3.12** | 软 | 只是因为我们手上的 wheel 是 `cp312`。换版本要换 wheel |
| **`torch_npu.contrib.transfer_to_npu`** | **硬**（DSA） | 没有它 `allow_internal_format` 这个属性都不存在，INT8 权重转 NZ 变成静默 no-op，`o_proj` 103 µs 而不是 58 µs。见 §7 第 10 条 |

**我们验过的具体版本**（不是要求，是「这套组合确实跑通了」）：

| | 版本 |
|---|---|
| 卡 | `Ascend910_9362`（A3），单 die |
| 驱动 | `Version=25.5.5`，`ascendhal 7.35.23`（`/usr/local/Ascend/driver/version.info`）；`npu-smi 25.5.5` |
| CANN toolkit | **9.2.0**（`innerversion V100R001C11B134`），装在 `/home/developer/Ascend/cann-9.2.0` |
| torch | **2.10.0**（`import torch` 打印 `2.10.0+cpu` —— **这是对的**，torch_npu 是 out-of-tree 后端，不要去找 `+npu` 的包） |
| torch_npu | **2.10.0.post4** |
| triton-ascend | **3.2.2**（它提供的 `triton` 模块自称 `3.2.0`） |
| sgl_kernel_npu | **2026.6.1** |
| Python | 3.12.9 |

⚠ **不需要**的东西，实测确认：本章的 venv 里**没有** sglang 的 `env.sh`、
**没有** `opp_custom/vendors/{customize,custom_transformer}`（那两套自定义 opp
包是整网 INT8 里别的算子用的）、**没有** transformers / vllm / sglang 的服务侧依赖。
两个用例在只 source 了 CANN `set_env.sh` 的环境里跑完。

### 00.2 最小依赖集

用例实际 import 的东西，逐条核对过：

| 来源 | 提供什么 |
|---|---|
| `torch` + `torch_npu` | 设备、`torch.ops.npu.*` 的内置算子、`npu_format_cast`、NPUGraph |
| `torch_npu.contrib.transfer_to_npu` | `torch.npu.config.allow_internal_format` 开关（**只有 DSA 用例 import**） |
| `sgl_kernel_npu` | `torch.ops.npu.causal_conv1d`（KDA）、`torch.ops.npu.batch_matmul_transpose`（DSA） |
| `triton`（triton-ascend） | 三个 Triton kernel 的编译执行 |
| **4 个 sglang 模块** | 见下表 |

| 用例 | sglang 模块 | 取什么 |
|---|---|---|
| KDA | `sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py` | `fused_sigmoid_gating_delta_rule_update` |
| KDA | `sglang/kernels/ops/attention/fla/fused_norm_gate.py` | `layer_norm_gated_fwd` |
| DSA | `sglang/srt/hardware_backend/npu/attention/kpool_indexer_npu.py` | `compress_pool_bf16`、`hadamard_transform_npu` |
| DSA | `sglang/srt/layers/attention/dsa/kpool_fp8_index.py` | `expand_pooled_groups_to_topk`、`append_kpool_tail_to_topk` |

### 00.3 ⚠ 那 4 个模块**不需要**整棵 sglang 树 —— 两条路都实测过

**结论：把 4 个 `.py` 连同 13 个空 `__init__.py` 和一个 17 行的
`sglang.srt.utils` 垫片拷出来就能跑，两个用例都跑通，算子清单与整棵树逐组相同。**

两条路我都在同一个干净 venv 里跑到出数，不是推断：

| | 额外要装的 pip 包 | 拷贝的文件 | KDA `--sections layer --reps 3` | DSA 同上 |
|---|---|---|---|---|
| **A. 整棵树上 `PYTHONPATH`** | **18 个顶层 + 传递依赖 = 30 个 dist-info，80 MB**（`orjson pybase64 requests urllib3 idna certifi packaging pillow starlette anyio torchvision==0.25.0 tqdm IPython traitlets pygments pydantic aiohttp msgspec`） | 0 | **283.06 µs/层** | **453.16 µs/层** |
| **B. 只拷这 4 个模块** | **0** | 4 个 `.py`（3437 行）+ 13 个空 `__init__.py` + 17 行垫片，共 188 KB | **289.26 µs/层** | **452.16 µs/层** |

四个数是**同一次会话里连着跑的 A/B**（§7 第 13 条：跨 run 比没有意义），
同一张空闲 die。两条路的**算子清单逐组相同**（同样 9 组 / 99 组，同样的 shape
和每层次数），µs 差落在噪声里 —— 拷贝不改变任何东西。

**为什么 A 那么贵**：`sglang/__init__.py` 会跑起来，它 import
`sglang.srt.utils.hf_transformers_patches` → `sglang.srt.utils.common`（`import orjson` 在第 82 行），
再 import `sglang.lang.api` → `pydantic`。这 18 个包**没有一个**被那 4 个模块用到，
纯粹是包 `__init__` 的过路费。

**B 具体怎么拷。** 关键是**四个 `__init__.py` 必须换成空文件**——
`sglang/`、`sglang/kernels/`、`sglang/kernels/ops/`、`sglang/kernels/ops/attention/`
在真树里是有内容的（105 / 76 / 45 / 151 行），照拷就把整棵树的依赖又拉回来了；
`sglang/srt/**` 那几层在真树里本来就没有 `__init__.py`（namespace package）。

```bash
SRC=<sglang-tree>/python          # 见 §00.4 的 commit
DST=<你的目录>/standalone
for d in sglang sglang/kernels sglang/kernels/ops sglang/kernels/ops/attention \
         sglang/kernels/ops/attention/fla sglang/srt sglang/srt/utils \
         sglang/srt/hardware_backend sglang/srt/hardware_backend/npu \
         sglang/srt/hardware_backend/npu/attention \
         sglang/srt/layers sglang/srt/layers/attention sglang/srt/layers/attention/dsa; do
  mkdir -p "$DST/$d"; : > "$DST/$d/__init__.py"        # 空文件，不是拷贝
done
cp $SRC/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py $DST/sglang/kernels/ops/attention/fla/
cp $SRC/sglang/kernels/ops/attention/fla/fused_norm_gate.py                $DST/sglang/kernels/ops/attention/fla/
cp $SRC/sglang/srt/hardware_backend/npu/attention/kpool_indexer_npu.py     $DST/sglang/srt/hardware_backend/npu/attention/
cp $SRC/sglang/srt/layers/attention/dsa/kpool_fp8_index.py                 $DST/sglang/srt/layers/attention/dsa/
```

四个文件里**唯一**一处跨树的顶层 import 是 `fused_norm_gate.py` 的
`from sglang.srt.utils import cdiv, cpu_has_amx_support, is_cpu, is_npu, next_power_of_2`。
把 `$DST/sglang/srt/utils/__init__.py` 写成：

```python
"""Five-function stand-in for sglang.srt.utils, for the oplab benches only."""
import os, platform, torch

def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()

def is_cpu() -> bool:
    return os.getenv("SGLANG_USE_CPU_ENGINE", "0") == "1" and \
           platform.machine().lower() in ("x86_64", "aarch64", "arm64")

def cpu_has_amx_support() -> bool:
    return False          # AMX 是 Intel 的，这台 aarch64 上永远 False

def cdiv(a: int, b: int) -> int:
    return -(a // -b)

def next_power_of_2(n: int):
    return 1 << (n - 1).bit_length() if n > 0 else 1
```

`cdiv` / `next_power_of_2` 与 `srt/utils/common.py` 逐字相同；`is_npu` / `is_cpu`
去掉了服务侧才有意义的分支；`cpu_has_amx_support` 只在 `is_cpu() and ...` 里被短路调用。

另外三个文件里指向树深处的 import（`runtime_context`、`forward_context`、
`sgl_kernel.fast_topk_v2`、`dp_attention` 等）**全都是函数体内的懒 import**，
在这两个用例走的那 6 个函数里一个都不会执行 —— 这也是拷贝可行的原因。

**如果你更愿意走 A（整棵树）**：树的版本是
`wt-int8-singlecard` 分支 commit `6fb999ca3df1c0954b2db717b097b3b7704c94ae`
（2026-08-31，`docs/docs/glm53_npu_support/oplab/` 就在这棵树里）。
A 也确实跑通了，只是要多装 30 个包。

### 00.4 一步一步装（我实际敲的命令，6 步）

用的临时目录是 `/var/tmp/glm53/`（本机 `/mnt/workspace` 只剩 17 GB，别往那儿装）。

```bash
# 0) 三个 wheel 先备好。torch / torch_npu 见 §00.1 的版本；
#    triton-ascend 和 sgl_kernel_npu 见 §00.7 第 1 条（都不在公开 PyPI 上）
W=${GLM53_ROOT}/env/wheels     # torch, torch_npu
P=${GLM53_ROOT}/env/pkg        # sgl_kernel_npu
TA=/var/tmp/glm53/wheels                                 # triton_ascend
IDX="-i https://repo.huaweicloud.com/repository/pypi/simple"
DIR=/var/tmp/glm53/cleanenv/.venv-oplab

# 1) 干净 venv（不要复用项目的 .venv-glm53）
/opt/buildtools/python-3.12.9/bin/python3 -m venv $DIR
$DIR/bin/pip install $IDX --upgrade pip

# 2) torch + torch_npu，成对装，版本必须配套
$DIR/bin/pip install $IDX \
    $W/torch-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl \
    $W/torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_aarch64.whl

# 3) triton-ascend —— 必须 --no-deps，理由见 §00.7 第 2 条
$DIR/bin/pip install $IDX --no-deps \
    $TA/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl

# 4) AOT 算子：causal_conv1d / batch_matmul_transpose
$DIR/bin/pip install $IDX $P/sgl_kernel_npu-2026.6.1-cp312-cp312-linux_aarch64.whl

# 5) 那些「没人声明但运行时要」的包，见 §00.7 第 3、4 条
#    scipy 必须钉 1.13.1
$DIR/bin/pip install $IDX 'numpy==1.26.4' 'scipy==1.13.1' pyyaml pybind11 decorator attrs psutil
```

装完 `pip list` 一共 **21 个包**（含 pip / setuptools），2.0 GB：

```
attrs 26.1.0        filelock 3.32.4   fsspec 2026.7.0   Jinja2 3.1.6
MarkupSafe 3.0.3    mpmath 1.3.0      networkx 3.6.1    numpy 1.26.4
decorator 5.3.1     psutil 7.2.2      pybind11 3.1.0    PyYAML 6.0.3
scipy 1.13.1        sympy 1.14.0      typing_extensions 4.16.0
sgl_kernel_npu 2026.6.1   torch 2.10.0   torch_npu 2.10.0.post4   triton_ascend 3.2.2
pip 26.2.1          setuptools 84.0.0
```

跑用例的环境（**注意 `PYTHONPATH` 用 `:$PYTHONPATH` 追加**）：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset LD_PRELOAD                       # 系统预置的 libgomp 会干扰 torch_npu
source /home/developer/Ascend/ascend-toolkit/set_env.sh    # 你的 CANN 路径
export PATH=/var/tmp/glm53/cleanenv/.venv-oplab/bin:$PATH
export PYTHONPATH=/var/tmp/glm53/standalone:$PYTHONPATH     # §00.3 的 B 路
                                       # 走 A 路就换成 <tree>/python
mkdir -p /var/tmp/glm53/oplab-run && cd /var/tmp/glm53/oplab-run   # 见 §7 第 14 条
export ASCEND_RT_VISIBLE_DEVICES=2     # 换成你那张空闲的 die
```

⚠⚠ **`PYTHONPATH` 只能追加，不能覆盖。** `set_env.sh` 往里塞了
`.../cann-9.2.0/python/site-packages` 和 `.../opp/built-in/op_impl/ai_core/tbe`，
写成 `PYTHONPATH=<tree>/python` 会把它们挤掉，然后 **在算子编译时**才炸出
`error code 500001` + `No module named 'tbe'` —— 错误信息里没有一个字提到
`PYTHONPATH`。README §1 也记了这一条。

### 00.5 验证清单

**（1）import 能通** —— 30 秒，不碰卡：

```bash
python -c "
import torch, torch_npu, sgl_kernel_npu, triton
print(torch.__version__, torch_npu.__version__, triton.__version__)
print(hasattr(torch.ops.npu,'causal_conv1d'), hasattr(torch.ops.npu,'batch_matmul_transpose'))
from sglang.kernels.ops.attention.fla.fused_norm_gate import layer_norm_gated_fwd
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
from sglang.srt.hardware_backend.npu.attention.kpool_indexer_npu import compress_pool_bf16, hadamard_transform_npu
from sglang.srt.layers.attention.dsa.kpool_fp8_index import append_kpool_tail_to_topk, expand_pooled_groups_to_topk
print('ALL FOUR OK')"
```

期望：`2.10.0+cpu 2.10.0.post4 3.2.0` / `True True` / `ALL FOUR OK`。

**（2）最小 smoke** —— 各 1–2 分钟：

```bash
python bench_kda_layer.py --sections layer --reps 3
python bench_dsa_layer.py --sections layer --reps 3
```

⚠ KDA 能过而 DSA 炸 `error code 500001`，是**正常的分辨点**：只有 DSA 用
`npu_format_cast`，它才会去懒加载 CANN 的 tbe。看 §00.7 第 3 条。

**（3）跑满 30 次，看数** ——`--sections layer` 不带 `--reps`：

| | 干净空闲 die 上应该落在 | 本次干净 venv 实测（die 2，30 reps） |
|---|---|---|
| KDA 单层 p50 | **285–300 µs** | **290.94 µs**（×34 = 9.892 ms，−5.0%） |
| DSA 单层 p50 | **440–470 µs** | **452.81 µs**（×11 = 4.981 ms，−7.7%） |

单个算子的锚点（同一次运行）：`MatMulV2 "1,4096;24896,4096"` 166.7 µs、
`MatMulV2 "1,8192;4096,8192"` 55.1 µs、`SparseFlashAttention` ~32.6 µs。

**（4）⚠ 真正的判据：`regress_against_network.py` 的清单检查。**
时间只是旁证，清单才是判据（理由见 §5.2、§7 第 15 条）：

```bash
PK=$(ls -dt /var/tmp/glm53/oplab/kda/layer/*ascend_pt | head -1)
PD=$(ls -dt /var/tmp/glm53/oplab/dsa/layer/*ascend_pt | head -1)
python regress_against_network.py --family KDA --profile "$PK" --steps 30
python regress_against_network.py --family DSA --profile "$PD" --steps 30
```

在干净 venv 里实测通过，**和 §0 记录的状态逐条一致**：

| | 组数 | 缺 | 多 | TOTAL ref/got |
|---|---|---|---|---|
| KDA | 9 组全部对上（op、shape、每层次数） | **0** | **0** | 306.1 / 291.1 = 0.95 |
| DSA | — | **1**（`Cast "1,1,1"`，1.3 µs） | **6**（`ClipByValueV2 "1;;"`、`FloorMod "1;1"`、`IndexCheck "2;1;1"`、`BroadcastTo ";1"`、`Sub "1;1"`、`Equal "1;"`） | 490.4 / 453.1 = 0.92 |

**这两行就是「环境搭对了」的判据。** 缺/多的条目数变了 —— 尤其是 KDA 不再是
0/0 —— 说明环境有问题（多半是某个 kernel 没编出来，或者 NZ 没生效），
而不是卡慢。TOTAL 那一列的比值变了只说明卡不同，见 §00.6。

### 00.6 换机器之后：哪些数会变，哪些不会

| | 跟机器走吗 | 说明 |
|---|---|---|
| **绝对微秒数**（`p50`、`×N 层`、`baseline_*.txt`） | **会变** | 跟卡型号、CANN / 驱动版本、die 是否空闲都有关。§7 第 9 条：die 被占会整体膨胀 ~1.7× 且不报错；§7 第 13 条：**同一张空闲 die 跨 run 单个小算子能差 34%** |
| **算子清单**（哪些 op、什么输入 shape、每层几次） | **不变** | 这是模型和这段代码决定的，不是硬件。`regress_against_network.py` 量的就是这个 |
| **kernel 总个数**（KDA 9/层、DSA 99/层） | **不变** | 同上 |
| **`p50/ref` 那一列的比值** | **会变** | 分母 `ref` 是我们这台机器的整网实测 |

⚠⚠ **所以：`reference_inventory_cfgI.json` 换机器后仍然是「清单」的标靶，
但不再是「时间」的标靶。** 别把我们的 289 µs / 445 µs 当成你们机器上的目标。
换机器后正确的做法是：

1. 先用清单检查确认**跑的是同一件事**（KDA 0 缺 0 多，DSA 1 缺 6 多）；
2. 然后在**你们自己的机器上**跑一次，把那次的数当作你们的基线；
3. 优化的判据是**同一次 run 里的 A/B**，不是「今天的数比 README 里的数小」
   （§7 第 13 条）。

`--ref-seq` / `--context-len` 这些参数**不要动**：`p50/ref` 里的 `ref` 是
按 cfgI 的 shape 采的，参数一动清单就对不上了（§7 第 9 条第一版误报就是这么来的）。

### 00.7 装的时候实际踩到的 5 个坑

按踩到的顺序，都是在干净 venv 里真炸过的：

1. **`triton-ascend` 和 `sgl_kernel_npu` 都不在公开 PyPI 上。**
   实测 `pip index versions triton-ascend` 在 pypi.org、
   `repo.huaweicloud.com`、`mirrors.aliyun.com`、`pypi.tuna.tsinghua.edu.cn`
   四个源上**全都是 "No matching distribution found"**。必须自己拿 wheel
   （triton-ascend 见 `https://gitcode.com/Ascend/triton-ascend/`；
   `sgl_kernel_npu` 是昇腾侧发的包，本机在 `env/pkg/` 下）。
   ⚠ 另外：机器上散落的 `/tmp/pip-unpack-*/triton_ascend-*.whl` **全是断掉的
   半截下载**（1 MB / 3 MB / 17 MB / 23 MB / 28 MB，完整的是 **270 MB**），
   `pip` 只说一句 `Wheel ... is invalid`。拿到 wheel 先 `python -c
   "import zipfile;zipfile.ZipFile('...')"` 验一下。

2. **`pip install triton_ascend...whl`（不带 `--no-deps`）会把自己装坏。**
   它的 metadata 写着 `Requires-Dist: triton==3.5.0`，pip 就从 PyPI 装了上游
   Triton，把 `site-packages/triton/` 覆盖掉，报
   `ModuleNotFoundError: No module named 'triton._C.libtriton.ascend'`。
   **只能 `--no-deps`**，然后手工补 `numpy` / `pybind11` /（下面的）`scipy==1.13.1`。
   同理，`pip install torchvision` 会顺手把 torch 升到 2.13.0，torch_npu 立刻变成
   `undefined symbol: _ZN5torch8autograd10deleteNodeEPNS0_4NodeE` ——
   **凡是可能碰 torch 的包都要 `--no-deps` 或钉版本**（`torchvision==0.25.0`）。

3. **DSA 用例炸 `error code 500001`，真因藏在报错的第 20 行。**
   ```
   RuntimeError: SetPrecisionMode:.../LazyInitAclops.cpp:223 ... error code is 500001
   ...
   Environment_Error_Import_Python_Module_Failed(EC0010): Failed to import Python
   module ModuleNotFoundError: No module named 'decorator'.
   ```
   `npu_format_cast` 会**懒加载 CANN 的 tbe**，而 tbe 是一堆 Python，缺哪个包就
   在这里炸。实测按顺序缺了 **`decorator` → `scipy` → `attrs` → `psutil`** 四个，
   每次只报一个，得装一个跑一次。
   **KDA 用例完全不受影响**（它不 `npu_format_cast`），所以「KDA 过、DSA 炸」
   不是用例的问题，是这四个包没装。
   ⚠ 这和 §1 里那条 `No module named 'tbe'` 是**同一个坑的不同外衣** ——
   `500001` 只说明「tbe 起不来」，具体原因每次不同，**一定要往下读到 EC0010 那行**。

4. **`pip install scipy` 会装上 1.18.1，然后 CANN 报 `module 'numpy' has no
   attribute 'long'`。** 新 scipy 要 `numpy>=2.0`，而 torch_npu / CANN 这套要
   `numpy 1.26.4`。**必须钉 `scipy==1.13.1`**（也正是 triton-ascend 自己声明的版本）。
   报错出现在 CANN 的 tbe 初始化里，跟 scipy 三个字毫无关系。

5. **`torch_npu` 2.10.0.post4 有两个没声明的运行时依赖。**
   `import torch_npu` 直接 `ModuleNotFoundError: No module named 'yaml'`
   （`torch_npu/npu/_memory_viz.py:10`），而且被包成
   `RuntimeError: Failed to load the backend extension: torch_npu`；
   triton-ascend 那边同样缺 `pybind11`
   （`triton/backends/ascend/utils.py:36`）。装 `pyyaml` + `pybind11` 解决。

另外两条不是坑但会吓人：`import torch` 打印 **`2.10.0+cpu`** 是正常的；
profiler 每次都会打一行 `Failed to get acl to npu flow events`，
数照样出（用例读的是 `kernel_details.csv`）。

清理：CANN 每次跑都会在 **cwd** 掉一个 `fusion_result.json`（仓库 `.gitignore`
已忽略），异常时还会掉几百 MB 的 `extra-info/data-dump/`，权限 `-r--------`，
`ls` 不留神就漏。**在 `/var/tmp/` 下建个空目录 `cd` 进去再跑**，详见 §7 第 14 条。

---

## 0. 一句话结论

| 层族 | 层数 | 用例单层 p50 | ×层数 | 整网实测 | 偏差 |
|---|---|---|---|---|---|
| **KDA** | 34 | **289.4 µs** | **9.841 ms** | 10.408 ms | **−5.0%** |
| KDA（真跑 34 个不同层） | 34 | — | **9.623 ms** | 10.408 ms | −7.1% |
| **DSA**（n=256） | 11 | **445.3 µs** | **4.898 ms** | 5.395 ms | **−4.8%** |
| DSA（真跑 11 个不同层，各自的池子） | 11 | — | **5.320 ms** | 5.395 ms | **+3.4%** |

⚠⚠ **但总数不是判据，算子清单才是。** 一次 replay 只要发的 kernel 集合不同，
就可能靠互相抵消落在正确的总数上；**而少发一个 kernel 只会让用例显得更快**。
判据是 `regress_against_network.py`（见 §1、§5.2）：

| | 整网参考算子组 | shape 与每层次数**都**对上 | 缺 | 多 |
|---|---|---|---|---|
| **KDA** | 8 | **8** | 0 | 1（已证明是归因盲区，§5.1） |
| **DSA** | 60 | **59** | 1（`Cast "1,1,1"`，1.3 µs，未解释） | 13（其中 7 组已证明是归因盲区，6 组是真差异，§5.2） |

单层版都比整网**略便宜**（−5%），因为单层跑没有和一步里的 MoE / mHC / head
抢内存系统 —— `hcpre_microbench` 上是同一个形状：微基准 28–30 µs，服务里 33.0 µs。
**这不是需要「调平」的误差，是两种测量的定义差。**

而「真建 N 个不同的层串起来跑」这一档（`family`）把 −5.0% / −4.8% 变成
**KDA −7.1%、DSA +3.4%** —— 两个方向相反。所以「单层 ×N 偏低是因为 cache 太热」
这个看起来很合理的解释**被自己的数据否掉了**（KDA 那边真跑 34 层反而更便宜）。
没有替它编第二个解释。

⚠ **别拿总偏差当进度条。** DSA 用例修保真度的过程里，总偏差从 −3.1% 变成 −4.8%
（中间对齐 `--ref-seq` 时一度到 −7.2%），而清单从「7 组缺失 + 3 个 shape 全错」
变成「1 组缺失 + 0 个 shape 错」。**−3.1% 那次更接近，是因为 SFA 高了 47%
在补偿别处偏低。**

---

## 1. 怎么跑

```bash
source ${GLM53_ROOT}/env/env.sh
export PYTHONPATH=${GLM53_ROOT}/wt-int8-singlecard/python:$PYTHONPATH
cd docs/docs/glm53_npu_support/oplab

ASCEND_RT_VISIBLE_DEVICES=0 python bench_kda_layer.py
ASCEND_RT_VISIBLE_DEVICES=0 python bench_dsa_layer.py
```

⚠⚠ **`PYTHONPATH` 必须用 `:$PYTHONPATH` 追加，不能覆盖。** `env.sh` 里 CANN 的
`set_env.sh` 往 `PYTHONPATH` 塞了 `.../cann-9.2.0/python/site-packages` 和
`.../op_impl/ai_core/tbe`。覆盖掉它，`npu_format_cast` 会在
`LazyInitAclops` 里炸出 `error code 500001` + `No module named 'tbe'`——
错误信息里没有一个字提到 `PYTHONPATH`。本用例开发时踩了这个，记在这里。

分段跑：

```bash
python bench_kda_layer.py --sections layer          # 只做回归判据
python bench_kda_layer.py --sections refs           # 只做带宽地板对照
python bench_kda_layer.py --sections family         # 真的建 34 个层（~11 GB）
python bench_dsa_layer.py --sections sweep          # 序列长度扫描
python bench_dsa_layer.py --ref-seq 32768 --sections layer

# 清单比对（判据），不是看总时间
P=$(ls -dt /var/tmp/glm53/oplab/dsa/layer/*ascend_pt | head -1)
python regress_against_network.py --family DSA --profile "$P" --steps 30
```

显存：KDA `layer/sweep/refs` < 2 GB，`family` ~11 GB；
DSA `layer/sweep/refs` ~2.2 GB（KV 池按部署的 1.25 M token 全量开），`family` ~19 GB。

**跑之前先 `npu-smi info` 看你那张 die 是不是空的**（见 §7 第 9 条）。

---

## 2. 这两个用例 import 了什么

只依赖 `torch` / `torch_npu` / `sgl_kernel_npu`，**外加几个 sglang 树里的
kernel 函数**（不是模型代码）。如实列出：

| 用例 | 从 sglang 里 import 的 | 是什么 |
|---|---|---|
| KDA | `sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update` | Triton kernel + 它的 launcher |
| KDA | `sglang.kernels.ops.attention.fla.fused_norm_gate.layer_norm_gated_fwd` | Triton kernel + 它的 launcher |
| DSA | `...hardware_backend.npu.attention.kpool_indexer_npu.compress_pool_bf16` | 纯 torch，pool 压缩 |
| DSA | `...hardware_backend.npu.attention.kpool_indexer_npu.hadamard_transform_npu` | 纯 torch，fp32 Hadamard |
| DSA | `...layers.attention.dsa.kpool_fp8_index.expand_pooled_groups_to_topk` | 纯 torch，pool→token 展开 |
| DSA | `...layers.attention.dsa.kpool_fp8_index.append_kpool_tail_to_topk` | Triton kernel |

**没有 import 的**：任何 model 文件、attention backend 类、scheduler、memory pool、
ForwardBatch、ServerArgs。DSA 用例里 `kpool_decode_update_index_cache` 那段
（ring 更新 + 压缩写 cache）是**照抄成本地代码**的，不是调用 —— 因为原函数挂在
memory pool 类上。抄的来源写在代码注释里。

如果算子团队想彻底断开 sglang：上面四个纯 torch 函数不到 60 行，可以直接内联；
两个 Triton kernel 就得连 `.py` 一起拷。

---

## 3. KDA 用例覆盖了什么

**34 层，每层 8 个算子（见 §5.1 的更正：其实是 9 个），合计 272 个 kernel / 10.408 ms。**

序列（`glm5_next.py: Glm5NextLinearAttention.forward`，融合投影那条腿）：

```
x [1,4096] bf16
 1 fused_qkvbfg_a_proj   W[24896,4096] bf16        MatMulV2 "1,4096;24896,4096"
   split → qkv[1,24576] | beta[1,64] | f_a,g_a[1,256]
 2 fused_fg_b_proj       bmm [2,1,128]×[2,8192,128] BatchMatMulV2
   → forget_gate[1,8192], norm_gate[1,8192]
 3 clamp(cache_indices, min=0)                     ClipByValueV2 "1;;"
 4 torch.ops.npu.causal_conv1d(run_mode=1)         causal_conv1d_4 (+ 内部 2 个 Cast)
   x[1,24576], w[4,24576] bf16, state[17,3,24576] bf16 window-major
 5 fused_sigmoid_gating_delta_rule_update(is_kda=True)
   q,k,v[1,1,64,128] · a[1,8192] · b[1,1,64] · ssm[17,64,128,128] fp32
 6 FusedRMSNormGated(128, "sigmoid")               layer_norm_gated_fwd_kernel
 7 o_proj                W[4096,8192] bf16         MatMulV2 "1,8192;4096,8192"
```

结构常数全部来自 `config.json`：hidden 4096，`linear_attn_config` 的
`num_heads 64 / head_dim 128 / short_conv_kernel_size 4 / gate_lower_bound -5.0`，
`kda_layers` 34 个。融合投影 24896 = 24576 (q|k|v) + 64 (beta) + 128 (f_a) + 128 (g_a)。

**KDA 的投影是 bf16 不是 INT8** —— W8A8 checkpoint 的 `ignore` 里列了全部八个子模块
加融合模块名，所以走 `MatMulV2` 而不是 `QuantBatchMatmulV3`。这是作者的决定，
不是漏掉（REPORT.md §6.4）。用例照此建权重。

---

## 4. DSA 用例覆盖了什么

**11 层，每层 90 个 kernel，合计 891 / 5.395 ms。** 用例跑出 93 个（差异见 §5.2）。

```
A 隐向量  DynamicQuant → QuantBatchMatmulV3 "1,4096;64,256,16,32;2048;1"
          split 1536 q_lora | 512 kv_lora；RmsNorm(1536)、RmsNorm(512)
          q_b_proj  QuantBatchMatmulV3 "1,1536;512,96,16,32;16384;1"
          W^UK 吸收  BatchMatMulV2 "64,1,256;64,512,256" → q 进 512 维隐空间
B 索引器  wq_b 1536→4096   MatMulV2 "1,1536;4096,1536"
          wk   4096→128    MatMulV2 "1,4096;128,4096"  + LayerNormV3(128)
          gate 4096→128    MatMulV2 "1,4096;128,4096"   ← 两个同 shape，n=22/step
          weights_proj 4096→32 fp32  MatMulV2 "1,4096;32,4096"
          fp32 Hadamard-128  MatMulV2 "32,128;128,128" / "1,128;128,128"
          4 token 压成 1 个 pool     ReduceMax/ReduceSum "1,4,128;1"
          ring + index cache 写      ScatterNdUpdate ×3
          npu_lightning_indexer(sparse_count=512) → 512 个 pool id
          512 pool → 2048 token      Add "1,512,1;4" + BroadcastTo "1,512,1;3"
          + 3 列 tail                → topk_indices [1,2051] int32
C 注意力  kv_buffer[loc] = k_nope    IndexPutV2 "1246656,1,512;..."
          npu_sparse_flash_attention(sparse_mode=3, attention_mode=2,
              layout TND / PA_BSND, sparse_block_size=1)
D 出口    W^UV 吸收  torch.ops.npu.batch_matmul_transpose → batch_matmul_transpose_0
          o_proj     QuantBatchMatmulV3 "1,16384;128,1024,16,32;4096;1"
```

池子按部署全量开：KV `[1246656,1,512]` bf16（PA_BSND 视图 `[19479,64,1,512]`），
index-K `[19480,64,1,128]` bf16，两者差一页是有意的。

---

## 5. 已知的对不上的地方（如实记，没有调参去凑）

### 5.1 KDA 家族其实是每层 9 个 kernel，不是 8 个

用例每层稳定跑出 **9** 个 kernel，比 cfgI 的 KDA 段多一个
`Cast "1"`（int32，1 个元素，1.2 µs）。

不是用例多做了事：**`torch.ops.npu.causal_conv1d` 自己会发两个 Cast** ——
一个是 `query_start_loc`（2 个元素 → `Cast "2"`），一个是 `cache_indices`
（1 个元素 → `Cast "1"`）。独立探针验过，跟外面加不加 `clamp` 无关。

那 cfgI 里它去哪了？在 **`--- unclassified ---` 段**：

```
84.7 us/step  n=59  1.4 us  Cast  AI_VECTOR_ "1"
```

`attribute_kernels.py` 是**按调用次数归属**的（34 次 → KDA）。这个 shape 被 KDA 的
34 次和别处的 25 次混在一起变成 59，于是整组落进 unclassified。

**独立复核**：另一个 session 在干净 die 上单独验过 —— trace 里这个 `Cast "1"`
紧贴在 `causal_conv1d_4` 前面，整组 64 次/步，其中 34 次属 KDA。

**结论：KDA 的真实开销是 10.408 + 34×1.4 ≈ 10.405 ms，不是 10.408。**
差 0.05 ms，不影响任何结论，但**基于计数的归属工具会漏掉与别人共享 shape 的算子**，
这条对以后读 attribution 表的人有用。

### 5.2 DSA 的保真度：判据是**算子清单**，不是总时间

**总数对上可能是巧合。** 一次 replay 只要发的 kernel 集合不同，就可能靠互相抵消
落在正确的总数上；而**少发一个 kernel 只会让用例显得更快**。所以判据用
`regress_against_network.py`（清单比对），不用总偏差：

```bash
P=$(ls -dt /var/tmp/glm53/oplab/dsa/layer/*ascend_pt | head -1)
python regress_against_network.py --family DSA --profile "$P" --steps 30
```

**现状：整网 60 个参考算子组，59 个在用例里 shape 逐字一致且每层次数一致。**

#### 修掉的（第一版确实是错的）

| 症状 | 根因 | 修法 |
|---|---|---|
| `LightningIndexer` 最后一维 `1,4096` 而不是 `1,128`；`Index "1,16384;…"` 而不是 `"1,512;…"`；`SparseFlashAttention` 第 5 个操作数 `1,16384` 而不是 `1,512` | **block table 宽度按扫描的最大序列长度开的，应该按 `--context-length` 开** | 新增 `--context-len`（默认 32768 → `[1,512]`）。**一个根因同时错了三个最大算子的 shape** |
| 缺 `GreaterEqual "1;"` ×2、`Less "1;"`；`LogicalAnd` 2/层而不是 5/层 | 有效性掩码只写了 2 个条件，源码是 5 个（`req>=0 & req<N & out_cache_loc!=0 & pos>=0 & pos<seq_lens`） | 照源码补全 |
| 缺 `SelectV2 "1;1;"`，多一个 `"1;1;1"` | `torch.where(c, t, python_int)` 和 `torch.where(c, t, tensor)` 是**两个不同的算子**。源码的 `_tail_scratch_row` 是 python int | 用 python int |
| `Mul "1,32;"` 1/层而不是 2/层 | `_kpool_head_gate_npu` 是 `w * n_heads**-0.5 * softmax_scale`，**两次标量乘**，我折成了一个常数 | 拆回两次 |
| 多 `Cast "4,128"` | `ape` 建成了 bf16，`compress_pool_bf16` 里 `ape.float()` 就多一次 cast | 建成 fp32 |
| `Cast "1,128"` 3/层而不是 5/层 | `k_norm` 是**显式 fp32 往返**（`.float()` → layer_norm → `.to(bf16)`）。直接把 bf16 喂给 `F.layer_norm` 配 fp32 权重**一次 cast 都不发**（实测） | 照源码显式往返 |

#### 剩下的 1 个「缺」

`Cast "1,1,1"`（1.3 µs/层）。**没找到它从哪来**，`topk_from_pooled_selection`、
`_expand_dsa_sparse_indices`、`_pad_topk_indices` 都逐行核过了，都不产生 `[1,1,1]`。
**如实记着，没有为了让清单好看去造一个。**

#### 剩下的 14 个「多」——分成两类，判据是全网计数

一个算子组是不是「用例多做的」，不能只看它不在 DSA 段里，**要看它在整份 profile
（所有 family 合计）的计数容不容得下 DSA 的那一份**：

**(a) 7 组是归因盲区，不是保真度缺口** —— 整网确实跑了，只是被
`attribute_kernels.py` 的「按调用次数归属」规则记到别的 family 去了：

| 组 | 用例 /层 | 整网合计 | 落在 |
|---|---|---|---|
| `Range ";;"` | 1 | 14 | unclassified（11 DSA + 3） |
| `DynamicQuant "1,4096"` | 1 | 14 | unclassified（11 DSA + 3 dense FFN） |
| `Fill "1;"` | 2 | 24 | unclassified |
| `Fill "2;"` | 1 | 96 | unclassified |
| `Cast "1"` | 2 | 64 | unclassified + head/global |
| `Cast "1,4096"` | 1 | 53 | unclassified |
| `IndexCheck "1;1"` | 1 | 17 | unclassified |

**(b) 6 组是真差异，而且方向和我预期的相反** —— 这些**源码里有、profile 里没有**：

| 组 | 用例 /层 | 整网合计 | 若 DSA 也发，合计应是 | 来自源码的哪一行 |
|---|---|---|---|---|
| `ClipByValueV2 "1;;"` | 3 | 34（全是 KDA） | 67 | `req.clamp(0,N-1)`、`pos.clamp(min=0)`、`page_col.clamp(0,W-1)` |
| `FloorMod "1;1"` | 2 | 45（全是 mHC） | 67 | `safe_pos % pool_size`、`pool_id % slots_per_page` |
| `BroadcastTo ";1"` | 2 | 45（全是 mHC） | 67 | 同上，标量取模的广播 |
| `Sub "1;1"` | 1 | 12（全是 dense FFN） | 23 | `start = safe_pos - safe_pos % pool_size` |
| `Equal "1;"` | 1 | 12（全是 dense FFN） | 23 | `safe_pos % pool_size == pool_size - 1` |
| `IndexCheck "2;1;1"` | 1 | 13（dense FFN + head/global） | 24 | `block_tables[rows, page_col]` 的越界检查 |

计数**容不下**，所以不是归因盲区：**整网那一步真的没有执行这几行标量下标运算。**
可是 `memory_pool_npu.py::kpool_decode_update_index_cache` 里这几行明明白白写着，
而同一个函数里的 `GreaterEqual` / `Less` / `LogicalAnd` / `SelectV2` /
`FloorDiv "1;"` / `Mul "1;"` / `FloorMod "1,4;1,4"` **全都对得上**。
也就是说这个函数的**大部分**在整网 profile 里，**只有 clamp / 标量取模 / 相等 / 相减
这几类不在**。

⚠ **这条我没解释掉，也没有去凑。** 要让它们消失，只能改成和源码不一样的写法，
而那恰恰是「调参数凑数」。合计约 23 µs/层（用例 452 µs 的 5%）。
**给算子团队的实际影响：这 6 组是簿记，不是候选，看 §6.3 的时候直接跳过。**
**给下一个人的提示**：值得查的是 cfgI 那份 profile 的代码版本与当前 worktree
是否在 kpool 这一段上有差，或者 GE 图融合有没有把这几类标量算子折掉。

#### 一条方法论

`--ref-seq 512` 时总偏差是 **−3.1%**，改成对齐的 256 之后**反而变成 −7.2%**。
512 那次更接近，**是因为 SFA 高了 47% 在补偿别处偏低**。
**不要拿总偏差变大当成改坏了** —— 清单对上之后总数是多少就是多少。

### 5.3 ⚠ 最重要的一条：cfgI 的 DSA 数字是**短上下文**数字

`tools/profile_server_decode.py` 的默认是 `--prompt-tokens 13`。cfgI 那份
attribution 是在 13 token 提示 + 若干解码 token 上采的，**序列长度是几百，不是 32k**。

所以 `DSA = 5.395 ms/step` 这个数**只在短上下文成立**。用例扫出来：

| n | µs/层 | ×11 (ms) | 相对 n=128 |
|---|---|---|---|
| 128 | 443.6 | 4.879 | +0.0% |
| 512 | 462.1 | 5.083 | +4.2% |
| 1 024 | 476.9 | 5.246 | +7.5% |
| 4 096 | 518.5 | 5.704 | +16.9% |
| 32 768 | 523.4 | 5.757 | +18.0% |
| 131 072 | 534.5 | 5.879 | +20.5% |
| 1 048 576 | 597.6 | 6.573 | **+34.7%** |

**引用 5.395 ms 的时候必须带上「短上下文」这个限定。** 在 32k（服务默认的
`--context-length`）上，同一份 DSA 是 5.757 ms（+18%）；在 1M 上是 6.573 ms（+35%）。

⚠ **上表是一次跑的绝对值，带跑间差异。** 另一次独立跑（同一 die、同一构型）给出
n=128 → 436.5 µs、32 768 → 516.9、131 072 → 528.3，**系统性低约 1.2%**。
**该引用的是「相对 n=128 的增幅」那一列和拐点位置，不是绝对微秒数** ——
增幅在两次跑之间一致（+18.0% vs +18.4% @32k），绝对值不一致。
这和 §7.x 记的是同一件事：**单次 profile 的噪声底是几百微秒，一个数不算数。**

### 5.4 用 `SparseFlashAttention` 反解出 cfgI 的上下文长度

整网 SFA 的中位数是 **32.08 µs**。用例的 SFA 对序列长度是单调的，
n=128 → 28.5 µs，n=256 → 34.4 µs，**32.08 落在两者之间 → cfgI 的上下文 ≈ 210**。
和 `--prompt-tokens 13` + 若干解码 token 完全吻合。

**所以 `--ref-seq` 的默认值是 256，不是 512。** 这是本轮最强的一条交叉验证：
一个采数时没记下来的参数，被算子时间本身反解出来了。

### 5.5 几个比 cfgI 便宜的算子

`BatchMatMulV2 "64,1,256;64,512,256"` 0.71×、`batch_matmul_transpose_0` 0.55×、
`MatMulV2 "1,1536;4096,1536"` 0.66×、`ClipByValueV2 "1;;"` 0.40×。
**没有去解释它们**：单层跑的内存系统状态跟整网不同，而这几个都是小算子，
整网里更容易被邻居拖慢。方向一致（全部偏便宜），量级也一致，所以按 §0 的
「测量定义差」处理，不当成算子差异。

---

## 6. 看到什么算正常 / 哪些算子不用优化 / 哪些是真候选

### 6.1 正常范围

* **KDA**：`layer` 段 285–300 µs/层，×34 = 9.7–10.2 ms。
  `MatMulV2 "1,4096;24896,4096"` 应该在 **164–172 µs**。
* **DSA**（默认 `--ref-seq 256 --context-len 32768`）：440–460 µs/层，×11 = 4.85–5.1 ms。
  `SparseFlashAttention` 应该在 **31 µs 附近**（整网 32.5）。**它大于 40 µs
  说明 `--ref-seq` 设大了**，不是算子变慢了。
* **先跑 `regress_against_network.py`。** 清单不对，时间就不用看。
* 两个用例的 shape 字符串必须和 cfgI **逐字相同**。不同就是用例建错了，
  不是机器慢了 —— 特别注意 DSA 那三个 `QuantBatchMatmulV3` 必须显示成
  `128,1024,16,32` 这种四维 NZ 形状（见 §7 第 10 条）。
* **p90 / p50 < 1.1**。分布一散就是 die 上有别人。

### 6.2 已经贴在带宽地板上，不用看了

`refs` 段用四个对照给出判决，而不是只给一个比值：

| 算子 | 权重 | 冷跑 p50 | 1.25 TB/s 地板 | 1.40 TB/s 地板 | 比值 |
|---|---|---|---|---|---|
| KDA `fused_qkvbfg_a_proj` `[1,4096]×[24896,4096]` | 194.5 MiB | 146.7 µs | 163.2 µs | 145.7 µs | **1.01×** |
| KDA `o_proj` `[1,8192]×[4096,8192]` | 64.0 MiB | 56.1 µs | 53.7 µs | 47.9 µs | 1.17× |
| DSA `o_proj` INT8 `16384→4096` | 64.0 MiB | 55.8 µs | 53.7 µs | — | **1.04×** |

（1.40 TB/s 是 GEMM 在本机实测能到的读带宽，见 §7.12；1.25 是 REPORT.md 沿用的
保守值，对 reduce 类算子合适，对 GEMM 偏低。）

**KDA 那两个 GEMM 加起来是 KDA 的 76%、整个 decode step 的 25%。它们已经比
「把同样的字节纯读一遍」还快**：194.5 MiB 的 `w.sum()` 要 **169.7 µs**，
同样字节的 GEMM 只要 **146.7 µs**。算子团队在这上面拿不到东西。唯一的杠杆是
**少读字节**（换更小的 dtype）或**少调用**，两者都是模型层面的改动。

DSA 侧同理：三个 `QuantBatchMatmulV3` 都在 0.93–1.12× cfgI，`o_proj` 冷跑
1.04× 地板，**INT8 NZ 已经是快路径**。另外两个大池子读也贴墙：
KV 池 1.19 GiB 读 1100.5 µs（1.08× 地板）、index 池 0.30 GiB 读 269.8 µs（1.06×）。

### 6.3 真正值得看的

按「不是带宽地板 + 绝对时间够大」排：

1. **`fused_sigmoid_gating_delta_rule`（KDA，34×33.3 µs = 1.13 ms/step）**。
   它读的是 `[64,128,128]` fp32 状态 = 4.19 MiB，地板 3.4 µs。**34.3 µs 是地板的
   10×。** 这是本轮 KDA 侧最大的非带宽项。Triton kernel，`grid=(1,4,64)`，
   `num_warps=1`。
2. **`SparseFlashAttention` 饱和后的 ~99 µs**（DSA，n≥4k，11×99 = 1.09 ms/step）。
   它读 2048 个 token × 512 维 bf16 = **2 MiB**。用例专门测了同样字节的
   gather 对照（`refs` 段的 `KV pool read 2048 tokens`）：**~7.2 µs**。
   **也就是说 SFA 有 ~92 µs 不是在读 KV。14× 于同字节的 gather。**
   这是本轮 DSA 侧最大的非带宽项，而且它在 n≥4k 之后**恒定**，
   所以省下来的每一微秒在所有上下文长度上都算数。
3. **`LightningIndexer`（DSA）**。短上下文 16.6 µs，1M 上 80.3 µs。
   **它是整个 DSA 唯一的 O(n) 项**（见 §6.4），长上下文那条线的斜率全在这里。
   ⚠ 但它在 1M 上**已经接近带宽地板**：262144 个 pool × 128 × bf16 = 64 MiB，
   1.25 TB/s 地板 53.7 µs，实测 81.1 = **1.51×**。所以这里的杠杆不是 kernel
   调优，而是**少读字节**：index cache 换 fp8（现在是 bf16，仓库里有
   `f06266470a npu: store the DSA index-K cache as bf16` 这条反向改动的历史）
   或者把 `index_kpool` 从 4 调大。两者都是**模型/配置**决定，不是算子决定。
4. **`causal_conv1d_4`（KDA，11.9–15.9 µs，见 §7.13）**。读 conv 窗口 2.39 MiB → 地板 2.0 µs，
   但真正被摸到的只有 1 个 slot 的 [3,24576] = 144 KiB。**接近纯固定开销。**
5. **DSA 的 ~50 个小算子合计约 120 µs/层 = 1.3 ms/step（占 DSA 的 27%）**，
   单个都在 1–8 µs。**这一堆是「总固定成本」问题，不是带宽问题** ——
   参见 §7 第 6 条：减少 kernel 个数是代理不是目标。

### 6.4 两个层族对序列长度的依赖完全不同 —— 这是这对用例最大的价值

**KDA 是 O(1)，而且是结构性的 O(1)。** 用例先证明再测量：
`sweep` 段打印每个长度下所有输入张量的 shape 指纹，四档**逐字节相同**
（KDA 的 decode 状态是固定的 `[64,128,128]` fp32 + `[3,24576]` conv 窗口，
序列长度不进入任何 shape、stride 或循环边界）。然后照样测：

| n | 1 024 | 4 096 | 32 768 | 131 072 |
|---|---|---|---|---|
| µs/层 | 291.6 | 292.4 | 298.3 | 291.5 |
| ×34 (ms) | 9.913 | 9.941 | 10.143 | 9.911 |

最大 +2.3%，而且**不单调**（131 072 比 1 024 还低 0.0%），
**这就是本机的噪声底**，不是长度效应。

> ⚠ 只测不证是不够的：四个完全一样的负载测出来一样，本身什么也没证明。
> 所以用例把「什么都没变」这件事**打印出来**。

**DSA 是两段：注意力本体被 topk 封顶，索引器 O(n)。** 用例把这两个机制分开了：

| n | 128 | 512 | 1 024 | 4 096 | 32 768 | 131 072 | 1 048 576 |
|---|---|---|---|---|---|---|---|
| `SparseFlashAttention` | 26.5 | 45.2 | 59.6 | 99.7 | 99.4 | 99.5 | **100.3** |
| `LightningIndexer` | 16.6 | 16.9 | 17.3 | 18.5 | 23.9 | 32.5 | **80.3** |

* SFA 一路涨到 n≈4k **就再也不动了** —— `index_topk=2048` 的封顶，实测到了。
* LightningIndexer 从 32k 到 1M 涨 3.2×，n 涨 32×。**给 n/4 个 pool 打分。**

n ≥ 4096（注意力已饱和）的最小二乘：**DSA 家族 ≈ 5.733 + 8.06e-7 · n ms/step**。
多卡 TP8 那条线拟的是**整步** `27.3 + 5.4e-6 · n`。两个不是同一个量
（一个是 TP1 的 DSA 家族，一个是 TP8 的整步），**它们的比值不能直接当证据**。

⚠ 还有一条**没解释掉的**：`BatchMatMulV2`、`batch_matmul_transpose_0`、
`MatMulV2 "1,1536;4096,1536"` 这三个 shape **不依赖 n**，却在 n=1M 时各涨了
2–5 µs（19.6→22.0、13.5→18.1、12.4→15.3）。最可能是索引器扫 1M 个 pool 把 L2
冲干净了，但**没有直接量过 L2 命中率，所以这是推断，不是结论**。

### 6.5 并发（不是长度）才是 KDA 的另一个轴

ssm 状态是 **4.19 MiB / 槽 / 层**，TP1 下 16 槽就是 2.34 GB。
`slots` 段扫 2/17/65/129 槽（8 MiB → 516 MiB 每层）：

| slots | 2 | 17 | 65 | 129 |
|---|---|---|---|---|
| µs/层 | 290.1 | 285.5 | 286.3 | 288.1 |

**平的。** 也就是说 REPORT.md §6.1 那条「L2 冲刷」机制在**单层隔离测量里复现不出来**
——池子再大，本层只摸 1 个槽。⚠ 这不证伪 §6.1（那条是整网 34 层交替时的现象），
但它说明**光把池子开大不会让 KDA 变慢**，变慢需要真的有别的东西在冲 L2。

`family` 段就是去测这个的：真建 34 个**不同的**层串起来跑一张图。结果
**9.623 ms，比「单层 ×34」的 9.841 还便宜 2.2%** —— 所以
**「单层 ×34 偏低是因为 cache 太热」这个假设被证伪了。** 剩下的 −6% 只能归给
与一步里其它层族的共存，跟 `hcpre_microbench` 的 28–30 vs 33.0 同源。

---

## 7. 踩坑清单

前 8 条来自本项目此前的实测（都写在 `int8_singlecard/REPORT.md` 里），
后 4 条是做这两个用例时新踩的。

1. **`tensor.is_cuda` 在这台机器上返回 `True`。** `torch_npu.contrib.transfer_to_npu`
   补的是属性本身，**连 import 之前建的张量也变**。判平台用 `device.type`，
   永远不要用 `is_cuda`。
   ⚠ `bench_dsa_layer.py` **必须** import `transfer_to_npu`（见第 10 条），
   所以这个用例里 `is_cuda` 是坏的。KDA 用例没 import，是好的。**同一个 repo 里
   两个文件行为不同——这正是不能靠 `is_cuda` 的理由。**

2. **「每 kernel 13.5 µs 固定 launch 开销」是错的。** 这台机器 device 侧单 kernel
   下限实测 **1.3~1.5 µs**（标量 Cast/Mul），`HcPost` 6.3 µs。13.5 是一次具体
   观测被误当成机器常数。本用例的 `refs` 段每次都重新测这个下限
   （`trivial torch.add on 1 element`，实测 1.4–1.7 µs），**因为它跟构型走，
   不是常数**。

3. **本后端上贪心输出不是 batch 不变的**，而且同一进程、同一宽度、重复跑都可能不同
   ——不确定的是调度器的组批。**batch > 1 上不存在可复现基线**，别用「输出逐位相同」
   当判据；宽度 1 是稳定的。（这两个用例不做精度判据，只做性能。）

4. **`causal_conv1d` 的 AOT 算子要求输入 contiguous，且响亮报错**
   （`x must be contiguous`）；但同族的 `causal_conv1d_update` 的 layout 约束是
   **静默**的。本项目已知四个算子约束里**三个是静默的**。
   在 KDA 用例里 `qkv` 是 24896 宽输出的**末维前缀切片**：bs=1 时恰好连续，
   bs≥2 不连续 —— 所以 `.contiguous()` 那行不是装饰。

5. **`custom::npu_dequant_swiglu_clamp_quant` 的 `clamp_limit` 和 `swiglu_mode`
   参数被静默忽略** —— 实测取 0/1/2/3 结果完全一样，且与「完全不 clamp」逐位相同。
   **要验这个算子必须放大输入让 clamp 真的咬到。**（不在这两个用例的范围内，
   但同一批算子里，列出来提醒。）

6. **`npu_clipped_swiglu` 融合版逐位相同、少 45 个 kernel、却净慢 65.6 µs。**
   「减少 kernel 个数」是**代理不是目标**；杠杆是减少**总固定成本**，
   当替换进来的 kernel 单次做的活不等价时代理就失效。看 §6.3 第 5 条的时候
   请带着这条。

7. **`torch.nn.functional.pad` 在这里降解成 `MemSet` + `PadV3`** —— 同样的 launch 数、
   3.3× device 时间。

8. **`HcPre` 类的算子跨服务重启会差 10%**（同一 shape 同一次数，33.0 vs 29.9 µs）。
   **单次测量的噪声底是几百微秒，别拿单次结果下结论。** 这两个用例默认
   30 次取 p50 并同时打 p90/max，就是为了这条。

--- 以下 4 条是做这两个用例时新踩的 ---

9. **die 被别人占着，测出来的数会整体膨胀 ~1.7×，而且没有任何报错。**
   实测（2026-08-31，die 0 上有别人的 8 die 服务、die 1 上有别人的 16 卡训练）：

   | | 空闲 die | 有人的 die 0 | 有人的 die 1 |
   |---|---|---|---|
   | `MatMulV2 "1,4096;24896,4096"` p50 | **166 µs** | 287 µs | 296 µs |
   | 同一算子 p90 | 168 | 387 | 551 |

   **1.7× 正好落在「20% 以内正常 / 2× 说明 shape 错了」这个判据的缝里**，
   所以两个用例都内置了哨兵。**它经过两轮误报才做对，两次误报的方向相反：**

   **第一版：中位数当主判据 → 把「参数不对齐」误报成「卡被占」。**
   在一张确认空闲的 die 上（3136 MiB，无其它进程），`--ref-seq` 用了 512 而不是
   对齐的 256，`IndexPutV2` 中位数 1.62×，哨兵喊「THIS DIE IS SHARED」。
   而它的**离散度只有 1.01×——干净**。
   → **中位数偏了但尾巴干净 = 参数或环境不匹配，不是卡被占。**

   **第二版：离散度当主判据，但没设下限 → 把「计时器精度」误报成「卡被占」。**
   `family` 段里 `LayerNormV3` p50 只有 1.26 µs，0.4 µs 的抖动就是 p90/p50 = 1.30。
   → **只对 ≥5 µs 的 kernel 做离散度判断**（`DISPERSION_FLOOR_US`）。

   **第三版：`family` 段根本不做这个判断。** 那一段把 N 个**不同的**层的 kernel
   汇进同一个分布，它的 p90/p50 量的是层与层之间的真实差异，不是干扰。
   同样在确认空闲的 die 上误报过 1.26×。
   → **离散度只在「每个样本都是同一份工作的重复」时有意义，也就是 `layer` 段。**

   最终形态，两个独立判据指向两件不同的事：

   | 信号 | 结论 |
   |---|---|
   | 离散度 p90/p50 > 1.25×（只看 ≥5 µs 的 kernel，只在 `layer` 段） | **卡被占** |
   | 中位数 > 1.25× 而离散度干净 | **参数/环境不匹配**（先查 `--ref-seq` / `--context-len` / NZ） |

   这是本项目「分布形状比均值更能说明问题」那条教训的两个方向：
   **均值会漏掉干扰，而离散度会把噪声和差异当成干扰。**

10. **`npu_format_cast` 转 FRACTAL_NZ 是个静默 no-op，除非先开 internal format ——
    而那个开关在 import `transfer_to_npu` 之前根本不存在。**

    ```python
    from torch_npu.contrib import transfer_to_npu   # 必须先 import
    torch.npu.config.allow_internal_format = True   # 否则这个属性都没有
    ```

    不开的话：权重停在 ND，profiler 打出 `"1,16384;16384,4096;4096;1"` 而不是
    `"1,16384;128,1024,16,32;4096;1"`，`o_proj` **103 µs 而不是 58 µs**，
    stderr 上只有一行 `Warning: Cannot create tensor with internal format`。
    **本用例现在会主动 `get_npu_format` 复查并在不是 NZ 时直接退出** ——
    因为这是第 4 条说的那种静默约束里最贵的一个。

11. **在一个 profiler session 里按「每个 case 均分 kernel 行数」来切分是错的，
    而且是静默错的。** 只要两个 case 的 launch 数不同，一个 case 的 kernel 就被
    算到隔壁头上。实测后果：一次 68 MiB 的读被报成 **7.06 µs = 它自己带宽地板的
    0.12×** —— 一个物理上不可能的数，但表格里看不出来。
    两个用例改成**在 case 之间插一个独有 shape 的哨兵 kernel**（`[1357]` 的
    `add_`），按哨兵切分，并且在哨兵个数对不上时**拒绝出数**而不是继续。

12. **这台机器的 L2 有 ~168 MiB，足以把一整个权重装进去；重复读同一个 buffer
    量出来的带宽是假的。** 实测 KDA `o_proj` 权重 64 MiB：

    | | p50 | 该权重的带宽地板 | 比值 |
    |---|---|---|---|
    | 热跑（同一个权重 30 次） | 29.8 µs | 53.7 µs | **0.56×** |
    | 冷跑（6 份不同权重轮转，> 2×L2） | 54.7 µs | 53.7 µs | 1.02× |

    **热跑比它自己的 HBM 地板还快 1.8 倍。** 服务里不是这个状态：一步要把 34 层
    不同的权重从 L2 前面流过去。`refs` 段现在**热冷都给**，
    并且**任何低于 1.0 的比值都当成警告而不是结论**。

    ⚠ **这条我自己先踩了一次，记下来。** 第一版用「> 2×L2」做冷跑轮转，
    194.5 MiB 的权重只需要 2 份 = 389 MiB，对 168 MiB 的 L2 仍然留了不少复用，
    量出 146.9 µs 就直接反推「≥1.39 TB/s，所以 1.25 常数偏低」。
    **判据应该是「把轮转做大，答案不再动」，不是「轮转 > 2×L2」。**
    改成 4 份（778 MiB）重测：**146.1 µs**，确实不动了 —— 结论碰巧站得住，
    但当时的证据不足以支持它。

    结论的正确版本，用两个独立测量钉住：

    | 测量 | 字节 | p50 | 达到的带宽 |
    |---|---|---|---|
    | GEMM `[1,4096]×[24896,4096]`，4 份轮转 | 194.5 MiB | 146.1 µs | **1.40 TB/s** |
    | `w.sum()` 同样的字节 | 194.5 MiB | 167.7 µs | 1.22 TB/s |
    | DSA KV 池整读 1.19 GiB（远超任何复用） | 1217.4 MiB | 1100.5 µs | 1.16 TB/s |

    **1.25 TB/s 不是一个数，是两个**：GEMM 走的读路径能到 ~1.40 TB/s，
    reduce 类走的能到 ~1.16–1.22 TB/s。用例打印地板时仍用 1.25（跟
    REPORT.md 一致，好比对），**但 §6.2 的判决用 1.40 复核过**，
    结论是那两个 GEMM 比 1.25 的表看起来**更贴墙**。

13. **同一张空闲 die、同一个脚本、隔几分钟跑两次，单个小算子能差 34%。**
    实测 `causal_conv1d_4`：一次 **11.89 µs**，另一次 **15.88 µs**（两次都是
    30 replay 的 p50，两次的 p90/p50 都 < 1.05，所以不是尾巴，是整个分布挪了）。
    `BatchMatMulV2` 同样在 8.66 和 11.58 之间跳。
    **这是第 8 条在这两个用例上的具体形态**：p50 稳、p90 稳，但**跨 run 不稳**。
    所以判一个算子改动有没有效，必须**同一次 run 里 A/B**，不能拿今天的数
    去比昨天的数。大 GEMM 不受影响（`MatMulV2 "1,4096;24896,4096"` 三次分别是
    166.4 / 165.2 / 168.3 µs，1.2% 内）——**受影响的正好是那些「固定开销主导」
    的小算子，也就是 §6.3 里真正值得优化的那些。**

14. **一次 AI core 异常会往你的当前目录扔 200 MB。** 本用例开发时踩到一个
    `aicore exception (507015)`，CANN 在 **cwd** 下建了
    `extra-info/data-dump/0/exception_info.<...>`（204 MB）+ 几个算子二进制，
    另外每次跑都会掉一个 `fusion_result.json`。
    **`/mnt/workspace` 只剩 17 GB，而且这些文件是 `-r--------`，`ls` 不留神就漏了。**
    跑完记得 `rm -rf extra-info fusion_result.json`，或者干脆在
    `/var/tmp/glm53/` 下建个空目录 `cd` 进去再跑
    （用例本身的 profiler 输出已经写在 `/var/tmp/glm53/oplab/`，可以用
    `OPLAB_PROF_DIR` 改）。

15. **「总数对上」和「清单对上」是两回事，而且前者会骗你。**
    DSA 用例第一版总偏差 **−3.1%**，看起来很好；用清单比对一查，
    **7 组算子整个缺失、3 个最大算子的输入 shape 全错、5 组次数不对**。
    缺一个 kernel 只会让用例显得更快，而错的 shape 会让算子团队去优化不存在的形状。
    把参数对齐（`--ref-seq` 512→256）之后总偏差**反而变大到 −7.2%**——
    512 那次更接近，是因为 SFA 高了 47% 在补偿别处偏低。
    **判据必须是清单**（`regress_against_network.py`），
    **总偏差变大不等于改坏了。**


---

## 8. 测量纪律（用例里已经做了的）

* **入图。** `torch_npu.npu.NPUGraph()` + `torch_npu.npu.graph(g)` 捕获，然后
  `g.replay()`。eager 的墙钟在这里是 **686 µs/层**，图 replay 是 **304 µs** ——
  eager 量的是 host，不是 device。`bench_kda_layer.py --sections eager` 会把这个
  差打出来。
* **捕获前先热身 ≥10 次。** Ascend 上第一次跑一个新 shape 要付编译和 tiling 选择，
  Triton kernel 还要 JIT。冷着捕获会把编译一起捕进去。
* **图捕获里不能有 `int(tensor)` / `.item()` / `.nonzero()` / `bool(t.any())`。**
  真实代码里这些都被刻意消掉了（DSA 的 run 分段、kpool 的 closing 掩码都是
  「全算 + 掩码」而不是「筛选」，就是为了这个），两个用例也一样。
* **读 profiler 的 device `Duration(us)`，不读墙钟。** 墙钟只在 `eager` 段出现，
  而且只是为了说明它不能用。
* **30 次取 p50，同时打 p90 和 max。** 本项目最关键的一条证据就是分布形状：
  一个 GEMM 中位数只降 2.4% 而 p90 从 338 掉到 183 —— 只看均值会把真相盖掉。

---

## 9. 参考文件

* `../int8_singlecard/data/kernel_attribution_cfgI.txt` — 两个用例的回归目标，
  全量 kernel 清单
* `../int8_singlecard/REPORT.md` — 构型 I 的来历，§6.1–§6.5 是 KDA 的五条优化
* `../tools/hcpre_microbench.py` — 同一类交付物的前作（`HcPre` 单算子），
  这两个用例的 `refs` 段就是抄它的
* `../tools/profile_server_decode.py` — cfgI 的采数驱动，
  **`--prompt-tokens` 默认 13**，见 §5.3
* `../layer_check/check_kda.py` / `check_dsa.py` — 精度侧的单层用例
  （这两个用例只管性能）

* `regress_against_network.py` + `reference_inventory_cfgI.json` —— 另一个 session
  写的**清单比对**工具（不是本用例的一部分，但**是 §5.2 的判据**）。
  它按 `(算子, 输入 shape)` 分组比对「整网每层」与「用例每次 replay」，
  比总时间严格得多。两个用例现在都以它的输出为准。
