# GLM-5.3-Flash 昇腾开发环境搭建（复现文档）

本文把 2026-08-27 一次实际搭建过程沉淀成可照抄的步骤。目标是让新机器 **一次做对**，
不用重踩里面每一个坑。

- **验证过的环境**：**Ubuntu 24.04.3 / glibc 2.39 / aarch64 / Atlas A3（`Ascend910_9362`）× 16 die**
  （2026-08-28 全流程重跑到 P0.5 通过；此前在 Ubuntu 20.04 上也验证过，需附录 B 绕行）
- 配套计划文档：[`PLAN.md`](./PLAN.md)

> ✅ 2026-08-28 更新：24.04 上已**逐条实测通过**，**附录 B 整段不需要**——
> 两个 `.run` 算子包直接安装成功，不需要 `--force`，`.so` 全部正常 dlopen。
> 相对 20.04 的差异全部记在**附录 D**。

---

## 0. 先确认硬件与 CANN

```bash
npu-smi info                                   # 应看到 die 数与 HBM
python3 -c "import torch,torch_npu;print(torch.npu.get_device_name(0))"   # 装完后再跑
```

**认 SoC 别靠猜。** `npu-smi` 对 A2(910B) 和 A3(910_93) 都显示 `Ascend910`，
必须用 `torch.npu.get_device_name(0)`：

| 返回 | 型号 | sgl-kernel-npu 该选哪档 |
|---|---|---|
| `Ascend910B*` | Atlas A2 | `910b` |
| `Ascend910_93*` | **Atlas A3** | **`a3`** ← 本项目 |

CANN 版本同样别只看包名：

```bash
cat $ASCEND_TOOLKIT_HOME/*/ascend_toolkit_install.info   # 外层包名，可能不准
cat $ASCEND_TOOLKIT_HOME/compiler/version.info          # ← 以组件为准
bash $ASCEND_TOOLKIT_HOME/query_pkg_version.sh          # 全组件一览
```

> 本次实测：`ascend_toolkit_install.info` 写 `version=9.2.0`，
> 但 `compiler` / `opp` / `hccl` / `bisheng-compiler` 等**全部组件都是 `9.1.0`**。
> → 按 **9.1.0** 选包。

本次环境（供对照）：

| 项 | 值 |
|---|---|
| SoC | `Ascend910_9362`（A3），16 die × 64 GB |
| CANN | 组件 9.1.0（外层标 9.2.0），装在 `/home/developer/Ascend/ascend-toolkit/` |
| driver | 25.5.5，`/usr/local/Ascend/driver` |
| CPU / RAM | 320 核 / 1.8 TB |

---

## 1. 网络规则（先设对，否则处处超时）

```bash
# 代理只对 GitHub / Anthropic 有效
export https_proxy=http://127.0.0.1:1056 http_proxy=http://127.0.0.1:1056

# 其他一律直连，必须先 unset，否则挂到超时
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

| 站点 | 走法 | 实测速度 |
|---|---|---|
| `repo.huaweicloud.com/repository/pypi/simple` | 直连 | **46 MB/s** ← pip 默认源 |
| `mirrors.huaweicloud.com/ascend/repos/pypi` | 直连 | 快（triton-ascend 在这） |
| `ports.ubuntu.com` | 直连 | 快 |
| `hf-mirror.com` / `gitcode.com` / `modelscope.cn` | 直连 | 通 |
| `github.com` **API** | 代理 | 1.4 s |
| `github.com` **release 下载** | ⚠ 代理只有 ~8 KB/s | 改用 `https://gh-proxy.com/<原url>` **直连**，~300 KB/s |
| `pypi.org` | ✗ 不通 | — |

---

## 2. Python 3.12 环境

**为什么必须 3.12**：sgl-kernel-npu 的发布矩阵里 `py311` 只配 `cann9.0.0`，
`py312` 才配 `cann9.1.0`。安装文档写的 "Only python==3.11" 对应的是 9.0.0 档。

24.04 上 `python3` 就是 3.12（本机 3.12.9），直接建 venv 即可，**不需要 uv**：

```bash
python3 -V                              # 期望 3.12.x
python3 -m venv $ROOT/.venv-glm53
```

> 20.04 上没有 3.12，当时用 uv 装的；uv 带来的坑见附录 A（A2）与附录 D。

---

## 3. torch 2.10.0 + torch_npu 2.10.0.post4

**坑 1：必须每条 pip 命令都显式带 `-i <华为源>`。**
华为源虽然配在 `/home/developer/.pip/pip.conf`，但**新版 pip（26.x）不再读这个 legacy 路径**
（`pip config list` 输出为空即可确认）。不带 `-i` 就会回落到 `pypi.org` —— 本机不通，
表现是**完全没有输出地挂死十几分钟**，很容易误判成"在下载"。

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
IDX=https://repo.huaweicloud.com/repository/pypi/simple
V=$ROOT/.venv-glm53

$V/bin/pip install -i $IDX "torch==2.10.0" "torch_npu==2.10.0.post4"
```

> 实测两个 wheel（139 MB + 35 MB）从华为源下完只要几秒。

**坑 2：torch_npu 的 wheel 只声明了 `torch==2.10.0`，运行期依赖一堆没写。**
不补齐会在 `import torch_npu` 时报 `No module named 'yaml'` / `'numpy'`。
`numpy` 必须钉 1.26（2.x 与 torch_npu 不兼容）：

```bash
$V/bin/pip install -i $IDX \
  pyyaml "numpy==1.26.4" decorator cffi psutil protobuf attrs "scipy==1.13.1" requests \
  absl-py cloudpickle ml-dtypes tornado packaging tzdata pybind11
```

> `pybind11` 是 2026-08-28 新补的：`sgl_kernel_npu` 的 KDA / causal-conv1d 在 import 期
> 就要 `pybind11`，缺了 §7 第 3 段会 6/6 全 FAIL（报 `No module named 'pybind11'`）。

**坑 3：不 `source set_env.sh` 就 import，会报 `Failed to load the backend extension`**，
真实原因被掩盖（`libhccl.so` 找不到）。

验证：

```bash
source /home/developer/Ascend/ascend-toolkit/set_env.sh
$ROOT/.venv-glm53/bin/python -c "
import torch, torch_npu
print(torch.__version__, torch_npu.__version__, torch.npu.device_count(),
      torch.npu.get_device_name(0))"
# 期望：2.10.0+cpu 2.10.0.post4 16 Ascend910_9362
```

---

## 4. sgl-kernel-npu 三件套

从 https://github.com/sgl-project/sgl-kernel-npu/releases 取（本次用 tag `20260826`），
**按 §0 的 SoC 与 CANN 结论选档**，本项目是 `cann9.1.0-a3-aarch64` + `py312`：

| 文件 | 提供什么 |
|---|---|
| `custom-ops-<tag>-torch2.10.0-cann9.1.0-a3-aarch64.zip` | `hc_pre`/`hc_post`/`hc_pre_sinkhorn`、`swiglu_clip_quant`、`moe_gating_top_k_hash` … |
| `ops-transformer-<tag>-torch2.10.0-cann9.1.0-a3-aarch64.zip` | `compressor`、`quant_lightning_indexer`、`sparse_attn_sharedkv`（+metadata） |
| `sgl-kernel-npu-<tag>-torch2.10.0-py312-cann9.1.0-a3-aarch64.zip` | 4 个 wheel：`sgl_kernel_npu` / `attentions` / `deep_ep` / `torch_memory_saver` |

下载走 gh-proxy（**不要**开代理）：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -L -O https://gh-proxy.com/https://github.com/sgl-project/sgl-kernel-npu/releases/download/20260826/<file>
```

### 4.1 装两个 CANN 算子包

装到**独立目录**，不污染共享 toolkit：

```bash
source /home/developer/Ascend/ascend-toolkit/set_env.sh
OPP=$ROOT/opp_custom && mkdir -p $OPP
unzip -o custom-ops-*.zip && unzip -o ops-transformer-*.zip
./CANN-custom_ops-none-linux.aarch64.run        --quiet --install-path=$OPP
./cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=$OPP
```

> 装完得到两个 vendor：`$OPP/vendors/customize` 与 `$OPP/vendors/custom_transformer`，
> 各自带一个 `bin/set_env.bash`，**两个都要 source**。
>
> ✅ 24.04 实测：两个 `.run` 都直接 `SUCCESS`，**不需要 `--force`**。
> （glibc < 2.34 的机器上 `ops-transformer` 会在 "validate shared libraries" 阶段失败，
> 需加 `--force` 跳过校验并配合附录 B。）

### 4.2 装 python wheel

```bash
unzip -o sgl-kernel-npu-*.zip
$V/bin/pip install -i $IDX \
  custom_ops-*.whl sgl_kernel_npu-*.whl attentions-*.whl \
  torch_memory_saver-*.whl deep_ep-*.whl
```

---

## 5. triton-ascend（KDA / causal-conv1d 靠它）

**坑 4：必须用稳定版 `3.2.2`，且必须 `--no-deps`。**

- nightly `3.6.0` 的 `triton/language/extra/` 里**只有 `cuda`/`hip`，没有 `cann`** → `sgl_kernel_npu` 全线 import 失败
- 不加 `--no-deps` 会连带装上游 `triton==3.5.0`，它的 `triton/_C/libtriton` 覆盖掉 triton-ascend 自带的，报
  `ImportError: cannot import name 'amd' from 'triton._C.libtriton'`

```bash
$V/bin/pip uninstall -y triton triton-ascend
rm -rf $V/lib/python3.12/site-packages/triton $V/lib/python3.12/site-packages/triton{-,_ascend-}*
$V/bin/pip install -i $IDX \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --no-deps "triton-ascend==3.2.2"

ls $V/lib/python3.12/site-packages/triton/language/extra/   # 必须看到 cann
```

---

## 6. 环境脚本

把 [`env.sh.example`](./env.sh.example) 复制成 `$ROOT/env.sh`（内容即本机在跑的那份），
之后一律 `source $ROOT/env.sh` 再干活，用 `npy` 代替 `python`：

```bash
cp docs/docs/glm53_npu_support/env.sh.example $ROOT/env.sh
source $ROOT/env.sh
npy -c "import torch_npu; print(torch.npu.get_device_name(0))"
```

---

## 7. 验收：算子可见性

跑 `probe/p0_5_ops.py`（本仓库 `docs/docs/glm53_npu_support/probe/` 下）。**期望**：

24.04 实测输出（2026-08-28）：

```
### 1. python 包
  [OK]   import custom_ops
  [OK]   import sgl_kernel_npu
  [OK]   import attentions            ← 24.04 上由 FAIL 变 OK
  [OK]   import torch_memory_saver
  [FAIL] import deep_ep               ← 见下方"可以不管的两项"

### 2. torch.ops.custom.*
  [OK] compressor / inplace_partial_rotary_mul / npu_hc_post / npu_hc_pre
  [OK] npu_moe_gating_top_k / npu_quant_lightning_indexer(+_metadata)
  [OK] npu_sparse_attn_sharedkv(+_metadata)
  [MISS] npu_mla_prolog_v3   ← 见下

### 3. KDA / conv1d
  [OK] causal_conv1d_fn_npu / causal_conv1d_update_npu
  [OK] fused_kda_gate_npu / chunk_gla_fwd_o_gk_npu
  [OK] chunk_gated_delta_rule_fwd_h_npu / kda_target_verify_npu
```

> 第 3 段若整段报 `No module named 'pybind11'`，是 §3 坑 2 的依赖没补齐，`pip install pybind11` 即可。

**`import deep_ep` 失败要修**（TP16 + EP / DeepEP 路径必须有它），修法见 §7.1。

**唯一可以不管的一项**：

| 现象 | 结论 |
|---|---|
| `torch.ops.custom.npu_mla_prolog_v3` MISS | 只在 `mla_preprocess.py:458` 的 `forward_mlaprolog` 用（融合 prolog，`SGLANG_NPU_USE_MLAPO` 默认关）。而 **`torch_npu.npu_mla_prolog_v3` 原生存在** —— 真要用改个命名空间即可 |

### 7.1 修 `import deep_ep`（wheel 的打包 bug）

**症状**：`ModuleNotFoundError: No module named 'deep_ep_cpp'`。

**不是 GLIBC，也不是缺依赖**。`deep_ep_cpp.cpython-312-aarch64-linux-gnu.so` 确实装进去了，
但它躺在 `site-packages/deep_ep/` **包目录内**，而 `deep_ep/__init__.py` 写的是

```python
from deep_ep_cpp import Config     # ← 顶层导入，不是 from .deep_ep_cpp
```

顶层导入只搜 `sys.path`，搜不到包目录里的东西。放一个 `.pth` 把该目录也加进 `sys.path` 即可：

```bash
SP=$VENV/lib/python3.12/site-packages
echo "$SP/deep_ep" > $SP/_deep_ep_cpp_path.pth

npy -c "import torch, torch_npu; from deep_ep import Buffer, Config; print('deep_ep OK')"
```

> 包里那个 `deep_ep/vendors/hwcomputing/bin/set_env.bash` **不要 source** —— 里面是 CI 的
> 硬编码路径（`/__w/sgl-kernel-npu/...`），在本机不存在。`deep_ep/__init__.py` 会在 import
> 时自己把正确的 `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH` 拼好。

> `import attentions` 在 20.04 上是 FAIL（GLIBC），24.04 上已 OK。它只被
> `python/sglang/multimodal_gen/`（diffusion）用，LLM 路径不碰，OK 与否都不影响本项目。

---

## 8. SGLang 本体与它的依赖

### 8.1 不要 `pip install -e python/`

`python/pyproject.toml` 是 **CUDA 变体**，会拉入 `torch`/`flashinfer`/`cuda-python`
顶掉 torch 2.10，把 torch_npu 绑到一个不存在的 torch 上，容器基本就废了。
**用 `PYTHONPATH` 跑源码树**：

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH=$REPO/python:$PYTHONPATH
npy -c "import sglang; print(sglang.__file__)"   # 必须指向 $REPO/python/sglang/...
```

### 8.2 装依赖时**必须**带 constraints —— 否则同样会被顶掉

**坑 5（2026-08-28 实测，差点重演"容器报废"）**：即使只装
`pyproject_npu.toml` 的 `dependencies`（里面**没有** torch），pip 依然会装上
**torch 2.13.0（CUDA 版）+ 15 个 `nvidia-*` 包 + 上游 `triton` 3.7.1**。

依赖链是：

```
timm==1.0.16  ->  torchvision 0.28.0  ->  torch==2.13.0（硬 pin）  ->  nvidia-* / triton
```

`torchvision` 对 torch 是**精确等值 pin**，所以只要 pip 选了新的 torchvision，
torch 就一定被换掉；顺带的上游 `triton` 还会覆盖掉 triton-ascend（见坑 4）。
注意 `pyproject_npu.toml` 的直接依赖里既没有 torch 也没有 torchvision，
**光看依赖表看不出这个风险**。

解法是钉一个 constraints 文件（torchvision 0.25.0 ↔ torch 2.10.0）：

```bash
cat > $ROOT/npu-constraints.txt <<'EOC'
torch==2.10.0
torchvision==0.25.0
EOC

python3 -c "import tomllib;d=tomllib.load(open('$REPO/python/pyproject_npu.toml','rb'));\
open('$ROOT/npu-requirements.txt','w').write('\n'.join(d['project']['dependencies'])+'\n')"

pip install -i $IDX -c $ROOT/npu-constraints.txt -r $ROOT/npu-requirements.txt
```

**动手装之前先 `--dry-run` 验一遍**，5 秒就能确认没被顶掉：

```bash
pip install --dry-run --report /tmp/r.json -i $IDX \
  -c $ROOT/npu-constraints.txt -r $ROOT/npu-requirements.txt
python3 -c "
import json; r=json.load(open('/tmp/r.json'))
bad=[(x['metadata']['name'],x['metadata']['version']) for x in r['install']
     if 'nvidia' in x['metadata']['name'] or x['metadata']['name'] in ('torch','triton')]
print('DANGEROUS:', bad or 'none')"
# 期望：DANGEROUS: none   （不带 -c 时会打印 torch 2.13.0 + 15 个 nvidia-*）
```

装完再确认一次：

```bash
npy -c "import torch,torch_npu;print(torch.__version__,torch_npu.__version__,torch.npu.device_count())"
# 期望仍是：2.10.0+cpu 2.10.0.post4 16
```

---

---

## 9. 起 DeepSeek-V4-Flash 服务（P0.7 冒烟用）

现成脚本：[`launch_dsv4_a3.sh.example`](./launch_dsv4_a3.sh.example)。

配方来自上游 PR [sgl-project/sglang#25144](https://github.com/sgl-project/sglang/pull/25144)
（`[NPU] Add Ascend NPU support for DeepSeek-V4`，已合入 main），按本机路径改写。
**A3 单节点 TP16 / DP16 / EP16 + DeepEP**，275 GB 权重直接放 HBM，
**不需要** `deepseek_v4_flash.mdx` 那套 A2 单卡 KT CPU 卸载。

两个必须知道的点：

| 点 | 说明 |
|---|---|
| `--quantization compressed-tensors` | PR 里写的是 `modelslim`，那是另一份权重。我们这份 modelscope 权重的 `config.json` 自述 `quant_method=compressed-tensors`，**照抄 PR 会被 SGLang 直接拒绝启动** |
| `INF_NAN_MODE_FORCE_DISABLE=1` | PR 标注为**必须**：不设会让 W8A8 溢出产生 NaN。这条对后面 GLM 的 W8A8（P5）同样适用 |

实测（2026-08-28）：权重 **28.08 GB/die**（16 个 rank 一致），NPU graph 捕获通过，约 2 分钟起好。
DP-attention 每个 rank 各存一份 attention/dense 权重，所以比纯 TP16 不开 DP（18.45 GB/die）高，
**两个数不可直接比**。

验收两步，**两步都过才算**（端口通不是验收信号）：

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:30000/health_generate   # 200
curl -s http://127.0.0.1:30000/generate -H 'Content-Type: application/json' \
  -d '{"text":"What is the capital of France?","sampling_params":{"max_new_tokens":64,"temperature":0}}'
# "text" 必须非空 —— 200 但 "text":"" 是**失败**
```

停服务（只 `kill -INT` 不够，主进程可能挂住 30 s）：

```bash
kill -INT ${PID}; sleep 5; kill -TERM ${PID}
```

## 附录 A：本次踩坑速查

| # | 现象 | 原因 / 解法 |
|---|---|---|
| A1 | `import torch_npu` → `Failed to load the backend extension` | 没 `source set_env.sh`；或缺 `yaml`/`numpy`（真实报错被掩盖，直接 `import torch_npu` 看原始栈） |
| A2 | uv 下载龟速 | uv 不读 pip.conf；用 `--default-index`，或 curl 直下 wheel |
| A3 | GitHub release 下载 ~8 KB/s | 代理对 release 很慢；改 `gh-proxy.com` 且**不开代理** |
| A4 | `sgl_kernel_npu` 全线 `No module named 'triton.language.extra.cann'` | 装成了 nightly triton-ascend；换 `3.2.2` |
| A5 | `cannot import name 'amd' from 'triton._C.libtriton'` | 上游 `triton` 覆盖了 triton-ascend；卸干净后 `--no-deps` 重装 |
| A6 | `pkill -f "uv pip install"` 把自己的 shell 也杀了 | `pkill -f` 会匹配到当前命令行本身（命令文本里含该字符串）。用 `pgrep -f "[u]v pip"` 之类的括号法 |
| A7 | 权重目录 `Permission denied` | 别人上传的权重是 `-rw-r-----`；需 owner `chmod -R a+rX <dir>` |
| A8 | pip 装依赖时开始下 `nvidia-*` / `torch 2.13` | `timm -> torchvision -> torch==2.13.0` 硬 pin。**立刻 Ctrl-C**，改用 §8.2 的 constraints |
| A9 | pip 长时间零输出后超时 | 没带 `-i $IDX`，回落到不通的 `pypi.org`（pip 26 不读 `~/.pip/pip.conf`）。见 §3 坑 1 |

## 附录 B：glibc < 2.34 的绕行（**24.04 不需要**）

**症状**：`custom-ops` / `ops-transformer` / `sgl_kernel_npu` 的 `.so` 全部
`version 'GLIBC_2.32' not found` 或 `GLIBC_2.34 not found`。

**原因**：这些 `.so` 在 CI 的 Ubuntu 22.04 镜像里编译，引用了
`__libc_single_threaded@GLIBC_2.32`、`dlopen/dlsym/dlclose/dlerror@GLIBC_2.34`、
`__pthread_key_create/pthread_once@GLIBC_2.34`，以及 `GLIBCXX_3.4.29`。
**与 CANN 版本无关**：`cann9.0.0` 档实测同样需要 2.32/2.34。

**无效的做法**（我验证过，别浪费时间）：
LD_PRELOAD 一个重导出这些符号的 shim。失败原因是 `.gnu.version_r` 里的版本需求
**指名要 `libc.so.6` 提供 `GLIBC_2.32`**，预加载另一个 soname 满足不了这个检查。

**有效的做法**：把 Ubuntu 22.04 的 glibc 解到独立 prefix，让整个进程跑在新 loader 下。
方向是对的 —— **glibc 向后兼容：老二进制跑新 glibc 安全，反之不行**，
所以昇腾 driver 那些 2.31 编的 `.so` 在 2.35 下没问题。不改任何 `.so`，不动系统，不需要 root。

```bash
P=$ROOT/sysroot22 && mkdir -p $P/debs && cd $P/debs
B=http://ports.ubuntu.com/ubuntu-ports/pool/main
curl -LO $B/g/glibc/libc6_2.35-0ubuntu3.14_arm64.deb
curl -LO $B/g/gcc-12/libstdc++6_12.3.0-1ubuntu1~22.04.3_arm64.deb
curl -LO $B/g/gcc-12/libgcc-s1_12.3.0-1ubuntu1~22.04.3_arm64.deb
mkdir -p $P/root && for d in *.deb; do dpkg-deb -x $d $P/root; done
```

然后把 `env.sh` 里的 `npy()` 换成：

```bash
SYSROOT=$ROOT/sysroot22/root
export NPU_LOADER=$SYSROOT/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1
export NPU_LIBPATH="$SYSROOT/lib/aarch64-linux-gnu:$SYSROOT/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH:/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu"
npy() { $NPU_LOADER --library-path "$NPU_LIBPATH" $VENV/bin/python "$@"; }
```

验证（20.04 上实测全部由 FAIL 变 OK）：

```bash
$NPU_LOADER --library-path "$NPU_LIBPATH" $SYSROOT/lib/aarch64-linux-gnu/libc.so.6 | head -1
# GNU C Library (Ubuntu GLIBC 2.35-0ubuntu3.14) stable release version 2.35.
```

**24.04 校验一下就能整段跳过**：`ldd --version` ≥ 2.34 且
`strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.29` 有输出即可。

---

## 附录 C：本次装出来的确切版本

| 包 | 版本 |
|---|---|
| Python | 3.12.9（24.04 系统自带；20.04 那轮是 uv 装的 3.12.14） |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post4 |
| triton-ascend | 3.2.2（**不是** nightly 3.6.0） |
| sgl-kernel-npu | 2026.6.1 |
| custom-ops | 1.0 |
| attentions | 0.2 |
| deep-ep | 1.0.0+146153e5.cann.9.1.0.b243 |
| torch-memory-saver | 0.0.8 |
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| pybind11 | 3.1.0（**新增**，`sgl_kernel_npu` import 期需要） |
| tzdata | 2026.3 |

---

## 附录 D：Ubuntu 24.04 相对 20.04 的差异（2026-08-28 实测）

| 项 | 20.04（旧） | **24.04（现在）** |
|---|---|---|
| glibc / libstdc++ | 2.31 / GLIBCXX < 3.4.29 → **必须**附录 B 的独立 loader | **2.39 / 13** → **附录 B 整段跳过** |
| `.run` 算子包安装 | `ops-transformer` 需 `--force` 跳过 so 校验 | **两个都直接 `SUCCESS`** |
| Python 3.12 | 系统没有，用 uv 装 | **系统自带 `python3` = 3.12.9**，`python3 -m venv` 即可，不用 uv |
| 包管理器 | uv（`--default-index`） | **plain pip**，但**每条都要显式 `-i $IDX`**（见 §3 坑 1） |
| `import attentions` | FAIL（GLIBC） | **OK** |
| `pybind11` | 未发现需要（可能被 uv 顺带装上） | **必须显式装**，否则 §7 第 3 段 6/6 FAIL |
| `import deep_ep` | FAIL（`deep_ep_cpp`） | 同样 FAIL，但**已定位并修好**：是 wheel 打包 bug，不是环境问题（§7.1） |

**24.04 上新踩的唯一一个坑**：pip 26 不读 `~/.pip/pip.conf`（legacy 路径），
不带 `-i` 就静默回落 pypi.org 挂死。用 `pip config list` 输出是否为空可以 5 秒确认。
