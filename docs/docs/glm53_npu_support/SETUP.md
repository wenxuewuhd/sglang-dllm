# GLM-5.3-Flash 昇腾开发环境搭建

照抄即可。**验证过的环境**：Ubuntu 24.04.3 / glibc 2.39 / aarch64 / Atlas A3（`Ascend910_9362`）× 16 die。

配套：[`PLAN.md`](./PLAN.md) 计划与算子结论 ｜ [`env.sh.example`](./env.sh.example) 环境脚本 ｜
[`probe/`](./probe/) 探测脚本 ｜ [`tools/`](./tools/) 权重转换与 golden 工具

---

## 0. 先认硬件与 CANN

**认 SoC 不能靠猜**：`npu-smi` 对 A2(910B) 和 A3(910_93) 都显示 `Ascend910`。

```bash
python3 -c "import torch,torch_npu;print(torch.npu.get_device_name(0))"
```

| 返回 | 型号 | sgl-kernel-npu 选哪档 |
|---|---|---|
| `Ascend910B*` | Atlas A2 | `910b` |
| `Ascend910_93*` | **Atlas A3** | **`a3`** ← 本项目 |

**认 CANN 版本也不能看包名**：`ascend_toolkit_install.info` 写 `version=9.2.0`，
但 `compiler`/`opp`/`hccl`/`bisheng-compiler` 等**全部组件都是 9.1.0`**。以组件为准：

```bash
cat $ASCEND_TOOLKIT_HOME/compiler/version.info      # ← 以这个为准
bash $ASCEND_TOOLKIT_HOME/query_pkg_version.sh      # 全组件一览
```

本机：toolkit 在 `/home/developer/Ascend/ascend-toolkit/`（**不是** `/usr/local/Ascend`，那里只有 driver）；
driver 25.5.5；CPU 320 核 / 内存 1.8 TB。

---

## 1. 网络规则（先设对，否则处处超时）

代理 `http://127.0.0.1:1056` **只对 GitHub / Anthropic 有效**。访问其他站点前必须：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

| 站点 | 走法 |
|---|---|
| `repo.huaweicloud.com/repository/pypi/simple` | 直连，**46 MB/s** ← pip 用这个 |
| `mirrors.huaweicloud.com/ascend/repos/pypi` | 直连（triton-ascend 在这） |
| `hf-mirror.com` / `gitcode.com` / `modelscope.cn` / `ports.ubuntu.com` | 直连 |
| `github.com` **API** | 代理 |
| `github.com` **release 下载** | 代理只有 ~8 KB/s → 改用 `https://gh-proxy.com/<原url>` 且**不开代理** |
| `pypi.org` | ✗ 不通 |

---

## 2. Python 3.12 venv

**必须 3.12**：sgl-kernel-npu 的 `py311` 只配 `cann9.0.0`，`py312` 才配 `cann9.1.0`
（安装文档写的 "Only python==3.11" 对应 9.0.0 档）。24.04 的 `python3` 就是 3.12：

```bash
export ROOT=/mnt/workspace/y00359136/work/glm53_dev/env
python3 -m venv $ROOT/.venv-glm53
```

---

## 3. torch 2.10.0 + torch_npu 2.10.0.post4

**坑 1：每条 pip 命令都要显式 `-i`。** 新版 pip 不读 `~/.pip/pip.conf`（`pip config list` 输出为空即可确认），
不带 `-i` 会回落到不通的 `pypi.org`，表现是**完全没有输出地挂死十几分钟**，很像"在下载"。

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
IDX=https://repo.huaweicloud.com/repository/pypi/simple
V=$ROOT/.venv-glm53
$V/bin/pip install -i $IDX "torch==2.10.0" "torch_npu==2.10.0.post4"
```

**坑 2：torch_npu 只声明了 `torch==2.10.0`，运行期依赖一堆没写。** 不补齐会在 `import torch_npu` 时
报 `No module named 'yaml'` / `'numpy'`。`numpy` 必须钉 1.26：

```bash
$V/bin/pip install -i $IDX \
  pyyaml "numpy==1.26.4" decorator cffi psutil protobuf attrs "scipy==1.13.1" requests \
  absl-py cloudpickle ml-dtypes tornado packaging tzdata pybind11
```

> `pybind11` 不能少 —— `sgl_kernel_npu` 的 KDA / causal-conv1d 在 **import 期**就要它。

**坑 3：不 `source set_env.sh` 就 import**，会报 `Failed to load the backend extension`，真实原因被掩盖。

```bash
source /home/developer/Ascend/ascend-toolkit/set_env.sh
$V/bin/python -c "import torch,torch_npu;print(torch.__version__,torch_npu.__version__,
  torch.npu.device_count(),torch.npu.get_device_name(0))"
# 期望：2.10.0+cpu 2.10.0.post4 16 Ascend910_9362
```

---

## 4. sgl-kernel-npu 三件套

从 https://github.com/sgl-project/sgl-kernel-npu/releases 取（当前 tag `20260826`），
按 §0 的结论选 **`cann9.1.0-a3-aarch64` + `py312`**：

| 文件 | 提供什么 |
|---|---|
| `custom-ops-<tag>-torch2.10.0-cann9.1.0-a3-aarch64.zip` | `hc_pre`/`hc_post`、`swiglu_clip_quant`、`moe_gating_top_k_hash` … |
| `ops-transformer-<tag>-...-a3-aarch64.zip` | `compressor`、`quant_lightning_indexer`、`sparse_attn_sharedkv` |
| `sgl-kernel-npu-<tag>-torch2.10.0-py312-cann9.1.0-a3-aarch64.zip` | 4 个 wheel：`sgl_kernel_npu`/`attentions`/`deep_ep`/`torch_memory_saver` |

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # gh-proxy 不能开代理
curl -L -O https://gh-proxy.com/https://github.com/sgl-project/sgl-kernel-npu/releases/download/20260826/<file>
```

### 4.1 两个 CANN 算子包 → 装到独立目录，不污染共享 toolkit

```bash
source /home/developer/Ascend/ascend-toolkit/set_env.sh
OPP=$ROOT/opp_custom && mkdir -p $OPP
unzip -o custom-ops-*.zip && unzip -o ops-transformer-*.zip
./CANN-custom_ops-none-linux.aarch64.run        --quiet --install-path=$OPP
./cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=$OPP
```

装完得到 `$OPP/vendors/customize` 与 `$OPP/vendors/custom_transformer`，各带一个
`bin/set_env.bash`，**两个都要 source**。

### 4.2 python wheel

```bash
unzip -o sgl-kernel-npu-*.zip
$V/bin/pip install -i $IDX custom_ops-*.whl sgl_kernel_npu-*.whl attentions-*.whl \
  torch_memory_saver-*.whl deep_ep-*.whl
```

---

## 5. triton-ascend

**坑 4：必须稳定版 `3.2.2`，且必须 `--no-deps`。**

- nightly `3.6.0` 的 `triton/language/extra/` 里**只有 `cuda`/`hip`，没有 `cann`** → `sgl_kernel_npu` 全线 import 失败
- 不加 `--no-deps` 会连带装上游 `triton`，它的 `_C/libtriton` 覆盖掉 triton-ascend 自带的，
  报 `cannot import name 'amd' from 'triton._C.libtriton'`

```bash
$V/bin/pip uninstall -y triton triton-ascend
rm -rf $V/lib/python3.12/site-packages/triton $V/lib/python3.12/site-packages/triton{-,_ascend-}*
$V/bin/pip install -i $IDX --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --no-deps "triton-ascend==3.2.2"

ls $V/lib/python3.12/site-packages/triton/language/extra/   # 必须看到 cann
```

---

## 6. 环境脚本

把 [`env.sh.example`](./env.sh.example) 复制成 `$ROOT/env.sh`，之后一律
`source $ROOT/env.sh` 再干活，用 `npy` 代替 `python`。

---

## 7. 验收：算子可见性

```bash
cd $REPO && npy docs/docs/glm53_npu_support/probe/p0_5_ops.py
```

期望：python 包 4/5 OK（`deep_ep` 见 §7.1）、`torch.ops.custom.*` 9/10、KDA/conv1d 6/6。

**唯一可以不管的**：`torch.ops.custom.npu_mla_prolog_v3` MISS —— 只在融合 prolog 用
（`SGLANG_NPU_USE_MLAPO` 默认关），且 `torch_npu.npu_mla_prolog_v3` 原生存在。

> 第 3 段若整段报 `No module named 'pybind11'`，是 §3 坑 2 的依赖没补齐。

### 7.1 修 `import deep_ep`（wheel 的打包 bug）

**症状**：`ModuleNotFoundError: No module named 'deep_ep_cpp'`。

不是缺依赖 —— `.so` 确实装进去了，但它躺在 `site-packages/deep_ep/` **包目录内**，
而 `deep_ep/__init__.py` 写的是顶层导入 `from deep_ep_cpp import Config`，只搜 `sys.path`。
放个 `.pth` 把该目录也加进去：

```bash
SP=$V/lib/python3.12/site-packages
echo "$SP/deep_ep" > $SP/_deep_ep_cpp_path.pth
npy -c "import torch,torch_npu; from deep_ep import Buffer, Config; print('deep_ep OK')"
```

> 包里那个 `deep_ep/vendors/hwcomputing/bin/set_env.bash` **不要 source** —— 里面是 CI 的硬编码路径。
> `__init__.py` 会在 import 时自己拼好正确的 `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH`。

---

## 8. SGLang 本体与依赖

### 8.1 不要 `pip install -e python/`

`python/pyproject.toml` 是 **CUDA 变体**，会拉入 `torch`/`flashinfer`/`cuda-python` 顶掉 torch 2.10，
把 torch_npu 绑到一个不存在的 torch 上。用 `PYTHONPATH` 跑源码树：

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH=$REPO/python:$PYTHONPATH
npy -c "import sglang; print(sglang.__file__)"   # 必须指向 $REPO/python/sglang/...
```

### 8.2 装依赖**必须**带 constraints

**坑 5**：即使只装 `pyproject_npu.toml` 的 `dependencies`（里面**没有** torch），pip 依然会装上
**torch 2.13.0（CUDA 版）+ 15 个 `nvidia-*` + 上游 `triton`**。依赖链是：

```
timm  ->  torchvision  ->  torch==2.13.0（精确等值 pin）  ->  nvidia-* / triton
```

`torchvision` 对 torch 是**等值 pin**，所以只要 pip 选了新的 torchvision，torch 就一定被换掉。
**光看 `pyproject_npu.toml` 的直接依赖看不出这个风险**（那里既没有 torch 也没有 torchvision）。

```bash
cat > $ROOT/npu-constraints.txt <<'EOC'
torch==2.10.0
torchvision==0.25.0
EOC

python3 -c "import tomllib;d=tomllib.load(open('$REPO/python/pyproject_npu.toml','rb'));\
open('$ROOT/npu-requirements.txt','w').write('\n'.join(d['project']['dependencies'])+'\n')"

pip install -i $IDX -c $ROOT/npu-constraints.txt -r $ROOT/npu-requirements.txt
```

**动手前先 `--dry-run` 验一遍**，5 秒就能确认：

```bash
pip install --dry-run --report /tmp/r.json -i $IDX \
  -c $ROOT/npu-constraints.txt -r $ROOT/npu-requirements.txt
python3 -c "
import json; r=json.load(open('/tmp/r.json'))
bad=[(x['metadata']['name'],x['metadata']['version']) for x in r['install']
     if 'nvidia' in x['metadata']['name'] or x['metadata']['name'] in ('torch','triton')]
print('DANGEROUS:', bad or 'none')"
# 期望 none；不带 -c 时会打印 torch 2.13.0 + 15 个 nvidia-*
```

装完再确认一次 `torch 2.10.0+cpu / torch_npu 2.10.0.post4 / device_count 16`。

---

## 9. 起服务

现成脚本：[`launch_dsv4_a3.sh.example`](./launch_dsv4_a3.sh.example)（DeepSeek-V4-Flash，
A3 单节点 TP16/DP16/EP16 + DeepEP）。GLM 的在 `$ROOT/run/launch_glm_bf16.sh`。

| 必须知道的 | |
|---|---|
| `--quantization` | DSv4 那份 modelscope 权重自述 `compressed-tensors`；**照抄上游 PR 的 `modelslim` 会被 SGLang 直接拒绝启动** |
| `--page-size` | **GLM 必须 64**（DSA pool 有 `assert page_size == 64`）；**DSv4 是 128**。照搬会启动失败 |
| `INF_NAN_MODE_FORCE_DISABLE=1` | **必须设**，否则 W8A8 溢出产生 NaN |
| 停服务 | `kill -INT $PID; sleep 5; kill -TERM $PID` —— 只 INT 主进程可能挂住 30 s |

验收两步，**两步都过才算**（端口通不是验收信号）：

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:${PORT}/health_generate   # 200
curl -s http://127.0.0.1:${PORT}/generate -H 'Content-Type: application/json' \
  -d '{"text":"What is the capital of France?","sampling_params":{"max_new_tokens":64,"temperature":0}}'
# "text" 必须非空 —— 200 但 "text":"" 是失败
```

---

## 10. 参考环境 `.venv-ref`（CPU，HF golden 的唯一来源）

`layer_check/`、`tools/golden_*.py`、`tools/logit_check.py` 的**参考侧全部依赖它**。
换机不建它，所有对拍都做不了。

⚠ **必须是独立 venv**：sglang 钉死 `transformers==5.12.1`，而 `glm5_next`
**只在 5.16.1 才有**（5.16.0 没有）。装到一起两边都坏。
官方权重 repo **不带** `modeling_*.py`，`trust_remote_code` 这条路是空的。

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY   # 代理只对 github/anthropic 有效
python3.12 -m venv $ROOT/.venv-ref
$ROOT/.venv-ref/bin/pip install -i <华为源> \
    torch==2.10.0 --index-url <CPU 轮子源>            # CPU 版，不要 torch_npu
$ROOT/.venv-ref/bin/pip install -i <华为源> transformers==5.16.1 accelerate safetensors
```

实测装成的版本（`$ROOT/.venv-ref`）：

| 包 | 版本 |
|---|---|
| python | 3.12.9 |
| torch | **2.10.0+cpu** |
| transformers | **5.16.1** |
| accelerate / safetensors / tokenizers / numpy | 1.14.0 / 0.8.0 / 0.23.1 / 2.5.2 |

### 用它的两个坑

**① 模型类不是 `AutoModelForCausalLM`。** GLM-5.3-Flash 的架构是
`Glm5NextForConditionalGeneration`（带视觉塔），5.16.1 **定义了它但没注册到那个 auto class**，
而且**顶层没导出**，只能从模块里拿：

```python
from transformers.models.glm5_next import Glm5NextForConditionalGeneration
```

用 `AutoModelForCausalLM` 会得到 `Unrecognized configuration class`。

**② 整模型别直接 `from_pretrained`。** bf16 要 599 GB（1.8 TB 机器上勉强可以），
**fp32 要 1.2 TB —— 实测加载到 26% 就吃掉 1.26 TB 并开始换页，跑不完**。
需要 fp32 参考时走**流式**：`layer_check/trace_reference.py` 在 meta device 上建模型、
逐层物化再退回，**峰值只有一层**，45 层 128 token 约 3.5 分钟。

---

## 附录 A：踩坑速查

| 现象 | 原因 / 解法 |
|---|---|
| `import torch_npu` → `Failed to load the backend extension` | 没 `source set_env.sh`；或缺 `yaml`/`numpy`（§3 坑 3） |
| pip 长时间零输出后超时 | 没带 `-i $IDX`，回落到不通的 pypi.org（§3 坑 1） |
| pip 开始下 `nvidia-*` / `torch 2.13` | `timm → torchvision → torch` 等值 pin。**立刻 Ctrl-C**，改用 §8.2 |
| `sgl_kernel_npu` 全线 `No module named 'triton.language.extra.cann'` | 装成了 nightly triton-ascend；换 3.2.2（§5） |
| `cannot import name 'amd' from 'triton._C.libtriton'` | 上游 triton 覆盖了 triton-ascend；卸干净后 `--no-deps` 重装 |
| `No module named 'deep_ep_cpp'` | wheel 打包 bug，放 `.pth`（§7.1） |
| KDA/conv1d 6 项全 `No module named 'pybind11'` | §3 坑 2 |
| 权重目录 `Permission denied` | 别人上传的权重是 `-rw-r-----`，需 owner `chmod -R a+rX` |
| `pkill -f "..."` 把自己的 shell 也杀了 | `pkill -f` 会匹配到当前命令行本身；用 `pgrep -f "[u]v pip"` 的括号法 |

## 附录 B：glibc < 2.34 的机器（24.04 不需要）

预编译 `.so` 在 CI 的 Ubuntu 22.04 上编译，需 **GLIBC ≥ 2.34 + GLIBCXX ≥ 3.4.29**，与 CANN 版本无关。
24.04（glibc 2.39）无此问题；20.04（2.31）上表现为全部 `.so` 报 `version 'GLIBC_2.32' not found`。

**无效的做法**：LD_PRELOAD 一个重导出符号的 shim —— `.gnu.version_r` 指名要 `libc.so.6` 提供该版本，
预加载别的 soname 满足不了。

**有效的做法**：把 22.04 的 glibc 解到独立 prefix，让整个进程跑在新 loader 下（glibc 向后兼容，
老二进制跑新 glibc 安全）。不改任何 `.so`、不动系统、不需要 root：

```bash
P=$ROOT/sysroot22 && mkdir -p $P/debs && cd $P/debs
B=http://ports.ubuntu.com/ubuntu-ports/pool/main
curl -LO $B/g/glibc/libc6_2.35-0ubuntu3.14_arm64.deb
curl -LO $B/g/gcc-12/libstdc++6_12.3.0-1ubuntu1~22.04.3_arm64.deb
curl -LO $B/g/gcc-12/libgcc-s1_12.3.0-1ubuntu1~22.04.3_arm64.deb
mkdir -p $P/root && for d in *.deb; do dpkg-deb -x $d $P/root; done
```

然后把 `env.sh` 的 `npy()` 换成在该 loader 下启动：
`$P/root/lib/aarch64-linux-gnu/ld-linux-aarch64.so.1 --library-path "<sysroot libs>:$LD_LIBRARY_PATH:/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu" $VENV/bin/python`

## 附录 C：确切版本

| 包 | 版本 | | 包 | 版本 |
|---|---|---|---|---|
| Python | 3.12.9 | | sgl-kernel-npu | 2026.6.1 |
| torch | 2.10.0+cpu | | custom-ops | 1.0 |
| torch_npu | 2.10.0.post4 | | attentions | 0.2 |
| triton-ascend | **3.2.2**（不是 nightly） | | deep-ep | 1.0.0+146153e5.cann.9.1.0.b243 |
| numpy | 1.26.4 | | torch-memory-saver | 0.0.8 |
| scipy | 1.13.1 | | pybind11 | 3.1.0 |
| torchvision | 0.25.0（**由 constraints 钉住**） | | transformers | 5.12.1（**参考环境是 5.16.1**） |
