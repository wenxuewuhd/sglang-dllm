# GLM-5.3-Flash 昇腾开发环境搭建（复现文档）

本文把 2026-08-27 一次实际搭建过程沉淀成可照抄的步骤。目标是让新机器 **一次做对**，
不用重踩里面每一个坑。

- **验证过的环境**：Ubuntu 20.04 / aarch64 / Atlas A3（`Ascend910_9362`）× 16 die
- **本文主线针对**：Ubuntu 24.04 —— 步骤相同，但**不需要附录 B 的 glibc 绕行**
- 配套计划文档：[`PLAN.md`](./PLAN.md)

> ⚠ 诚实说明：正文步骤 1–7 在 **20.04 上逐条验证过**（附加附录 B 的绕行）。
> 24.04 上我**没有实测过**，但 24.04 的 glibc 2.39 / libstdc++ 13 覆盖了所有依赖，
> 附录 B 应当可以整段删掉。若 24.04 上出现新问题，请回来补充本文。

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

Ubuntu 24.04 自带 python3.12，可直接 `python3.12 -m venv`。若没有，用 uv：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
python3 -m pip install --user uv
export PATH=$HOME/.local/bin:$PATH
export UV_PYTHON_INSTALL_DIR=$ROOT/.uv-python
uv python install 3.12
uv venv --python 3.12 $ROOT/.venv-glm53
```

---

## 3. torch 2.10.0 + torch_npu 2.10.0.post4

**坑 1：uv 不读 `/home/developer/.pip/pip.conf`。**
pip 的华为源配在那里，uv 默认走 pypi.org（本机 ~30 KB/s，等于卡死）。
且 `UV_INDEX_URL` 在 uv 0.12 已废弃，要用 `--default-index`。

最省事的做法是 curl 直接下 wheel 再本地装：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
IDX=https://repo.huaweicloud.com/repository/pypi/simple
mkdir -p $ROOT/wheels && cd $ROOT/wheels
# 从 $IDX/torch/ 与 $IDX/torch-npu/ 的索引页取 href 后 curl -O
#   torch-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl          (139 MB)
#   torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_aarch64.whl (35 MB)

export VIRTUAL_ENV=$ROOT/.venv-glm53
uv pip install --default-index $IDX $ROOT/wheels/*.whl
```

**坑 2：torch_npu 的 wheel 只声明了 `torch==2.10.0`，运行期依赖一个没写。**
不补齐会在 `import torch_npu` 时报 `No module named 'yaml'` / `'numpy'`：

```bash
uv pip install --default-index $IDX \
  pyyaml numpy decorator cffi psutil protobuf attrs scipy requests \
  absl-py cloudpickle ml-dtypes tornado packaging
```

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
> ⚠ 在 glibc < 2.34 的机器上，`ops-transformer` 会在 "validate shared libraries" 阶段失败，
> 需加 `--force` 跳过校验并配合附录 B。**24.04 上不应出现这个问题**。

### 4.2 装 python wheel

```bash
unzip -o sgl-kernel-npu-*.zip
uv pip install --default-index $IDX \
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
uv pip uninstall triton triton-ascend
rm -rf $VENV/lib/python3.12/site-packages/triton{,-*,_ascend-*}
uv pip install --default-index $IDX --index-strategy unsafe-best-match \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --no-deps "triton-ascend==3.2.2"

ls $VENV/lib/python3.12/site-packages/triton/language/extra/   # 必须看到 cann
```

---

## 6. 环境脚本

把下面存成 `$ROOT/env.sh`，之后一律 `source env.sh` 再干活：

```bash
#!/bin/bash
ROOT=/mnt/workspace/y00359136/glm5.3_single_card_dev
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset LD_PRELOAD                     # 系统预置的 libgomp 会干扰 torch_npu 的 FunctionLoader

source /home/developer/Ascend/ascend-toolkit/set_env.sh
source $ROOT/opp_custom/vendors/customize/bin/set_env.bash
source $ROOT/opp_custom/vendors/custom_transformer/bin/set_env.bash
export ASCEND_CUSTOM_OPP_PATH=$ROOT/opp_custom/vendors/custom_transformer:$ROOT/opp_custom/vendors/customize

export VENV=$ROOT/.venv-glm53
npy() { $VENV/bin/python "$@"; }     # 24.04 直接用解释器；20.04 见附录 B
```

---

## 7. 验收：算子可见性

跑 `probe/p0_5_ops.py`（本仓库 `docs/docs/glm53_npu_support/probe/` 下）。**期望**：

```
### 1. python 包
  [OK]   import custom_ops
  [OK]   import sgl_kernel_npu
  [OK]   import torch_memory_saver
  attentions / deep_ep 见下方"可以不管的两项"

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

**可以不管的三项**（本次已确认不阻塞 LLM 路径）：

| 现象 | 结论 |
|---|---|
| `import attentions` 失败 | 只被 `python/sglang/multimodal_gen/`（diffusion）使用，LLM 路径不碰 |
| `import deep_ep` 失败（`No module named 'deep_ep_cpp'`） | 只被 `token_dispatcher/deepep.py` 使用；单卡冒烟不需要。**TP16 + EP 阶段要回头解决** |
| `torch.ops.custom.npu_mla_prolog_v3` MISS | 只在 `mla_preprocess.py:458` 的 `forward_mlaprolog` 用（融合 prolog，`SGLANG_NPU_USE_MLAPO` 默认关）。而 **`torch_npu.npu_mla_prolog_v3` 原生存在** —— 真要用改个命名空间即可 |

---

## 8. SGLang 本体

**不要 `pip install -e python/`** —— 那会装 `python/pyproject.toml`（CUDA 变体），
拉入 `torch`/`flashinfer`/`cuda-python` 顶掉 torch 2.10，把 torch_npu 绑到一个不存在的 torch 上，
容器基本就废了。

用 `PYTHONPATH`，或先换成 `python/pyproject_npu.toml`。

---

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
| Python | 3.12.14 |
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
