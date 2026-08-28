# Environment: how to reproduce and how to run the tests

There are **two** environments here and they do different jobs. Only one of them needs
an NPU.

| | Purpose | Needs an NPU? | Python |
|---|---|---|---|
| **Test env** | run the pytest suite in this package against the torch reference | no | any Python 3.10+ with `torch` (CPU) and `pytest` |
| **Target env** | run the same suite against the delivered Ascend operators | yes | the `.venv-glm53` recipe below |

---

## 1. Test env — the only thing you need to start

The reference implementations and tests import **nothing but `torch`**. No sglang, no
torch_npu, no CANN.

```bash
python -m pip install "torch" pytest          # CPU torch is fine
cd <this directory>
python -m pytest tests -q
```

Verified on this machine with **torch 2.7.1+cpu and pytest 9.0.3**: **60 passed, 1
skipped** in 6.3 s. The skip is the OP-3 `rope_dim>0` case (see the table below). The
reference modules were separately smoke-tested under **torch 2.10.0+cpu** in
`.venv-ref`, so the suite is not pinned to one torch version.

> The reference venv shipped with this project,
> `/mnt/workspace/y00359136/work/glm53_dev/env/.venv-ref` (CPU torch 2.10.0 +
> transformers 5.16.1, the first release containing `glm5_next`), has torch but **not**
> pytest. Add it with
> `.../.venv-ref/bin/pip install -i https://repo.huaweicloud.com/repository/pypi/simple pytest`
> — note the explicit `-i`, see §3.

Once an operator lands:

```bash
GLM53_OP_BACKEND=npu python -m pytest tests -q
```

That is the only switch. `reference/backend.py` is the single file that names the
delivered operators; if a name or signature differs from the spec, change it there and
nowhere else.

| Env var | Default | Meaning |
|---|---|---|
| `GLM53_OP_BACKEND` | `reference` | `reference` or `npu` |
| `GLM53_NPU_DEVICE` | `npu:0` | which die the NPU adapters use |
| `GLM53_TOL_SLACK` | `2.0` | multiplier on the measured noise floor (see [ACCEPTANCE.md](ACCEPTANCE.md)) |
| `GLM53_TOL_ABS_MIN` | `1e-6` | absolute floor under the measured floor |
| `GLM53_TEST_ROPE_CONVENTION` | `0` | enable the OP-3 `rope_dim>0` regression test; off because this package's RoPE convention is unverified (see [specs/op3](specs/op3_kv_norm_rope_cache_rope0.md) §8) |

---

## 2. Target env — build it from SETUP.md, do not re-derive it

**The authoritative, step-by-step recipe is
[`../SETUP.md`](../SETUP.md)** (Chinese; it is a transcript of one successful build on
this exact machine, with every trap it hit). Follow it rather than this section. What
follows is the index plus the traps you cannot afford to skim past.

| SETUP.md § | What it gives you |
|---|---|
| §0 | How to identify the SoC and the *real* CANN version |
| §1 | Network rules (see §3 below) |
| §2 | Python 3.12 and why 3.11 will not do |
| §3 | torch 2.10.0 + torch_npu 2.10.0.post4 and its undeclared runtime deps |
| §4 | The three `sgl-kernel-npu` release artifacts, incl. the two CANN op `.run` packages |
| §5 | triton-ascend **3.2.2** (not nightly, and `--no-deps`) |
| §6 | `env.sh` and the `npy` wrapper |
| §7 | Operator-visibility acceptance probe (`probe/p0_5_ops.py`) |
| §8 | SGLang itself via `PYTHONPATH`, and the constraints file |
| App. B | glibc < 2.34 workaround — **not needed on Ubuntu 24.04** |
| App. C | The exact version table that was built |

### The facts you will need for a build target

| | |
|---|---|
| Hardware | Atlas A3, SoC **`Ascend910_9362`**, 16 die × 64 GB |
| **Compile kernels for** | **`ascend910_93`** |
| CANN | components are **9.1.0** (the outer package name claims 9.2.0 — trust `$ASCEND_TOOLKIT_HOME/compiler/version.info`, not `ascend_toolkit_install.info`) |
| driver | 25.5.5 |
| OS | Ubuntu 24.04.3 / glibc 2.39 / aarch64 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4 |
| Python | 3.12.9 |
| Vendor op packages | `/mnt/workspace/y00359136/work/glm53_dev/env/opp_custom/vendors/{customize,custom_transformer}` |
| Runtime venv | `/mnt/workspace/y00359136/work/glm53_dev/env/.venv-glm53` |
| Entry point | `source /mnt/workspace/y00359136/work/glm53_dev/env/env.sh`, then use **`npy`** instead of `python` |

`npu-smi` displays `Ascend910` for both A2 (910B) and A3 (910_93). **Identify the SoC
with `torch.npu.get_device_name(0)`, not with `npu-smi`** (SETUP.md §0).

### Where the existing operators live

Both vendor packages ship op-info JSON, aclnn headers, and some AscendC source:

```
$OPP/vendors/custom_transformer/
    op_api/include/aclnnop/aclnn_compressor.h            # OP-2 aclnn signature
    op_proto/inc/compressor_proto.h                      # OP-2 IR
    op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json
    op_impl/ai_core/tbe/custom_transformer_impl/ascendc/compressor/arch22/
        rms_norm.h                                       # OP-2: the norm to extend
        compressor_block_vec_perf.h                      # OP-2: :1256-1260 call site
$OPP/vendors/customize/
    op_api/include/aclnn_dequant_swiglu_clamp_quant.h    # OP-4 aclnn signature
    op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json
    op_impl/ai_core/tbe/customize_impl/ascendc/dequant_swiglu_clamp_quant/
        dequant_swiglu_clamp_quant.h                     # OP-4: :606-648 SwiGluGate
```

Each vendor dir has its own `bin/set_env.bash`. **Both must be sourced.**

---

## 3. Two traps that are worth repeating out of SETUP.md

**pip needs an explicit `-i`.** pip 26 no longer reads the legacy `~/.pip/pip.conf`, so
without `-i` it silently falls back to `pypi.org`, which is unreachable here. The symptom
is *no output at all* for ten-plus minutes, which reads as "downloading".

```bash
pip install -i https://repo.huaweicloud.com/repository/pypi/simple <pkg>
```

**Never let anything reinstall torch.** `torch_npu` is bound to `torch==2.10.0`; if pip
replaces torch, the container is effectively dead. The live example: `timm → torchvision
→ torch==2.13.0` (an exact pin) pulls in a CUDA torch plus 15 `nvidia-*` packages plus an
upstream `triton` that shadows triton-ascend. Neither torch nor torchvision appears in
the direct dependency list, so you cannot see this coming from the requirements file.
Use a constraints file and dry-run first (SETUP.md §8.2):

```bash
pip install --dry-run --report /tmp/r.json -i $IDX -c $ROOT/npu-constraints.txt -r ...
# then assert the report contains no torch / triton / nvidia-* installs
```

**Also:** the global proxy works only for GitHub and Anthropic. `unset http_proxy
https_proxy HTTP_PROXY HTTPS_PROXY` before anything else; the Huawei mirrors are
direct-connect and fast (46 MB/s), and GitHub *release downloads* are faster through
`https://gh-proxy.com/<original-url>` with the proxy **off**.

---

## 4. House rules on this machine

* **Do not install anything into `.venv-glm53`.** It is the working runtime.
* **Do not kill running processes.** A 16-die SGLang server may be up; `npu-smi info`
  showing ~56 GB HBM in use on every die means it is.
* Because of that, an NPU probe may not be runnable at any given moment. Where this
  package cites a measurement it was taken rather than re-measured, and it says so —
  see [specs/op3](specs/op3_kv_norm_rope_cache_rope0.md) §1.
