#!/bin/bash
# H20, 4K/1.5K bs=72, radix prefix cache DISABLED + per-step device/host timing.
# Same config as launch_npu_norad.sh minus the ascend attention backend, so the
# two are directly comparable.
#
# Why --disable-radix-cache: the prefix tree never hits at 4K with distinct
# prompts, but every denoise step still re-walks each request's 4K+ token
# sequence through it. This is device-independent Python, so H20 pays the same
# tax; on 910B removing it cut scheduler host time by 90% (208.6 -> 21.2 ms).
#
#   DEV=0 bash launch_h20_norad.sh      # GPU 0 (default), port 31400
#
# Knobs: DEV (GPU), PORT, MODEL, MRR.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEV=${DEV:-0}
PORT=${PORT:-31400}
MODEL=${MODEL:-/workspace/models/LLaDA/LLaDA2.1-mini/}
MRR=${MRR:-72}

export CUDA_VISIBLE_DEVICES=$DEV
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export SGLANG_ENABLE_DLLM_MIXED_BATCH=1
export SGLANG_DLLM_FDFO_STEPS_PER_ROUND=2
export SGLANG_DEBUG_DLLM_STEP_TIMING=1
export SGLANG_DEBUG_DLLM_STEP_TIMING_INTERVAL=${TIMING_INTERVAL:-50}

echo "H20 DEV=$DEV PORT=$PORT MRR=$MRR (no-radix + step timing)"

exec python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dllm-algorithm JointThreshold --dllm-fdfo --mem-fraction-static 0.78 --max-running-requests "$MRR" --disable-radix-cache --cuda-graph-config '{"decode":{"backend":"full","max_bs":72,"bs":[1,8,16,32,48,56,64,72]}}' --port "$PORT"
