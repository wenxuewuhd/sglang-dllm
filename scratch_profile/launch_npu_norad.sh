#!/bin/bash
# 910B NPU, 4K/1.5K bs=72, radix prefix cache DISABLED + per-step device/host timing.
#
# Why --disable-radix-cache: at 4K context with distinct prompts the prefix tree
# never hits, but cache_unfinished_req still re-walks every request's 4K+ token
# sequence through the tree every denoise step -- measured at 72% of scheduler
# CPU (py-spy). Disabling it cut host 208.6 -> 21.2 ms/step with device
# unchanged, and output throughput 1058 -> 1706 tok/s.
#
#   DEV=7 bash launch_npu_norad.sh      # card 7 (default), port 31600
#
# Knobs: DEV (NPU card), PORT, MODEL, MRR.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEV=${DEV:-7}
PORT=${PORT:-31600}
MODEL=${MODEL:-/workspace/models/LLaDA/LLaDA2.1-mini/}
MRR=${MRR:-72}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=$DEV
export SGLANG_ENABLE_DLLM_MIXED_BATCH=1
export SGLANG_DLLM_FDFO_STEPS_PER_ROUND=2
export SGLANG_DEBUG_DLLM_STEP_TIMING=1
export SGLANG_DEBUG_DLLM_STEP_TIMING_INTERVAL=${TIMING_INTERVAL:-50}

echo "NPU DEV=$DEV PORT=$PORT MRR=$MRR (no-radix + step timing)"

exec python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --attention-backend ascend --dllm-algorithm JointThreshold --dllm-fdfo --mem-fraction-static 0.78 --max-running-requests "$MRR" --disable-radix-cache --cuda-graph-config '{"decode":{"backend":"full","max_bs":72,"bs":[1,8,16,32,48,56,64,72]}}' --port "$PORT"
