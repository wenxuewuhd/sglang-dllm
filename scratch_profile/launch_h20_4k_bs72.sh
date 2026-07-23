#!/bin/bash
# H20 strict bs=72 capture for the 4K/1.5K PK (same config as the 910B strict run,
# minus the ascend backend; MIXED_BATCH on, no K=2 freeze).
# Trace lands in scratch_profile/profiles/prof_h20_4k_bs72 automatically.
set -uo pipefail
#   DEV=3 bash launch_h20_4k_bs72.sh    # pick GPU 3
REPO=/workspace/code/sglang-dllm
MODEL=${MODEL:-/workspace/models/LLaDA/LLaDA2.1-mini/}
PORT=${PORT:-31600}
DEV=${DEV:-0}
export CUDA_VISIBLE_DEVICES=$DEV
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export SGLANG_ENABLE_DLLM_MIXED_BATCH=1
export SGLANG_TORCH_PROFILER_DIR=$REPO/scratch_profile/profiles/prof_h20_4k_bs72
mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
exec python -m sglang.launch_server \
  --model-path "$MODEL" --trust-remote-code \
  --dllm-algorithm JointThreshold --dllm-fdfo \
  --mem-fraction-static 0.78 --max-running-requests 72 \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":72,"bs":[1,8,16,32,48,56,64,72]}}' \
  --port "$PORT"
