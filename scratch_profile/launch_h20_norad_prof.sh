#!/bin/bash
# H20, 4K/1.5K bs=72, no-radix, WITH the profiler armed for bench_serving.
# Same config as launch_h20_norad.sh plus SGLANG_TORCH_PROFILER_DIR so
# `bench_serving --profile` can trigger a capture.
#
#   DEV=0 bash launch_h20_norad_prof.sh
#
# Then run bench_serving with --profile --profile-start-step N --profile-steps M
# (N is a FORWARD count, so it delays the capture until steady state).
# Output is a kineto trace.json -- use perfetto / torch key_averages, NOT
# scratch_profile/analyze.py (that one parses Ascend msprof CSVs).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEV=${DEV:-0}
PORT=${PORT:-31400}
MODEL=${MODEL:-/workspace/models/LLaDA/LLaDA2.1-mini/}
MRR=${MRR:-72}
PROF_DIR=${PROF_DIR:-$REPO/scratch_profile/profiles/prof_h20_norad_bench}

export CUDA_VISIBLE_DEVICES=$DEV
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export SGLANG_ENABLE_DLLM_MIXED_BATCH=1
export SGLANG_DLLM_FDFO_STEPS_PER_ROUND=2
export SGLANG_TORCH_PROFILER_DIR=$PROF_DIR
mkdir -p "$PROF_DIR"

echo "H20 DEV=$DEV PORT=$PORT MRR=$MRR (no-radix, profiler -> $PROF_DIR)"

exec python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --dllm-algorithm JointThreshold --dllm-fdfo --mem-fraction-static 0.78 --max-running-requests "$MRR" --disable-radix-cache --cuda-graph-config '{"decode":{"backend":"full","max_bs":72,"bs":[1,8,16,32,48,56,64,72]}}' --port "$PORT"
