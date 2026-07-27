#!/bin/bash
# 910B NPU, 4K/1.5K bs=72, no-radix, WITH the profiler armed for bench_serving.
# Same config as launch_npu_norad.sh plus SGLANG_TORCH_PROFILER_DIR (so
# `bench_serving --profile` can trigger a capture) and Level2 per-kernel detail.
#
# Recommended defaults (2026-07-27, both 910B and H20, 4K/1.5K):
#   --disable-radix-cache : at 4K with distinct prompts the prefix tree never
#     hits, yet every denoise step re-walks each request's 4K+ tokens through it
#     -- 72% of scheduler CPU (py-spy). Off: host 208.6 -> 21.2 ms/step, device
#     unchanged, output 1058 -> 1811 tok/s. Turn back ON only for workloads with
#     genuinely shared prefixes (common system prompt / few-shot RL rollouts).
#   FDFO_STEPS=1 : K=2 exists to amortize per-round host work over N forwards;
#     with radix off there is little left to amortize and its costs dominate
#     (bs>=64 skips the early exit so it always runs the extra forward, and the
#     frozen batch delays admission). K=1 measured +10.7% on 910B / +16.9% on
#     H20, TTFT -40%. K=2 is a tuning option, not the default.
#
#   DEV=7 bash launch_npu_norad_prof.sh
#
# Then run bench_serving with --profile --profile-start-step N --profile-steps M
# (N is a FORWARD count, so it delays the capture until steady state).
# Traces land in $PROF_DIR; analyze with scratch_profile/analyze.py.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEV=${DEV:-7}
PORT=${PORT:-31500}
MODEL=${MODEL:-/workspace/models/LLaDA/LLaDA2.1-mini/}
MRR=${MRR:-72}
FDFO_STEPS=${FDFO_STEPS:-1}
PROF_DIR=${PROF_DIR:-$REPO/scratch_profile/profiles/prof_npu_norad_bench}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=$DEV
export SGLANG_ENABLE_DLLM_MIXED_BATCH=1
export SGLANG_DLLM_FDFO_STEPS_PER_ROUND=$FDFO_STEPS
export SGLANG_NPU_PROFILER_LEVEL2=1
export SGLANG_TORCH_PROFILER_DIR=$PROF_DIR
mkdir -p "$PROF_DIR"

echo "NPU DEV=$DEV PORT=$PORT MRR=$MRR $FDFO_STEPS (no-radix, profiler -> $PROF_DIR)"

exec python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code --attention-backend ascend --dllm-algorithm JointThreshold --dllm-fdfo --mem-fraction-static 0.78 --max-running-requests "$MRR" --disable-radix-cache --cuda-graph-config '{"decode":{"backend":"full","max_bs":72,"bs":[1,8,16,32,48,56,64,72]}}' --port "$PORT"
