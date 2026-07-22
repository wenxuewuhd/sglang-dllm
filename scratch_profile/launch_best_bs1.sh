#!/bin/bash
# Launch LLaDA2.1-mini in the SINGLE-REQUEST (bs=1) latency-optimal recipe.
#
# Counterpart to launch_best.sh (which is the large-batch throughput recipe).
# At bs=1 the throughput machinery is pure overhead, so this is deliberately the
# OPPOSITE configuration:
#   - SYNC mode via --no-dllm-fdfo. NOTE: --dllm-fdfo defaults to TRUE, so you
#     must explicitly disable it; simply omitting it leaves FDFO ON.
#     FDFO/mixing/K=2 exist to keep a *large* batch full and hide dispatch
#     bubbles ACROSS many concurrent requests; with one request there is nothing
#     to fill, and the K=2 freeze only adds latency. Plain sync denoise is best.
#   - decode graph captured at bs=1 only (not the 96-128 dense buckets): smaller
#     capture, no wasted high buckets, lowest per-step latency.
#   - mem-fraction 0.85: memory is abundant with a single request, so give the
#     KV pool room (matches the concurrency-1 baseline).
#
# The per-forward kernel optimizations (fused split_qkv, fused argmax-softmax
# denoise reduction, ascend-backend D2H-sync elision) are on by default on this
# branch and DO help at bs=1 -- they shave the small-operator tax on every step.
#
#   DEV=7 bash launch_best_bs1.sh          # card 7, port 31400, bs=1
#
# Knobs: DEV (NPU card), PORT, MODEL, MEMFRAC (--mem-fraction-static).
set -euo pipefail

DEV=${DEV:-7}
PORT=${PORT:-31400}
MODEL=${MODEL:-/workspace/models/LLaDA/LLaDA2.1-mini/}
MEMFRAC=${MEMFRAC:-0.85}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=$DEV

echo "DEV=$DEV PORT=$PORT MEMFRAC=$MEMFRAC (SYNC mode, bs=1 latency-optimal)"

# SYNC mode: --no-dllm-fdfo (fdfo defaults ON), no SGLANG_ENABLE_DLLM_MIXED_BATCH,
# no STEPS_PER_ROUND.
exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name LLaDA2.1-mini \
  --trust-remote-code \
  --attention-backend ascend \
  --dtype bfloat16 \
  --dllm-algorithm JointThreshold \
  --no-dllm-fdfo \
  --mem-fraction-static "$MEMFRAC" \
  --max-running-requests 1 \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":1,"bs":[1]}}' \
  --port "$PORT"
