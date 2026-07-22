#!/bin/bash
# Launch LLaDA2.1-mini in the tuned SINGLE-CARD optimal recipe:
#   FDFO scheduling + mixed prefill/decode batching (K=2 frozen rounds)
#   + dense decode-graph buckets clustered in the 96-128 hot zone.
#
# This is the empirically-verified single-card ceiling (2026-07). Long-seq
# ~2100 tok/s, gsm8k ~3800 tok/s, no memory-edge / retract.
#
#   DEV=0 bash launch_best.sh          # card 0, port 31400, bs=128
#
# Knobs: DEV (NPU card), PORT, MODEL, MRR (--max-running-requests),
#        MEMFRAC (--mem-fraction-static).
#
# Why these numbers (all measured, see LLaDA2_910B_perf_stage1.md):
#  - mem-fraction 0.75: 0.85 over-provisions the KV pool 8x and OOMs at bs=128;
#    0.75 leaves the ~4GB runtime the sustained-full-batch activations need.
#  - graph max_bs=128 with dense top buckets [96,104,112,120,128]: graph costs
#    9.76GB here; pushing max_bs to 160 costs 12.2GB and leaves only 1.07GB
#    runtime -> long-seq retracts with NO throughput gain. 128 is the sweet spot.
#  - dense hot-zone buckets cut pad waste in the 112-127 range (+5% vs step-16).
set -euo pipefail

DEV=${DEV:-0}
PORT=${PORT:-31400}
MODEL=${MODEL:-/workspace/models/LLaDA/LLaDA2.1-mini/}
MRR=${MRR:-128}
MEMFRAC=${MEMFRAC:-0.75}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=$DEV

# Mixed prefill/decode batching: fold newly-arrived prefills into the running
# decode batch, freeze the mix for K=2 denoise rounds before re-admitting.
export SGLANG_ENABLE_DLLM_MIXED_BATCH=1
export SGLANG_DLLM_FDFO_STEPS_PER_ROUND=2

echo "DEV=$DEV PORT=$PORT MRR=$MRR MEMFRAC=$MEMFRAC (FDFO + mixed-batch K=2 + dense graph@128)"

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --attention-backend ascend \
  --dllm-algorithm JointThreshold \
  --dllm-fdfo \
  --mem-fraction-static "$MEMFRAC" \
  --max-running-requests "$MRR" \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":128,"bs":[1,2,4,8,16,32,64,96,104,112,120,128]}}' \
  --port "$PORT"
