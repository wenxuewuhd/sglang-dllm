#!/bin/bash
# GPU counterpart of launch_best.sh -- runs LLaDA2.1-mini (dLLM) on CUDA.
#
# The NPU scripts pin --attention-backend ascend, which does not exist on GPU.
# dLLM is supported on GPU by the flashinfer / flash-attention backends (both
# implement is_dllm_extend()), so this uses fa3 and drops every NPU-only knob:
#   - no --attention-backend ascend           (use fa3 / flashinfer)
#   - no SGLANG_NPU_* / SGLANG_TORCH_PROFILER  (those are Ascend-only)
#   - no ascend --cuda-graph-config buckets    (CUDA graph is native; --cuda-graph-max-bs)
# The dLLM scheduling knobs (FDFO, mixed batch, K=2) are device-agnostic and
# work on GPU too, so they are kept.
#
#   DEV=0 bash launch_best_gpu.sh              # GPU 0, port 31400, bs=128, mixed-batch K=1
#   DEV=0 KSTEPS=2 bash launch_best_gpu.sh     # also try K=2 (see note below)
#
# Knobs: DEV (CUDA device), PORT, MODEL, MRR, MEMFRAC, ATTN (fa3|flashinfer),
#        MIXED (1=mixed prefill/decode on), KSTEPS (FDFO steps per round).
set -euo pipefail

DEV=${DEV:-0}
PORT=${PORT:-31400}
MODEL=${MODEL:-/workspace/models/LLaDA/LLaDA2.1-mini/}
MRR=${MRR:-128}
MEMFRAC=${MEMFRAC:-0.85}     # GPU: no Ascend overhead; give the KV pool room
ATTN=${ATTN:-fa3}            # fa3 or flashinfer; both handle is_dllm_extend()
MIXED=${MIXED:-1}           # mixed prefill/decode batching (helps GPU: fills the batch)
KSTEPS=${KSTEPS:-1}         # FDFO steps/round. Default 1 on GPU.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=$DEV

# Mixing (batch-fill) is the half that helps GPU; K>1 (frozen rounds) exists to
# hide NPU dispatch bubbles and on GPU takes the per-row step fallback
# (vectorized_decoding is gated to NPU), so its benefit is marginal-to-negative.
# Default K=1; measure K=2 before trusting it on GPU.
export SGLANG_ENABLE_DLLM_MIXED_BATCH=$MIXED
export SGLANG_DLLM_FDFO_STEPS_PER_ROUND=$KSTEPS

echo "GPU=$DEV PORT=$PORT MRR=$MRR MEMFRAC=$MEMFRAC ATTN=$ATTN MIXED=$MIXED K=$KSTEPS"

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --attention-backend "$ATTN" \
  --dllm-algorithm JointThreshold \
  --dllm-fdfo \
  --mem-fraction-static "$MEMFRAC" \
  --max-running-requests "$MRR" \
  --cuda-graph-max-bs 128 \
  --port "$PORT"
