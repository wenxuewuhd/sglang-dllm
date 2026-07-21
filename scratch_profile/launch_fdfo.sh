#!/bin/bash
# Launch LLaDA2.1-mini in the tuned large-batch FDFO recipe, with Level2 NPU
# profiling armed so /start_profile produces a timeline trace_view.json (host
# dispatch gaps visible), matching the reference capture.
#
#   DEV=7 bash launch_fdfo.sh        # card 7, port 31400, bs=128
#
# Knobs: DEV (NPU card), PORT, MODEL, MRR (--max-running-requests),
#        MEMFRAC (--mem-fraction-static), LEVEL2 (1=Level2 timeline / 0=Level1).
set -euo pipefail

DEV=${DEV:-7}
PORT=${PORT:-31400}
MODEL=${MODEL:-/workspace/models/LLaDA/LLaDA2.1-mini/}
MRR=${MRR:-128}
MEMFRAC=${MEMFRAC:-0.75}
LEVEL2=${LEVEL2:-1}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROF_DIR=${PROF_DIR:-$REPO/scratch_profile/profiles}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=$DEV
export SGLANG_NPU_PROFILER_LEVEL2=$LEVEL2
export SGLANG_TORCH_PROFILER_DIR=$PROF_DIR
mkdir -p "$PROF_DIR"

echo "DEV=$DEV PORT=$PORT MRR=$MRR MEMFRAC=$MEMFRAC LEVEL2=$LEVEL2"
echo "PROF_DIR=$PROF_DIR"

# FDFO on (no --no-dllm-fdfo). mem-fraction 0.75 keeps runtime memory for the
# sustained-full-batch activations (0.85 OOMs at bs=128).
exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --trust-remote-code \
  --attention-backend ascend \
  --dllm-algorithm JointThreshold \
  --mem-fraction-static "$MEMFRAC" \
  --max-running-requests "$MRR" \
  --dllm-fdfo \
  --port "$PORT"
