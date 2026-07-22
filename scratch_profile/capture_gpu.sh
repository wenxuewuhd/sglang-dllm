#!/bin/bash
# Capture N consecutive dLLM forwards from a GPU sglang server (launched via
# launch_best_gpu.sh) into a kineto trace you can open in perfetto.
#
# The capture driver (capture_fdfo.py) is device-agnostic: it holds the batch
# full with a feeder, then arms the server's /start_profile for `steps` forwards.
# On GPU the server records a CUDA (kineto) trace; on NPU it records an Ascend
# (msprof) trace -- same driver, different backend picks the profiler.
#
#   PORT=31400 STEPS=4 bash capture_gpu.sh          # 4 forwards -> ./gpu_trace
#
# Knobs: PORT, STEPS (num forwards), WARMUP (sec to steady-state), OUT (dir).
#
# Output: OUT/<...>/*.pt.trace.json  (kineto). Drag it into https://ui.perfetto.dev.
#
# ALIGNING WITH THE NPU CAPTURE: the NPU trace was a clean Level2 device-kernel
# timeline. GPU has no "Level2" toggle (that is msprof/NPU-only); kineto detail
# is controlled by with_stack / record_shapes instead. capture_fdfo.py now
# defaults BOTH off, so no per-op Python call stacks (the "level1" look) and no
# shape-metadata Free inflation. The GPU analog of the NPU Level2 device row is
# the CUDA stream track in perfetto (the actual kernels); the CPU thread track
# is the host aten tree (kept shallow by with_stack=off). Pass --with-stack to
# capture_fdfo.py only if you actually want the Python stacks back.
#
# NOTE: scratch_profile/analyze.py does NOT work on GPU traces -- it parses the
# Ascend msprof kernel_details.csv, which CUDA does not produce. For a GPU
# per-kernel breakdown use perfetto's built-in slice aggregation, or
# torch.profiler's key_averages() table. The timeline (trace_view) is the same
# idea on both; only the offline CSV analyzer is NPU-specific.
set -euo pipefail

PORT=${PORT:-31400}
STEPS=${STEPS:-4}
WARMUP=${WARMUP:-75}
OUT=${OUT:-./gpu_trace}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export PYTHONPATH=$REPO/python:${PYTHONPATH:-}
mkdir -p "$OUT"

echo "capturing $STEPS forwards from port $PORT (warmup ${WARMUP}s) -> $OUT"
python "$REPO/scratch_profile/capture_fdfo.py" \
  --port "$PORT" --bs 128 --steps "$STEPS" --warmup "$WARMUP" --out "$OUT"

echo "done. trace under $OUT ; open the *.trace.json in https://ui.perfetto.dev"
