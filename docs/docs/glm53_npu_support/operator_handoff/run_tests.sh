#!/usr/bin/env bash
# Run the operator handoff test suite.
#
#   ./run_tests.sh                 # torch reference (CPU, no NPU needed)
#   GLM53_OP_BACKEND=npu ./run_tests.sh
#
# Needs only: torch (CPU is fine) and pytest. See ENVIRONMENT.md.
set -euo pipefail

# The global proxy here only reaches GitHub/Anthropic; unset it so nothing hangs.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
echo "backend = ${GLM53_OP_BACKEND:-reference}   python = $($PY -c 'import sys;print(sys.executable)')"
$PY -c "import torch; print('torch', torch.__version__)"
exec $PY -m pytest tests "$@"
