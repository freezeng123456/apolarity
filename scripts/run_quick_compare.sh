#!/usr/bin/env bash
# Quick 5-minute single-monomial backend comparison on T4.
# Source cuda_env.sh and run a small benchmark to confirm correctness + speedup.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=cuda_env.sh
source "$SCRIPT_DIR/cuda_env.sh"

cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$APOLARITY_PYTHON" experiments/quick_compare_5min.py "$@"
