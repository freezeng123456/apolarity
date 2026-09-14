#!/bin/bash
set -e
source /etc/profile
set -u
set -o pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SWEEP_ROOT="$1"
SWEEP_MODE="$2"
SWEEP_PYTHON="/data02/run01/scv7tsq/semigroup-beijing-20260911-r1/venv/bin/python"
cd "$SWEEP_ROOT"
exec "$SWEEP_PYTHON" "$SWEEP_ROOT/source/experiments/torch_benchmarks/bscc_sweep.py" "$SWEEP_MODE" --root "$SWEEP_ROOT" --index "${SLURM_ARRAY_TASK_ID:-0}"
