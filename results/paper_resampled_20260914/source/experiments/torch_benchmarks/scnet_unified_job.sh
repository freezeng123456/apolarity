#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
root=$1
mode=$2
code="$root/source/experiments/torch_benchmarks"
export PYTHONPATH="$code"
export TMPDIR="$root/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd "$code"
if [ "$mode" = preflight ]; then
    exec /work/home/zenghang/apolarity_envs/torch251-cu118-20260912/bin/python -u scnet_unified.py preflight --root "$root"
fi
exec /work/home/zenghang/apolarity_envs/torch251-cu118-20260912/bin/python -u scnet_unified.py pair --root "$root" --index "$SLURM_ARRAY_TASK_ID"
