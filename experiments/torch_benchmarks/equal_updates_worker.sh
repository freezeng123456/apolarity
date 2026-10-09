#!/usr/bin/env bash
set -e
source /etc/profile
set -u
set -o pipefail
: "${SLURM_JOB_ID:?Requires an allocated GPU}"
: "${APOLARITY_SOURCE:?Requires a frozen source export}"
: "${APOLARITY_PYTHON:?Requires the pinned interpreter}"
: "${APOLARITY_OUTPUT:?Requires a unique output root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$APOLARITY_SOURCE"
case "${1:?Expected preflight or cell}" in
    preflight)
        exec "$APOLARITY_PYTHON" experiments/torch_benchmarks/equal_updates_preflight.py \
            --out "$APOLARITY_OUTPUT/preflight"
        ;;
    cell)
        [[ -f "$APOLARITY_OUTPUT/preflight/PASS.json" ]]
        exec "$APOLARITY_PYTHON" experiments/torch_benchmarks/equal_updates_campaign.py \
            cell --out "$APOLARITY_OUTPUT" --index "${SLURM_ARRAY_TASK_ID:?Requires array index}"
        ;;
    *) exit 2 ;;
esac
