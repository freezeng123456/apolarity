#!/bin/bash
set -eo pipefail
source /etc/profile
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
TASK_ROOT=/data02/run01/scv7tsq/apolarity/20261021-memory-peak30s-r2
PYTHON=/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python
cd "$TASK_ROOT"
if [ "$1" = gate ]; then
    exec "$PYTHON" campaign.py gate
fi
exec "$PYTHON" campaign.py lane --index "$SLURM_ARRAY_TASK_ID"
