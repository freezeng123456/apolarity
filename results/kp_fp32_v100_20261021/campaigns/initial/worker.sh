#!/bin/bash
set -eo pipefail
source /etc/profile
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COHORT_ROOT=/data02/run01/scv7tsq/apolarity/20261020-kp-fp32-new-seeds-r1
FORMAL_ROOT=/data02/run01/scv7tsq/apolarity/20261020-kp-formal-lr1e-4-r1
export PYTHONPATH="$FORMAL_ROOT/experiments/torch_benchmarks"
PYTHON=/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python
cd "$COHORT_ROOT"
if [ "$1" = plan ]; then
    exec "$PYTHON" new_seed_campaign.py plan --root "$COHORT_ROOT" --formal-root "$FORMAL_ROOT"
elif [ "$1" = gate ]; then
    exec "$PYTHON" new_seed_campaign.py preflight --root "$COHORT_ROOT" --formal-root "$FORMAL_ROOT"
else
    exec "$PYTHON" new_seed_campaign.py lane --root "$COHORT_ROOT" --formal-root "$FORMAL_ROOT" \
        --lane "$SLURM_ARRAY_TASK_ID"
fi
