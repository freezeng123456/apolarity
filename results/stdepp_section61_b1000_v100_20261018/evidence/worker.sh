#!/usr/bin/env bash
set -e
source /etc/profile
set -u
set -o pipefail
: "${SLURM_JOB_ID:?Requires allocation}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
APO_SCALE_ROOT=/data02/run01/scv7tsq/apolarity/20261018-stdepp-section61-b1000-r1
APO_SCALE_PYTHON=/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python
cd "$APO_SCALE_ROOT/source"
if [[ "$1" == preflight ]]; then
 exec "$APO_SCALE_PYTHON" experiments/torch_benchmarks/stdepp_fixed_batch_campaign.py preflight --out "$APO_SCALE_ROOT/results"
fi
exec "$APO_SCALE_PYTHON" "$APO_SCALE_ROOT/driver.py"
