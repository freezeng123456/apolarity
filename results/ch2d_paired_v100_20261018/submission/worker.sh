#!/usr/bin/env bash
set -e
source /etc/profile
set -u
set -o pipefail
: "${SLURM_JOB_ID:?Requires allocation}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CH_ROOT=/data02/run01/scv7tsq/apolarity/20261018-ch2d-paired-r1
CH_PYTHON=/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python
cd "$CH_ROOT/source"
case "${1:?Expected stage}" in
 preflight) exec "$CH_PYTHON" experiments/torch_benchmarks/ch2d_paired_campaign.py preflight --out "$CH_ROOT/results/preflight" ;;
 formal) exec "$CH_PYTHON" "$CH_ROOT/driver.py" ;;
 audit) exec "$CH_PYTHON" "$CH_ROOT/audit.py" ;;
 *) exit 2 ;;
esac
