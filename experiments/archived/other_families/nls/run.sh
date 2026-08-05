#!/usr/bin/env bash
# DIAGNOSTIC ONLY: H=128 implementation check; not jsc_v2 or paper evidence.
# Real baselines carry the field as a split-real (Re/Im) pair (tanh RVPINN).
set -euo pipefail; cd "$(dirname "$0")"
source ../../scripts/cuda_env.sh
PY="${APOLARITY_PYTHON:-/usr/bin/python3.11}"
C="--seconds 600 --seeds 2 --depth 4 --lr-schedule cosine --history"
CPLX=complex_sinh,siren,fourier,tanh
mkdir -p data
echo "WARNING: diagnostic-only output (not jsc_v2); do not use for paper." >&2
$PY exp_nls_schrodinger.py $C --hidden 128 --variants $CPLX        --sweeps 1,2,4 --out data/nls_h128.csv
