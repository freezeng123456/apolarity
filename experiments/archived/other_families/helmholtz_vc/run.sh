#!/usr/bin/env bash
# DIAGNOSTIC ONLY: H=128 implementation check; not jsc_v2 or paper evidence.
set -euo pipefail; cd "$(dirname "$0")"
source ../../scripts/cuda_env.sh
PY="${APOLARITY_PYTHON:-/usr/bin/python3.11}"
C="--seconds 600 --seeds 2 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
mkdir -p data
echo "WARNING: diagnostic-only output (not jsc_v2); do not use for paper." >&2
$PY exp_helmholtz_vc.py $C --hidden 128 --variants $REAL        --sweeps 2,4,6 --out data/helmvc_h128.csv
