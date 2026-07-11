#!/usr/bin/env bash
# DIAGNOSTIC ONLY: H=128 implementation checks; not jsc_v2 or paper evidence.
set -euo pipefail; cd "$(dirname "$0")"
source ../../scripts/cuda_env.sh
PY="${APOLARITY_PYTHON:-/usr/bin/python3.11}"
C="--seconds 600 --seeds 2 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
mkdir -p data
echo "WARNING: diagnostic-only output (not jsc_v2); do not use for paper." >&2

# --- isotropic plate + beam, modes 1..3 ---
$PY exp_plate_beam.py $C --hidden 128 --variants $REAL        --kind both --modes 1,2,3 --out data/plate_beam_h128.csv

# --- anisotropic mixed-mode plate (m, m+1), m=2..4 ---
$PY exp_plate_beam.py $C --hidden 128 --variants $REAL        --kind mix --modes 2,3,4 --out data/platemix_h128.csv
