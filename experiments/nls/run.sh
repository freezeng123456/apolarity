#!/usr/bin/env bash
# Cubic nonlinear Schrodinger (genuinely complex-valued field).
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--hidden 32 --depth 4 --lr-schedule cosine"
V=complex_sinh,siren,fourier,tanh
mkdir -p data
$PY exp_nls_schrodinger.py --seconds 70 --seeds 2 $C --variants $V --sweeps 1,2,4 --out data/nls_v3.csv
$PY exp_nls_schrodinger.py --seconds 70 --seeds 2 $C --variants $V --sweeps 2 --history --out data/nls.csv
