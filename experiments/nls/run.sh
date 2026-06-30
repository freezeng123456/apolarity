#!/usr/bin/env bash
# Cubic nonlinear Schrodinger (complex-valued) -- 600s width study.
# Real baselines carry the field as a split-real (Re/Im) pair (tanh RVPINN).
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 600 --seeds 2 --depth 4 --lr-schedule cosine --history"
CPLX=complex_sinh,siren,fourier,tanh
mkdir -p data
$PY exp_nls_schrodinger.py $C --hidden 128 --variants $CPLX        --sweeps 1,2,4 --out data/nls_h128.csv
$PY exp_nls_schrodinger.py $C --hidden 64  --variants complex_sinh --sweeps 1,2,4 --out data/nls_h64.csv
