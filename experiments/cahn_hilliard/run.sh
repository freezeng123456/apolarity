#!/usr/bin/env bash
# Cahn-Hilliard (4th & 6th order, nonlinear) -- 600s width study.
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 600 --seeds 2 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
mkdir -p data
$PY exp_cahn_hilliard.py $C --hidden 128 --variants $REAL        --a 2,3 --orders 4,6 --out data/cahn_hilliard_h128.csv
$PY exp_cahn_hilliard.py $C --hidden 64  --variants complex_sinh --a 2,3 --orders 4,6 --out data/cahn_hilliard_h64.csv
