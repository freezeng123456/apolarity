#!/usr/bin/env bash
# Polyharmonic order sweep (1D + 2D) -- 600s width study.
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 600 --seeds 2 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
mkdir -p data

# --- 1D order axis (omega0=pi default), orders 2..10 ---
$PY exp_polyharmonic.py $C --hidden 128 --variants $REAL        --dim 1 --orders 2,4,6,8,10 --out data/poly1d_h128.csv
$PY exp_polyharmonic.py $C --hidden 64  --variants complex_sinh --dim 1 --orders 2,4,6,8,10 --out data/poly1d_h64.csv

# --- 2D order axis (omega0=10), orders 2..6 ---
$PY exp_polyharmonic.py $C --hidden 128 --variants $REAL        --dim 2 --orders 2,4,6 --omega0 10 --out data/poly2d_h128.csv
$PY exp_polyharmonic.py $C --hidden 64  --variants complex_sinh --dim 2 --orders 2,4,6 --omega0 10 --out data/poly2d_h64.csv
