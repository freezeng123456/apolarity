#!/usr/bin/env bash
# Polyharmonic 2D order sweep -- 1200s (20 min) width study (JSC main text).
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 1200 --seeds 5 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
mkdir -p data
$PY exp_polyharmonic.py $C --hidden 128 --variants $REAL \
  --dim 2 --orders 2,4,6 --omega0 10 --out data/poly2d_h128.csv
$PY exp_polyharmonic.py $C --hidden 64  --variants complex_sinh \
  --dim 2 --orders 2,4,6 --omega0 10 --out data/poly2d_h64.csv
