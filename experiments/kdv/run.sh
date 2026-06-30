#!/usr/bin/env bash
# Linearized KdV / dispersive wave (3rd order) -- 600s width study.
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 600 --seeds 2 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
mkdir -p data
$PY exp_kdv_dispersive.py $C --hidden 128 --variants $REAL        --sweeps 2,3,4,5,6 --out data/kdv_h128.csv
$PY exp_kdv_dispersive.py $C --hidden 64  --variants complex_sinh --sweeps 2,3,4,5,6 --out data/kdv_h64.csv
