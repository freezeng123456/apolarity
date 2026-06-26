#!/usr/bin/env bash
# Linearized KdV / dispersive wave (3rd order) -- the dispersive counter-case.
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--hidden 32 --depth 4 --lr-schedule cosine"
V=complex_sinh,fourier,siren,mscale
mkdir -p data
$PY exp_kdv_dispersive.py --seconds 70 --seeds 3 $C --variants $V --sweeps 2,3,4,5,6 --out data/kdv_v3.csv
$PY exp_kdv_dispersive.py --seconds 70 --seeds 2 $C --variants $V --sweeps 4 --history --out data/kdv.csv
