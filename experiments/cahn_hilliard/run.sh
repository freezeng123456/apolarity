#!/usr/bin/env bash
# Cahn-Hilliard (4th & 6th order, nonlinear phase field).
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--hidden 32 --depth 4 --lr-schedule cosine"
V=complex_sinh,fourier,siren,mscale
mkdir -p data
$PY exp_cahn_hilliard.py --seconds 70 --seeds 2 $C --variants $V --a 2,3 --orders 4,6 --out data/cahn_hilliard_v3.csv
$PY exp_cahn_hilliard.py --seconds 70 --seeds 2 $C --variants $V --a 2 --orders 6 --history --out data/cahn_hilliard.csv
