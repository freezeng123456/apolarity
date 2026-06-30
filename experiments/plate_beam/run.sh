#!/usr/bin/env bash
# Kirchhoff plate / Euler-Bernoulli beam / mixed-mode plate (4th order) -- 600s width study.
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 600 --seeds 2 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
mkdir -p data

# --- isotropic plate + beam, modes 1..3 ---
$PY exp_plate_beam.py $C --hidden 128 --variants $REAL        --kind both --modes 1,2,3 --out data/plate_beam_h128.csv
$PY exp_plate_beam.py $C --hidden 64  --variants complex_sinh --kind both --modes 1,2,3 --out data/plate_beam_h64.csv

# --- anisotropic mixed-mode plate (m, m+1), m=2..4 ---
$PY exp_plate_beam.py $C --hidden 128 --variants $REAL        --kind mix --modes 2,3,4 --out data/platemix_h128.csv
$PY exp_plate_beam.py $C --hidden 64  --variants complex_sinh --kind mix --modes 2,3,4 --out data/platemix_h64.csv
