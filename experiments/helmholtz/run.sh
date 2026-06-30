#!/usr/bin/env bash
# High-wavenumber Helmholtz (+ anisotropic) -- 600s width study.
# Real baselines @128, complex sinh @{128,64}, depth 4, 2 seeds, --history.
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 600 --seeds 2 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
mkdir -p data

# --- isotropic wavenumber sweep ---
$PY exp_helmholtz_highk.py $C --hidden 128 --variants $REAL        --sweeps 2,4,6,8,10 --out data/helmholtz_h128.csv
$PY exp_helmholtz_highk.py $C --hidden 64  --variants complex_sinh --sweeps 2,4,6,8,10 --out data/helmholtz_h64.csv

# --- anisotropic (1,4) gradient-pathology case ---
$PY exp_helmholtz_highk.py $C --hidden 128 --variants $REAL        --aniso --out data/helmholtz_aniso_h128.csv
$PY exp_helmholtz_highk.py $C --hidden 64  --variants complex_sinh --aniso --out data/helmholtz_aniso_h64.csv
