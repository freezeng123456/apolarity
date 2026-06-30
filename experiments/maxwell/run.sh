#!/usr/bin/env bash
# Time-harmonic Maxwell, lossy medium (complex-valued) -- 600s width study.
# Real baselines carry the field as a split-real (Re/Im) pair (tanh RVPINN).
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 600 --seeds 2 --depth 4 --lr-schedule cosine --history"
CPLX=complex_sinh,siren,fourier,tanh
mkdir -p data
$PY exp_maxwell.py $C --hidden 128 --variants $CPLX        --sweeps 2,4,6 --out data/maxwell_h128.csv
$PY exp_maxwell.py $C --hidden 64  --variants complex_sinh --sweeps 2,4,6 --out data/maxwell_h64.csv
