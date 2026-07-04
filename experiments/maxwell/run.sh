#!/usr/bin/env bash
# Time-harmonic Maxwell, lossy medium -- 1200s width study (JSC main text).
# Real baselines: split-real (Re/Im) RVPINN at literal width H.
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 1200 --seeds 5 --depth 4 --lr-schedule cosine --history"
CPLX=complex_sinh,siren,fourier,tanh
mkdir -p data
$PY exp_maxwell.py $C --hidden 128 --variants $CPLX        --sweeps 2,4,6 --out data/maxwell_h128.csv
$PY exp_maxwell.py $C --hidden 64  --variants complex_sinh --sweeps 2,4,6 --out data/maxwell_h64.csv
