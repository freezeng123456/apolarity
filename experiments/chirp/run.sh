#!/usr/bin/env bash
# Non-separable radial chirp (a=1,2,3) -- 1200s width study (JSC main text).
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 1200 --seeds 5 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
mkdir -p data
$PY exp_chirp.py $C --hidden 128 --variants $REAL        --sweeps 1,2,3 --out data/chirp_h128.csv
$PY exp_chirp.py $C --hidden 64  --variants complex_sinh --sweeps 1,2,3 --out data/chirp_h64.csv
