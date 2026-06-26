#!/usr/bin/env bash
# Non-separable radial chirp -- spatially varying local frequency.
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--hidden 32 --depth 4 --lr-schedule cosine"
V=complex_sinh,fourier,siren,mscale
mkdir -p data
$PY exp_chirp.py --seconds 70 --seeds 3 $C --variants $V --sweeps 2,4,6,8 --out data/chirp_v3.csv
$PY exp_chirp.py --seconds 70 --seeds 2 $C --variants $V --sweeps 4 --history --out data/chirp.csv
