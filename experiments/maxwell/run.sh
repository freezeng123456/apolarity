#!/usr/bin/env bash
# Time-harmonic Maxwell (2D TM mode) in a lossy medium (complex-valued).
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--hidden 32 --depth 4 --lr-schedule cosine"
V=complex_sinh,siren,fourier,tanh
mkdir -p data
$PY exp_maxwell.py --seconds 60 --seeds 2 $C --variants $V --sweeps 2,4,6 --out data/maxwell_v3.csv
$PY exp_maxwell.py --seconds 70 --seeds 2 $C --variants $V --sweeps 4 --history --out data/maxwell.csv
