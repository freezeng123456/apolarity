#!/usr/bin/env bash
# Polyharmonic order sweep (1D to order 10, 2D to order 6) -- headline order axis.
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--hidden 32 --depth 4 --lr-schedule cosine"
V=complex_sinh,fourier,siren,mscale,real_sinh
mkdir -p data
# accuracy tables
$PY exp_polyharmonic.py --seconds 85  --seeds 3 $C --variants $V --dim 1 --orders 2,4,6,8,10 --out data/poly1d_v3.csv
$PY exp_polyharmonic.py --seconds 100 --seeds 2 $C --variants $V --dim 2 --orders 2,4,6 --omega0 10 --out data/poly2d_v3.csv
# convergence traces (representative order)
$PY exp_polyharmonic.py --seconds 70 --seeds 2 $C --variants $V --dim 1 --orders 4 --history --out data/poly1d.csv
$PY exp_polyharmonic.py --seconds 70 --seeds 2 $C --variants $V --dim 2 --orders 6 --history --out data/poly2d.csv
