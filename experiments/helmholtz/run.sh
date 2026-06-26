#!/usr/bin/env bash
# High-wavenumber Helmholtz (isotropic frequency sweep + anisotropic case).
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--hidden 32 --depth 4 --lr-schedule cosine"
V=complex_sinh,fourier,siren,mscale
mkdir -p data
$PY exp_helmholtz_highk.py --seconds 60 --seeds 2 $C --variants $V --sweeps 2,4,6,8,10 --out data/helmholtz_v3.csv
$PY exp_helmholtz_highk.py --seconds 60 --seeds 2 $C --variants $V --aniso          --out data/helmholtz_aniso_v3.csv
$PY exp_helmholtz_highk.py --seconds 70 --seeds 2 $C --variants $V --sweeps 6 --history --out data/helmholtz.csv
