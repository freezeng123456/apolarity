#!/usr/bin/env bash
# Variable-coefficient ("scattering") Helmholtz in a heterogeneous medium.
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--hidden 32 --depth 4 --lr-schedule cosine"
V=complex_sinh,fourier,siren,mscale
mkdir -p data
$PY exp_helmholtz_vc.py --seconds 60 --seeds 3 $C --variants $V --sweeps 2,4,6 --out data/helmvc_v3.csv
$PY exp_helmholtz_vc.py --seconds 70 --seeds 2 $C --variants $V --sweeps 6 --history --out data/helmvc.csv
