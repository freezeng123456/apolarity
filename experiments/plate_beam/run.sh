#!/usr/bin/env bash
# Kirchhoff plate / Euler-Bernoulli beam (iso) + anisotropic mixed-mode plate.
set -u; cd "$(dirname "$0")"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--hidden 32 --depth 4 --lr-schedule cosine"
V=complex_sinh,fourier,siren,mscale
mkdir -p data
$PY exp_plate_beam.py --seconds 70 --seeds 2 $C --variants $V --kind both --modes 1,2,3 --out data/plate_beam_v3.csv
$PY exp_plate_beam.py --seconds 90 --seeds 2 $C --variants $V --kind mix  --modes 2,3,4 --out data/platemix_v3.csv
$PY exp_plate_beam.py --seconds 70 --seeds 2 $C --variants $V --kind both --modes 2 --history --out data/plate_beam.csv
$PY exp_plate_beam.py --seconds 70 --seeds 2 $C --variants $V --kind mix  --modes 3 --history --out data/platemix.csv
