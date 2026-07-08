#!/usr/bin/env bash
# Wang et al. (2021) Helmholtz Eq.(8): u = sin(a1*pi*x)*sin(a2*pi*y) on (-1,1)^2.
# Three (a1,a2) settings: (1,1), (1,2), (1,4) -- isotropic -> mild -> stiff aniso.
# Protocol: 1200s (20 min), 5 seeds, depth 4, width bracketing (real@128, cs@{128,64}).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 1200 --seeds 5 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
HD=experiments/helmholtz
mkdir -p "$HD/data" experiments/logs

echo "[wang2021-helm] START $(date -Iseconds)"
cd "$ROOT/$HD"

echo "--- wang aniso h128 (1,1),(1,2),(1,4) ---"
$PY -u exp_helmholtz_highk.py $C --hidden 128 --variants "$REAL" \
  --wang-aniso --aniso-pairs "1,1,1,2,1,4" \
  --out data/helmholtz_wang2021_h128.csv

echo "--- wang aniso h64 (complex_sinh only) ---"
$PY -u exp_helmholtz_highk.py $C --hidden 64 --variants complex_sinh \
  --wang-aniso --aniso-pairs "1,1,1,2,1,4" \
  --out data/helmholtz_wang2021_h64.csv

echo "[wang2021-helm] DONE $(date -Iseconds)"
