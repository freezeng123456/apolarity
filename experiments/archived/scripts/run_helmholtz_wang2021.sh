#!/usr/bin/env bash
# Wang et al. (2021) Helmholtz Eq.(8): u = sin(a1*pi*x)*sin(a2*pi*y) on (-1,1)^2.
# Three (a1,a2) settings: (1,1), (1,2), (1,4) -- isotropic -> mild -> stiff aniso.
# DIAGNOSTIC ONLY: H=128 implementation checks; not jsc_v2 or paper evidence.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/scripts/cuda_env.sh"
PY="${APOLARITY_PYTHON:-/usr/bin/python3.11}"
C="--seconds 1200 --seeds 5 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
HD=experiments/archived/other_families/helmholtz
mkdir -p "$ROOT/$HD/data" "$ROOT/experiments/logs"

echo "WARNING: diagnostic-only output (not jsc_v2); do not use for paper." >&2
echo "[wang2021-helm] START $(date -Iseconds)"
cd "$ROOT/$HD"

echo "--- wang aniso h128 (1,1),(1,2),(1,4) ---"
$PY -u exp_helmholtz_highk.py $C --hidden 128 --variants "$REAL" \
  --wang-aniso --aniso-pairs "1,1,1,2,1,4" \
  --out data/helmholtz_wang2021_h128.csv

echo "[wang2021-helm] DONE $(date -Iseconds)"
