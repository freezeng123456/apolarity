#!/usr/bin/env bash
# v4: complete the v3-consistent suite.  v3 already covers chirp, helmholtz,
# helmvc, kdv, maxwell, platemix, poly1d, poly2d (kept as-is).  This run adds the
# families that were missing from v3 plus the new comparability cases, under the
# SAME config (cosine LR floor lr*0.1, hidden 32, depth 4) so all numbers match:
#
#   * cahn_hilliard   -- 4th/6th-order nonlinear phase field (was pre-v3)
#   * plate_beam      -- isotropic Kirchhoff plate + Euler-Bernoulli beam (iso)
#   * nls             -- cubic NLS, domain aligned to Raissi x in[-5,5], t in[0,pi/2]
#   * helmholtz_aniso -- anisotropic (a1,a2)=(1,4) Wang-2021 Helmholtz
#   * wave1d          -- PINNacle Wave1d-C multi-frequency wave
#
# Writes *_v3.csv so the aggregator picks up one consistent set.
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/emlnn/bin/python
mkdir -p results
SCHED="--lr-schedule cosine"
REALV="complex_sinh,fourier,siren,mscale"   # real SOTA contenders
CPLXV="complex_sinh,siren,fourier,tanh"     # complex-valued: split-real reals

echo "[v4] start $(date)  complete the v3 suite"

run() {  # name script seconds seeds variants extra...
  local name=$1 script=$2 secs=$3 seeds=$4 vars=$5; shift 5
  echo "[v4] === ${name} (${secs}s x ${seeds} seeds) === $(date)"
  $PY "experiments/${script}" --seconds "$secs" --seeds "$seeds" --hidden 32 --depth 4 \
      $SCHED --variants "$vars" "$@" --out "results/${name}_v3.csv" \
      > "results/run_${name}_v4.log" 2>&1
  echo "[v4] done ${name} rc=$? $(date)"
}

# new comparability cases first (cheap, novelty preserved if cut short)
run helmholtz_aniso exp_helmholtz_highk.py 60 2 "$REALV"  --aniso
run nls            exp_nls_schrodinger.py 70 2 "$CPLXV"  --sweeps 1,2,4
# complete the consistent set
run plate_beam     exp_plate_beam.py      70 2 "$REALV"  --kind both --modes 1,2,3
run cahn_hilliard  exp_cahn_hilliard.py   70 2 "$REALV"  --a 2,3 --orders 4,6

echo "[v4] ALL DONE $(date)"
