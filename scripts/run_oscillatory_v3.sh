#!/usr/bin/env bash
# v3 consolidated run (focus A: solidify main results, B: extend examples).
#
# Unified schedule for EVERY benchmark: cosine LR with a gentle floor (lr_final =
# lr*0.1, the osc_common default) -- this keeps the high-order convergence gains
# of cosine while no longer starving slow problems like KdV (the v2 regression).
#
# New in v3:
#   * poly1d  -- 1D controlled ORDER sweep to order 10 (cheap single-term operator,
#                frequency-matched omega0=pi); the headline order-axis evidence.
#   * maxwell -- time-harmonic Maxwell in a lossy medium (genuinely complex).
#   * helmvc  -- variable-coefficient ("scattering") Helmholtz (heterogeneous medium).
#   * platemix-- anisotropic (m, m+1) plate modes at fixed order 4.
#   * kdv     -- re-run under the fixed schedule (v2 KdV was hurt by the old floor).
#   * poly2d / helmholtz -- re-run for a single consistent config.
#
# Writes *_v3.csv so earlier results are preserved.  New examples run FIRST.
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/emlnn/bin/python
mkdir -p results
SCHED="--lr-schedule cosine"
ORDERV="complex_sinh,fourier,siren,mscale,real_sinh"  # headline: full baseline set
REALV="complex_sinh,fourier,siren,mscale"             # real SOTA contenders
CPLXV="complex_sinh,siren,fourier,tanh"               # complex-valued: split-real reals

echo "[v3] start $(date)  unified cosine-LR (floor lr*0.1)"

run() {  # name script seconds seeds variants extra...
  local name=$1 script=$2 secs=$3 seeds=$4 vars=$5; shift 5
  echo "[v3] === ${name} (${secs}s x ${seeds} seeds) === $(date)"
  $PY "experiments/${script}" --seconds "$secs" --seeds "$seeds" --hidden 32 --depth 4 \
      $SCHED --variants "$vars" "$@" --out "results/${name}_v3.csv" \
      > "results/run_${name}_v3.log" 2>&1
  echo "[v3] done ${name} rc=$? $(date)"
}

# --- B: new examples first (novelty preserved if the run is cut short) ---
# Headline: 1D controlled ORDER sweep to order 10 (high orders need more steps).
run poly1d    exp_polyharmonic.py   85  3 "$ORDERV" --dim 1 --orders 2,4,6,8,10
# True complex-valued, linear: time-harmonic Maxwell (lossy).
run maxwell   exp_maxwell.py        60  2 "$CPLXV"  --sweeps 2,4,6
# Variable-coefficient / scattering Helmholtz (heterogeneous medium).
run helmvc    exp_helmholtz_vc.py   60  3 "$REALV"  --sweeps 2,4,6
# Anisotropic (m, m+1) plate modes at fixed order 4.
run platemix  exp_plate_beam.py     90  2 "$REALV"  --kind mix --modes 2,3,4

# --- A: solidify the main results under the unified schedule ---
# Dispersive KdV re-run with the fixed (floored) schedule.
run kdv       exp_kdv_dispersive.py 70  3 "$REALV"  --sweeps 2,3,4,5,6
# 2D controlled ORDER sweep (omega0=10 proven for orders 2-6).
run poly2d    exp_polyharmonic.py   100 2 "$ORDERV" --dim 2 --orders 2,4,6 --omega0 10
# High-wavenumber Helmholtz frequency axis.
run helmholtz exp_helmholtz_highk.py 60 2 "$REALV"  --sweeps 2,4,6,8,10

echo "[v3] ALL DONE $(date)"
