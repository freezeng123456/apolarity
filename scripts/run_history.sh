#!/usr/bin/env bash
# Convergence-trace runs for the paper figures.  For each PDE family we re-run a
# single REPRESENTATIVE case (the swept-parameter panels in the paper come from
# the full v3/v4 aggregates; these runs only add the rel-L2 / loss vs training
# time traces) with --history under the SAME protocol as the accuracy tables:
# cosine LR floor lr/10, hidden 32, depth 4, 2 seeds, 70 s budget.
#
# Writes results/hist/<fam>.csv plus results/hist/<fam>_history.json (the trace
# sidecar consumed by experiments/common/plot_convergence.py).
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/emlnn/bin/python
mkdir -p results/hist
SECS=70
SEEDS=2
COMMON="--seconds $SECS --seeds $SEEDS --hidden 32 --depth 4 --lr-schedule cosine --history"
REALV="complex_sinh,fourier,siren,mscale"
POLYV="complex_sinh,fourier,siren,mscale,real_sinh"
NLSV="complex_sinh,siren,fourier,tanh"
MAXV="complex_sinh,tanh,siren,fourier"

run() {  # name script variants extra...
  local name=$1 script=$2 vars=$3; shift 3
  echo "[hist] === ${name} === $(date)"
  $PY "experiments/${script}" $COMMON --variants "$vars" "$@" \
      --out "results/hist/${name}.csv" > "results/hist/run_${name}.log" 2>&1
  echo "[hist] done ${name} rc=$? $(date)"
}

echo "[hist] start $(date)"

# --- order axis (fixed frequency, vary differential order) ---
run poly2d  exp_polyharmonic.py    "$POLYV" --dim 2 --orders 6
run poly1d  exp_polyharmonic.py    "$POLYV" --dim 1 --orders 4

# --- frequency axis (2nd-order Helmholtz, const + variable coeff) ---
run helmholtz exp_helmholtz_highk.py "$REALV" --sweeps 6
run helmvc    exp_helmholtz_vc.py    "$REALV" --sweeps 6

# --- non-separable expressivity (radial chirp) ---
run chirp   exp_chirp.py           "$REALV" --sweeps 4

# --- high-order real (plate / beam / mixed-mode plate) ---
run plate_beam exp_plate_beam.py   "$REALV" --kind both --modes 2
run platemix   exp_plate_beam.py   "$REALV" --kind mix  --modes 3

# --- dispersive counter-case (3rd-order KdV) ---
run kdv     exp_kdv_dispersive.py  "$REALV" --sweeps 4

# --- nonlinear high-order phase field (6th-order Cahn-Hilliard) ---
run cahn_hilliard exp_cahn_hilliard.py "$REALV" --a 2 --orders 6

# --- genuinely complex-valued fields ---
run nls     exp_nls_schrodinger.py "$NLSV" --sweeps 2
run maxwell exp_maxwell.py         "$MAXV" --sweeps 4

echo "[hist] ALL DONE $(date)"
