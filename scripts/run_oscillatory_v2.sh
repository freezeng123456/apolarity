#!/usr/bin/env bash
# v2 run: cosine LR decay (uniform), 3 seeds, longer/order-scaled budgets, wider
# sweeps, plus the new controlled polyharmonic order sweep. Writes *_v2.csv so the
# original v1 results are preserved. Competitive variant sets keep total ~5h.
set -u
cd "$(dirname "$0")/.."
PY=/root/apolenv/bin/python
mkdir -p results
SEEDS=${SEEDS:-3}
SCHED="--lr-schedule cosine"
COMP="complex_sinh,fourier,siren"            # competitive SOTA contenders
POLYV="complex_sinh,fourier,siren,real_sinh"  # + same-activation ablation

echo "[v2] start $(date) seeds=${SEEDS} cosine-LR"

run() {  # name script seconds variants extra...
  local name=$1 script=$2 secs=$3 vars=$4; shift 4
  echo "[v2] === ${name} (${secs}s) === $(date)"
  $PY "experiments/${script}" --seconds "$secs" --seeds "$SEEDS" --hidden 32 --depth 4 \
      $SCHED --variants "$vars" "$@" --out "results/${name}_v2.csv" \
      > "results/run_${name}_v2.log" 2>&1
  echo "[v2] done ${name} rc=$? $(date)"
}

# Flagship: controlled ORDER sweep at fixed frequency (orders 2,4,6).
run polyharmonic   exp_polyharmonic.py    150 "$POLYV" --orders 2,4,6
# Main battlefield: 4th-order plate/beam, extended to mode 4.
run plate_beam     exp_plate_beam.py      130 "$COMP"  --modes 1,2,3,4
# High-wavenumber Helmholtz, extended to k=10.
run helmholtz_highk exp_helmholtz_highk.py 80 "$COMP"  --sweeps 2,4,6,8,10
# Dispersive KdV, extended to k=6.
run kdv_dispersive exp_kdv_dispersive.py   90 "$COMP"  --sweeps 2,3,4,5,6

echo "[v2] ALL DONE $(date)"
