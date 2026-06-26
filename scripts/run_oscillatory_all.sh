#!/usr/bin/env bash
# Run the full oscillatory high-order PINN benchmark suite sequentially.
# Each benchmark writes its own results/<name>.csv (+ .json) and a run log, so
# partial progress is preserved if interrupted.
set -u
cd "$(dirname "$0")/.."
PY=/root/apolenv/bin/python
mkdir -p results
SECONDS_PER=${SECONDS_PER:-60}
SEEDS=${SEEDS:-2}
HIDDEN=${HIDDEN:-32}
DEPTH=${DEPTH:-4}
COMMON="--seconds ${SECONDS_PER} --seeds ${SEEDS} --hidden ${HIDDEN} --depth ${DEPTH}"

echo "[driver] start $(date) | seconds=${SECONDS_PER} seeds=${SEEDS} hidden=${HIDDEN}"

run() {  # name script extra-args
  local name=$1; shift
  local script=$1; shift
  echo "[driver] === ${name} === $(date)"
  $PY "experiments/${script}" $COMMON "$@" \
      --out "results/${name}.csv" > "results/run_${name}.log" 2>&1
  echo "[driver] done ${name} rc=$? $(date)"
}

run helmholtz_highk exp_helmholtz_highk.py --sweeps 2,4,6,8
run kdv_dispersive  exp_kdv_dispersive.py  --sweeps 2,3,4,5
run cahn_hilliard   exp_cahn_hilliard.py   --a 2,3 --orders 4,6
run plate_beam      exp_plate_beam.py      --modes 1,2,3
run nls             exp_nls_schrodinger.py --sweeps 1,2,4

echo "[driver] ALL DONE $(date)"
