#!/usr/bin/env bash
# Run the compact, preregistered jsc_v3 task list in order.
#
# The batch is intentionally serial: every atomic task owns one H20 stream,
# writes a validated bundle, and is only followed by the next task after the
# previous task has completed.  The outer process can be launched with nohup.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${APOLARITY_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${APOLARITY_PYTHON:-/usr/bin/python3}"
fi
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src"

LOG_DIR="$ROOT/experiments/logs"
mkdir -p "$LOG_DIR"

run_task() {
  local task_id="$1"
  shift
  local task_dir="$ROOT/experiments/results/jsc_v3/$task_id"
  local log="$LOG_DIR/jsc_v3_${task_id}_formal.log"

  if [[ -f "$task_dir/VALIDATED" ]]; then
    echo "[skip] $task_id already has VALIDATED" | tee -a "$log"
    return 0
  fi

  echo "[begin] $task_id $(date -Is)" | tee -a "$log"
  "$PY" "$ROOT/scripts/run_jsc_atomic.py" "$@" 2>&1 | tee -a "$log"
  echo "[end] $task_id $(date -Is)" | tee -a "$log"
}

run_task poly_d2_o2 poly --dim 2 --order 2
run_task poly_d2_o4 poly --dim 2 --order 4
run_task poly_d2_o6 poly --dim 2 --order 6
run_task chirp_a1 chirp --sweep 1
run_task chirp_a2 chirp --sweep 2
run_task chirp_a3 chirp --sweep 3
run_task maxwell_a2 maxwell --sweep 2
run_task maxwell_a4 maxwell --sweep 4
run_task maxwell_a6 maxwell --sweep 6

echo "[batch-complete] $(date -Is)"
