#!/usr/bin/env bash
# Launch exactly one preregistered jsc_v2 atomic task.
#
# Examples:
#   bash scripts/run_jsc_main3.sh poly --dim 3 --order 6
#   bash scripts/run_jsc_main3.sh chirp --sweep 2
#   bash scripts/run_jsc_main3.sh maxwell --sweep 4
#   bash scripts/run_jsc_main3.sh --dry-run poly --dim 2 --order 4
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/cuda_env.sh"
PY="${APOLARITY_PYTHON:-/usr/bin/python3.11}"
export PYTHONUNBUFFERED=1

DRY_RUN=0
SMOKE=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
elif [[ "${1:-}" == "--smoke" ]]; then
  SMOKE=1
  shift
fi

if [[ $# -lt 1 ]]; then
  echo "usage: $0 [--dry-run|--smoke] {poly|chirp|maxwell} setting-args" >&2
  exit 2
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  exec "$PY" "$ROOT/scripts/run_jsc_atomic.py" "$@" --dry-run
fi

LOG_DIR="$ROOT/experiments/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAMILY="$1"
LOG="$LOG_DIR/jsc_v2_${FAMILY}_${STAMP}.log"
PID_FILE="$LOG.pid"

EXTRA=()
if [[ "$SMOKE" -eq 1 ]]; then
  EXTRA+=(--smoke)
fi

setsid nohup "$PY" "$ROOT/scripts/run_jsc_atomic.py" "$@" "${EXTRA[@]}" \
  >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" >"$PID_FILE"

echo "started pid=$PID"
echo "log=$LOG"
echo "pid_file=$PID_FILE"
echo "The runner never commits or pushes results."
