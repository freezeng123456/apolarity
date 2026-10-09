#!/usr/bin/env bash
# Submit from the verified source checkout; scheduler options are chosen after
# inspecting the live Beijing allocation and its limits.
set -e
source /etc/profile
set -u
set -o pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm GPU allocation}"
: "${APOLARITY_SOURCE:?Absolute frozen checkout path is required}"
: "${APOLARITY_PYTHON:?Absolute pinned environment interpreter is required}"
: "${APOLARITY_OUTPUT:?Absolute campaign output path is required}"
stage="${1:?Expected preflight or run}"
case "$stage" in
    preflight) limit=3600 ;;
    run) limit=50400 ;;
    *) exit 2 ;;
esac
[[ "$APOLARITY_SOURCE" == /* && "$APOLARITY_PYTHON" == /* && "$APOLARITY_OUTPUT" == /* ]]
cd "$APOLARITY_SOURCE"
mkdir -p "$APOLARITY_OUTPUT/scheduler"
log="$APOLARITY_OUTPUT/scheduler/${stage}_${SLURM_JOB_ID}.log"
[[ ! -e "$log" ]]
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
detach=()
if command -v setsid >/dev/null 2>&1; then detach=(setsid); fi
nohup "${detach[@]}" timeout --signal=TERM --kill-after=60 "$limit" \
    "$APOLARITY_PYTHON" experiments/torch_benchmarks/beijing_v100.py \
    "$stage" --out "$APOLARITY_OUTPUT" > "$log" 2>&1 < /dev/null &
worker_pid=$!
echo "$worker_pid" > "$APOLARITY_OUTPUT/scheduler/${stage}_${SLURM_JOB_ID}.pid"
kill -0 "$worker_pid"
printf 'job=%s stage=%s pid=%s log=%s\n' "$SLURM_JOB_ID" "$stage" "$worker_pid" "$log"
nvidia-smi
# Keep the allocation alive until the detached process exits.
set +e
wait "$worker_pid"
result=$?
printf '%s\n' "$result" > "$APOLARITY_OUTPUT/scheduler/${stage}_${SLURM_JOB_ID}.exit"
exit "$result"
