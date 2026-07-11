#!/usr/bin/env bash
# Launch exactly one preregistered jsc_v2 Maxwell task.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/scripts/cuda_env.sh"

if [[ $# -ne 2 || "${1:-}" != "--sweep" ]]; then
  echo "usage: $0 --sweep SWEEP" >&2
  exit 2
fi

SWEEP="$2"
exec bash "$ROOT/scripts/run_jsc_main3.sh" maxwell --sweep "$SWEEP"
