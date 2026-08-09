#!/usr/bin/env bash
# Launch exactly one preregistered jsc_v3 Poly task.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT/scripts/cuda_env.sh"

if [[ $# -ne 4 || "${1:-}" != "--dim" || "${3:-}" != "--order" ]]; then
  echo "usage: $0 --dim DIM --order ORDER" >&2
  exit 2
fi

DIM="$2"
ORDER="$4"
exec bash "$ROOT/scripts/run_jsc_main3.sh" poly --dim "$DIM" --order "$ORDER"
