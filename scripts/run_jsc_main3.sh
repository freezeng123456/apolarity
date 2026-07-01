#!/usr/bin/env bash
# JSC main-text width study: polyharmonic 2D, chirp (a=1,2,3), Maxwell.
# 1200 s wall-clock, 5 seeds (Maxwell: keep seeds 0-1, add 2-4).
# History: rel-L2 every 20 training steps (eval time excluded from budget).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONUNBUFFERED=1
export GIT_AUTHOR_NAME=freezeng
export GIT_AUTHOR_EMAIL=freezeng@tencent.com
export GIT_COMMITTER_NAME=freezeng
export GIT_COMMITTER_EMAIL=freezeng@tencent.com
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 1200 --seeds 5 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
CPLX=complex_sinh,siren,fourier,tanh
LOG_DIR="$ROOT/experiments/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/jsc_main3_20min_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== jsc_main3 start $(date -Iseconds) pid=$$ ==="
echo "log=$LOG"

git_commit_task() {
  local msg="$1"
  shift
  cd "$ROOT" || exit 1
  git add "$@"
  if git diff --cached --quiet; then
    echo "[git] nothing to commit for: $msg"
    return 0
  fi
  GIT_AUTHOR_NAME="$GIT_AUTHOR_NAME" \
  GIT_AUTHOR_EMAIL="$GIT_AUTHOR_EMAIL" \
  GIT_COMMITTER_NAME="$GIT_COMMITTER_NAME" \
  GIT_COMMITTER_EMAIL="$GIT_COMMITTER_EMAIL" \
  git commit -m "$(cat <<EOF
$msg
EOF
)"
  git push origin HEAD
  echo "[git] pushed: $msg"
}

merge_csv_hist() {
  local base="$1" extra="$2"
  "$PY" - "$base" "$extra" <<'PY'
import csv, json, sys
from pathlib import Path

def load_csv(p):
    p = Path(p)
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))

def load_hist(p):
    p = Path(p)
    if not p.exists():
        return []
    return json.loads(p.read_text())

def save_csv(rows, p):
    p = Path(p)
    keys = sorted({k for r in rows for k in r})
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def save_hist(rows, p):
    p = Path(p)
    with p.open("w") as f:
        json.dump(rows, f, indent=2)

base_csv, extra_csv = sys.argv[1], sys.argv[2]
base = Path(base_csv)
extra = Path(extra_csv)
id_keys = ("problem", "order", "sweep", "variant", "seed", "rep")

def dedupe(rows):
    out, seen = [], set()
    for r in rows:
        key = tuple(str(r.get(k, "")) for k in id_keys if k in r or k == "rep")
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

rows = dedupe(load_csv(base_csv) + load_csv(extra_csv))
save_csv(rows, base_csv)
with base.with_suffix(".json").open("w") as f:
    json.dump(rows, f, indent=2)

hbase = base.with_name(base.stem + "_history.json")
hextra = extra.with_name(extra.stem + "_history.json")
hrows = dedupe(load_hist(hbase) + load_hist(hextra))
save_hist(hrows, hbase)
print(f"[merge] {base.name}: {len(rows)} csv rows, {len(hrows)} history traces")
PY
}

# --- Polyharmonic 2D only ---
cd "$ROOT/experiments/polyharmonic"
mkdir -p data
echo "--- polyharmonic 2D h128 ---"
$PY -u exp_polyharmonic.py $C --hidden 128 --variants "$REAL" \
  --dim 2 --orders 2,4,6 --omega0 10 --out data/poly2d_h128.csv
echo "--- polyharmonic 2D h64 ---"
$PY -u exp_polyharmonic.py $C --hidden 64 --variants complex_sinh \
  --dim 2 --orders 2,4,6 --omega0 10 --out data/poly2d_h64.csv
git_commit_task "Add polyharmonic 2D width study (1200s, 5 seeds, step-based history)." \
  experiments/common/osc_common.py \
  experiments/polyharmonic/run.sh \
  experiments/polyharmonic/data/poly2d_h128.csv \
  experiments/polyharmonic/data/poly2d_h128.json \
  experiments/polyharmonic/data/poly2d_h128_history.json \
  experiments/polyharmonic/data/poly2d_h64.csv \
  experiments/polyharmonic/data/poly2d_h64.json \
  experiments/polyharmonic/data/poly2d_h64_history.json \
  scripts/run_jsc_main3.sh

# --- Chirp a=1,2,3 ---
cd "$ROOT/experiments/chirp"
mkdir -p data
echo "--- chirp h128 ---"
$PY -u exp_chirp.py $C --hidden 128 --variants "$REAL" \
  --sweeps 1,2,3 --out data/chirp_h128.csv
echo "--- chirp h64 ---"
$PY -u exp_chirp.py $C --hidden 64 --variants complex_sinh \
  --sweeps 1,2,3 --out data/chirp_h64.csv
git_commit_task "Add chirp width study a=1,2,3 (1200s, 5 seeds, step-based history)." \
  experiments/chirp/run.sh \
  experiments/chirp/data/chirp_h128.csv \
  experiments/chirp/data/chirp_h128.json \
  experiments/chirp/data/chirp_h128_history.json \
  experiments/chirp/data/chirp_h64.csv \
  experiments/chirp/data/chirp_h64.json \
  experiments/chirp/data/chirp_h64_history.json

# --- Maxwell: seeds 0-1 done; add seeds 2-4 then merge ---
cd "$ROOT/experiments/maxwell"
mkdir -p data
if [[ -f data/maxwell_h128.csv ]]; then
  cp -a data/maxwell_h128.csv "data/maxwell_h128_seeds01_${STAMP}.csv"
  cp -a data/maxwell_h128_history.json "data/maxwell_h128_seeds01_${STAMP}_history.json" 2>/dev/null || true
fi
if [[ -f data/maxwell_h64.csv ]]; then
  cp -a data/maxwell_h64.csv "data/maxwell_h64_seeds01_${STAMP}.csv"
  cp -a data/maxwell_h64_history.json "data/maxwell_h64_seeds01_${STAMP}_history.json" 2>/dev/null || true
fi
echo "--- maxwell h128 seeds 2-4 ---"
$PY -u exp_maxwell.py --seconds 1200 --seeds 3 --seed-start 2 --depth 4 --lr-schedule cosine --history \
  --hidden 128 --variants "$CPLX" --sweeps 2,4,6 --out data/maxwell_h128_s234.csv
echo "--- maxwell h64 seeds 2-4 ---"
$PY -u exp_maxwell.py --seconds 1200 --seeds 3 --seed-start 2 --depth 4 --lr-schedule cosine --history \
  --hidden 64 --variants complex_sinh --sweeps 2,4,6 --out data/maxwell_h64_s234.csv
if [[ -f data/maxwell_h128_seeds01_${STAMP}.csv ]]; then
  cp -a "data/maxwell_h128_seeds01_${STAMP}.csv" data/maxwell_h128.csv
  cp -a "data/maxwell_h128_seeds01_${STAMP}_history.json" data/maxwell_h128_history.json 2>/dev/null || true
  merge_csv_hist data/maxwell_h128.csv data/maxwell_h128_s234.csv
fi
if [[ -f data/maxwell_h64_seeds01_${STAMP}.csv ]]; then
  cp -a "data/maxwell_h64_seeds01_${STAMP}.csv" data/maxwell_h64.csv
  cp -a "data/maxwell_h64_seeds01_${STAMP}_history.json" data/maxwell_h64_history.json 2>/dev/null || true
  merge_csv_hist data/maxwell_h64.csv data/maxwell_h64_s234.csv
fi
git_commit_task "Add Maxwell width study seeds 2-4 merged (1200s, step-based history)." \
  experiments/maxwell/run.sh \
  experiments/maxwell/exp_maxwell.py \
  experiments/maxwell/data/maxwell_h128.csv \
  experiments/maxwell/data/maxwell_h128.json \
  experiments/maxwell/data/maxwell_h128_history.json \
  experiments/maxwell/data/maxwell_h128_s234.csv \
  experiments/maxwell/data/maxwell_h128_s234.json \
  experiments/maxwell/data/maxwell_h128_s234_history.json \
  experiments/maxwell/data/maxwell_h64.csv \
  experiments/maxwell/data/maxwell_h64.json \
  experiments/maxwell/data/maxwell_h64_history.json \
  experiments/maxwell/data/maxwell_h64_s234.csv \
  experiments/maxwell/data/maxwell_h64_s234.json \
  experiments/maxwell/data/maxwell_h64_s234_history.json \
  "experiments/maxwell/data/maxwell_h128_seeds01_${STAMP}.csv" \
  "experiments/maxwell/data/maxwell_h128_seeds01_${STAMP}_history.json" \
  "experiments/maxwell/data/maxwell_h64_seeds01_${STAMP}.csv" \
  "experiments/maxwell/data/maxwell_h64_seeds01_${STAMP}_history.json"

echo "=== jsc_main3 done $(date -Iseconds) ==="
