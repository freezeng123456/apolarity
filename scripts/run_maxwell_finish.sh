#!/usr/bin/env bash
# Finish Maxwell JSC main-text width study: add seeds 3-4 (1200 s each run).
# Merges into data/maxwell_h{128,64}.* then commits and pushes.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
export GIT_AUTHOR_NAME=freezeng
export GIT_AUTHOR_EMAIL=freezeng@tencent.com
export GIT_COMMITTER_NAME=freezeng
export GIT_COMMITTER_EMAIL=freezeng@tencent.com

if [[ -x /root/miniconda3/envs/emlnn/bin/python ]]; then
  PY=/root/miniconda3/envs/emlnn/bin/python
elif [[ -n "${APOLARITY_PYTHON:-}" && -x "$APOLARITY_PYTHON" ]]; then
  PY="$APOLARITY_PYTHON"
elif [[ -x /usr/bin/python3.11 ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/scripts/cuda_env.sh"
  PY=/usr/bin/python3.11
else
  echo "[maxwell-finish] no suitable python (need emlnn or python3.11+torch)" >&2
  exit 1
fi

C="--seconds 1200 --seeds 2 --seed-start 3 --depth 4 --lr-schedule cosine --history"
CPLX=complex_sinh,siren,fourier,tanh
MD=experiments/maxwell
mkdir -p "$MD/data" experiments/logs

echo "=== maxwell finish start $(date -Iseconds) py=$PY ==="

"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA required'" || {
  echo "[maxwell-finish] torch/CUDA check failed" >&2
  exit 1
}

cd "$MD"
echo "--- maxwell h128 seeds 3-4 ---"
"$PY" -u exp_maxwell.py $C --hidden 128 --variants "$CPLX" \
  --sweeps 2,4,6 --out data/maxwell_h128_s34.csv
echo "--- maxwell h64 seeds 3-4 ---"
"$PY" -u exp_maxwell.py $C --hidden 64 --variants complex_sinh \
  --sweeps 2,4,6 --out data/maxwell_h64_s34.csv

cd "$ROOT"
"$PY" - "$MD/data/maxwell_h128.csv" "$MD/data/maxwell_h128_s34.csv" <<'PY'
import csv, json, sys
from pathlib import Path

def load_csv(p):
    p = Path(p)
    return list(csv.DictReader(p.open())) if p.exists() else []

def load_hist(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else []

def save_csv(rows, p):
    keys = sorted({k for r in rows for k in r})
    with Path(p).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def save_hist(rows, p):
    Path(p).write_text(json.dumps(rows, indent=2))

def dedupe(rows, id_keys):
    out, seen = [], set()
    for r in rows:
        key = tuple(str(r.get(k, "")) for k in id_keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

base_csv, extra_csv = sys.argv[1], sys.argv[2]
base = Path(base_csv)
id_keys = ("problem", "order", "sweep", "variant", "seed", "rep")
rows = dedupe(load_csv(base_csv) + load_csv(extra_csv), id_keys)
save_csv(rows, base_csv)
base.with_suffix(".json").write_text(json.dumps(rows, indent=2))
hbase = base.with_name(base.stem + "_history.json")
hextra = Path(extra_csv).with_name(Path(extra_csv).stem + "_history.json")
hrows = dedupe(load_hist(hbase) + load_hist(hextra), id_keys)
save_hist(hrows, hbase)
seeds = sorted({int(r["seed"]) for r in rows})
print(f"[merge] {base.name}: {len(rows)} rows, seeds={seeds}, {len(hrows)} history traces")
PY

"$PY" - "$MD/data/maxwell_h64.csv" "$MD/data/maxwell_h64_s34.csv" <<'PY'
import csv, json, sys
from pathlib import Path

def load_csv(p):
    p = Path(p)
    return list(csv.DictReader(p.open())) if p.exists() else []

def load_hist(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else []

def save_csv(rows, p):
    keys = sorted({k for r in rows for k in r})
    with Path(p).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def save_hist(rows, p):
    Path(p).write_text(json.dumps(rows, indent=2))

def dedupe(rows, id_keys):
    out, seen = [], set()
    for r in rows:
        key = tuple(str(r.get(k, "")) for k in id_keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

base_csv, extra_csv = sys.argv[1], sys.argv[2]
base = Path(base_csv)
id_keys = ("problem", "order", "sweep", "variant", "seed", "rep")
rows = dedupe(load_csv(base_csv) + load_csv(extra_csv), id_keys)
save_csv(rows, base_csv)
base.with_suffix(".json").write_text(json.dumps(rows, indent=2))
hbase = base.with_name(base.stem + "_history.json")
hextra = Path(extra_csv).with_name(Path(extra_csv).stem + "_history.json")
hrows = dedupe(load_hist(hbase) + load_hist(hextra), id_keys)
save_hist(hrows, hbase)
seeds = sorted({int(r["seed"]) for r in rows})
print(f"[merge] {base.name}: {len(rows)} rows, seeds={seeds}, {len(hrows)} history traces")
PY

cd "$ROOT"
git add \
  experiments/maxwell/exp_maxwell.py \
  experiments/maxwell/run.sh \
  experiments/maxwell/README.md \
  experiments/maxwell/data/maxwell_h128.csv \
  experiments/maxwell/data/maxwell_h128.json \
  experiments/maxwell/data/maxwell_h128_history.json \
  experiments/maxwell/data/maxwell_h128_s34.csv \
  experiments/maxwell/data/maxwell_h128_s34.json \
  experiments/maxwell/data/maxwell_h128_s34_history.json \
  experiments/maxwell/data/maxwell_h64.csv \
  experiments/maxwell/data/maxwell_h64.json \
  experiments/maxwell/data/maxwell_h64_history.json \
  experiments/maxwell/data/maxwell_h64_s34.csv \
  experiments/maxwell/data/maxwell_h64_s34.json \
  experiments/maxwell/data/maxwell_h64_s34_history.json \
  experiments/README.md \
  README.md \
  scripts/run_maxwell_finish.sh \
  scripts/run_jsc_main3.sh

if git diff --cached --quiet; then
  echo "[git] nothing to commit"
else
  git commit -m "$(cat <<'EOF'
Complete Maxwell width study to 5 seeds (1200s, seeds 3-4 merged).

Fix exp_maxwell.py to honor --seed-start; document the three 20-minute
JSC main-text experiments (poly2d, chirp, Maxwell).
EOF
)"
  git push origin HEAD
  echo "[git] pushed Maxwell finish $(date -Iseconds)"
fi

echo "=== maxwell finish done $(date -Iseconds) ==="
