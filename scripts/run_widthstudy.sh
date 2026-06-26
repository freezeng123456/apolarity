#!/usr/bin/env bash
# 600s WIDTH-ROBUSTNESS study (refreshes the old 60-100s results).
#
# Parameter-count matching of real vs complex nets has no settled convention, so
# instead of rescaling widths we fix every REAL baseline at width 128 and run the
# complex sinh net at TWO widths, 64 and 128, that bracket them: a complex weight
# carries ~2x the real DOF, so complex@64 ~ half and complex@128 ~ 2x the real
# baselines' equivalent parameters.  If complex@64 ~ complex@128 the method is
# insensitive to width.  The real_sinh baseline is removed.
#
# Variants (depth 4 everywhere):
#   real-valued    : complex_sinh@{64,128}, fourier@128, siren@128, mscale@128
#   complex-valued : complex_sinh@{64,128}, siren@128, fourier@128, tanh@128 (split)
#
# Per family: full physics sweep, 600s/run, 2 seeds, --history.  Each family x
# width writes its own CSV on completion, so a crash only loses the running one.
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/emlnn/bin/python
OUT=results/width
mkdir -p "$OUT"
C="--seconds 600 --seeds 2 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
CPLX=complex_sinh,siren,fourier,tanh

GIT_ID=(-c user.name=freezeng -c user.email=freezeng@tencent.com)

gitpush() {  # name -- force-add this family's outputs, commit, push (never aborts)
  local name=$1 f files=()
  for f in "${OUT}/${name}_h128.csv" "${OUT}/${name}_h128.json" \
           "${OUT}/${name}_h128_history.json" "${OUT}/${name}_h64.csv" \
           "${OUT}/${name}_h64.json" "${OUT}/${name}_h64_history.json" \
           "${OUT}/run_${name}_h128.log" "${OUT}/run_${name}_h64.log"; do
    [ -e "$f" ] && files+=("$f")
  done
  [ ${#files[@]} -eq 0 ] && { echo "[git] no files for ${name}"; return; }
  git add -f "${files[@]}" 2>/dev/null
  if git "${GIT_ID[@]}" commit -m "width-study: ${name} (600s, complex@{64,128} vs SOTA@128)" >/dev/null 2>&1; then
    echo "[git] committed ${name}"
  else
    echo "[git] nothing to commit for ${name}"
  fi
  if timeout 180 git push origin master >/dev/null 2>&1; then
    echo "[git] pushed ${name} $(date)"
  else
    echo "[git] push FAILED for ${name} (continuing) $(date)"
  fi
}

run() {  # name script variants128 extra...
  local name=$1 script=$2 v128=$3; shift 3
  echo "[w] === ${name} @128 === $(date)"
  $PY "experiments/${script}" $C --hidden 128 --variants "$v128" "$@" \
      --out "${OUT}/${name}_h128.csv" > "${OUT}/run_${name}_h128.log" 2>&1
  echo "[w] done ${name}@128 rc=$? $(date)"
  echo "[w] === ${name} @64 === $(date)"
  $PY "experiments/${script}" $C --hidden 64 --variants complex_sinh "$@" \
      --out "${OUT}/${name}_h64.csv" > "${OUT}/run_${name}_h64.log" 2>&1
  echo "[w] done ${name}@64 rc=$? $(date)"
  gitpush "${name}"
}

echo "[w] START $(date)"
# headline order / frequency axes first (most informative if cut short)
run poly1d         exp_polyharmonic.py    "$REAL" --dim 1 --orders 2,4,6,8,10
run poly2d         exp_polyharmonic.py    "$REAL" --dim 2 --orders 2,4,6 --omega0 10
run helmholtz      exp_helmholtz_highk.py "$REAL" --sweeps 2,4,6,8,10
run helmholtz_aniso exp_helmholtz_highk.py "$REAL" --aniso
# genuinely complex-valued
run nls            exp_nls_schrodinger.py "$CPLX" --sweeps 1,2,4
run maxwell        exp_maxwell.py         "$CPLX" --sweeps 2,4,6
# remaining real-valued families
run helmvc         exp_helmholtz_vc.py    "$REAL" --sweeps 2,4,6
run chirp          exp_chirp.py           "$REAL" --sweeps 2,4,6,8
run plate_beam     exp_plate_beam.py      "$REAL" --kind both --modes 1,2,3
run platemix       exp_plate_beam.py      "$REAL" --kind mix --modes 2,3,4
run kdv            exp_kdv_dispersive.py  "$REAL" --sweeps 2,3,4,5,6
run cahn_hilliard  exp_cahn_hilliard.py   "$REAL" --a 2,3 --orders 4,6
echo "[w] ALL DONE $(date)"
