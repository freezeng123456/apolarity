#!/usr/bin/env bash
# 600s WIDTH-ROBUSTNESS study.
#
# Parameter-count matching of real vs complex nets has no settled convention, so
# instead of rescaling widths we fix every REAL baseline at width 128 and run the
# complex sinh net at TWO widths, 64 and 128, that bracket them: a complex weight
# carries ~2x the real DOF, so complex@64 ~ half and complex@128 ~ 2x the real
# baselines' equivalent parameters.  If complex@64 ~ complex@128 the method is
# insensitive to width.
#
# Variants (depth 4 everywhere):
#   real-valued    : complex_sinh@{64,128}, fourier@128, siren@128, mscale@128
#   complex-valued : complex_sinh@{64,128}, siren@128, fourier@128, tanh@128 (split)
#
# Each family writes into its own folder experiments/<dir>/data/, full physics
# sweep, 600s/run, 2 seeds, --history.  Outputs land per family so a crash only
# loses the running one.
set -u
cd "$(dirname "$0")/.."
PY=/root/miniconda3/envs/emlnn/bin/python
C="--seconds 600 --seeds 2 --depth 4 --lr-schedule cosine --history"
REAL=complex_sinh,fourier,siren,mscale
CPLX=complex_sinh,siren,fourier,tanh

GIT_ID=(-c user.name=freezeng -c user.email=freezeng@tencent.com)

gitpush() {  # name outdir -- add this family's outputs, commit, push (never aborts)
  local name=$1 outdir=$2 f files=()
  for f in "${outdir}/${name}_h128.csv" "${outdir}/${name}_h128.json" \
           "${outdir}/${name}_h128_history.json" "${outdir}/${name}_h64.csv" \
           "${outdir}/${name}_h64.json" "${outdir}/${name}_h64_history.json"; do
    [ -e "$f" ] && files+=("$f")
  done
  [ ${#files[@]} -eq 0 ] && { echo "[git] no files for ${name}"; return; }
  git add "${files[@]}" 2>/dev/null
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

run() {  # name script outdir variants128 extra...
  local name=$1 script=$2 outdir=$3 v128=$4; shift 4
  mkdir -p "$outdir"
  echo "[w] === ${name} @128 === $(date)"
  $PY "$script" $C --hidden 128 --variants "$v128" "$@" \
      --out "${outdir}/${name}_h128.csv" > "${outdir}/run_${name}_h128.log" 2>&1
  echo "[w] done ${name}@128 rc=$? $(date)"
  echo "[w] === ${name} @64 === $(date)"
  $PY "$script" $C --hidden 64 --variants complex_sinh "$@" \
      --out "${outdir}/${name}_h64.csv" > "${outdir}/run_${name}_h64.log" 2>&1
  echo "[w] done ${name}@64 rc=$? $(date)"
  gitpush "${name}" "${outdir}"
}

PH=experiments/polyharmonic;   PLB=experiments/plate_beam
echo "[w] START $(date)"
# headline order / frequency axes first (most informative if cut short)
run poly1d         "$PH/exp_polyharmonic.py"           "$PH/data"  "$REAL" --dim 1 --orders 2,4,6,8,10
run poly2d         "$PH/exp_polyharmonic.py"           "$PH/data"  "$REAL" --dim 2 --orders 2,4,6 --omega0 10
run helmholtz      experiments/helmholtz/exp_helmholtz_highk.py experiments/helmholtz/data "$REAL" --sweeps 2,4,6,8,10
run helmholtz_aniso experiments/helmholtz/exp_helmholtz_highk.py experiments/helmholtz/data "$REAL" --aniso
# genuinely complex-valued
run nls            experiments/nls/exp_nls_schrodinger.py experiments/nls/data "$CPLX" --sweeps 1,2,4
run maxwell        experiments/maxwell/exp_maxwell.py     experiments/maxwell/data "$CPLX" --sweeps 2,4,6
# remaining real-valued families
run helmvc         experiments/helmholtz_vc/exp_helmholtz_vc.py experiments/helmholtz_vc/data "$REAL" --sweeps 2,4,6
run chirp          experiments/chirp/exp_chirp.py         experiments/chirp/data "$REAL" --sweeps 2,4,6,8
run plate_beam     "$PLB/exp_plate_beam.py"               "$PLB/data" "$REAL" --kind both --modes 1,2,3
run platemix       "$PLB/exp_plate_beam.py"               "$PLB/data" "$REAL" --kind mix --modes 2,3,4
run kdv            experiments/kdv/exp_kdv_dispersive.py  experiments/kdv/data "$REAL" --sweeps 2,3,4,5,6
run cahn_hilliard  experiments/cahn_hilliard/exp_cahn_hilliard.py experiments/cahn_hilliard/data "$REAL" --a 2,3 --orders 4,6
echo "[w] ALL DONE $(date)"
