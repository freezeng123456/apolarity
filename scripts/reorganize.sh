#!/usr/bin/env bash
# One-shot project reorganization: one experiment family per subfolder, each
# self-contained (script + run.sh + README + data/), shared harness kept at the
# experiments/ root, and all superseded / exploratory material swept into
# experiments/archived/ (documented in archived/progress.md).
#
# Run from the repo root AFTER scripts/run_history.sh has finished (so nothing is
# still writing into results/).  Idempotent-ish: re-running is harmless because
# every move is guarded.
set -u
cd "$(dirname "$0")/.."
E=experiments
mkdir -p "$E/archived/scripts" "$E/archived/old_results" "$E/core_method/data"

mv_if() { for f in "$@"; do [ -e "$f" ] && mv -f "$f" "$DST/" || true; done; }

# --- per-family: move the script and its canonical data (v3 + history) ---
fam() {  # famdir script  [data-stems...]   (stems are looked up in results/ and results/hist/)
  local fam=$1 script=$2; shift 2
  mkdir -p "$E/$fam/data"; DST="$E/$fam"
  mv_if "$E/$script"
  DST="$E/$fam/data"
  for s in "$@"; do
    mv_if results/"$s".csv results/"$s".json
    mv_if results/hist/"$s".csv results/hist/"$s".json results/hist/"$s"_history.json
  done
}

fam helmholtz     exp_helmholtz_highk.py  helmholtz_v3 helmholtz_aniso_v3 helmholtz
fam helmholtz_vc  exp_helmholtz_vc.py     helmvc_v3 helmvc
fam chirp         exp_chirp.py            chirp_v3 chirp
fam polyharmonic  exp_polyharmonic.py     poly1d_v3 poly2d_v3 poly1d poly2d
fam plate_beam    exp_plate_beam.py       plate_beam_v3 platemix_v3 plate_beam platemix
fam kdv           exp_kdv_dispersive.py   kdv_v3 kdv
fam cahn_hilliard exp_cahn_hilliard.py    cahn_hilliard_v3 cahn_hilliard
fam nls           exp_nls_schrodinger.py  nls_v3 nls
fam maxwell       exp_maxwell.py          maxwell_v3 maxwell

# --- core-method (paper Section 5) scripts ---
DST="$E/core_method"
mv_if "$E/benchmark_single_monomial.py" "$E/train_pinn_monomial.py" \
      "$E/train_pinn_ch_sixth_order.py" "$E/profile_complex_waring_steps.py" \
      "$E/generate_paper_tables.py"

# --- cross-family roll-up next to the shared aggregator ---
DST="$E"; mv_if results/aggregate_final.txt

# --- archived (superseded scripts) ---
DST="$E/archived"
mv_if "$E/exp_complex_vs_real.py" "$E/exp_oscillatory_suite.py" \
      "$E/aggregate_oscillatory.py" "$E/pinn_5min_compare.py" \
      "$E/quick_compare_5min.py"
DST="$E/archived/scripts"
mv_if scripts/run_oscillatory_v2.sh scripts/run_oscillatory_all.sh \
      scripts/run_quick_compare.sh

# --- archived data: sweep everything left in results/ (diag, *_v2, older
#     duplicates, logs, scratch) into one place ---
DST="$E/archived/old_results"
mv_if results_cmp.log results_fair.log
if [ -d results ]; then
  shopt -s dotglob nullglob 2>/dev/null || true
  for f in results/*; do
    [ "$f" = "results/hist" ] && continue
    mv -f "$f" "$E/archived/old_results/" 2>/dev/null || true
  done
  for f in results/hist/*; do mv -f "$f" "$E/archived/old_results/" 2>/dev/null || true; done
fi

echo "[reorg] done.  experiments/ now holds one folder per family + archived/."
