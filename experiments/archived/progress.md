# Archived experiments and project history

This folder keeps exploratory work that is **not** part of the final paper
(`docs/paper/jsc_paper_main.tex`), for provenance only. Nothing here is imported
by the live experiments; the maintained code is in `experiments/<family>/` with
the shared harness `experiments/common/osc_common.py`.

## Project layout (after the 2026-06-30 reorganization)

```
experiments/
  common/        shared harness (osc_common.py)
  tools/         plot_width.py, build_width_tables.py  (figures + LaTeX tables)
  <family>/      exp_*.py + run.sh + README.md + data/   (one per PDE family)
  core_method/   microbenchmark + manufactured 6th-order PINN (paper Section 5)
  archived/      this folder
```

The live (paper) experiments are, one family per folder: `helmholtz`,
`helmholtz_vc`, `chirp`, `polyharmonic`, `plate_beam`, `kdv`, `cahn_hilliard`,
`nls`, `maxwell`, plus `core_method`.

## Current data: the 600 s width study

The canonical numbers in the paper come from the width-robustness study and live
with each experiment in `experiments/<family>/data/` as `<stem>_h128.*` (complex
`sinh` + real baselines at width 128) and `<stem>_h64.*` (complex `sinh` at width
64), each `.csv` + `.json` + `_history.json`. Real baselines are
Fourier/SIREN/MscaleDNN (`tanh` split-real for the complex-valued families).

## Archived scripts (superseded, not in the paper)

- **`exp_oscillatory_suite.py`** — the original monolithic suite running all
  oscillatory PDEs in one script. Superseded by the per-family `exp_*.py` files
  on the shared `osc_common.py` harness.
- **`exp_complex_vs_real.py`** — the early "should we use complex parameters?"
  study (complex vs. real vs. parameter-matched real on single-monomial PDEs).
  Its conclusion is now demonstrated by the full benchmark suite + width study.
- **`aggregate_oscillatory.py`, `aggregate_osc.py`** — aggregators for the older
  runs; superseded by `tools/plot_width.py` / `tools/build_width_tables.py`.
- **`pinn_5min_compare.py`, `quick_compare_5min.py`** — early 5-minute backend
  sanity checks; the maintained microbenchmark is
  `core_method/benchmark_single_monomial.py`.

## Removed during reorganization (recoverable from git history)

The pre-width-study outputs (`results/` with the `*_v2`/`*_v3` runs, the `diag/`
parameter-tuning sweeps, and all `*.log` driver logs), the earlier benchmark
drivers (`scripts/run_oscillatory_{all,v2,v3,v4}.sh`, `run_history.sh`,
`run_quick_compare.sh`, `reorganize.sh`), the legacy `plot_convergence.py`, and
the `*.png` figure previews were deleted. The maintained driver is
`scripts/run_widthstudy.sh`.

## Key findings from the archived sweeps (kept for the record)

- **Frequency matching.** An order-`m` operator amplifies an initialization
  frequency `ω` like `ω^m`; an over-large `omega0` buries the learning signal.
  Setting `omega0 ≈ |∇|` (π in 1D, 2π/10 in 2D) unlocked orders 8–10.
- **Cosine LR floor.** A too-low tail (`lr_final = lr/100`) starved slow problems
  such as KdV; the floor was raised to `lr_final = lr/10`.
- **Wave1d-C dropped.** The manufactured all-faces-Dirichlet harness is ill-posed
  for the hyperbolic wave equation (low interior loss, high L2 for every method),
  so that benchmark was removed rather than reported misleadingly.
