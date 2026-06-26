# Archived experiments and superseded artifacts

This folder records exploratory work that is **not** part of the final paper
(`docs/paper/jsc_paper_main.tex`). It is kept for provenance and reproducibility
only. Nothing here is imported by the live experiments; the maintained code is in
`experiments/<family>/` and the shared harness `experiments/osc_common.py`.

The live (paper) experiments are, one family per folder:

| folder | family | paper section |
|---|---|---|
| `helmholtz/`      | high-wavenumber Helmholtz (+ anisotropic) | high-frequency 2nd order |
| `helmholtz_vc/`   | variable-coefficient (scattering) Helmholtz | high-frequency 2nd order |
| `chirp/`          | non-separable radial chirp | high-frequency 2nd order |
| `polyharmonic/`   | polyharmonic order sweep (1D + 2D) | high-order real |
| `plate_beam/`     | Kirchhoff plate / Euler–Bernoulli beam / mixed-mode plate | high-order real |
| `kdv/`            | linearized KdV / dispersive wave | high-order real |
| `cahn_hilliard/`  | Cahn–Hilliard (4th/6th order, nonlinear) | high-order real |
| `nls/`            | cubic nonlinear Schrödinger | complex-valued |
| `maxwell/`        | time-harmonic Maxwell (lossy) | complex-valued |
| `core_method/`    | direct-derivative microbenchmark + manufactured 6th-order PINN | method experiments |

## Archived scripts (superseded, not in the paper)

- **`exp_oscillatory_suite.py`** — the original monolithic suite that ran all
  oscillatory PDEs in one script. Superseded by the per-family `exp_*.py` files,
  which use the shared `osc_common.py` harness, frequency-matched initialization,
  and the floored cosine schedule. Output: `old_results/oscillatory_suite.*`.
- **`exp_complex_vs_real.py`** — the early "should we use complex parameters?"
  study (complex vs. real vs. parameter-matched real on single-monomial PDEs).
  Its conclusion (complex parameters help on balanced high-order operators) is
  now demonstrated properly by the full benchmark suite. Output:
  `old_results/complex_vs_real_5min.*`, `complex_vs_real_fair_seeds.*`,
  `strong_real_baseline.*`.
- **`aggregate_oscillatory.py`** — aggregator for `exp_oscillatory_suite.py`.
  Superseded by `experiments/aggregate_osc.py`, which also checks the
  advantage-factor acceptance criterion.
- **`pinn_5min_compare.py`** — early 5-minute backend comparison on the 4D
  sixth-order PDE. Superseded by `core_method/train_pinn_ch_sixth_order.py`
  (the canonical Section 5.4 run).
- **`quick_compare_5min.py`** — quick single-monomial backend sanity check.
  A development convenience; the maintained microbenchmark is
  `core_method/benchmark_single_monomial.py`.
- **`scripts/run_oscillatory_v2.sh`, `run_oscillatory_all.sh`,
  `run_quick_compare.sh`** — earlier benchmark drivers. The maintained drivers
  are `scripts/run_oscillatory_v3.sh` and `run_oscillatory_v4.sh` (accuracy
  tables) and `scripts/run_history.sh` (convergence traces for the figures).

## Archived data (`old_results/`)

Superseded numeric outputs: the `diag/` parameter-tuning sweeps (frequency
matching, cosine-schedule floor, order-8 `omega0`), the pre-`v3` runs
(`*_v2.*`, and the unsuffixed `helmholtz_highk.*`, `kdv_dispersive.*`,
`plate_beam.*`, `cahn_hilliard.*`, `nls.*` duplicates), and all `*.log` driver
logs. The **canonical** numbers that appear in the paper live with their
experiment in `experiments/<family>/data/` (the `*_v3.*` CSV/JSON plus the
`*_history.json` convergence traces), and the cross-family roll-up is
`experiments/aggregate_final.txt`.

## Key findings from the archived sweeps (kept for the record)

- **Frequency matching.** An order-`m` operator amplifies an initialization
  frequency `ω` like `ω^m`; an over-large `omega0` buries the learning signal.
  Setting `omega0 ≈ |∇|` (π in 1D, 2π in 2D) unlocked orders 8–10. (`diag/poly_om*`.)
- **Cosine LR floor.** A too-low tail (`lr_final = lr/100`) starved slow problems
  such as KdV; the floor was raised to `lr_final = lr/10`. (`diag/kdv_*`.)
- **Wave1d-C dropped.** The manufactured all-faces-Dirichlet harness is ill-posed
  for the hyperbolic wave equation (low interior loss, high L2 for every method),
  so that benchmark was removed rather than reported misleadingly.
