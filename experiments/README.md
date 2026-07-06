# Experiments

One experiment family per subfolder, each self-contained:

```
<family>/
  exp_<name>.py     # the experiment (imports the shared harness ../common/osc_common.py)
  run.sh            # exact commands to reproduce this family's data
  README.md         # problem, literature source, outputs
  data/             # <stem>_h128.* and <stem>_h64.* (CSV + JSON + history traces)
```

## Families (paper Section "Numerical experiments on oscillatory and high-order PDEs")

| folder | family | operator order |
|---|---|---|
| `helmholtz/`     | high-wavenumber Helmholtz (+ anisotropic) | 2 |
| `helmholtz_vc/`  | variable-coefficient (scattering) Helmholtz | 2 |
| `chirp/`         | non-separable radial chirp | 2 |
| `polyharmonic/`  | polyharmonic order sweep (1D + 2D) | 2–10 |
| `plate_beam/`    | Kirchhoff plate / Euler–Bernoulli beam / mixed-mode plate | 4 |
| `kdv/`           | linearized KdV / dispersive wave | 3 |
| `cahn_hilliard/` | Cahn–Hilliard (nonlinear phase field) | 4 / 6 |
| `nls/`           | cubic nonlinear Schrödinger (complex-valued) | 2 |
| `maxwell/`       | time-harmonic Maxwell, lossy medium (complex-valued) | 2 |
| `core_method/`   | direct-derivative microbenchmark + manufactured 6th-order PINN | — |

## Shared code

- `common/osc_common.py` — architectures (complex `sinh`, Fourier-features,
  SIREN, MscaleDNN, split-real `tanh`), the single complex-Waring Taylor-jet
  derivative backend, and the `train_eval` loop (equal wall-clock budget, floored
  cosine schedule, optional `--history` convergence logging). Every `exp_*.py`
  imports it via a one-line path bootstrap.
- `tools/plot_width.py` — builds the per-family paper figures
  `docs/paper/figures/fig_<key>.pdf` from each family's `data/`.
- `tools/build_width_tables.py` — writes the per-family LaTeX tables
  `docs/paper/tables/w_<key>.tex`.

## JSC main-text experiments (three 20-minute runs)

Going forward, **only three** oscillatory width studies are maintained for the
JSC paper main text. Each uses **1200 s (20 min)** wall-clock, **5 seeds**,
depth 4, step-based history (rel-\(L^2\) every 20 training steps; eval time
excluded from the budget):

| folder | sweep | driver |
|---|---|---|
| `polyharmonic/` (2D only) | orders 2, 4, 6 | `experiments/polyharmonic/run.sh` |
| `chirp/` | \(a=1,2,3\) | `experiments/chirp/run.sh` |
| `maxwell/` | \(a=2,4,6\) | `experiments/maxwell/run.sh` |

Protocol (all three): real baselines at width 128; complex \(\sinh\) at widths
64 and 128. See `scripts/run_jsc_main3.sh` for the batch driver; use
`scripts/run_maxwell_finish.sh` to add missing Maxwell seeds and push.

The remaining families under `experiments/` keep their archived **600 s, 2-seed**
width-study data for the full-suite supplement; they are not scheduled for reruns.

## Archived width study (600 s, 2 seeds)

Real baselines run at width 128; the complex `sinh` net runs at **both** 64 and
128 (a complex weight carries ~2× the real DOF, so these bracket the baselines).
If complex@64 ≈ complex@128 the method is insensitive to width. Depth 4, 2 seeds,
600 s wall-clock per (problem, variant, seed).

## Reproducing the paper

```bash
# everything, in one driver (long):
bash scripts/run_widthstudy.sh
# or a single family:
bash experiments/helmholtz/run.sh
# regenerate figures + tables from the data already in each family's data/:
python experiments/tools/plot_width.py
python experiments/tools/build_width_tables.py
```

Superseded / exploratory experiments are kept in `archived/` and documented in
`archived/progress.md`.
