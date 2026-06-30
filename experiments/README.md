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

## The current study (width robustness, 600 s)

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
