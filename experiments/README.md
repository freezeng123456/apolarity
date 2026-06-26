# Experiments

One experiment family per subfolder. Each family folder is self-contained:

```
<family>/
  exp_<name>.py     # the experiment (imports the shared harness ../osc_common.py)
  run.sh            # exact commands to reproduce this family's CSVs + history
  README.md         # problem, literature source, outputs
  data/             # <name>_v3.csv/.json (accuracy) + <name>_history.json (traces)
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

## Shared code (at this level)

- `osc_common.py` — architectures (complex `sinh`, Fourier, SIREN, MscaleDNN,
  real `sinh`/`tanh`), the single complex-Waring Taylor-jet derivative backend,
  and the `train_eval` loop (equal wall-clock budget, floored cosine schedule,
  optional `--history` convergence logging).
- `plot_convergence.py` — builds the per-family paper figures
  (`docs/paper/figures/fig_<key>.pdf`) from each family's `data/`.
- `aggregate_osc.py` — cross-family roll-up and advantage-factor check.

## Reproducing the paper

```bash
# accuracy tables (all families, ~hours)
bash scripts/run_oscillatory_v3.sh
bash scripts/run_oscillatory_v4.sh
# convergence traces for the figures
bash scripts/run_history.sh
# regenerate figures
python experiments/plot_convergence.py
```

Superseded / exploratory experiments are documented in
`archived/progress.md`.
