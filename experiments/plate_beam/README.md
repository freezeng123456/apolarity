# Kirchhoff plate / Euler-Bernoulli beam (4th order)

**Problem.** Biharmonic structural models:
- beam \(u'''' = \lambda u\) (Euler-Bernoulli), \(u=\sin(m\pi x)\);
- plate \(\Delta^2 u = \lambda u\) (Kirchhoff), \(u=\sin(m\pi x)\sin(m\pi y)\);
- mixed-mode plate with anisotropic \((m,m+1)\) modes at fixed order 4.
Sweeps the mode number \(m\).

**Source.** Biharmonic plate/beam PINN benchmarks (Vahab et al. 2022, biharmonic
PINNs; classical structural eigenproblems).

**Why it matters.** A real, physically grounded 4th-order family: tests the
high-order advantage at fixed order while increasing spatial frequency.

## Outputs (`data/`)
- `plate_beam_v3.csv` (iso plate+beam), `platemix_v3.csv` (anisotropic).
- `plate_beam_history.json`, `platemix_history.json` -- traces.

## Reproduce
```bash
bash run.sh
```
Figures: `fig_plate.pdf`, `fig_beam.pdf`, `fig_platemix.pdf`.
