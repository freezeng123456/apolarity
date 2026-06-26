# Cubic nonlinear Schrodinger (complex-valued)

**Problem.** \(i u_t + \tfrac12 u_{xx} + |u|^2 u = 0\) on \(x\in[-5,5]\),
\(t\in[0,\pi/2]\) with the Raissi et al. soliton/initial data; \(u\) is genuinely
**complex**. Sweeps the frequency/amplitude setting.

**Source.** Raissi, Perdikaris & Karniadakis (2019), the canonical PINN NLS
benchmark (domain and IC matched to their setup).

**Why it matters.** A genuinely complex-valued field where the complex `sinh`
network is native; real baselines must split into (Re, Im) channels. Tests the
method on its "home turf" against split-real reals (`tanh`, SIREN, Fourier).

## Outputs (`data/`)
- `nls_v3.csv` -- accuracy over the sweep.
- `nls_history.json` -- traces at sweep 2.

## Reproduce
```bash
bash run.sh
```
Figure: `fig_nls.pdf`.
