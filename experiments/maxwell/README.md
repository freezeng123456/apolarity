# Time-harmonic Maxwell, lossy medium (complex-valued)

**Problem.** TM-mode reduction to a complex Helmholtz equation
\(\Delta E + \kappa^2 E = f\) on \((-1,1)^2\), \(\kappa^2=(a\pi)^2(1+i\beta)\),
\(\beta=0.2\) (loss tangent → complex permittivity → genuinely complex \(E\)).
Manufactured plane wave \(E=e^{i a\pi(x+y)}\), Dirichlet \(=E_\star\). Real baselines
use a split-real (Re/Im) pair (RVPINN).

**Source.** Jiang 2024 (lossy TM variant).

**Sweep.** wavenumber \(a\in\{2,4,6\}\). Init \(\omega_0=\max(10,2\pi a)\),
\(\sigma=\max(2,\pi a)\).

## Outputs (`data/`)
Width study (600 s, 2 seeds): `maxwell_h{128,64}.{csv,json}` + `*_history.json`.
Real baselines here are `siren,fourier,tanh` (split-real).

## Reproduce
```bash
bash run.sh
```
Figure: `docs/paper/figures/fig_maxwell.pdf` (via `experiments/tools/plot_width.py`).
