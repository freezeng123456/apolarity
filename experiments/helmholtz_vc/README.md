# Variable-coefficient (scattering) Helmholtz

**Problem.** \(\Delta u + \kappa^2(x)u = f\) on \((-1,1)^2\) with a spatially varying
coefficient \(\kappa^2(x)=(a\pi)^2(1+0.5\sin\pi x\sin\pi y)\) (a \(\pm50\%\) lens),
manufactured \(u=\sin(a\pi x)\sin(a\pi y)\), Dirichlet \(0\). Probes robustness to
medium heterogeneity rather than a single clean eigenmode.

**Source.** PINNacle (Hao 2024) heterogeneous-medium family.

**Sweep.** background wavenumber \(a\in\{2,4,6\}\). Init \(\omega_0=\max(10,2\pi a)\),
\(\sigma=\max(2,\pi a)\).

## Outputs (`data/`)
Width study (600 s, 2 seeds): `helmvc_h{128,64}.{csv,json}` + `*_history.json`.

## Reproduce
```bash
bash run.sh
```
Figure: `docs/paper/figures/fig_helmvc.pdf` (via `experiments/tools/plot_width.py`).
