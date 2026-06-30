# High-wavenumber Helmholtz (+ anisotropic)

**Problem.** \(\Delta u + \kappa^2 u = f\) on \((-1,1)^2\), \(\kappa=a\pi\), manufactured
\(u=\sin(a\pi x)\sin(a\pi y)\), homogeneous Dirichlet. Source \(f=-(a\pi)^2 u\).
The *anisotropic* case fixes \((a_1,a_2)=(1,4)\), \(u=\sin(\pi x)\sin(4\pi y)\) — the
classic gradient-pathology construction.

**Source.** Wang–Teng–Perdikaris 2021; PINNacle (Hao 2024).

**Sweep.** wavenumber \(a\in\{2,4,6,8,10\}\). Init \(\omega_0=\max(10,2\pi a)\),
\(\sigma=\max(2,\pi a)\).

## Outputs (`data/`)
Width study (600 s, 2 seeds): `helmholtz_h{128,64}.{csv,json}` +
`*_history.json`, and `helmholtz_aniso_h{128,64}.*`. Real baselines
(Fourier/SIREN/MscaleDNN) at width 128; complex `sinh` at 128 and 64.

## Reproduce
```bash
bash run.sh
```
Figure: `docs/paper/figures/fig_helmholtz.pdf` (via `experiments/tools/plot_width.py`).
