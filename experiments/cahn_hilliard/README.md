# Cahn–Hilliard (4th & 6th order, nonlinear)

**Problem.** On \((x,t)\in(-1,1)^2\), \(\Delta=\partial_x^2\):
- 4th: \(u_t = M[\Delta(u^3)-\Delta u-\gamma\Delta^2 u]\)
- 6th: \(\;+\,\kappa\Delta^3 u\)

with \(M=\gamma=\kappa=1\). The nonlinear flux \(\Delta(u^3)=3u^2u_{xx}+6u(u_x)^2\) is
built from single-monomial partials, so the whole residual runs through the fast
Taylor-jet. Manufactured \(u=\sin(a\pi x)\cos(a\pi t)\).

**Source.** Raissi 2019 / PINNacle (Hao 2024).

**Sweep.** amplitude/frequency \(a\in\{2,3\}\), order \(\in\{4,6\}\). Init
\(\omega_0=\max(10,2\pi a)\), \(\sigma=\max(2,\pi a)\).

## Outputs (`data/`)
Width study (600 s, 2 seeds): `cahn_hilliard_h{128,64}.*` (problems `ch4_a*`,
`ch6_a*`; `.csv` + `.json` + `_history.json`).

## Reproduce
```bash
bash run.sh
```
Figure: `docs/paper/figures/fig_cahn.pdf` (6th order; via
`experiments/tools/plot_width.py`).
