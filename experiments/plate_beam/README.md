# Kirchhoff plate / Euler–Bernoulli beam / mixed-mode plate (4th order)

**Problem.** 4th-order real oscillatory eigenmodes.
- Plate (2D biharmonic): \(\Delta^2 w=S^2 w\), \(w=\sin(m\pi x)\sin(n\pi y)\),
  \(S=(m^2+n^2)\pi^2\); simply-supported \(w=0,\ \Delta w=0\).
- Beam (1D): \(w''''=(m\pi)^4 w\), \(w=\sin(m\pi x)\); \(w=0,\ w''=0\).
- Mixed plate: anisotropic \((m,m+1)\) modes (non-separable frequency).

**Source.** Vahab 2022 (plate/beam vibration).

**Sweep.** mode \(m\): plate/beam \(\{1,2,3\}\); mixed \(\{2,3,4\}\). Order fixed at
4; only oscillation rises. Init \(\omega_0=\max(10,2\pi f)\), \(\sigma=\max(2,\pi f)\),
\(f=\max(m,n)\).

## Outputs (`data/`)
Width study (600 s, 2 seeds): `plate_beam_h{128,64}.*` (problems `plate_m*`,
`beam_m*`) and `platemix_h{128,64}.*` (`.csv` + `.json` + `_history.json`).

## Reproduce
```bash
bash run.sh
```
Figures: `docs/paper/figures/fig_plate.pdf`, `fig_beam.pdf`, `fig_platemix.pdf`
(via `experiments/tools/plot_width.py`).
