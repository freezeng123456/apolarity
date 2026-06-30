# Non-separable radial chirp

**Problem.** \(-\Delta u + u = f\) on \((-1,1)^2\), manufactured radial chirp
\(u=\sin(\tfrac{a\pi}{2}(x^2+y^2))\) whose local frequency \(|\nabla\phi|=a\pi r\)
grows with radius — so \(u\) is **not** a single Fourier mode. Dirichlet \(=u_\star\).

**Source.** Chirp / space-varying-frequency expressivity test (Tancik 2020,
Liu 2020). Removes the "separable sine" confound present in the other families.

**Sweep.** chirp rate \(a\in\{2,4,6,8\}\). Init \(\omega_0=\max(10,2\pi a)\),
\(\sigma=\max(2,\pi a)\).

## Outputs (`data/`)
Width study (600 s, 2 seeds): `chirp_h{128,64}.{csv,json}` + `*_history.json`.

## Reproduce
```bash
bash run.sh
```
Figure: `docs/paper/figures/fig_chirp.pdf` (via `experiments/tools/plot_width.py`).
