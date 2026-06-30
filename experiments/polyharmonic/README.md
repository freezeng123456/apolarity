# Polyharmonic order sweep (1D + 2D)

**Problem.** Controlled order axis at fixed frequency.
- 2D: \(\Delta^m u=(-2\pi^2)^m u\) on \((-1,1)^2\), \(u=\sin\pi x\sin\pi y\).
- 1D: \(d^{2m}u/dx^{2m}=(-\pi^2)^m u\) on \((-1,1)\), \(u=\sin\pi x\).
Navier (simply-supported) BCs: \(\Delta^j u=0\), \(j=0..m-1\). Only the operator
**order** changes across the sweep — no frequency confound.

**Source.** Vahab 2022 (high-order generalization of the biharmonic benchmark).

**Sweep.** order \(2m\): 1D \(\{2,4,6,8,10\}\), 2D \(\{2,4,6\}\). Init 1D
\(\omega_0=\pi\); 2D \(\omega_0=10\); \(\sigma=\pi\). (An order-\(m\) operator
amplifies init frequency like \(\omega^m\), so \(\omega_0\) must sit at the target.)

## Outputs (`data/`)
Width study (600 s, 2 seeds): `poly1d_h{128,64}.*` and `poly2d_h{128,64}.*`
(`.csv` + `.json` + `_history.json`).

## Reproduce
```bash
bash run.sh
```
Figures: `docs/paper/figures/fig_poly1d.pdf`, `fig_poly2d.pdf` (via
`experiments/tools/plot_width.py`).
