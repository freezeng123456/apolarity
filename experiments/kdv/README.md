# Linearized KdV / dispersive wave (3rd order)

**Problem.** \(u_t + \delta\,u_{xxx}=f\) on \((x,t)\in(-1,1)^2\), \(\delta=1\),
manufactured \(u=\sin(k\pi x)\cos(k\pi t)\), Dirichlet \(=u_\star\). The odd 3rd-order
dispersion term is where the Taylor-jet backend and complex `sinh` are exercised.

**Source.** Raissi 2019 (KdV); here the linearized dispersive term isolates the
3rd-order operator.

**Sweep.** wavenumber \(k\in\{2,3,4,5,6\}\). Init \(\omega_0=\max(10,2\pi k)\),
\(\sigma=\max(2,\pi k)\).

## Outputs (`data/`)
Width study (600 s, 2 seeds): `kdv_h{128,64}.{csv,json}` + `*_history.json`.

## Reproduce
```bash
bash run.sh
```
Figure: `docs/paper/figures/fig_kdv.pdf` (via `experiments/tools/plot_width.py`).
