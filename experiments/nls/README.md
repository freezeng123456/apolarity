# Cubic nonlinear Schrödinger (complex-valued)

**Problem.** \(i u_t + \tfrac12 u_{xx} + |u|^2 u = f\), \(u:\mathbb{R}^2\to\mathbb{C}\),
manufactured bright soliton \(u=\operatorname{sech}(x)\,e^{ikt}\) (\(f=(\tfrac12-k)u\),
\(f=0\) at \(k=\tfrac12\)). Physical domain \(x\in[-5,5]\), \(t\in[0,\pi/2]\); networks
take normalized inputs. Real baselines carry the field as a split-real (Re/Im)
pair (RVPINN).

**Source.** Raissi–Perdikaris–Karniadakis 2019.

**Sweep.** temporal frequency \(k\in\{1,2,4\}\). Init \(\omega_0=\max(10,2kL_T)\),
\(\sigma=\max(2,kL_T)\), \(L_T=\pi/4\).

## Outputs (`data/`)
Width study (600 s, 2 seeds): `nls_h{128,64}.{csv,json}` + `*_history.json`.
Real baselines here are `siren,fourier,tanh` (split-real).

## Reproduce
```bash
bash run.sh
```
Figure: `docs/paper/figures/fig_nls.pdf` (via `experiments/tools/plot_width.py`).
