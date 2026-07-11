# Cubic nonlinear Schrödinger (complex-valued)

**Problem.** \(i u_t + \tfrac12 u_{xx} + |u|^2 u = f\), \(u:\mathbb{R}^2\to\mathbb{C}\),
manufactured bright soliton \(u=\operatorname{sech}(x)\,e^{ikt}\) (\(f=(\tfrac12-k)u\),
\(f=0\) at \(k=\tfrac12\)). Physical domain \(x\in[-5,5]\), \(t\in[0,\pi/2]\); networks
take normalized inputs. Real baselines carry the field as a split-real (Re/Im)
pair (RVPINN).

**Source.** Raissi–Perdikaris–Karniadakis 2019.

**Sweep.** temporal frequency \(k\in\{1,2,4\}\). Init \(\omega_0=\max(10,2kL_T)\),
\(\sigma=\max(2,kL_T)\), \(L_T=\pi/4\).

## Experiment status

This family is outside the formal `jsc_v2` grid. `data/` is empty, there are no
formal results, and its paper figure and table are **TBD**.

The only formal methods in this repository are `complex_sinh`, SIREN,
mFF-PINN, and MscaleDNN-2-sin, compared only through the `jsc_v2` atomic
runner. The family-local launcher below is retained for implementation
diagnosis only:

```bash
bash run.sh
```

Outputs from this `run.sh`, historical width studies, or archived runners
cannot be used as paper evidence.
