# Linearized KdV / dispersive wave (3rd order)

**Problem.** \(u_t + \delta\,u_{xxx}=f\) on \((x,t)\in(-1,1)^2\), \(\delta=1\),
manufactured \(u=\sin(k\pi x)\cos(k\pi t)\), Dirichlet \(=u_\star\). The odd 3rd-order
dispersion term is where the Taylor-jet backend and complex `sinh` are exercised.

**Source.** Raissi 2019 (KdV); here the linearized dispersive term isolates the
3rd-order operator.

**Sweep.** wavenumber \(k\in\{2,3,4,5,6\}\). Init \(\omega_0=\max(10,2\pi k)\),
\(\sigma=\max(2,\pi k)\).

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
