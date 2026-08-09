# Variable-coefficient (scattering) Helmholtz

**Problem.** \(\Delta u + \kappa^2(x)u = f\) on \((-1,1)^2\) with a spatially varying
coefficient \(\kappa^2(x)=(a\pi)^2(1+0.5\sin\pi x\sin\pi y)\) (a \(\pm50\%\) lens),
manufactured \(u=\sin(a\pi x)\sin(a\pi y)\), Dirichlet \(0\). Probes robustness to
medium heterogeneity rather than a single clean eigenmode.

**Source.** PINNacle (Hao 2024) heterogeneous-medium family.

**Sweep.** background wavenumber \(a\in\{2,4,6\}\). Init \(\omega_0=\max(10,2\pi a)\),
\(\sigma=\max(2,\pi a)\).

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
