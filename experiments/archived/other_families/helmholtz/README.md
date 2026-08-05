# High-wavenumber Helmholtz (+ anisotropic / Wang 2021)

**Problem.** \(\Delta u + \kappa^2 u = f\) on \((-1,1)^2\).

- **Isotropic sweep:** \(u=\sin(a\pi x)\sin(a\pi y)\), \(\kappa=a\pi\), \(a\in\{2,4,6,8,10\}\).
- **Wang (2021) Eq. (8):** \(u=\sin(a_1\pi x)\sin(a_2\pi y)\), \(\Delta u + k^2 u = q\)
  with \(k=1\). Canonical triple: \((a_1,a_2)\in\{(1,1),(1,2),(1,4)\}\).

**Source.** Wang–Teng–Perdikaris 2021 (gradient pathology); isotropic extension as in PINN literature.

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

Outputs from this `run.sh`, historical Wang/width-study scripts, or archived
runners cannot be used as paper evidence.
