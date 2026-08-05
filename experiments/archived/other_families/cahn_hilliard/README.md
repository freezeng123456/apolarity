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
