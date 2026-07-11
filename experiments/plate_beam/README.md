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

## Experiment status

This family is outside the formal `jsc_v2` grid. `data/` is empty, there are no
formal results, and its paper figures and tables are **TBD**.

The only formal methods in this repository are `complex_sinh`, SIREN,
mFF-PINN, and MscaleDNN-2-sin, compared only through the `jsc_v2` atomic
runner. The family-local launcher below is retained for implementation
diagnosis only:

```bash
bash run.sh
```

Outputs from this `run.sh`, historical width studies, or archived runners
cannot be used as paper evidence.
