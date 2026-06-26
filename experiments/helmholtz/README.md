# High-wavenumber Helmholtz

**Problem.** \(-\Delta u - a^2\pi^2 u = f\) on \([0,1]^2\) with manufactured
solution \(u=\sin(a\pi x)\sin(a\pi y)\); the wavenumber \(a\) is swept. The
anisotropic variant uses distinct \((a_1,a_2)\) per axis.

**Source.** Spectral-bias / high-frequency PINN benchmark (Wang et al. 2021,
"understanding and mitigating gradient pathologies"; PINNacle, Hao et al. 2024).

**Why it matters.** The canonical *frequency axis* (2nd order). Tests whether the
complex `sinh` advantage tracks wavenumber against the real spectral-bias remedies
(Fourier features, SIREN, MscaleDNN).

## Outputs (`data/`)
- `helmholtz_v3.csv` -- accuracy vs wavenumber; `helmholtz_aniso_v3.csv` -- anisotropic.
- `helmholtz_history.json` -- traces at \(a=6\).

## Reproduce
```bash
bash run.sh
```
Figure: `fig_helmholtz.pdf`.
