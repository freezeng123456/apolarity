# Variable-coefficient Helmholtz (scattering)

**Problem.** \(-\Delta u - \kappa(x)^2 u = f\) on \([0,1]^2\) with a spatially
varying wavenumber \(\kappa(x)\) (heterogeneous medium), manufactured solution.
Sweeps the mean wavenumber.

**Source.** Heterogeneous-medium / scattering Helmholtz, a standard harder variant
of the constant-coefficient benchmark (PINNacle, Hao et al. 2024).

**Why it matters.** Confirms the frequency-axis advantage survives spatially
varying coefficients (no global Fourier basis matches the local wavelength).

## Outputs (`data/`)
- `helmvc_v3.csv` -- accuracy vs mean wavenumber.
- `helmvc_history.json` -- traces at sweep 6.

## Reproduce
```bash
bash run.sh
```
Figure: `fig_helmvc.pdf`.
