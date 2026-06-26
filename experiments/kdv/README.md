# Linearized KdV / dispersive wave (3rd order)

**Problem.** \(u_t + u_{xxx} = 0\) on a periodic strip with a travelling
sinusoidal solution \(u=\sin(k(x-ct))\), \(c=-k^2\); sweeps the wavenumber \(k\).

**Source.** Korteweg-de Vries dispersive benchmark (Raissi et al. 2019 use the
nonlinear KdV; here the linear dispersion term isolates the odd, 3rd-order operator).

**Why it matters.** A deliberate *counter-case*: an **odd**-order operator. It
checks whether the complex `sinh` advantage is specific to even, sign-balanced
operators (it is largest there) versus odd dispersive ones.

## Outputs (`data/`)
- `kdv_v3.csv` -- accuracy vs wavenumber.
- `kdv_history.json` -- traces at wavenumber 4.

## Reproduce
```bash
bash run.sh
```
Figure: `fig_kdv.pdf`.
