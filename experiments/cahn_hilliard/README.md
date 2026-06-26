# Cahn-Hilliard phase field (4th / 6th order, nonlinear)

**Problem.** Cahn-Hilliard-type evolution \(u_t = \Delta(\,\cdot\,)\) with a
manufactured oscillatory solution; the spatial order (4 or 6) and frequency \(a\)
are both swept. Nonlinear (cubic) reaction term included.

**Source.** Cahn-Hilliard phase-field PINN benchmark (high-order, nonlinear;
PINNacle, Hao et al. 2024).

**Why it matters.** Combines high order *and* nonlinearity, the most demanding real
family: shows the order advantage persists when the residual is nonlinear.

## Outputs (`data/`)
- `cahn_hilliard_v3.csv` -- accuracy over (order, frequency).
- `cahn_hilliard_history.json` -- traces at 6th order, \(a=2\).

## Reproduce
```bash
bash run.sh
```
Figure: `fig_cahn.pdf`.
