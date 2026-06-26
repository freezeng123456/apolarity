# Polyharmonic eigenmode -- controlled order sweep

**Problem.** \((-\Delta)^{m/2} u = \lambda\, u\) on \([0,1]^d\) with a sinusoidal
eigenmode \(u=\prod_i \sin(k_i\pi x_i)\); the differential **order** \(m\) is swept
at fixed frequency. Isolates the effect of operator order on trainability.

**Source.** Standard polyharmonic/biharmonic manufactured solution (cf. biharmonic
PINN benchmarks, Vahab et al. 2022); here generalized to arbitrary even order.

**Why it matters.** This is the paper's headline *order axis*: it shows complex
`sinh`'s advantage growing monotonically with \(m\) (1D to order 10, 2D to order 6)
while every real baseline collapses.

## Outputs (`data/`)
- `poly1d_v3.csv`, `poly2d_v3.csv` -- accuracy vs order (the table).
- `poly1d_history.json`, `poly2d_history.json` -- rel-L2 / loss vs time traces.

## Reproduce
```bash
bash run.sh        # writes data/*.csv + data/*_history.json
```
Figure: `python ../plot_convergence.py` -> `docs/paper/figures/fig_poly{1,2}d.pdf`.
