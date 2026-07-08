# High-wavenumber Helmholtz (+ anisotropic / Wang 2021)

**Problem.** \(\Delta u + \kappa^2 u = f\) on \((-1,1)^2\).

- **Isotropic sweep:** \(u=\sin(a\pi x)\sin(a\pi y)\), \(\kappa=a\pi\), \(a\in\{2,4,6,8,10\}\).
- **Wang (2021) Eq. (8):** \(u=\sin(a_1\pi x)\sin(a_2\pi y)\), \(\Delta u + k^2 u = q\)
  with \(k=1\). Canonical triple: \((a_1,a_2)\in\{(1,1),(1,2),(1,4)\}\).

**Source.** Wang–Teng–Perdikaris 2021 (gradient pathology); isotropic extension as in PINN literature.

## Outputs (`data/`)

| stem | protocol | content |
|------|----------|---------|
| `helmholtz_h{128,64}` | 600s, 2 seeds | isotropic \(a=2..10\) |
| `helmholtz_aniso_h{128,64}` | 600s, 2 seeds | legacy single \((1,4)\), name `helm_aniso_1_4` |
| `helmholtz_wang2021_h{128,64}` | **1200s, 5 seeds** | Wang triple \((1,1),(1,2),(1,4)\) |

## Reproduce

```bash
bash run.sh                                    # archived 600s suite
bash ../../scripts/run_helmholtz_wang2021.sh   # Wang (1,1),(1,2),(1,4) @ 1200s
```
Figure: `docs/paper/figures/fig_helmholtz.pdf` (via `experiments/tools/plot_width.py`).
