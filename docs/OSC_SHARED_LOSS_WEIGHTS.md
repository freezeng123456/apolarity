# Shared boundary weights for Chirp and Maxwell

The Chirp and Maxwell comparisons use the same normalized objective for
Vanilla PINN and Complex Sinh,

```text
L = L_PDE + lambda_bc L_bc.
```

The boundary coefficient was selected independently for each setting from
`{1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3}`. Every grid point used 30 seconds per
method, training seed 0, and evaluation seed 54321. The ranking score is the
geometric mean of the Vanilla and Sinh relative-L2 errors.

| setting | selected weight | Vanilla screen L2 | Sinh screen L2 | geometric mean |
|---|---:|---:|---:|---:|
| `chirp_a1` | 1 | 3.439e-3 | 1.340e-3 | **2.147e-3** |
| `chirp_a2` | 0.1 | 8.095e-3 | 1.492e-3 | **3.475e-3** |
| `chirp_a3` | 0.01 | 2.057e-2 | 7.656e-3 | **1.255e-2** |
| `maxwell_a2` | 0.1 | 1.046 | 4.118e-3 | **6.563e-2** |
| `maxwell_a4` | 0.1 | 1.003 | 1.121e-2 | **1.060e-1** |
| `maxwell_a6` | 0.01 | 1.002 | 6.234e-2 | **2.500e-1** |

Using the selected weights, each method then ran for 1200 seconds with final
evaluation seed 12345:

| setting | Vanilla L2 | Sinh L2 | Vanilla loss | Sinh loss |
|---|---:|---:|---:|---:|
| `chirp_a1` | 3.710e-4 | **1.432e-4** | 2.457e-7 | 4.958e-8 |
| `chirp_a2` | **1.731e-4** | 3.669e-4 | 1.021e-7 | 3.276e-8 |
| `chirp_a3` | 9.319e-3 | **6.512e-4** | 4.475e-7 | 3.972e-8 |
| `maxwell_a2` | 5.941e-3 | **2.698e-4** | 1.130e-5 | 5.381e-8 |
| `maxwell_a4` | 9.339e-3 | **1.988e-3** | 3.811e-5 | 1.436e-6 |
| `maxwell_a6` | 1.626e-1 | **1.055e-2** | 6.284e-4 | 2.079e-7 |

The formal runs are single-seed diagnostic comparisons, matching the Poly
weight-search protocol. The Maxwell Vanilla results should be interpreted as
optimization evidence: despite the 20-minute budget, `a=4` and `a=6` remain
near relative error 1, while the native-complex Sinh model converges. The
weight grid is not a substitute for multi-seed validation.
