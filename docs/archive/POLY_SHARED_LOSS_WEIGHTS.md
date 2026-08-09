# Shared loss weights for the polyharmonic comparison

## Objective

For order `2m` and dimension `d`, both Vanilla PINN and Complex Sinh use exactly the same
dimensionless objective

```text
L = L_PDE + sum_{j=0}^{m-1} lambda_j L_bc,j,

L_PDE  = MSE((Delta^m u - f) / S^m),
L_bc,j = MSE((Delta^j u - Delta^j u_exact) / S^j),
S       = d pi^2.
```

Consequently, o2 has one boundary weight, o4 has two, and o6 has three.  This
replaces the old implementation, which summed all boundary terms and then
multiplied the sum by one coefficient.  Under the old objective, all terms had
the same implicit coefficient and the total boundary pressure increased with
the number of boundary conditions.

Weights are selected on training seed 0 with evaluation seed 54321.  Every
candidate vector is shared by both methods.  The symmetric selection score is
the geometric mean of the two validation relative-L2 errors.  Once frozen,
the final comparison uses evaluation seed 12345 and does not retune.

The weight search is dimension-specific: the normalization changes with `d`,
so a vector selected for d=2 must not be reused for d=3 without a new search.

## d=3 power-of-ten grid and formal runs

The d=3 audit uses the same predeclared grid

```text
G = {1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3}.
```

It contains 7, 49, and 343 Cartesian points for o2, o4, and o6,
respectively. Each point used 30 seconds per method, training seed 0, and
evaluation seed 54321. The selected vectors and screen scores are:

| problem | weights | Vanilla relative L2 | Sinh relative L2 | geometric mean |
|---|---|---:|---:|---:|
| d3 o2 | `[1e-1]` | 1.024e-1 | 5.030e-2 | **7.178e-2** |
| d3 o4 | `[1e-1, 1e0]` | 7.452e-1 | 1.436e-1 | **3.272e-1** |
| d3 o6 | `[1e-1, 1e-1, 1e0]` | 9.991e-1 | 4.035e-1 | **6.349e-1** |

Using those frozen vectors, the 1200-second single-seed formal runs gave:

| problem | Vanilla relative L2 | Sinh relative L2 | Vanilla loss | Sinh loss |
|---|---:|---:|---:|---:|
| d3 o2 | 2.108e-3 | 2.182e-3 | 2.695e-7 | 1.858e-7 |
| d3 o4 | 2.380e-2 | **1.183e-2** | 6.001e-5 | 8.985e-6 |
| d3 o6 | 4.968e-1 | **4.328e-2** | 2.677e-2 | 1.140e-4 |

The d3 o6 screen is especially low-fidelity: sixth-order direct AD fits only
about 14 Vanilla updates in 30 seconds, and 548 updates in 1200 seconds.
These results are therefore diagnostic single-seed evidence; multi-seed runs
and longer budgets are still required for paper-grade claims.

## d=2 power-of-ten grid audit for o4 and o6

The latest audit restricts every boundary coefficient to an integer power of
ten and searches the full range

```text
G = {1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3}.
```

Each Cartesian point uses training seed 0, evaluation seed 54321, and 30
wall-clock seconds per method.  All 49 o4 points and all 343 o6 points are
present, with two finite method rows per point.  The shared ranking metric is
the geometric mean of the Vanilla and Sinh validation relative-L2 errors.

The selected vectors are

```text
o4: [lambda_u, lambda_Delta_u] = [1e-1, 1e1]
o6: [lambda_u, lambda_Delta_u, lambda_Delta2_u] = [1e-2, 1e0, 1e1]
```

| problem | weights | Vanilla relative L2 | Sinh relative L2 | geometric mean |
|---|---|---:|---:|---:|
| o4 | `[1e-1, 1e1]` | 1.011e-1 | 2.053e-2 | **4.556e-2** |
| o6 | `[1e-2, 1e0, 1e1]` | 5.532e-1 | 1.443e-1 | **2.825e-1** |

At the o4 winner, the final raw losses `(PDE, u, Delta u)` are
`(1.812e-3, 1.488e-2, 2.821e-5)` for Vanilla and
`(7.311e-5, 1.305e-4, 9.567e-7)` for Sinh.  After weighting, the boundary
contributions are `(1.488e-3, 2.821e-4)` and
`(1.305e-5, 9.567e-6)`, respectively.

At the o6 winner, the final raw losses `(PDE, u, Delta u, Delta^2 u)` are
`(1.681e-1, 2.067e-1, 1.159e-2, 3.663e-4)` for Vanilla and
`(5.889e-4, 1.328e-2, 1.060e-3, 1.702e-5)` for Sinh.  The weighted boundary
contributions are `(2.067e-3, 1.159e-2, 3.663e-3)` and
`(1.328e-4, 1.060e-3, 1.702e-4)`, respectively.

Neither winning vector touches the `1e-3` or `1e3` search boundary, so this
range is adequate for the present 30-second screen.  In particular, the o6
winner lowers the short-budget Vanilla error from the previous near-zero-solution
region (about 1.0) to 0.553.  The screen still contains only about 35 direct-AD
Vanilla updates for o6; fresh longer-budget and multi-seed runs are required
before paper use.

## Earlier Cartesian-grid audit for o4 and o6

The earlier o4/o6 weights below were obtained by a staged profile-and-scale
search.  A later audit replaces that heuristic for future comparisons with the
predeclared per-component grid

```text
G = {0.01, 0.03, 0.1, 0.3, 1, 3}.
```

Every Cartesian product point uses training seed 0, evaluation seed 54321,
and 30 wall-clock seconds for each method.  The ranking key is the geometric
mean of the Vanilla and Sinh validation relative-L2 errors, followed by the
maximum error and weight sum as tie breakers.  The audit contains all 36 o4
points and all 216 o6 points, with no missing or non-finite results.

For o4, the grid winner is

```text
[lambda_u, lambda_Delta_u] = [0.3, 3.0].
```

| rank | weights | Vanilla relative L2 | Sinh relative L2 | geometric mean |
|---:|---|---:|---:|---:|
| 1 | `[0.3, 3.0]` | 1.385e-1 | 4.637e-3 | **2.534e-2** |
| 2 | `[0.03, 1.0]` | 1.118e-1 | 2.166e-2 | 4.921e-2 |
| 3 | `[0.03, 3.0]` | 1.533e-1 | 1.673e-2 | 5.065e-2 |

For o6, the 30-second winner is `[3.0, 0.01, 0.1]`.  Direct sixth-order AD
fits only about 35 Vanilla updates into 30 seconds, so the top five grid points
were re-run for 90 seconds under the same shared protocol.  The confirmation
ranking selects

```text
[lambda_u, lambda_Delta_u, lambda_Delta2_u] = [3.0, 0.01, 0.03].
```

| confirmation rank | weights | Vanilla relative L2 | Sinh relative L2 | geometric mean |
|---:|---|---:|---:|---:|
| 1 | `[3.0, 0.01, 0.03]` | 9.988e-1 | 7.915e-2 | **2.812e-1** |
| 2 | `[3.0, 0.03, 0.01]` | 9.988e-1 | 8.048e-2 | 2.835e-1 |
| 3 | `[3.0, 0.03, 0.03]` | 9.988e-1 | 8.529e-2 | 2.919e-1 |

The 90-second confirmation is a candidate-selection run, not a formal paper
comparison.  In particular, all five confirmed o6 candidates leave Vanilla
near relative error 1 under this short budget.  The selected vectors must be
used in a fresh 180-second final comparison before replacing the historical
formal metrics below.

## Earlier staged search and formal runs

### Poly d2 o4

The two weights correspond to `[u, Delta u]` after derivative normalization.
The ratio sweep fixes their sum at 0.6 and tests increasingly strong emphasis
on the Laplacian boundary condition.  Boundary extensions establish an
interior optimum, and a subsequent scale sweep tests total weights 0.2, 0.6,
and 1.8.

The frozen vector is

```text
[lambda_u, lambda_Delta_u] = [0.02, 0.58].
```

With a 180-second wall-clock budget per method, training seed 0, and final
evaluation seed 12345:

| method | relative L2 | `L_PDE` | `L_u` | `L_Delta_u` | steps |
|---|---:|---:|---:|---:|---:|
| Vanilla tanh, direct AD | 5.150e-2 | 4.970e-6 | 3.184e-3 | 4.747e-6 | 1575 |
| Complex Sinh | **5.208e-3** | 1.741e-6 | 2.941e-5 | 1.379e-6 | 744 |

Complex Sinh is 9.89 times more accurate in this single-seed controlled run.
Both methods use the same loss vector and improve through the end of training.
Multi-seed confirmation remains necessary before using the row as final paper
evidence.

### Poly d2 o2

The scalar sweep tests `lambda_u` in
`{0.01, 0.03, 0.1, 0.3, 1, 3}`.  The shared geometric-mean validation score is
best at

```text
lambda_u = 0.3.
```

The 180-second final results are:

| method | relative L2 | `L_PDE` | `L_u` | steps |
|---|---:|---:|---:|---:|
| Vanilla tanh, direct AD | 3.009e-3 | 3.292e-6 | 4.551e-6 | 9854 |
| Complex Sinh | **1.901e-4** | 1.810e-7 | 6.617e-8 | 2986 |

Complex Sinh is 15.83 times more accurate by the final-model rule.  The
Vanilla history briefly reaches `7.755e-4` near 179 seconds and then spikes to
the reported final value; no best-checkpoint value is substituted into the
table.

### Poly d2 o6

The three weights correspond to `[u, Delta u, Delta^2 u]`.  Because sixth-order
direct AD is expensive (about 0.86 seconds per step), the search is staged:

1. fix the total boundary weight at 0.9 and sweep five monotone profiles;
2. re-run the two best profiles for 90 seconds;
3. keep the winning profile and check total scales 0.3, 0.9, and 2.7.

The frozen vector is

```text
[lambda_u, lambda_Delta_u, lambda_Delta2_u] = [0.02, 0.10, 0.78].
```

The 180-second final results are:

| method | relative L2 | `L_PDE` | `L_u` | `L_Delta_u` | `L_Delta2_u` | steps |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla tanh, direct AD | 4.670e-1 | 5.174e-4 | 6.309e-2 | 2.187e-2 | 3.648e-4 | 210 |
| Complex Sinh | **1.187e-1** | 2.173e-5 | 3.115e-4 | 4.481e-4 | 9.112e-6 | 237 |

Complex Sinh is 3.93 times more accurate.  Vanilla improves from `0.703` at
90 seconds to `0.467` at 180 seconds, so the result indicates slow convergence
rather than a stationary zero solution.  Its remaining error is concentrated
in the lower-order boundary conditions; only 210 sixth-order direct-AD steps
fit in the shared wall-clock budget.

## Current shared weight table

| atomic problem | shared boundary weights |
|---|---|
| `poly_d2_o2` | `[0.3]` |
| `poly_d2_o4` | `[1e-1, 1e1]` |
| `poly_d2_o6` | `[1e-2, 1e0, 1e1]` |
| `poly_d3_o2` | `[1e-1]` |
| `poly_d3_o4` | `[1e-1, 1e0]` |
| `poly_d3_o6` | `[1e-1, 1e-1, 1e0]` |

These weights are shared across the two methods within each atomic problem.
The o4/o6 180-second tables
above still describe the earlier staged-search vectors and must not be quoted
as results for the new power-grid vectors.  Fresh formal runs and
multi-seed confirmation remain necessary before final paper use.
