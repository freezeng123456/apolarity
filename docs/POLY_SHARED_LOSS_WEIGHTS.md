# Shared loss weights for the 2D polyharmonic comparison

## Objective

For order `2m`, both Vanilla PINN and Complex Sinh use exactly the same
dimensionless objective

```text
L = L_PDE + sum_{j=0}^{m-1} lambda_j L_bc,j,

L_PDE  = MSE((Delta^m u - f) / S^m),
L_bc,j = MSE((Delta^j u - Delta^j u_exact) / S^j),
S       = 2 pi^2.
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

## Poly d2 o4

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

## Poly d2 o2

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

## Poly d2 o6

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

## Frozen 2D Poly weight table

| atomic problem | shared boundary weights |
|---|---|
| `poly_d2_o2` | `[0.3]` |
| `poly_d2_o4` | `[0.02, 0.58]` |
| `poly_d2_o6` | `[0.02, 0.10, 0.78]` |

These weights are shared across the two methods.  The comparison remains a
single-training-seed study and requires multi-seed confirmation before final
paper use.
