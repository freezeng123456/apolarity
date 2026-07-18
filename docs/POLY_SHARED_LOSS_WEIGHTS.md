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

The same preregistered procedure will next produce one shared scalar for o2
and one shared three-component vector for o6.
