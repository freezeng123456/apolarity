# PDE-loss weight ablation: Sinh-gradient alignment

Status: diagnostic only. These one-seed results are not formal paper evidence.

## Question

The strict Vanilla PINN and Complex Sinh runs used the same nominal objective

```text
L = L_int + 100 L_bc,
```

but their initial PDE derivatives have very different parameter-gradient
scales. This ablation changes only the strict Vanilla boundary coefficient so
that its initial

```text
||grad(w_bc L_bc)|| / ||grad(L_int)||
```

matches Complex Sinh under the original coefficient `w_bc=100`. Network,
initialization, collocation points, optimizer, direct-AD backend, wall-clock
budget, and evaluation set are unchanged.

For each problem,

```text
w_aligned = 100 * ratio_sinh / ratio_vanilla_at_100.
```

## Measured initial ratios and aligned coefficients

| problem | Sinh ratio at 100 | Vanilla ratio at 100 | aligned Vanilla `w_bc` |
|---|---:|---:|---:|
| `poly_d2_o4` | 44.9320 | 3173.1529 | 1.416006 |
| `chirp_a2` | 1518.2030 | 44602.0509 | 3.403886 |
| `maxwell_a4` | 1519.7271 | 323.8452 | 469.275824 |

## Seed-0, 180-second results

| problem | default `w_bc=100` L2 | Sinh-aligned L2 | final `L_int` | final `L_bc` |
|---|---:|---:|---:|---:|
| `poly_d2_o4` | 0.999251 | **0.739341** | 2.460e-3 | 1.339e-3 |
| `chirp_a2` | 1.779957 | **0.006976** | 2.529e-5 | 1.072e-5 |
| `maxwell_a4` | **2.216312** | 5.377546 | 6.353 | 0.3029 |

## Interpretation

The Chirp result shows that its original strict-Vanilla failure was primarily
an objective-balancing failure, not an inability of the tanh network to learn
the solution. Poly also leaves the zero-solution basin, although matching
Sinh's initial ratio is not sufficient for high accuracy within 180 seconds.

Maxwell demonstrates that a gradient ratio inherited from one architecture is
not a generally valid weight for another architecture. Complex Sinh and the
split-real tanh pair have different representations and gradient evolution;
raising the tanh boundary coefficient to 469 makes its PDE residual and L2
error substantially worse. The next PDE-loss ablation should therefore balance
each architecture using its own gradients (and preferably update the balance
during training) instead of transferring the Sinh ratio.

Raw outputs and convergence traces are in
`experiments/results/pde_loss_ablation_sinh_aligned/`.
