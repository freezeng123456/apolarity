# PDE-loss weight tuning and controlled comparison

## Scope and protocol

This experiment tunes only the scalar boundary coefficient in

```text
L = L_int + w_bc L_bc.
```

All architectures, initializations, collocation points, residual
normalizations, optimizers, learning-rate schedules, and wall-clock budgets
remain unchanged. The coarse search uses training seed 0, 30 seconds per
candidate, and a held-out evaluation set generated with seed 54321. After the
weights are frozen, the controlled comparison uses a separate evaluation set
generated with seed 12345 and 180 seconds per run.

This separation prevents selecting a weight directly on the reported test
set. The results still use one training seed, so they should be treated as a
controlled ablation rather than final multi-seed paper statistics.

## Reverse alignment check

The first check applies the previously selected strict-Vanilla weights to
Complex Sinh. It changes only `w_bc`.

| problem | transferred `w_bc` | Complex Sinh L2 | final `L_int` | final `L_bc` |
|---|---:|---:|---:|---:|
| `poly_d2_o4` | 1.416006 | 1.145e-3 | 6.263e-6 | 1.450e-6 |
| `chirp_a2` | 3.403886 | 7.510e-4 | 8.357e-6 | 8.313e-8 |
| `maxwell_a4` | 469.275824 | 8.011e-1 | 1.040e-1 | 1.367e-6 |

Lowering the coefficient dramatically improves Poly and Chirp relative to the
original `w_bc=100` Sinh runs. Transferring the very large Maxwell coefficient
does the opposite. A coefficient that is suitable for one architecture and
problem is therefore not portable to another.

## Frozen coefficients

The 30-second sweeps and boundary extensions produce the following frozen
values. No coefficient is changed after looking at the 180-second test runs.

| method | `poly_d2_o4` | `chirp_a2` | `maxwell_a4` |
|---|---:|---:|---:|
| strict Vanilla PINN | 0.3 | 0.1 | 0.03 |
| Complex Sinh | 1.0 | 0.1 | 0.03 |
| problem-specific baseline | MIM: 1.0 | WIRE: 0.001 | PWNN: 0.03 |

WIRE remains close to relative error one throughout the sweep. Its selected
coefficient is the predeclared lower stopping point, not evidence of a
well-balanced solution: reducing it further would effectively remove the
boundary condition while the model still fails to learn the target.

## Seed-0, 180-second controlled comparison

The table reports the final model for every run. Best-so-far values are not
mixed into the final column.

| problem | tuned Vanilla | tuned Complex Sinh | tuned problem-specific baseline |
|---|---:|---:|---:|
| `poly_d2_o4` | 7.776e-2 | **8.326e-4** | MIM: 1.085e-2 |
| `chirp_a2` | 8.660e-4 | **3.432e-4** | WIRE: 1.187 |
| `maxwell_a4` | 1.062 | 2.224e-3 | **PWNN: 1.527e-4** |

The main conclusions change substantially after tuning:

- Strict Vanilla can learn Chirp very accurately. Its earlier failure was
  primarily loss imbalance, not a fundamental expressivity failure.
- Complex Sinh is strongest on Poly and Chirp under the controlled budget.
- Strict Vanilla still remains near the zero-solution basin on Maxwell;
  changing only `w_bc` is insufficient.
- PWNN is the strongest Maxwell baseline by a wide margin, as expected from
  its problem-matched plane-wave representation.

The histories also show optimization noise. Complex Sinh Poly reaches
`6.190e-4` at 101.6 seconds and finishes at `8.326e-4`; MIM Poly reaches
`4.359e-3` near 179.8 seconds and finishes at `1.085e-2`; PWNN Maxwell reaches
`1.326e-4` and finishes at `1.527e-4`. Formal tables should consistently use
either final models or a predeclared checkpoint rule.

## Other weights that remain tunable

This run deliberately does not search the following degrees of freedom:

- separate weights for individual boundary conditions (for example `u` and
  Laplacian boundary terms in Poly);
- weights for real and imaginary residual components in complex problems;
- the fixed residual normalization scales used by each PDE;
- architecture regularization, including the Complex Sinh imaginary-parameter
  penalty;
- adaptive methods such as gradient-norm balancing, uncertainty weighting,
  augmented Lagrangian constraints, or causal/residual-based reweighting;
- optimizer, learning-rate, sampling, width, and frequency hyperparameters.

A defensible next step is a multi-seed confirmation of the frozen scalar
weights. More elaborate weighting should be evaluated as a separate ablation,
not folded into this comparison after seeing its test outcomes.

Raw tuning runs are in `experiments/results/pde_weight_tuning/`; the formal
outputs are in `experiments/results/pde_weight_tuned_full/`.
