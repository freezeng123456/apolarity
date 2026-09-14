# WDD implementation: single partials and real-direction PDE training

## The two constructions

`torch_pinn.monomials.monomial` evaluates an individual partial derivative with
`waring_batched` or the `nested_jvp` reference. Its schedule is always the same
roots-of-unity Waring construction. Nonreal roots require complex arithmetic;
when all roots are real, the same construction uses real arithmetic. There is
no selector for polarization, no `auto` method, and no serial benchmark option.

`torch_pinn.operators.evaluate` and `torch_pinn.problem_ch2d.components` implement
the PDE construction `shared_jet_linear`. The real direction sets are prescribed
by the paper's power-sum identities: five for 1D KdV-gPINN, six for 2D KdV, and
three spatial directions for Cahn–Hilliard. These functions do not call the
monomial schedule generator. The only alternative training method is
`nested_jvp`, the nested coordinate-JVP baseline with lower-primal reuse.

Both use `torch_pinn.jets.mlp_jet`, whose coefficients satisfy
`q[k] = D_v^k u / k!`. Reconstruction weights include the required factorials.
The Taylor-mode kernel supports the real and complex inputs required by these
two WDD constructions; the PDE API requires real parameters and inputs. Parameter
backpropagation differentiates ordinary Torch operations. Boundary derivatives
and the Cahn–Hilliard time derivative use a common coordinate-JVP path.
Independent references are validation utilities, not additional experiment methods.

## Current PDE training entry

Use `train_fixed_wall.py` for all three problems. Its defaults are:

- Four hidden layers of width 128 with tanh activations; `--depth 5` counts
  affine layers including the scalar output layer.
- Adam with constant learning rate 1e-5 for `kdv1d`/`kdv2d`, or 1e-4 for `ch2d`.
- A 1200-second training budget and 400 uniformly sampled interior points per update.
- Fresh uniform initial/boundary points at every update: 128 per face in 1D,
  256 per face in 2D. Conditions on the same face use the same points.
- Independent interior and constraint RNG streams, matched between paired methods.
- Fixed independent space–time and terminal evaluation sets, with 10,000 points each.
- Offline diagnostics, full checkpoints and model-only snapshots following the
  frozen paper protocol. A parameter-preserving warm-up precedes timing.

```bash
python train_fixed_wall.py --case kdv1d --method shared_jet_linear --out /absolute/new/kdv1d
python train_fixed_wall.py --case kdv2d --method nested_jvp --out /absolute/new/kdv2d
python train_fixed_wall.py --case ch2d --method shared_jet_linear --out /absolute/new/ch2d
```

`--constraint-sampling` accepts only `resampled`. `--max-steps` is a cap for small
correctness checks, not a fixed-update paper experiment. A full matrix uses three
problems, two methods, and five seeds:

```bash
python run_fixed_wall_matrix.py --out /absolute/new/matrix --dry-run
python run_fixed_wall_matrix.py --out /absolute/new/cpu-check --device cpu --seeds 71 --smoke
```

The dry run prints commands without launching jobs. The small CPU matrix performs
three updates per method/problem and checks paired sampling and evaluation points.

## Single-partial timing

`first_group.py` accepts only `nested_jvp` and `waring_batched`. The full
`run_first_group_matrix.py` now contains 20 cells corresponding to the retained
paper comparisons; historical serial records remain in the original result snapshot.
All derivative-value timings exclude parameter backpropagation. Original input
sampling, network initialization, direction counts and arithmetic types are retained.

## Validation and result provenance

```bash
python -m pytest -q tests
python ../../docs/paper/verify_experiment_identities.py
```

The tests compare loss values, parameter gradients and Adam updates with the exact
frozen source used for the paper. They also verify fresh paired constraint batches,
evaluation isolation, real-only PDE inputs, and rejection of retired method options.
The identity script checks both current and frozen KdV reconstruction weights.
The frozen training source is under
`results/paper_resampled_20260914/source/experiments/torch_benchmarks/` at the
repository root. The repository retains compact per-run summaries and measured
error curves for the four paper examples; large model and per-update artifacts
are omitted.
