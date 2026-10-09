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

Use `train_equal_updates.py` for all three problems. The paper protocol uses:

- Four hidden layers of width 128 with tanh activations; `--depth 5` counts
  affine layers including the scalar output layer.
- Adam with constant learning rate 1e-5 for `kdv1d`/`kdv2d`, or 1e-4 for `ch2d`.
- Exactly 25,000, 20,000, and 1,500 updates for `kdv1d`, `kdv2d`, and `ch2d`,
  respectively, with 400 uniformly sampled interior points per update.
- Fresh uniform initial/boundary points at every update: 128 per face in 1D,
  256 per face in 2D. Conditions on the same face use the same points.
- Independent interior and constraint RNG streams, matched between paired methods.
- Fixed independent space–time and terminal evaluation sets, with 10,000 points each.
- Online numeric error curves and endpoint summaries, with no model or optimizer
  checkpoints. A parameter-preserving warm-up precedes timing.

```bash
python train_equal_updates.py --case kdv1d --method shared_jet_linear --updates 25000 --out /absolute/new/kdv1d
python train_equal_updates.py --case kdv2d --method nested_jvp --updates 20000 --out /absolute/new/kdv2d
python train_equal_updates.py --case ch2d --method shared_jet_linear --updates 1500 --eval-every 10 --out /absolute/new/ch2d
```

The paired methods share initialization and resampled training points at each
update. A full PDE matrix uses three problems, two methods, and five seeds.
The campaign also lists the historical single-partial memory supplement:

```bash
python equal_updates_campaign.py plan --out /absolute/new/matrix
python train_equal_updates.py --case kdv1d --method shared_jet_linear --device cpu \
  --width 4 --depth 2 --batch 3 --constraint-side 2 --updates 3 --eval-every 1 \
  --eval-points 16 --out /absolute/new/cpu-check
```

The plan prints commands without launching jobs. The CPU command is a three-update
correctness check. The old `train_fixed_wall.py` entry and its result snapshots are
retained as historical evidence, not the current paper protocol.

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
Current equal-update curves and provenance are under
`results/paper_equal_updates_v100_20261008/`. The earlier frozen training source is under
`results/paper_resampled_20260914/source/experiments/torch_benchmarks/` at the
repository root. The repository retains compact per-run summaries and measured
error curves for the four paper examples; large model and per-update artifacts
are omitted.
