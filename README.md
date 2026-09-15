# Waring decomposition of derivatives (WDD)

The WDD method uses two explicit constructions. Section 4.1 evaluates
individual partial derivatives with the roots-of-unity Waring formula
(`waring_batched`). Sections 4.2–4.4 evaluate PDE residuals with the prescribed
**real** directions and fixed reconstruction coefficients displayed in the
paper (`shared_jet_linear`). Both are compared with nested coordinate JVPs
(`nested_jvp`). There is no automatic selection between Waring and polarization
formulas.

The single-partial formula uses complex arithmetic when its roots are nonreal;
pure and square-free derivatives naturally have real roots. PDE directions and
parameters remain real. The two constructions share the same normalized Taylor
arithmetic, and parameter gradients use ordinary reverse-mode differentiation.
Common boundary derivatives, and the Cahn–Hilliard time derivative, use coordinate
JVPs in both compared methods.

## Paper and reproducibility

- [Manuscript](docs/paper/jsc_paper_main.pdf) and [LaTeX source](docs/paper/jsc_paper_main.tex).
- [Current code and commands](experiments/torch_benchmarks/README.md).
- [Compact PDE records and their frozen source](results/paper_resampled_20260914/README.md).
- [Historical experiment ledger](docs/HISTORICAL_EXPERIMENT_LEDGER.md), recording
  superseded studies whose large raw artifacts were removed during compaction.
- [PDE figure/table reproduction](docs/paper/build_resampled_assets.py) and
  [individual-partial table reproduction](docs/paper/build_t4_experiment_assets.py).
- [Polynomial-identity checks](docs/paper/verify_experiment_identities.py) and
  tests comparing the current implementation with the frozen paper source.

The current PDE entry is `experiments/torch_benchmarks/train_fixed_wall.py`.
It resamples interior, initial and boundary training points at every update.
Its defaults match the paper: four hidden layers of width 128, 400 interior
points, 1200 seconds, and Adam learning rate 1e-5 for KdV or 1e-4 for
Cahn–Hilliard. Evaluation points remain fixed and independent of training.

## Install and check

```bash
python -m pip install -e '.[test]'
python -m pytest
python experiments/torch_benchmarks/train_fixed_wall.py --device cpu \
  --case kdv1d --method shared_jet_linear --width 4 --depth 2 --batch 3 \
  --constraint-side 2 --max-steps 3 --eval-points 16 --trajectory-seconds 0 \
  --out /absolute/new/paper-smoke
```

The CPU command is a small correctness check, not a performance result. To
inspect the full 30-run PDE plan without launching it:

```bash
python experiments/torch_benchmarks/run_fixed_wall_matrix.py \
  --out /absolute/new/paper-pde --dry-run
```

The installed package contains only `torch_pinn`. Historical JAX implementations
are not installed by default. Obsolete automatic-selection source snapshots and
superseded training entry points are retained in a checksummed archive under
[archive/retired_code_20260914](archive/retired_code_20260914/README.md).
Original numerical records and frozen source snapshots have not been rewritten.
