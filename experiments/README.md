# Experiments

One PDE family is kept per subfolder:

```
<family>/
  exp_<name>.py     # the experiment (imports the shared harness ../common/osc_common.py)
  run.sh            # historical/family diagnostic launcher; not a formal runner
  README.md         # problem, literature source, current protocol status
  data/             # currently empty
```

All `experiments/*/data/` directories have been cleared. There are currently no
formal results, and all paper figures and tables are **TBD**.

## Families

| folder | family | status |
|---|---|---|
| `polyharmonic/` | Poly, \(d=2,3\), order \(2,4,6\) | formal `jsc_v2` |
| `chirp/` | non-separable radial chirp, \(a=1,2,3\) | formal `jsc_v2` |
| `maxwell/` | time-harmonic Maxwell, \(a=2,4,6\) | formal `jsc_v2` |
| `helmholtz/` | high-wavenumber Helmholtz (+ anisotropic) | diagnostic only |
| `helmholtz_vc/` | variable-coefficient Helmholtz | diagnostic only |
| `plate_beam/` | Kirchhoff plate / Euler–Bernoulli beam | diagnostic only |
| `kdv/` | linearized KdV / dispersive wave | diagnostic only |
| `cahn_hilliard/` | Cahn–Hilliard | diagnostic only |
| `nls/` | cubic nonlinear Schrödinger | diagnostic only |
| `core_method/` | derivative and training diagnostics | diagnostic only |

Historical or archived runners and every family-local `run.sh` are retained
only for implementation diagnosis. They do not implement the formal evidence
pipeline, and their outputs cannot be cited as paper evidence.

## Frozen formal methods and capacity

The only formal comparison contains:

- `complex_sinh` (Complex Sinh);
- SIREN;
- mFF-PINN;
- MscaleDNN-2-sin.

The capacity reference is Complex Sinh \(H=128\), counted by true trainable
real degrees of freedom (each complex parameter counts as two real degrees of
freedom). The three external baselines receive automatically selected integer
widths whose trainable parameter counts differ from the reference by no more
than \(5\%\). Maxwell counts both split-real baseline networks. Formal \(H=64\)
outputs are rejected and are not part of the discussion.

## The only formal protocol: `jsc_v2`

The complete preregistered grid is:

- Poly: \(d\in\{2,3\}\) and order \(\in\{2,4,6\}\); \(d=3\), order 6 is a
  required setting;
- Chirp: \(a\in\{1,2,3\}\);
- Maxwell: \(a\in\{2,4,6\}\).

Formal tasks must be launched one setting at a time:

```bash
bash scripts/run_jsc_main3.sh poly --dim 3 --order 6
bash scripts/run_jsc_main3.sh chirp --sweep 2
bash scripts/run_jsc_main3.sh maxwell --sweep 4
```

These are three independent examples, not a batch command. The canonical
outputs live under `experiments/results/jsc_v2/<task_id>/`. Every formal bundle
must pass the validator before it can be consumed:

```bash
python scripts/validate_jsc_results.py \
  experiments/results/jsc_v2/poly_d3_o6
```

`validate_jsc_results.py` checks the protocol metadata, the four methods, the
five seeds, unique keys, parameter-budget tolerance, finite metrics, and
history traces. Figure and table builders accept only validated
`protocol_id=jsc_v2` bundles. Since no formal bundle currently exists, their
paper outputs remain **TBD**.
