# Experiments

One PDE family is kept per subfolder:

```
<family>/
  exp_<name>.py     # the experiment (imports the shared harness ../common/osc_common.py)
  run.sh            # historical/family diagnostic launcher; not a formal runner
  README.md         # problem, literature source, current protocol status
  data/             # currently empty
```

All `experiments/*/data/` directories have been cleared. The active paper
scope is limited to the three formal problem families below. Other families,
auxiliary result bundles, and non-formal runners are kept under
`experiments/archived/` and are not part of the active inventory.

## Families

| folder | family | status |
|---|---|---|
| `polyharmonic/` | Poly, \(d=2,3\), order \(2,4,6\) | formal `jsc_v3` (pow10 weights) |
| `chirp/` | non-separable radial chirp, \(a=1,2,3\) | formal `jsc_v3` (pow10 weights) |
| `maxwell/` | time-harmonic Maxwell, \(a=2,4,6\) | formal `jsc_v3` (pow10 weights) |

The other families are archived in `experiments/archived/other_families/`.
Their historical runners and outputs are retained only for diagnosis; they do
not implement the active formal evidence pipeline and cannot be cited as part
of the active paper inventory.

## Frozen formal methods and literal width

The only formal comparison contains:

- `complex_sinh` (Complex Sinh);
- SIREN;
- mFF-PINN;
- MscaleDNN-2-sin.

Every formal method uses literal hidden width \(H=128\). Trainable real degrees
of freedom are recorded separately (each complex parameter counts as two real
degrees of freedom), but they do not change the width. Maxwell counts both
split-real \(H=128\) baseline networks. Any formal output with a width other
than 128 is rejected.

## Frozen v2 results and pending v3 protocol

The completed 1200-second results under `experiments/results/jsc_v2/` are
retained as the fixed-`bc_weight=100` historical bundle. They are not mixed with
the next run because the boundary loss profile is changing.

The next formal run is `jsc_v3`, with the archived-search-informed
`pow10_reasonable_v1` boundary profile. The profile is frozen in
`experiments/common/boundary_weights.py` and includes:

```text
Poly d2/o2 [0.1], d2/o4 [0.1, 10], d2/o6 [0.01, 1, 10]
Poly d3/o2 [0.1], d3/o4 [0.1, 1], d3/o6 [0.1, 0.1, 1]
Chirp a1/a2/a3 [1], [0.1], [0.01]
Maxwell a2/a4/a6 [0.1], [0.1], [0.01]
```

No `jsc_v3` training has been launched yet.

## The formal task grid

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

These are three independent examples, not a batch command. The v3 canonical
outputs will live under `experiments/results/jsc_v3/<task_id>/`. Every formal bundle
must pass the validator before it can be consumed:

```bash
python scripts/validate_jsc_results.py \
  experiments/results/jsc_v3/poly_d3_o6
```

`validate_jsc_results.py` checks the protocol metadata, the boundary profile, the
four methods, the five seeds, unique keys, literal \(H=128\), finite metrics, and
history traces. Figure and table builders for the next run must accept only
validated `protocol_id=jsc_v3` bundles. The old v2 figure remains a historical
record until v3 results are available.
