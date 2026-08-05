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
| `polyharmonic/` | Poly, \(d=2,3\), order \(2,4,6\) | formal `jsc_v2` |
| `chirp/` | non-separable radial chirp, \(a=1,2,3\) | formal `jsc_v2` |
| `maxwell/` | time-harmonic Maxwell, \(a=2,4,6\) | formal `jsc_v2` |

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
five seeds, unique keys, literal \(H=128\), finite metrics, and history traces.
Figure and table builders accept only validated `protocol_id=jsc_v2` bundles.
The active results are the validated bundles under
`experiments/results/jsc_v2/<task_id>/`.
