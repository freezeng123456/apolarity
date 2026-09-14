# Resampled-constraint PDE experiments, 2026-09-14

This is the only PDE dataset used by the current manuscript. All 30 runs
completed: three problems × two derivative backends × five paired seeds.
The training source is frozen at
`b94fd9ce20f1f6c79bd6195ab6d79097f74c8044` and included under `source/`.

Each run uses a single RTX 3080, PyTorch 2.5.1 with CUDA 11.8, float32 eager
execution, four width-128 tanh hidden layers, and a 1200-second training
budget. Adam has constant learning rate 1e-5 for the KdV problems and 1e-4
for Cahn–Hilliard. Seeds are 20260919–20260923.

Every update draws 400 uniform interior points and fresh uniform points on
each prescribed initial/boundary surface: 128 points per surface in 1D KdV,
256 in the two-dimensional problems. Multiple conditions on one surface use
the same points. Initial/boundary sampling has a dedicated random stream;
paired methods share initialization and the sample sequence at each common
update index. The two independent 10,000-point test sets stay fixed.

| Problem | Paired update-count speedup, mean ± sample SD |
|---|---:|
| 1D gradient-enhanced KdV | 7.57 ± 0.26 |
| 2D mixed-derivative KdV | 5.57 ± 0.18 |
| Cahn–Hilliard | 1.91 ± 0.07 |

`cells/` retains the configuration and final summary for every run. The 30
files in `curves/` contain 17,998 measured error observations and are sufficient
to regenerate the paper figures. Aggregate statistics, paired comparisons,
environment records, the frozen source, and the original completion and audit
ledgers are also retained. `LOCAL_AUDIT.json` records the earlier complete-result
and CPU endpoint-error checks; the maximum recorded absolute CPU/GPU difference
in endpoint relative errors is 2.98e-7.

Checkpoints, predictions, evaluation arrays, per-update training logs, and
repeated sampling hashes were removed from Git after the aggregate values and
curves had been verified. The original sealed run remains identified in the
provenance ledger. No fixed-update appendix results are included.

From the repository root, regenerate the manuscript assets with:

```bash
python docs/paper/build_resampled_assets.py
python docs/paper/verify_experiment_identities.py
python docs/paper/build_t4_experiment_assets.py
```

The first script reads all 30 run summaries and verifies curve endpoints,
then regenerates tables, figures, paired statistics, and asset manifests.
The identity script uses exact polynomial arithmetic to check the displayed
directional decompositions and their relation to the frozen implementation.
The third script regenerates the combined Section 4.1 table from the original derivative measurements.
