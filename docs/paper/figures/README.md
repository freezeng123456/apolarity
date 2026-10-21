# Current paper figures

All three manuscript figures compare **WDD (blue)** with **STDE++ (red)**.
The horizontal axis is training time and the vertical axis is RE on a
logarithmic scale. At common update indices, each method's five-seed
arithmetic mean RE is plotted against its arithmetic mean training time.
Shading extends upward by one sample standard deviation. Every recorded
point is used without smoothing, interpolation, extrapolation, or a fixed
time cutoff.

| Figure | Updates | Seeds | Data |
|---|---:|---|---|
| `resampled_wall_kdv1d` | 25,000 | 20260919–20260923 | WDD: 2026-10-08; STDE++: 2026-10-16 |
| `resampled_wall_kdv2d` | 20,000 | 20260919–20260923 | WDD: 2026-10-08; STDE++: 2026-10-16 |
| `equal_updates_kp2d` | 20,000 | 20260936–20260940 | KP-II campaigns recovered on 2026-10-21 |

The two KdV datasets are in `results/paper_equal_updates_v100_20261008/`
and `results/stdepp_pde_v100_20261016/`. The ten paired comparisons have
matching initial-parameter, complete sample-sequence, and test-set hashes.
The hardware/runtime configuration matches, although the two methods were
measured in separate batches. The KP-II dataset and cohort history are in
`results/kp_fp32_v100_20261021/`.

Regenerate the current figures and tables from the repository root:

```bash
python docs/paper/build_equal_updates_assets.py
python docs/paper/build_kp_experiment_assets.py
```

Both scripts accept `--check` to verify current assets against the numeric
evidence. Manifests are in `docs/paper/data/equal_updates/` and
`docs/paper/data/kp/`. Tables display final RE and training time as means
and sample standard deviations. Speedups are the arithmetic means of the
five paired STDE++/WDD time ratios.

Earlier Nested JVP and Cahn–Hilliard results remain as historical records;
they are not part of the current manuscript figures or source bundle.
