# Current paper figures

The two current `resampled_wall_{kdv1d,kdv2d}` figures use the completed
V100 experiments in `results/paper_equal_updates_v100_20261008/`.
Their existing filenames and manuscript labels are retained. The historical
CH figure is retained but is no longer included in the current manuscript. Generate the
plots with that directory's `plot.py`, then install the figures and build
the endpoint tables with `docs/paper/build_equal_updates_assets.py`.

All 30 runs resample interior and initial/boundary training points at every
update. Both methods complete 25,000 updates for 1D KdV, 20,000 for 2D KdV,
and 1,500 for Cahn–Hilliard. Each plot shows the arithmetic mean error
against mean training time across five seeds at common update indices.
Shading extends from the mean to the mean plus one sample standard
deviation. Every recorded point is plotted without interpolation,
smoothing, extrapolation, or a fixed time cutoff. Errors are evaluated
online on fixed independent test points; no checkpoints are saved.

The corresponding tables use the final recorded iterate and report
training time and space–time RE; terminal error is not displayed.
Plots label the same space–time error as RE. Speedups are means of the five
paired Nested-JVP/WDD training-time ratios. PDF and PNG versions are
supplied, and `docs/paper/data/equal_updates/asset_manifest.json` records
the source dataset and asset hashes.

The third current figure, `equal_updates_kp2d`, uses
`results/kp_fp32_v100_20261021/`: WDD (blue) and STDE++ (red), seeds
20260936–20260940, and 20,000 updates per run. It follows the same
common-update arithmetic-mean and upper-standard-deviation shading rule
as the KdV figures. Regenerate it and its table with
`python docs/paper/build_kp_experiment_assets.py`; add `--check` to verify
the installed asset hashes. The result README records the cohort selection
and retains all seven observed seed pairs.
