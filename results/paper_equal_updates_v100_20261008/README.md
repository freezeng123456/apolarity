# V100 experiments with equal update counts

All 30 runs completed successfully on Beijing BSCC-N26, using Tesla
V100-SXM2-32GB GPUs. Each method uses five paired seeds, 20260919–20260923.
The targets are 25,000 Adam updates for 1D KdV, 20,000 for 2D KdV, and
1,500 for Cahn–Hilliard. Interior and initial/boundary points are resampled
at every update. Paired initial-parameter and full sample-sequence hashes
match for all 15 comparisons. These are exact optimizer-update counts;
there is no fixed training-time stopping rule.

`data/curves.csv` contains all 6,030 online measurements. Every curve
reaches its prescribed final update. Time is the sum of synchronized
training-update durations, excluding setup, warm-up, error evaluation and
curve output. Errors use the same fixed independent 10,000-point
space–time and terminal test sets within each paired comparison.
`data/pde-results.csv` and `data/pde-summary.csv` record numeric endpoints
and seed summaries. The latter's speedup is the mean of the five paired
Nested-JVP/WDD training-time ratios.

Run `python plot.py` with NumPy and Matplotlib to reproduce the three
individual vector PDFs and PNGs in `figures/`, plus a combined overview.
The horizontal axis remains training time, and the vertical axis is RE
(the space–time relative L2 error). At each common update index,
plot the arithmetic mean of the five times against the arithmetic mean of
the five space–time errors. Shading extends from the mean error to the
mean plus one sample standard deviation. Plot every recorded point with
no smoothing, interpolation, extrapolation, or 1,200-second truncation.
Each method's curve ends at its own mean completion time.

The numeric recovery archive is 289,676 bytes, with SHA-256
`f254df3c5829cb94fccec201cf2c40906f6e567309dee9df1d553172da920e02`.
All 150 original checksummed result files were verified after transfer.
`data/provenance.json` retains source/archive identities and paired hashes;
`figures/plot_manifest.json` records plotting versions and endpoints.
No model or optimizer weights are saved or required for plotting.

The current paper tables display space–time RE only. Previously recorded
terminal errors remain in the immutable numeric evidence. Mean training
GPU memory is measured separately in `results/pde_memory_v100_20261009/`
and retained as a measurement record; the paper does not display memory results.
