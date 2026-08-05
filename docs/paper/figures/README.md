# Active real-time accuracy figure

`jsc_realtime_accuracy.pdf` is the paper-ready vector figure and
`jsc_realtime_accuracy.png` is its 300-dpi preview. The curves are generated
from the validated active `experiments/results/jsc_v2` bundles by
`experiments/tools/plot_jsc_realtime_accuracy.py`.

Rebuild it with:

```bash
python experiments/tools/plot_jsc_realtime_accuracy.py \
  --results experiments/results/jsc_v2 \
  --out-dir docs/paper/figures \
  --tasks poly_d2_o4 chirp_a2 maxwell_a4
```

The default three panels use the central setting of each active family:

1. Polyharmonic, `d=2`, order `4`;
2. Chirp, `a=2`;
3. Maxwell, `a=4`.

The solid curve is the five-seed median of the raw held-out relative (L^2)
error, the shaded region is the seed interquartile range, the dotted curve is
the median best-so-far error, and `×` marks the median final checkpoint. The
figure is intentionally limited to the four active formal methods and excludes
`auto` and every path under `experiments/archived`.

The full audit, including all 12 active task final medians and the missing
boundary/direct-autodiff evidence, is in
`docs/paper/active_three_main_experiments_audit_zh.md`.
