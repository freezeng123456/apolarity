# Paper figures

`fig_poly_d2`, `fig_poly_d3`, `fig_chirp`, and `fig_maxwell` are the four
benchmark figures included by `docs/paper/jsc_paper_main.tex`. Each exists as a
vector `.pdf` for the manuscript and a 300-dpi `.png` preview, and each is
generated from the validated `experiments/results/jsc_v2` bundles by

```bash
python experiments/tools/plot_width.py
```

The script draws at the manuscript's own text width (370.4 pt), so
`\includegraphics[width=\linewidth]` scales the result by one and the type sizes
in the script are the sizes that reach the page. Style follows the group
convention: serif text with the matching mathtext font, inward ticks on all four
sides, a light two-level grid, open markers, and a single legend row below the
panels.

Every figure has the same three panels:

1. (a) final five-seed mean relative `L2` error against the sweep parameter,
   with the seedwise range shaded;
2. (b) the error history of the central setting of the family;
3. (c) the normalized interior residual history of the same setting.

Panels (b) and (c) are framed on the seed-mean curves, so a seedwise band leaves
the panel wherever one seed diverges; the manuscript states this in the shared
protocol. PDF output is written with a null `CreationDate` so that regenerating
an unchanged figure does not churn the committed file.

`jsc_realtime_accuracy.*` and `jsc_v3_realtime_accuracy.*` are separate
diagnostic figures that the manuscript does not include. They come from
`experiments/tools/plot_jsc_realtime_accuracy.py` and
`experiments/tools/plot_jsc_v3_realtime_accuracy.py`, and their audit is in
`docs/paper/active_three_main_experiments_audit_zh.md`.
