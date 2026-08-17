# Paper figures

`fig_poly_d2`, `fig_poly_d3`, `fig_chirp`, and `fig_maxwell` are the four
benchmark figures included by `docs/paper/jsc_paper_main.tex`. Each is written
as a vector `.pdf` for the manuscript, a 300-dpi `.png` preview, and an `.svg`,
and each is generated from the validated `experiments/results/jsc_v2` bundles by

```bash
python experiments/tools/plot_width.py
```

Layout and styling follow the group's reference plotting implementation: a row
of lettered panels, a legend row above them, error bars on the profile panel, a
star on the best entry, a two-level grid, inward ticks on all four sides, open
markers with white faces, and serif type with the matching mathtext font.

The one deliberate departure from the reference is scale. The canvas is the
manuscript text block itself, 370.38 pt, so `\includegraphics[width=\linewidth]`
neither enlarges nor shrinks the figure and the type sizes in the script are the
sizes that reach the page. A tight bounding box is not used, because its size
depends on the tick labels and would give the four figures four different
widths, four different scale factors, and four different effective type sizes.
`check_width` in the script fails the build if a saved PDF drifts from the text
width.

Every figure has the same three panels:

1. (a) final five-seed mean relative `L2` error against the sweep parameter,
   with sample-deviation bars and a star on the lowest mean at each value;
2. (b) the error history of the central setting of the family;
3. (c) the normalized interior residual history of the same setting.

Panels (b) and (c) draw the seed-mean curve with the seedwise range shaded, and
are framed on the mean curves, so a band leaves the panel wherever one seed
diverges; the manuscript states this in the shared protocol.

Throughput is deliberately not plotted against a sweep. The recorded source
commit differs between settings, so a trend across a sweep would not be
attributable to the sweep; the manuscript reports throughput one setting at a
time in its own table. PDF and SVG output carry a null creation date, so
regenerating an unchanged figure does not churn the committed file.

`jsc_realtime_accuracy.*` and `jsc_v3_realtime_accuracy.*` are separate
diagnostic figures that the manuscript does not include. They come from
`experiments/tools/plot_jsc_realtime_accuracy.py` and
`experiments/tools/plot_jsc_v3_realtime_accuracy.py`, and their audit is in
`docs/paper/active_three_main_experiments_audit_zh.md`.
