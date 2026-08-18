# Paper figures

`fig_v3_cost` and `fig_v3_history` are the two figures included by
`docs/paper/jsc_paper_main.tex`. Each exists as a vector `.pdf` for the
manuscript, a 300-dpi `.png` preview, and an `.svg`, and both are generated from
the validated `experiments/results/jsc_v3` bundles by

```bash
python experiments/tools/plot_v3_figures.py
```

The companion tables come from

```bash
python experiments/tools/build_v3_tables.py
```

`fig_v3_cost` reports the cost of one optimizer step and its peak memory against
the derivative order for the two backends of the `jsc_v3` protocol, together
with the ratio between them. `fig_v3_history` reports the held-out relative `L2`
error against training time on one representative setting of each family, as the
three-seed median with the seedwise range shaded; the panels are framed on the
median curves.

Style lives in `experiments/tools/paper_style.py` and follows the group's
reference plotting implementation: serif text with matching mathtext, a
two-level grid, inward ticks on all four sides, open markers with white faces,
and one legend row above the panels. Figures are drawn at the manuscript text
width, 370.38 pt, so `\includegraphics[width=\linewidth]` neither enlarges nor
shrinks them; `check_width` fails the run if a saved PDF drifts from that width.
PDF and SVG output carry a null creation date so that regenerating an unchanged
figure does not churn the committed file.

The `jsc_v2` figures that compared four architectures under the earlier protocol
are no longer part of the manuscript, which compares derivative backends rather
than architectures. Their data is still under `experiments/results/jsc_v2`, and
`experiments/tools/plot_width.py` with `experiments/tools/build_width_tables.py`
still regenerate them on demand.

`jsc_realtime_accuracy.*` and `jsc_v3_realtime_accuracy.*` are separate
diagnostic figures that the manuscript does not include. They come from
`experiments/tools/plot_jsc_realtime_accuracy.py` and
`experiments/tools/plot_jsc_v3_realtime_accuracy.py`.
