# Canvas: oscillatory PINN results (Chinese)

`oscillatory-pinn-results-zh.canvas.tsx` is a **Cursor Canvas** — a self-contained
React component that renders the 600 s width-study results as an interactive page:
per-method principles, the experimental protocol, the high-level "hardest sweep"
comparison, and one collapsible card per PDE family (settings + results table +
relative-`L2` trend chart).

## How to view it

Open the file in **Cursor** and use *Open Canvas* (the preview opens beside the
chat). The component imports the Cursor Canvas SDK primitives (`Card`, `Table`,
`LineChart`, …), so it renders inside that environment rather than a plain
browser.

## Notes

- This is a **snapshot** of the results for distribution with the repo; it is not
  wired into the build and nothing imports it.
- The numbers mirror the per-family data in `experiments/<family>/data/` and the
  figures in `docs/paper/figures/`. If the experiments are re-run, update the
  `FAMILIES` data block in the `.tsx` (or regenerate the paper figures/tables via
  `experiments/tools/`).
