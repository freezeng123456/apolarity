# Additional STDE++ PDE measurements (2026-10-16)

These are completed STDE++ measurements for all three PDE examples, five
paired seeds each. They are additional comparisons; the current paper's
PDE figures still compare WDD with Nested JVP. STDE++ ran on 2026-10-16;
the WDD/Nested reference runs are the existing 2026-10-08 records. GPU
model, CPU quota, runtime, initialization and sample hashes match, but
these were two batches rather than one simultaneous three-method rerun.

`pde-all-seed-results.csv` contains all three methods' endpoints;
`pde-comparison.csv` and `pde-paired-ratios.csv` contain their summaries.
`curves.csv` and `summaries.json` retain the 15 new STDE++ runs. The existing
Nested/WDD curves are in `../paper_equal_updates_v100_20261008/data/curves.csv`.
`BASELINE_PAIRS.json` preserves the matched initialization/sample/test hashes,
and `SOURCE_MANIFEST.json` identifies the measured source snapshot.

1D KdV uses two 7-jets and a 2-jet; 2D KdV uses a 9-jet and two 3-jets.
The CH port adapts §6.1 to four real spatial 4-jets, sharing the lower
coefficients and the common time/boundary derivative paths. WDD uses three
spatial 4-jets. CH's paired timing ratio is approximately 1.03, with enough
variation that it should be described as essentially tied. Its fresh
three-method paired repeat is recorded separately under
`../ch2d_paired_v100_20261018/`.

These files contain numeric results and provenance, without model weights.
