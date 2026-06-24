# JSC Revision Checklist

## Completed in this revision

- Reframed the paper as a scientific-computing method for one fixed high-order single-monomial partial derivative.
- Removed unsupported claims that the experiments benchmark against STDE; STDE is now discussed as related complementary work.
- Replaced the strong C2 wall-clock claim with the supported direction-count/complexity claim and measured timing caveat.
- Aligned the microbenchmark table with `results/quick_compare_T4_20260602_0303.txt`.
- Aligned the PINN experiment description with `results/pinn_5min_20260602_0436.txt` (`hidden=32`, `depth=4`, `complex128`).
- Corrected the repeated-index polarization count for `(2,2,2)` from the raw sign-count interpretation to the coalesced count used by the implementation.
- Added a related-work positioning paragraph, cost discussion, and numerical-stability caveat.
- Added `tests/test_paper_patterns.py` to lock the pattern table counts.
- Added `experiments/generate_paper_tables.py` for CSV-to-LaTeX table snippets.

## Remaining risks before external submission

- The PINN table currently reports one seed; add multiple seeds and uncertainty if time allows.
- The saved quick benchmark is a text log. For final submission, regenerate it as CSV/JSON with `benchmark_single_monomial.py` and generate table rows from that file.
- No STDE baseline is implemented in this repository. Do not claim STDE benchmark results unless a separate implementation and saved result are added.
- `waring_real_jet` appears only in historical logs and is not part of the current public backend API.
- Consider adding one additional manufactured PDE/operator pattern to broaden the JSC numerical evidence.
