# Full-complete snapshot

This directory contains the complete WAR/real-tanh-autodiff 60-second loss-weight search.

- 5 tasks, 497 candidates, 994 method runs.
- All JSON results are `status=complete`; no failed or running cells were reported.
- Every JSON result contains a 13-point history with final loss and rel_error.
- Raw point logs were preserved byte-for-byte. The six pre-fix logs are not rewritten; their final metrics are explicitly sourced from the paired JSON in `analysis/final_metrics.csv`.
- `TOP10.md` and `top10_full.csv` contain WAR, real AD, shared geometric-mean, and shared minimax rankings for every task.
- `verification.json` records the counts and the remote completion state.
