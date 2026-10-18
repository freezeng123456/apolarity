# CH paired-repeat submission (2026-10-18)

This directory records a submitted campaign, not completed training results.
At the saved 20:15:28 CST snapshot, array job 2385668 had all five seed groups
RUNNING. V100 correctness gate 2385667 completed with exit 0:0; dependent
audit job 2385669 was waiting for training to finish. Later scheduler state
must be checked separately instead of inferred from this launch snapshot.

Seeds 20260919–20260923 each run Nested JVP, WDD and STDE++, sequentially on
the same allocated V100 and ten CPU cores in separate processes. Five seed
groups may run concurrently. All original training settings are retained:
1,500 Adam updates, batch 400, four hidden layers of width 128, resampled
constraints, learning rate 1e-4, and fixed independent test sets.

`submission/RUN_PLAN.json` fixes method order and anomaly criteria.
`submission/driver.py`, `audit.py`, and `worker.sh` preserve the scheduler
controller used with its frozen `source/` export. The portable training
entry is `experiments/torch_benchmarks/ch2d_paired_campaign.py`.
It supports `plan`, `preflight` and `cell`; running a cell requires a
preflight matching the source fingerprint.

The dependent audit checks finite values, exact updates, matched sampling
hashes, same-seed curve agreement, within-run timing jitter, cross-seed
timing deviations and paired time ratios. A flagged seed receives one
complete three-method confirmation repeat. Both attempts are retained;
no result is selected by final error or smoothness. If the confirmation
remains abnormal, it is reported rather than silently replaced.

No checkpoints are saved. Optional CLI GPU telemetry failed for four groups
because its scheduler ordinals were remapped inside the allocation;
PyTorch independently verified the V100 and produced finite curves.
The missing optional telemetry is retained in `launch-status.json`.
