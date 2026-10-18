# Fixed B=1000 V100 partial-derivative measurements

Formal job 2385480 completed with exit 0:0 on 2026-10-18. All eight unique
targets and all three paired seeds completed; no runs or timed calls were
excluded. The paper displays the final five targets, as specified by the author.

The network has 16 inputs and three tanh hidden layers of width 128.
Both methods use `torch_pinn.sparse_jets` with fixed prepared tensors.
Direction generation and fixed-tensor conversion are outside timing; no
parameter backward pass is measured. Each seed receives ten warm-ups and
thirty synchronized wall-clock calls, alternating method order each round.
Reported times are mean and sample SD across three seed medians. Speedups
are mean paired ratios STDE++/WDD. Taylor coefficient counts are per input
point at the target derivative order; WDD counts include conjugate reuse.

`comparison.json` and `comparison.csv` summarize all eight targets.
`evidence/` preserves the 140 verified original result files, including raw
timings, small derivative-value arrays, configs, hashes and the completed
job receipt. These arrays contain derivative values, not model weights.
`audit.json` and `recovery-receipt.json` record validation and transfer checks.
The frozen source manifest identifies the measured code; the current source
also includes the later CH repeat controller, which was not part of this job.

Rebuild or check the current paper table from the repository root:

```bash
python docs/paper/build_stdepp_experiment_assets.py
python docs/paper/build_stdepp_experiment_assets.py --check
```

There are no B=100 measurements, checkpoints, profiler traces or duplicated
source archives in this result directory.
