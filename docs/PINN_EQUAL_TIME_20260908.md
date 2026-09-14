# Two actual ten-minute training runs

User clarification: train both methods for ten minutes and compare loss versus
elapsed time, not updates versus time from the earlier 1000-step pilot.

- One Tesla T4, two sequential cells: coordinate JVP then shared Taylor jet with
  linear reconstruction. Eager PyTorch 2.0.1+cu118, float32, TF32/AMP disabled.
- Each starts afresh, uses seed 20260918, and runs for 600 seconds of training
  wall time. No step cap. Time includes periodic evaluation, bookkeeping and
  checkpointing, not only optimizer computation. Initial setup/assessment and
  final diagnostics are reported outside the budget. An in-flight operation may
  finish just beyond the deadline; report actual times, do not relabel as exact.
- Same previous KdV-gPINN IBVP, three hidden layers of width 128, tanh,
  Adam lr=0.001/betas=(0.9,0.999)/eps=1e-8, B=400 fresh uniform interior points,
  128 fixed initial points and 128 times for each of the three boundary traces.
- Same initialization and ordered training-batch stream; verify the overlapping
  batch-hash prefix. Methods may complete different numbers of updates.
- Numerical model, derivative evaluators and training objective unchanged from
  migration source 04185306310d83a139bb0fb1f15ee2e406ba0dce.
- Main y-axis: fixed-point full objective, independently evaluated by coordinate
  JVP for both models. Fixed 400 interior points, RNG seed 20260999, and identical
  IC/BC data. L = MSE(f) + 0.001[MSE(f_x)+MSE(f_t)] + 10 MSE(IC)
  + 10[MSE(u_left)+MSE(u_right)+MSE(ux_right)]. No smoothing or envelopes.
- Assess at roughly ten-second training-wall intervals; timestamps are actual
  model snapshot times, not fabricated exact grid times. Evaluations do not
  update model/optimizer or consume the training RNG.
- Also retain raw resampled-batch training loss, all loss components, independent
  solution relative L2, 257x101 final test predictions, checkpoints, optimizer
  state, sampler state, timings and hashes. Checkpoints at each assessment.
- Single paired seed only: an equal-time comparison, not multi-seed statistical
  evidence. Report late fluctuations and failures as measured; do not tune,
  restart, cherry-pick a best checkpoint or extend either budget after seeing it.
- New entry train_equal_time.py leaves the historical train.py and existing raw
  result roots unchanged. CPU tests and a short same-runtime GPU smoke precede
  formal training. Full local recovery and hash verification; no GitHub push or
  manuscript changes requested.
