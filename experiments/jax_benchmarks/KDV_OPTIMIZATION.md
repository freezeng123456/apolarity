# One bounded KdV optimization round

Frozen 2026-09-08 before measuring the candidates. This is a new experiment,
separate from `t4b_pde_shared_20260908`; original code/results remain unchanged.

## Priority and hypotheses

1. `shared_jet_fused`: batch all six existing real directions in one order-four
   jet evaluation, retaining the original normalized reconstruction. Two
   formerly second-order directions now compute unused third/fourth coefficients.
   Hypothesis: fewer separate dispatch sequences outweigh that extra arithmetic.
2. `shared_jet_linear`: same fused directions/order, but return a tuple of raw
   first/second/fourth derivatives; reconstruct with three fixed linear maps.
   Hypothesis: removing factorial normalization, stacking and some elementwise
   reconstruction operations yields an additional speed improvement.
3. If neither is sufficiently faster, the fixed implementation remains unsuitable
   as a speed-win example under this protocol. Stop optimizing this case; move to
   the user-approved two-dimensional Cahn-Hilliard residual in a separate run.

All identities have exact rational monomial tests through degree four. Six is
still a trajectory count, not a claimed minimal Waring rank of the residual.

## Frozen comparison

- Baseline `nested_jvp` and original control `shared_jet` call the original
  `pde_shared.py` implementation without edits. No automatic AD-baseline selection.
- Screen: B in {100,1600}, four methods (baseline, original, fused, linear),
  three seeds {20260908,20260909,20260910}: eight cells, 24 records, 720 samples.
- Each record: 10 warmups, 30 synchronized value-only wall-clock samples.
- Model, inputs, sampling, dtype, precision and all coefficients unchanged:
  width 128, depth 4 (three hidden layers), tanh, float32, highest matmul precision,
  normalized Gaussian spatial direction with uniform radius/time on [0,1].
- Same isolated JAX/jaxlib 0.4.38 T4-B environment, one GPU/process at a time;
  no outer JIT, parameter backward, custom derivative rules or chunking.
- New seeds are not introduced. Baseline/control are remeasured in this run;
  previous timings are not pooled. Every cell has its own process and full outputs.
- A separate full-width B=3 gate checks all four methods and three seeds against
  independent per-term JVP derivatives before timing. Full timed-batch output and
  component arrays plus parameter/input hashes are checked after the screen.

## Decision, fixed before timing

A single candidate must have `JVP median time / candidate median time >= 1.10`
for **every matched seed at both endpoint batches**, with all correctness checks
passing at rtol=1e-3, atol=1e-6. If linear passes it is selected; otherwise fused
is selected only if it passes. No mixture of different winners by batch/seed.

If neither passes, stop KdV. If one passes, confirm it against JVP at B=400 with
the same three seeds, warmups and repetitions (two cells, six records, 180 samples).
The same per-seed 1.10 ratio gate applies to confirmation. A failed confirmation
also stops KdV; there is no further tuning sweep or threshold relaxation.

All configs, raw timings, component arrays, decision records, failures and hashes
are retained and fully recovered locally. Correctness/runtime failures are not
treated as performance failures; diagnose their cause before a scientific decision.
