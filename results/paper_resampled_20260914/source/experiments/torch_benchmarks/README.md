# PyTorch-only derivative and PINN benchmark

This is the active migration path. Historical JAX source/results remain intact.
One framework, eager execution, no torch.compile, no JAX runtime dependency,
no third-party differentiation library. Do not pool old JAX timings with these.

## Shared mathematical work, different derivative engines

- `nested_jvp`: official torch.func.jvp/vmap; selected derivative chains reuse
  lower primal values through has_aux, not a full high-order derivative tensor.
- `shared_jet_linear`: one batched Taylor propagation through Linear/tanh,
  all needed lower coefficients retained, then fixed linear reconstruction.
- `shared_jet`: the same torch jet engine with explicit termwise arithmetic in
  reconstruction, a diagnostic ablation, not the old JAX execution schedule.

Taylor coefficients are normalized: q[k]=D_v^k u/k!. The PyTorch linear maps
absorb the required factorials. Thus the mathematics agrees with the prior raw
JAX reconstruction without pretending the backend's intermediate format is identical.
The tanh series uses y'=x'(1-y^2); every operation is differentiable PyTorch
arithmetic. It is a Linear/tanh-specific Taylor implementation, not a new general
automatic-differentiation library and not custom reverse derivative code.
No archived compiler wrappers or unsupported performance claims are imported.

## APIs

`torch_pinn.model.MLP`: same original NumPy initialization convention.
`torch_pinn.monomials.monomial`: individual mixed partials, coordinate JVP or
rank-optimal complex/real directions. The old six first-group targets retain
their original direction counts. Historical first-group timing matrices are not rerun.
`torch_pinn.operators.evaluate`: scalar PDE loss for kdv2d/gkdv1d, with optional
all-components output. **kdv2d now returns mean(R^2), not the old residual vector**.
`torch_pinn.problem.loss`: full KdV-gPINN loss including initial and boundary data.
All loss/gradient/update paths are shared except the PDE derivative calculator.

## Training problem

    f = u_t + u u_x + 0.0025 u_xxx = 0, x in [-1,1], t in [0,1]
    exact u = 0.75 sech^2(5(x+0.5-0.25t))
    L = mean(f^2) + .001(mean(f_x^2)+mean(f_t^2)) + 10 L_IC + 10 L_BC

IC: exact t=0 data. BC: u at both endpoints and u_x at the right endpoint,
with exact nonzero traces (not falsely periodic/zero). L_BC is the sum of three
separately averaged errors. Interior exact values are evaluation-only, never
supervised training targets. Boundary derivatives use a common coordinate JVP.
Three hidden width-128 tanh layers, real float32, no input rescaling, TF32/AMP off.
Adam lr=.001, betas=(.9,.999), eps=1e-8, foreach=False, no clipping or adaptive weights.
Interior batches use an identical replayable uniform rectangle sequence;
128 IC points and 128 boundary times remain fixed.

## Entry points

Run from this directory, or pass its absolute path to the Python interpreter:

    python -m pytest -q tests
    python verify.py --device cuda --out /absolute/new/gate
    python benchmark.py --device cuda --case gkdv1d --method nested_jvp --out /absolute/new/cost
    python train.py --device cuda --method shared_jet_linear --out /absolute/new/train

Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python on CUDA. A GPU run
requires a T4 under this migration protocol; CPU is available for tests and smoke.

`benchmark.py` measures loss, loss+backward, and Adam step separately. The
gkdv1d task includes all IC/BC terms; kdv2d remains PDE-only because no 2D IBVP
has been approved. Outputs are synchronized and include all wall-clock samples.
Initial model/state are restored after warmup; step timings perform real updates.
Cold calls and allocator-memory semantics are separate fields. For loss-only
timing parameter graph construction remains enabled, as in the training forward.

`train.py` saves raw metrics, validation, initial/final/periodic full checkpoints
(including optimizer and sampling state), point hashes and predictions.
Validation uses a common independent coordinate-JVP path. Report error versus
steps/time, target attainment and failures, not only loss decrease. Optimizer-loop
time includes sampling/transfer/forward/backward/update; total time additionally
includes evaluation/checkpoint/hash/logging. No resume CLI is claimed yet.

## Frozen migration acceptance, not the full publication matrix

`run_migration.py --out /absolute/new/acceptance` runs one sequential GPU job:

1. CPU tests in the target Python, including exact coefficients, Taylor orders,
   individual mixed partials, full residuals, parameter gradients, Adam and
   five double-precision parameter-direction finite differences.
2. GPU correctness: 2 cases x 3 methods x 3 seeds at full width/B=3 = 18 records;
   full PINN loss/grad/first-Adam-state checks plus 20 updates per method/seed.
3. B=400 cost check: 2 cases x 2 primary methods x 3 seeds x 3 phases = 36 records,
   5 warmups and 15 synchronized samples each (540 timings).
4. Matched full-width KdV-gPINN pilot: both primary methods, independent seed
   20260918, B=400, at most 1000 steps or 600 optimizer seconds each. Evaluate
   every 100 steps; this is a migration/learning check, not a three-seed formal
   convergence or speed claim. Preserve unsuccessful target attainment.

Float32 gate: each component/parameter-gradient leaf rtol=1e-3, atol=1e-6;
global parameter-gradient relative L2 <=1e-3. Double tests have tighter tolerances.
Failure aborts the fixed matrix and preserves its root. Raw results and hashes
must be fully recovered; no manuscript edit or GitHub push is part of migration.

## Implementation caveat caught during migration

In local torch 2.14.0, jvp(lambda z: z * .0025) on a float32 scalar produced a
float64 tangent. The independent unexpanded residual now uses an explicit
same-dtype tensor dispersion coefficient, and a float32 regression test covers it.
No tolerance or output dtype cast was used to hide the issue. Runtime versions
are recorded; passing local tests does not replace target-runtime GPU tests.
The same constant is constructed outside vmap: torch 2.0.1 cannot dispatch
Tensor.new_tensor from a BatchedTensor. The failed target-runtime test root is
retained, and the corrected version receives a fresh full acceptance run.
