# 2D KdV-type prescribed-trace PINN protocol, v1

Status: designed before pilot execution. Pilot and formal outputs are separate.
This supplements, and does not alter, the verified 1D KdV-gPINN experiment.

## Continuous problem and independent derivation

The unforced equation is

\[
u_{ty}+u_{xxxy}+3(u_{xy}u_x+u_yu_{xx})-u_{xx}+2u_{yy}=0,
\quad (x,y,t)\in[-1,1]^2\times[0,1].
\]

This is the 2D KdV equation appearing in Section 4.3.2, Eq. (20), of
[Shi et al., STDE](https://openreview.net/pdf?id=J2wI2rCG2u).
Only the equation is inherited: this experiment does not reproduce that
paper's training setup, boundary conditions, or solver comparison.

An independently derived exact solution is

\[
u_\star(x,y,t)=\tanh\bigl((x+y)/2-t\bigr).
\]

For the general ansatz `u=U(s), s=kx+ly+wt`, the residual is
`k^3 l U'''' + 6 k^2 l U' U'' + (lw-k^2+2l^2) U''`.
Integrating once gives
`k^3 l U''' + 3 k^2 l (U')^2 + (lw-k^2+2l^2) U' = C`.
Taking `U=A tanh(s)` with `S=sech(s)^2`, `U'=A S` and
`U'''=4A S-6A S^2` makes the constant zero and both coefficients vanish
when `A=2k` and `w=k^2/l-2l-4k^3`, for nonzero k,l.
Choose `k=l=1/2`, `A=1`, `w=-1`. Automated float64 substitution independently
checks all PDE terms, not just the reduced ordinary differential equation.

This solution is an oblique single-phase kink. It exercises mixed derivatives
in three input coordinates but is not evidence of general two-dimensional
wave interactions or a new analytic-solution theorem.

## Constraints and interpretation

Prescribe exact compatible traces for:

- initial `u(x,y,0)`;
- `u(-1,y,t)`, `u(1,y,t)`, and `u_x(1,y,t)`;
- bottom `u(x,-1,t)` and `u_y(x,-1,t)`.

No top `y=1` value enters training; it remains in the independent test grid.
Writing `v=u_y` gives `v_t+v_xxx+3(v u_x)_x-u_xx+2v_y=0`, with
`u(x,y,t)=u(x,-1,t)+integral_{-1}^y v(x,z,t) dz`. This motivates the bottom
anchor and inflow trace, and three x traces corresponding to a third-order
x operator. It does not prove well-posedness of this nonlinear mixed-derivative
IBVP. The experiment is explicitly a compatible prescribed-trace solution
benchmark. There are no interior solution labels, forcing terms, periodic
assumptions, or homogeneous substitutions for the nonzero traces.

Loss: mean squared PDE residual plus 10 times each of the six separately
averaged constraint errors (one initial and five boundary traces).
All constraint surfaces use a 16 x 16 uniform grid, 256 points per surface.
The x-right and y-bottom derivative conditions reuse the corresponding grids.
There is no residual-gradient penalty in this 2D case.

## Representation and baseline

Use the existing six real directions `(1,1,0)`, `(1,-1,0)`, `(1,2,0)`,
`(1,-2,0)`, `(0,1,1)`, `(0,1,-1)` through order four. Reuse lower-order
coefficients to reconstruct the residual terms. No minimum-direction claim
is made for the full nonlinear residual. The coordinate-JVP implementation
also reuses lower-order nested-chain values. Both methods use identical
coordinate-JVP boundary derivatives and reverse-mode parameter gradients.
Independent evaluation uses per-term JVPs rather than either shared schedule.

## Frozen execution settings

- T4-B, single GPU sequential cells, PyTorch 2.0.1+cu118, float32 eager.
- No torch.compile, outer JIT, TF32, mixed precision or alternate libraries.
- Three inputs, three hidden tanh layers of width 128, scalar output.
- Adam betas (0.9,0.999), epsilon 1e-8, foreach false.
- Learning rate `1e-3*(1-k/10000)`, zero-based update index k.
- Resample 400 points uniformly on the rectangular space-time box each step.
- Each paired seed uses identical initialization and every sampled batch.
- Initialization uses seed+2; sampling uses a separate NumPy generator seed+1.
- Independent fixed-point full-loss evaluation every 50 updates, including
  step 0 and the last step, on 400 fixed points (generator seed 20260999).
- Validation solution grid 33 x 33 x 11; final test grid 65 x 65 x 21.
- All periodic models, optimizer states, sampling states and predictions saved.
- Last iterate reported; no checkpoint selection based on test accuracy.

## Stages and predeclared decisions

1. CPU tests: analytic PDE/traces, float64 values and parameter gradients,
   five parameter-direction finite differences, first Adam state, evaluation
   frequency invariance, old regressions.
2. T4 correctness: three seeds, both methods, full-width component/full-loss,
   parameter gradients, first Adam parameter and state checks, 20 finite updates.
   Per-tensor rtol 1e-3, atol 1e-6; global gradient relative L2 <= 1e-3.
3. Full-objective cost: three seeds, two methods, loss / loss+backward /
   Adam-step; 10 warmups and 30 synchronized samples per entry (540 samples).
   Initialization and optimizer state reset after warmup. All phases construct
   the training parameter graph. No scalar-only residual timing is substituted.
4. Pilot: seed 20260920, 1000 updates per method, but the 10000-step learning-rate
   horizon is retained. It is a short prefix, not a different fast-decay run.
   Both runs must have finite outputs, final fixed loss <= 10% of initial,
   final validation relative L2 <= 0.1, and matching initialization and batches.
   There is deliberately no speedup acceptance threshold.
5. If accepted: formal seeds 20260921, 20260922, 20260923, 10000 updates each,
   six runs on the same T4. Order JVP/shared, shared/JVP, JVP/shared.
   Formal starts require the exact same source commit and source-file hashes
   as the accepted pilot. No post-hoc parameter adjustments or seed exclusions.

Failure aborts the stage and preserves records. Any numerical redesign requires
a new named protocol and output root; do not overwrite a failed run. Per-process
timeouts: correctness 600s, cost 900s, each pilot 1200s, each formal 7200s.
The total authorized formal update count in this v1 matrix is 60000.

## Timing, audit and delivery

Training wall time includes sampling, transfer, finite-gradient checks,
optimization, hashing, logs and every scheduled assessment/checkpoint including
the final one. It excludes initialization/initial assessment and post-training
test diagnostics. This last-assessment inclusion is stated explicitly rather
than silently equated with the older 1D timing implementation. Stage timing is
a separate diagnostic and excludes those non-stage costs.

Audit checks file hashes, initialization/batch pairing, finite and changed
parameters, Adam state, update counts, learning rates, all recorded fixed-point
losses via independent CPU replay, and saved prediction errors. Recover complete
raw records and retain both failures and non-advantageous results. No GitHub
push, extra model families or concurrent GPU jobs are part of this request.
