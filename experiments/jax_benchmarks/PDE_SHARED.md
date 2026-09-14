# Shared real jets: frozen second-group protocol

Protocol `pde_shared_real_no_outer_jit_v1`, frozen 2026-09-08 before GPU timing.
Only the two equations and network/point initialization conventions are reused
from the existing benchmark. The shared directions and evaluator implementation
are ours; this is not an STDE comparison. No manuscript changes in this run.

## Tasks and reconstruction

KdV uses coordinates `(x,y,t)` and returns the entire length-B residual vector:
`u_ty + u_xxxy + 3(u_xy*u_x + u_y*u_xx) - u_xx + 2*u_yy`.
Four order-4 jets along `(1,±1,0),(1,±2,0)` and two order-2 jets along
`(0,1,±1)` supply all seven partials, retaining lower coefficients.

Gradient-enhanced KdV uses `(x,t)`, `f=u_t+u*u_x+0.0025*u_xxx` and returns
`mean(f²)+0.001*(mean(f_x²)+mean(f_t²))`. This includes the two input derivatives
of f, but excludes network-parameter backward and training. Five order-4 jets
along `(1,0),(1,±1),(1,±2)` supply u and all eight partials.

Coefficients are normalized by k!: `T_k(v)=D_v^k u/k!`.
For either quartic mixed term, the reconstruction is
`4*(T4(1,1)-T4(1,-1)) - (T4(1,2)-T4(1,-2))/2`.
The pure-x gKdV jet supplies u and x derivatives through order four.
Exact rational monomial tests through total degree four verify all shared
component identities. Six/five are unique trajectory counts, not a nonlinear
residual Waring rank or a proved optimum.

## Fixed comparison and execution

- Formal baseline: coordinate nested JVP, fixed by the user. AD chains reuse
  lower primal derivatives via `has_aux`; no full high-order derivative tensor.
- Our method: shared real directional jets, batched within each maximum order.
- Internal ablation: termwise real jets using the same per-term identities;
  12 direction calls per case with no sharing across terms. gKdV also evaluates
  the primal separately. This is explicitly a no-sharing ablation.
- Independent diagnostic only: coordinate JVP versus nested reverse grad at
  B=100, seed 20260918, 3 warmups and 7 samples. Never used to switch the baseline
  or pooled with formal timings.
- Formal matrix: 2 cases × B in {100,400,1600} × 3 methods = 18 isolated processes;
  3 matched seeds {20260908,20260909,20260910} = 54 records. Each record has
  10 warmups then 30 synchronized wall-clock samples (1620 formal samples).
- NVIDIA H20, one GPU and one process at a time; JAX/jaxlib 0.4.38, real float32,
  highest matmul precision, tanh MLP width 128, depth 4 (three hidden layers).
- Point sampler: spatial direction is normalized Gaussian, radius uniform
  [0,1], time uniform [0,1]. In two spatial dimensions this is NOT volume-uniform
  disk sampling. Point seed is seed+1; parameter seed is seed+2.
- All parameters and points are dynamic. No whole-function JIT, no custom
  activation rules, no chunking, no parameter backward. JAX's own primitive
  compilation remains enabled. First process call is recorded separately;
  warmed timing includes Python dispatch, derivative computation and assembly.
- Initialization, input hashing, correctness checks, component extraction and
  host serialization are outside timing. Allocator high-water marks are only
  process-level diagnostics, not isolated operation-peak memory claims.

## Gates and provenance

Before timing, a separate full-width B=3 GPU process checks all four evaluators
on all three formal seeds against independent per-term coordinate-JVP chains.
For gKdV, f_x/f_t and the objective are independently checked by differentiating
the unexpanded f. All component/value comparisons use rtol=1e-3, atol=1e-6.
CPU tests additionally cover exact polynomial identities and two small networks.

Every timing cell saves full component arrays and output for each seed, parameter
and point hashes, all 30 raw durations, source commit/file hashes and runtime.
After each group, every component and assembled output is compared to JVP at
the full timed batch, with matched input hashes. Any failure stops the matrix;
the failed root is retained and cannot be overwritten or silently resumed.
Each subprocess has a 3600-second timeout. Logs, progress, terminal markers and
SHA256SUMS are retained. Code runs from a detached Git worktree; outputs are
separate. A local bundle transfers the pinned commit without a GitHub push.

Entry points: `pde_benchmark.py --verify` for the gate and
`run_pde_matrix.py --out <new-absolute-canonical-root>` for the full matrix.
