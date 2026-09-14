# Second benchmark proposal: shared jets for the 2D KdV-type residual

Design only. No new evaluator or GPU experiment has been launched. The next
step requires approval of this scope, then implementation and verification.

## Purpose and target

The first group measures a single monomial derivative. The second should test
the operator--symbol correspondence and reuse of lower coefficients across
several derivative terms, without adding a PINN training problem yet.

Use the residual already present in the repository, in coordinates `(x,y,t)`:

\[
u_{ty}+u_{xxxy}+3(u_{xy}u_x+u_yu_{xx})-u_{xx}+2u_{yy}.
\]

The nonlinear products are assembled after recovering their derivative
factors; the entire nonlinear residual is not a constant-coefficient linear
operator and is not assigned a Waring rank.

## Six real trajectories

Use directions `(1,1,0)`, `(1,-1,0)`, `(1,2,0)`, `(1,-2,0)` through order four,
and `(0,1,1)`, `(0,1,-1)` through order two. Batch directions of the same
maximum order. Retain all needed output coefficients from each jet; do not
call the network again to request a lower coefficient.

With the manuscript's normalized Taylor coefficients, the quartic identity is

\[
u_{xxxy}=4\{T_4(\bx;(1,1,0))-T_4(\bx;(1,-1,0))\}
-\tfrac12\{T_4(\bx;(1,2,0))-T_4(\bx;(1,-2,0))\}.
\]

The first-order coefficients along `(1,±1,0)` yield `u_x,u_y` by half-sum
and half-difference. The second-order half-difference yields `u_xy`.
The second-order sums along `(1,±1,0)` and `(1,±2,0)` equal `u_xx+u_yy`
and `u_xx+4u_yy`; solve these two equations for `u_xx,u_yy`.
The second-order half-difference along `(0,1,±1)` yields `u_ty`.

Exact rational verification on all 35 three-variable monomials of total degree
at most four passed all 245 coefficient checks (seven reconstructed
derivatives). This verifies the identities, not finite-precision backend
accuracy, conditioning, speed or global minimality. These real directions
are a polynomial power-sum construction under Proposition 3.1, rather than
the roots-of-unity schedule verbatim. No complex computation is required.

## Comparators and implementation checks

- Main baseline: official nested coordinate `jax.jvp`, batched over points,
  returning the same residual vector. Compute each required derivative value
  once for residual assembly; inspect opportunities to reuse primal outputs
  in nested JVP calls. Do not construct a full high-order derivative tensor.
- Proposed method: six real shared jets, with the same network and points.
- One internal ablation: termwise real Taylor formulas, using the same
  quartic directions but without cross-term jet reuse. The seven terms use
  `1+1+1+2+1+2+4=12` termwise jet evaluations. Twelve counts repeated calls;
  six counts distinct trajectories with their maximum required orders.
  Neither is the Waring rank of the nonlinear residual.

The current `src/jax_apolarity_bench/residuals.py::_partials` loops over
individual monomials. `derivatives.py::directional_taylor` discards lower
coefficients, and `nested_partial_one` uses nested `jax.grad`, unlike the
first-group JVP baseline. These helpers must not be used unchanged as the
new shared-jet implementation or as a supposedly matched JVP baseline.
This turn does not modify those functions.

Before timing: test all seven derivatives and the assembled residual against
analytic polynomial fixtures and the JVP implementation on identical tanh
networks. Validate Taylor factorial normalization and unique jet counts.
Check both individual derivatives and the residual to avoid cancellation
masking an incorrect component. This is a correctness gate, not an expanded
precision study. No parameter-gradient correctness or speed claim is made.

## Frozen candidate matrix

- H20, JAX 0.4.38, scalar tanh network, three width-128 hidden layers.
- Input dimension three, rather than the first group's sixteen.
- All three implementations use float32 and no outer whole-function JIT.
- 100, 400 and 1600 points; same parameters and inputs per matched seed.
- Three seeds; ten warmups and thirty synchronized timed calls per seed;
  exclude setup/cold call and record them separately.
- Two primary implementations plus the one internal ablation: nine isolated
  cells, 27 seed records, 810 timed calls. No overlapping GPU cells or
  direction chunking; separate order-four and order-two batches are part of
  the declared representation, not memory-dependent chunking.
- Report full residual-vector evaluation, including nonlinear assembly.
  No loss reduction, parameter backward, optimizer or PDE solution training.
- One results table: points, JVP time, termwise time, shared-jet time,
  speedup versus JVP and versus termwise. No plots.

Defer g-KdV and two-dimensional polyharmonic operators. This keeps the second
group focused on the additional benefit of operator-level/shared evaluation.
