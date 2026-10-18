# STDE++ reproduction on the apolarity benchmark suite

## Method attribution

The repository contains two separately labelled partial-evaluation variants:
**STDE++ deterministic multi-jet (Section 6.1)** and **STDE++ prime single
curved jet (Section 6.2.4)**. The earlier PDE reproduction uses the published
low-dimensional PDE jet constructions. Section 6.1 attributes its multiple-linear-jet construction
to Griewank et al. (2000). This is an exact deterministic baseline. There is no
coordinate sampling or stochastic derivative error in these low-dimensional
cases.

The Section 6.1 results are **not** measurements of the random-prime single-jet
algorithm of Section 6.2.4; the latter is measured separately by
`stdepp_prime_campaign.py`. The journal paper retains both constructions, and
its Table 1 notes that the polynomial construction has a large prefactor. For
a sixth-order maximally mixed partial, its tabulated prime construction uses
jet order 560, whereas the exponential single-jet construction uses order 321.
Using either would be a separate benchmark variant and must be named as such.

Primary sources:

- Shi, Hu, Lin, Kawaguchi, [STDE++: Polynomial-Time Amortization for Linear
  Differential Operators](https://jmlr.org/papers/v27/25-1474.html), JMLR
  27(114):1–50, 2026; [paper PDF](https://jmlr.org/papers/volume27/25-1474/25-1474.pdf).
- [Official implementation](https://github.com/sail-sg/stde), source snapshot
  inspected at commit `fae88663b1d2f1b0666d4d250c38a3c34b852ac0`;
  relevant file `stde/equations.py`.

The public repository identifies itself as the NeurIPS 2024 STDE implementation.
Its inspected source contains the PDE constructions, but no random-prime
construction/precomputation script was found. The current port uses PyTorch
because the apolarity comparison uses eager PyTorch; it does not claim to
reproduce the official JAX runtime or compiler behavior.

## Individual mixed partials

Let `alpha` list the repeated zero-based coordinate indices, let `k=len(alpha)`,
and write `q_k(v)=D_v^k u/k!`. The implementation uses the centered polarization
identity equivalent to the multiple-linear-jet construction:

\[
D_\alpha u=2^{-k}\sum_{\varepsilon\in\{-1,1\}^k}
\left(\prod_j\varepsilon_j\right)
q_k\left(\sum_j\varepsilon_j e_{\alpha_j}\right).
\]

Repeated occurrences of an axis are grouped by binomial coefficients. A
direction and its negative are paired by homogeneity; integer scalar multiples
are then combined using `q_k(c v)=c^k q_k(v)`. All coefficients are formed as
exact integers with a common denominator `2^k` before conversion to the run
dtype. This avoids evaluating redundant jets and preserves the same identity.

| Target | Target/jet order | Real linear jets after grouping |
| --- | ---: | ---: |
| `111111` | 6 | 1 |
| `111222` | 6 | 6 |
| `112233` | 6 | 13 |
| `123456` | 6 | 32 |
| `11223344` | 8 | 40 |
| `111222333` | 9 | 28 |

All jets share an eager PyTorch contraction across direction, Taylor order,
and collocation axes. The existing WDD Taylor arithmetic is used for these
linear jets, so the mathematical representation is the changed component.
No parameter backward pass is included in the individual-partial value timing.

## PDE jet schedules

Curved input coefficients are normalized: a raw tangent `x^(r)` is supplied
as `x^(r)/r!`, and the propagated coefficient `q_r` represents `y^(r)/r!`.
`curved_jet` uses ordinary differentiable PyTorch operations and the existing
`tanh_series` recurrence, so standard parameter backpropagation applies.

### 1D gradient-enhanced KdV

Appendix I.4, Eqs. (110)–(111), and the official `highord1d_res`, `case == 4`,
use two 7-jets and one 2-jet:

- `a`: raw tangent `x^(1)=e_x`, all others zero, through order 7.
- `b`: raw tangents `x^(1)=e_x`, `x^(4)=e_t`, through order 7.
- `c`: raw tangent `x^(1)=e_t`, through order 2.

In normalized output coefficients the reconstruction is
`u=a_0`, `u_x=a_1`, `u_xx=2a_2`, `u_xxx=6a_3`,
`u_xxxx=24a_4`, `u_t=24(b_4-a_4)`, `u_tx=24(b_5-a_5)`,
`u_txxx=144(b_7-a_7)`, and `u_tt=2c_2`.

The PDF's Eq. (111) prints `y_1^(8)` in the `u_txxx` correction, despite
defining only a 7-jet. The official code subtracts `series_out1[6]`, namely
`y_1^(7)`, which agrees with direct expansion. This port follows the verified
seventh-order correction.

### 2D KdV

Appendix I.4, Eqs. (100)–(101), and the official `KdV2d_res`, `case == 3`,
give the alternative lower-order construction: one 9-jet and two 3-jets.

- `a`: raw tangents `x^(2)=e_x`, `x^(3)=e_y`, through order 9.
- `b`: raw tangents `x^(1)=e_y`, `x^(2)=e_t`, through order 3.
- `c`: raw tangent `x^(1)=e_y`, through order 3.

The reconstruction is `u_x=2a_2`, `u_y=6a_3`, `u_xx=8a_4`,
`u_xy=12a_5`, `u_yy=2c_2`, `u_yyy=6c_3`,
`u_ty=(6b_3-u_yyy)/3`, and
`u_xxxy=(9!a_9-280u_yyy)/1260`.
The official default is a different 13-jet construction (`case == 4`);
the present choice is explicitly the paper's lower-order alternative.

The paper's Eq. (101) and the official lower-order code print divisor `840`.
For the stated raw input tangents, direct expansion gives
`a_9=u_xxxy/288+u_yyy/1296`, hence the correct raw ninth-order coefficient
is `1260u_xxxy+280u_yyy`. The port corrects the divisor to `1260` and
checks the complete objective against independent coordinate derivatives.

### 2D Cahn–Hilliard

This PDE is not in the published STDE/STDE++ experimental suite. We adapt
the same deterministic multiple-linear-jet method, with four fourth-order
spatial jets along `e_x`, `e_y`, `e_x+e_y`, and `e_x-e_y`.
The first two also provide the spatial gradient and Laplacian. Exact
polarization gives

\[
u_{xxyy}=\frac{D_{e_x+e_y}^4u+D_{e_x-e_y}^4u
                    -2u_{xxxx}-2u_{yyyy}}{12},
\qquad
\Delta^2u=u_{xxxx}+2u_{xxyy}+u_{yyyy}.
\]

The time JVP, boundary derivatives, constraints, forcing function, objective
weights, sampling, and optimizer use the unchanged common implementation.
This is documented as an adaptation, not an additional official STDE++ case.

## Validation

`tests/test_stdepp.py` checks each centered polarization identity in exact
integer arithmetic on every homogeneous monomial of the relevant degree.
All six derivatives and their parameter gradients are checked independently
against the analytic tanh-derivative polynomial of a one-neuron MLP. Curved
coefficients are checked against independent scalar nested JVPs, including
factorial normalization. Every complete PDE objective, its component losses,
all parameter gradients, and one Adam update are compared with the existing
coordinate-JVP reference in float64. Formal hardware gates additionally check
the paper's float32 shapes before full training is submitted.

## Prepared values-only partial profiling and optimization

`stdepp_optimized_partials.py` profiles the original prepared six-partial
baseline, checks independent float64 and full-size float32 agreement, then
compares `baseline`, `precast`, and `optimized` under identical timing bounds.
The CLI stages are `profile`, `preflight`, and `cell --index 0..5`, each with
`--out ROOT`; dependent stages require completed matching-source receipts.

Both methods use the same unchanged `mlp_jet` kernel and receive the same
preparation policy. Fixed real model parameters and inputs are snapshotted and
converted outside timing. Verified conjugate pairs contribute
`2*Re(c*q(v))`; the reduced WDD evaluation counts are 1, 3, 5, 32, 14, 10,
while real STDE++ counts remain 1, 6, 13, 32, 40, 28. These fixed snapshots
are detached and are intended only for derivative-value measurements, not
parameter training. No PDE backend is changed.

Each seed has ten warm-ups and thirty synchronized wall-clock calls per
method and variant. Three cyclic variant orders and alternating method
orders put every evaluation in each of the six absolute positions five
times. CPU/CUDA profiling occurs outside formal timings and is retained as
compact reports and bounded compressed traces. Device-duration sums and
trace gaps are not interpreted as exclusive unprofiled wall-time fractions.
The six-target single-allocation sequential repeat uses fresh processes and
preserves the frozen source, seeds, timing counts and successful gate records.

## Current fixed-batch paper measurements

The current §4.1 uses `stdepp_fixed_batch_campaign.py`: eight unique targets,
fixed B=1000 and width 128, measured with the same `torch_pinn.sparse_jets`
backend for both methods. Both methods receive fixed, preconverted tensors;
preparation is outside timing. Three paired seeds each have ten warm-ups and
thirty synchronized calls with alternating method order. The final five
targets are displayed in the paper; all eight completed targets remain in
`results/stdepp_section61_b1000_v100_20261018/` for reproducibility. Run
`python docs/paper/build_stdepp_experiment_assets.py --check` to verify the
paper table against the saved raw timings.

`ch2d_paired_campaign.py` defines the separate five-seed CH repeat submitted
on 2026-10-18. For each seed, Nested JVP, WDD and STDE++ run sequentially
on one V100 with unchanged 1,500-update training. Its plan fixes method
order and numerical/timing anomaly checks before submission. A flagged
seed group receives one full three-method confirmation repeat; every
attempt is retained. Submission records are under
`results/ch2d_paired_v100_20261018/`. They are not completed scientific results
and do not replace the current PDE figures.
All raw samples from the preceding parallel trial are retained separately.

## Section 6.2.4 prime single-jet measurements

`torch_pinn/prime_single_jet.py` implements one distinct prime order for each
of the k derivative occurrences, including repeated coordinate indices. The
sixth-order schedule is exactly Table 1: 71, 73, 79, 83, 103, 151, summing to
560. The paper does not tabulate orders eight and nine. Fixed seeded uniform
rejection searches extend the construction for those orders; the first
accepted lists sum to 12,228 and 70,469. The full intervals, attempts, seeds,
precomputation times and exact uniqueness certificates are recorded. These
lists are neither author-supplied nor asserted to be optimal.

The input curve is represented with ordinary coefficients,
`x(t)=x+sum_i e_alpha[i]*t**p_i`. Its raw tangent at order p_i is implicitly
`p_i!*e_alpha[i]`. Exact unbounded-integer DP verifies that the only
nonnegative solution of `sum_i p_i*m_i=sum_i p_i` is the all-ones vector.
Consequently the normalized output coefficient q_L is the target derivative
itself, with no floating-point factorial conversion or underflow.

All three methods in this campaign—WDD with conjugate reuse, Section 6.1,
and Section 6.2.4—use the same `torch_pinn/sparse_jets.py` backend. It skips
only structural coefficients outside `S intersect (L-S)`, where S is the
additive semigroup of input orders, and uses vectorized coefficient-pair
contractions for the normalized tanh rule. This retains 64, 256 and 512
orders for the prime schedules, respectively, without pretending to execute
every coefficient up to L. The same preparation, target pruning and
contractions apply to the linear-jet methods. These new times should be
compared within this campaign, rather than directly to the earlier eager
scalar-sum backend.

The CLI stages `stdepp_prime_campaign.py preflight`, `pilot`, and
`cell --index 0..5` each take `--out ROOT`. Successful matching-source gates
are required. Every method uses three paired seeds, ten warm-ups and thirty
synchronized wall-clock calls; all six method permutations occur five times.
Prime selection, certification, dtype conversion and support preparation are
recorded separately from evaluation. The V100 campaign runs all six cases
sequentially in one GPU/ten-CPU allocation, using a fresh process per case.
Only values and metadata are stored; no PDE or parameter training is run.
These finite validation checks do not establish an asymptotic complexity
claim or reproduce the official JAX system's runtime.
