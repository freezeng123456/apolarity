# STDE/STDE++ benchmark taxonomy and candidate selection

## Scope and source basis

This note inventories the numerical benchmark families in the official STDE implementation and in STDE++. It is intended to guide a category-complete comparison for the apolarity/Waring method. It does not modify the manuscript.

Primary sources used:

- Official STDE repository: <https://github.com/sail-sg/stde>, locally inspected at commit `fae88663b1d2f1b0666d4d250c38a3c34b852ac0`, especially `README.md` and `stde/equations.py`.
- Shi, Hu, Lin, and Kawaguchi, *Stochastic Taylor Derivative Estimator: Efficient amortization for arbitrary differential operators*, NeurIPS 2024: <https://papers.nips.cc/paper_files/paper/2024/file/dd2eb5250696753ea37141bbd89bb569-Paper-Conference.pdf>.
- Shi, Hu, Lin, and Kawaguchi, *STDE++: Polynomial-Time Amortization for Linear Differential Operators*, JMLR 2026: <https://www.jmlr.org/papers/v27/25-1474.html>. The local PDF inspected is `/Users/zenghang/Desktop/STDE++.pdf`.

The central distinction is between (i) an individual partial derivative, (ii) a structured sum such as a Laplacian, (iii) a trace/Hessian-diagonal estimator in high dimension, and (iv) a complete PINN training problem. These are related but should not be presented as one identical benchmark.

## What STDE benchmarks

### 1. Inseparable and effectively high-dimensional PDEs

STDE uses nonlinear, inseparable exact solutions on a high-dimensional unit ball with zero boundary data. The operator families are Poisson, Allen--Cahn, and sine--Gordon:

\[
  \mathcal L u=\Delta u,\qquad
  \mathcal L u=\Delta u+u-u^3,\qquad
  \mathcal L u=\Delta u+\sin u.
\]

The two-body exact solution couples neighboring coordinates through terms such as `sin(x_i+cos(x_{i+1})+x_{i+1}cos(x_i))`; the three-body solution couples triples through `exp(x_i x_{i+1}x_{i+2})`. The reported dimensions range from 100 to 1M, with the headline experiments at 100K and 1M. The derivative order of the PDE operator is two; the differential term is a Laplacian, hence a structured sum of diagonal second derivatives rather than one mixed monomial. STDE samples coordinate terms and is compared with SDGD implementations, forward-over-backward differentiation, and the exact Forward Laplacian. The principal metrics are training speed/iterations per second, peak memory, and solution error (including an Allen--Cahn ablation over randomization batch size). Hutchinson-type estimators are discussed theoretically as dense STDEs, but they are not a separate method row in the main Allen--Cahn performance table.

This is a necessary high-dimensional trace/diagonal-operator category, but it is not an ideal direct test of a method whose theorem targets one partial derivative. A fair comparison must state whether both methods estimate the same Laplacian, whether the random estimator is included, and whether the comparison is for derivative evaluation or end-to-end PDE training.

### 2. Time-dependent semilinear parabolic PDEs

STDE evaluates semilinear heat, Allen--Cahn, and sine--Gordon equations of the form

\[
  \partial_t u=\mathcal L u,
\]

with an analytic initial condition and a test value at terminal time. The reported dimensions are 10, 100, 1K, and 10K. The spatial operator is a Laplacian plus a nonlinear reaction term, while time contributes a first derivative. The code includes `SemilinearHeatTime`, `AllenCahnTime`, and `SineGordonTime`. STDE reports terminal-value error, training throughput, and peak memory. This category tests time dependence and high-dimensional operator estimation, not high-order mixed partial recovery.

### 3. Low-dimensional high-order PDEs

STDE tests mixed derivatives in 2D KdV and 2D Kadomtsev--Petviashvili equations, and a 1D gradient-enhanced KdV equation. The reported equations include terms such as `u_{ty}`, `u_{xxxy}`, `u_{xxxx}`, and nonlinear products such as `(u_y u_x)_x`. The differential operators reach orders three and four. Some constructions use Taylor jets of orders five, seven, nine, or thirteen to encode and separate these lower-order partial derivatives; the jet order must not be confused with the order of the target derivative.

These are the closest STDE benchmarks to apolarity's direct target: low-dimensional, deterministic directional Taylor-jet evaluations of prescribed mixed partials. STDE compares pushforward/jet evaluation against repeated backward-mode AD. STDE++ additionally presents alternative lower-order pushforwards that recover several terms from shared jets. Metrics are speed/iterations per second across network depth and width; the reported baseline is repeated backward AD, with an alternative STDE construction in some tables.

### 4. Gradient-enhanced PINN (gPINN)

For gPINN, STDE differentiates the residual and adds a residual-gradient penalty. In the Allen--Cahn example, the penalty contains third derivatives of the form `\partial_{x_j}\partial_{x_i}^2u`; the associated jet construction reaches order seven when several terms are recovered together. STDE also randomizes the Laplacian coordinate and the penalty indices in the high-dimensional setting. This is an end-to-end training benchmark with an additional parameter-gradient/regularization workload, not merely a forward derivative microbenchmark.

### 5. Weight sharing and implementation ablations

STDE reports weight-sharing block sizes for the inseparable Allen--Cahn task and compares PyTorch, JAX, stacked backward AD, HVP-based parallelization, and sparse STDE. These are implementation/scaling ablations rather than distinct mathematical PDE categories. They should be reported only when the experimental goal is systems scaling.

### 6. STDE++ construction/precomputation check

STDE++ additionally reports a small computational check for its polynomial-time single-jet construction. For derivative orders three through seven, the paper lists sampled primes and the resulting jet orders, and verifies the required uniqueness condition with dynamic programming and a Diophantine solver. This is an algorithm-construction or precomputation experiment, not a PINN benchmark and not a runtime comparison for derivative evaluation. It motivates an order-and-pattern sweep for apolarity, but its numbers should not be mixed with the PDE training results.

## Candidate matrix for apolarity

The following suite covers the important categories while allowing the Waring method to be tested where its mathematical advantage is relevant.

| Category | Candidate task | Why include it | Expected relation to apolarity | Baseline and metrics |
|---|---|---|---|---|
| Pure high-order derivative | `u_{1^p}` for orders 4, 8, and 12 | Isolates high-order Taylor propagation without mixed-index combinatorics | Strong advantage: one direction for every order | Nested AD and Taylor-mode; value/backward time, peak memory, relative error |
| Repeated-index mixed derivative | `u_{11223344}` and `u_{111222333}`; optionally order 6 control | Directly tests the Waring-rank reduction with high order and repeated exponents | Strong advantage: rank 27 vs grouped polarization 80, and rank 16 vs 63 for the two existing targets | Deterministic STDE++/Griewank grouped jets vs apolarity; same precision, batch, network, and hardware |
| Low-order mixed control | `u_{12}`, `u_{112}`, or `u_{1234}` | Prevents a suite consisting only of favorable high-order cases | The advantage may be small or absent; this is an honest control | Nested AD and grouped polarization; time, memory, error |
| Square-free mixed derivative | `u_{123456}` or a moderate-order square-free term | Tests the unfavorable regime in which Waring rank grows exponentially | Important limitation: rank is `2^{p-1}` for a square-free monomial | Include one moderate order only; report direction count and cost, not a claim of universal superiority |
| Structured diagonal operator | Laplacian and a moderate-dimensional sum of diagonal derivatives | Connects to Poisson/Allen--Cahn/Sine--Gordon residuals | Requires operator-level reuse or summation; cannot be inferred from one-monomial rank alone | Compare deterministic apolarity term-by-term with a clearly specified structured baseline; report total residual cost |
| High-dimensional stochastic/trace regime | 10K--100K-dimensional Laplacian on two-body or three-body manufactured solutions | Completes the STDE high-dimensional category | STDE is the natural strong baseline; apolarity should be framed as a deterministic reference or limitation unless an operator-level construction is added | STDE and non-randomized deterministic method, where possible; memory, speed, variance/error vs sample budget |
| High-order PDE residual | 2D KdV/KP-like residual with `u_{ty}`, `u_{xxxy}`, or `u_{xxxx}` | Tests several derivative terms in one residual and shared jet computation | Favorable when terms contain repeated indices, but the full residual also contains nonlinear products | Compare residual evaluation and training cost; separate derivative cost from nonlinear assembly |
| gPINN/parameter-gradient | g-KdV or a manufactured high-order PDE with residual-gradient penalty | Tests reverse differentiation through the derivative estimator | Potential advantage at high order, but custom VJP implementation must be identical in scope | Forward+parameter backward time, peak memory, residual/solution error |

## Recommended minimum experimental package

For a first paper-quality suite, use five layers:

1. **Derivative microbenchmarks:** pure high order, repeated-index mixed high order, and one low-order mixed control.
2. **A limitation control:** one square-free mixed derivative, chosen at moderate order so the comparison remains computationally feasible.
3. **A structured-operator benchmark:** a moderate-dimensional Laplacian or a manufactured PDE residual, with the operator assembly defined explicitly.
4. **A high-dimensional STDE-style benchmark:** one two-body Allen--Cahn or sine--Gordon problem at 10K/100K dimensions, reported primarily as a scope boundary and operator-level comparison.
5. **One end-to-end high-order PINN/gPINN benchmark:** KdV/KP or a manufactured analogue containing repeated-index mixed derivatives.

The strongest headline evidence should come from layers 1 and 5: they match the theorem and the motivation that independent Taylor coefficients can be evaluated in parallel. Layers 2--4 are essential for scientific honesty and category coverage. The high-dimensional trace category should not be advertised as a direct Waring-rank win unless a separate structured-operator algorithm is proved and implemented.

## Fair-comparison rules

- Keep network architecture, parameter values, input batch, precision, compiler warm-up, and hardware fixed.
- Compare deterministic apolarity with deterministic grouped-jet/Taylor baselines for exact derivative recovery. Put randomized STDE in a separate stochastic/operator-level experiment.
- Report both direction/jet count and actual wall-clock cost; a lower direction count does not by itself imply a lower cost for structured sums.
- Separate first-call compilation time from steady-state time, and report forward value evaluation separately from parameter-gradient evaluation.
- For stochastic methods, report sample budget and variance/error, not only mean runtime.
- For complex directions, state the holomorphic-extension assumption and use the same real-valued network class or a controlled holomorphic activation class in all exact comparisons.

## Source pointers

- Official STDE repository README: `work/stde_upstream/README.md`.
- Official STDE equation definitions: `work/stde_upstream/stde/equations.py` (including `highord1d`, `KdV`, `KdV2d`, `SemilinearHeatTime`, and the two-/three-body problems).
- STDE++ Appendix I.1: inseparable Poisson, Allen--Cahn, and sine--Gordon; two-body and three-body exact solutions.
- STDE++ Appendix I.2: time-dependent semilinear parabolic equations.
- STDE++ Appendix I.4: 2D KdV, 2D KP, 1D g-KdV, alternative jet constructions, and amortized gPINN.
- STDE++ Appendix I.4.3: randomized high-dimensional Laplacian and gPINN terms.
