# JSC paper design plan

Target journal: **Journal of Scientific Computing**.

This document is a paper-design blueprint, not a manuscript draft.  It follows the structure commonly used in JSC numerical-method papers: motivation, related work, mathematical formulation, algorithm, analysis/properties, implementation, numerical experiments, and discussion.

---

## 0. Working title and thesis

### Working title

**Exact Single-Monomial High-Order Partial Derivatives via Waring Directional Schedules and Taylor-Mode Automatic Differentiation**

### One-sentence thesis

For a single multi-index derivative \(\partial^\alpha u_\theta(x)\), monomial Waring decompositions provide rank-optimal complex directional schedules, and Taylor-mode automatic differentiation evaluates these directional probes exactly and efficiently for neural-network parametrizations.

### Scope statement

The paper is about **one single monomial partial derivative at a time**.  It is not about Laplacian powers, trace contractions, or general differential-operator sums.

---

## 1. Abstract plan

The abstract should contain exactly these elements:

1. Problem: high-order mixed partial derivatives in scientific ML are expensive with nested automatic differentiation.
2. Gap: existing efficient methods mostly target full operators, contractions, stochastic estimators, or low-order operators; single monomial partials lack a rank-aware deterministic backend.
3. Method: reinterpret monomial Waring decompositions as directional-derivative schedules and evaluate the probes by Taylor-mode AD.
4. Theory: prove the coefficient-extraction identity and complex rank optimality via monomial Waring rank/apolarity.
5. Experiments: compare against nested coordinate AD and polarization on value and backward computations.
6. Result: large speedups for repeated-index high-order derivatives, with square-free limitations clearly characterized.

No detailed numerical claims should be inserted until the final benchmark grid is complete.

---

## 2. Introduction

### 2.1 Opening problem

Scientific computing and scientific machine learning often require derivatives such as

\[
\partial_{i_1}\cdots\partial_{i_p}u_\theta(x)
\]

for neural-network functions.  Standard nested AD is exact but costly for high order.

### 2.2 Why single monomial partials deserve separate treatment

Many works focus on differential operators, contractions, Laplacians, or stochastic estimators.  A single mixed partial has a different algebraic structure: it is a single coefficient of a symmetric high-order derivative tensor.  This coefficient-extraction viewpoint suggests a rank-structured directional-probe strategy.

### 2.3 Main idea

Define

\[
T_p(x;v)=\frac{1}{p!}D^p u_\theta(x)[v,\ldots,v].
\]

Find directions and weights such that

\[
\partial^\alpha u_\theta(x)=\sum_r c_rT_p(x;v_r).
\]

Use monomial Waring decompositions to choose the smallest possible number of complex directions, then evaluate all \(T_p\) by Taylor-mode AD.

### 2.4 Contributions

The contributions should be stated as follows:

1. **Derivative interpretation of monomial Waring decompositions.**  We formulate single mixed partial computation as coefficient extraction from directional Taylor coefficients.
2. **Rank-optimal complex directional schedules.**  We derive a roots-of-unity schedule with \(R_\mathbb C(\alpha)=\prod_{j=1}^n(a_j+1)\) directions and connect its minimality to monomial Waring rank.
3. **Taylor-mode implementation for neural networks.**  We implement the directional probes with Taylor jets, supporting exact value and parameter-gradient computation.
4. **Backend selection and benchmarking.**  We compare nested AD, real polarization, complex Waring schedules, and automatic pattern selection across order, exponent pattern, network size, dtype, and value/backward modes.
5. **Characterization of when the method wins.**  We show that repeated-index high-order derivatives benefit most, while square-free high-order derivatives have no rank advantage over polarization.

---

## 3. Related work

This section should be organized by method class, not chronologically.

### 3.1 Higher-order automatic differentiation

Discuss nested reverse AD, forward-mode/JVP, Taylor-mode AD, JAX jet, TaylorDiff.jl, TorchJet-like systems.  Emphasize that Taylor-mode computes directional high-order derivatives but does not by itself choose optimal directions for a target multi-index.

### 3.2 High-order derivatives in scientific ML and PINNs

Discuss expensive high-order PDE residuals, PINN derivative bottlenecks, and methods that accelerate operator computation.

### 3.3 Differential-operator acceleration

Discuss DOF/Forward-Laplacian-like approaches and related forward-propagation methods.  Distinguish them from single monomial partials.

### 3.4 Stochastic derivative estimators

Discuss STDE and randomized contraction estimators.  Distinguish stochastic contraction from deterministic exact single partial computation.

### 3.5 Waring decompositions and symmetric tensor rank

Introduce monomial Waring rank literature: Carlini--Catalisano--Geramita, Buczyńska--Buczyński--Teitler, Carlini--Kummer--Oneto--Ventura, Han--Moon.  State clearly that this literature was not originally about numerical differentiation.

---

## 4. Mathematical formulation

### 4.1 Single monomial partial as a tensor coordinate

Let \(A=D^p u(x)\) be a symmetric \(p\)-linear form.  A single mixed partial \(\partial^\alpha u(x)\) is a coordinate of \(A\).

### 4.2 Directional Taylor coefficients

Define \(T_p(x;v)\) and show its expansion:

\[
T_p\left(x;\sum_j z_je_{i_j}\right)
=\sum_{|\beta|=p}\frac{\partial^\beta u(x)}{\beta!}z^\beta.
\]

Therefore

\[
\partial^\alpha u(x)=\alpha![z^\alpha]T_p\left(x;\sum_j z_je_{i_j}\right).
\]

### 4.3 Directional schedules

Define a directional schedule as a formula

\[
\partial^\alpha u(x)=\sum_{r=1}^R c_rT_p(x;v_r)
\]

valid for all sufficiently smooth \(u\).

Explain that such schedules are coefficient-extraction formulas for homogeneous polynomials.

---

## 5. Waring directional schedules

### 5.1 Roots-of-unity construction

Let active exponents be

\[
1\le a_0\le a_1\le\cdots\le a_n.
\]

Choose the minimum-exponent variable as the base and define

\[
v_\zeta=e_{i_0}+\sum_{j=1}^n\zeta_je_{i_j},
\qquad \zeta_j^{a_j+1}=1.
\]

Then

\[
\partial^\alpha u(x)=
\frac{\alpha!}{\prod_{j=1}^n(a_j+1)}
\sum_\zeta \left(\prod_{j=1}^n\zeta_j\right)T_p(x;v_\zeta).
\]

### 5.2 Proof as coefficient filtering

Show roots-of-unity orthogonality kills all non-target monomials in the directional Taylor polynomial.

### 5.3 Minimality over complex schedules

State theorem:

\[
R_\mathbb C(\alpha)=\prod_{j=1}^n(a_j+1).
\]

Explain proof path through monomial Waring rank / apolarity.  Avoid overclaiming original algebraic geometry results.

### 5.4 Pattern taxonomy

Include a table:

| pattern | example | complex rank | polarization count | expected behavior |
|---|---|---:|---:|---|
| pure | 11111111 | 1 | small | strongest win |
| repeated binary | 11112222 | 5 | 12 | strong win |
| repeated multi-support | 11223344 | 27 | 40 | moderate win |
| square-free | 12345678 | 128 | 128 | no rank advantage |

---

## 6. Taylor-mode implementation

### 6.1 Jet representation

Represent every intermediate value as

\[
y(t)=y_0+y_1t+\cdots+y_pt^p.
\]

### 6.2 Primitive rules

State rules for `Linear` and `Tanh`.  Keep formulas concise; full implementation details can go to appendix.

### 6.3 Complex-valued directions

Explain why complex directions are valid for analytic activations such as `tanh`, and why the final result is real up to round-off for real-valued networks.

### 6.4 Parameter gradients

The schedule is built from differentiable tensor operations; therefore losses involving \(\partial^\alpha u_\theta\) can be backpropagated to \(\theta\).  Experiments must test both value and backward modes.

### 6.5 Backend selection

Define candidate backends:

- direct coordinate autodiff
- real polarization + jet
- complex Waring + jet
- automatic selection

Initial auto rule:

\[
\text{use complex Waring if }R_\mathbb C(\alpha)\le \tau R_{pol}(\alpha),
\]

with \(\tau\) tuned empirically, initially around 0.7 for backward workloads.

---

## 7. Numerical experiments

This section should be designed as a JSC numerical study, not a software demo.

### 7.1 Experimental setup

Report:

- hardware and GPU
- PyTorch version
- dtype
- network architecture
- batch size
- derivative order
- random seeds
- timing methodology
- memory measurement method

### 7.2 Accuracy verification

Compare every method against direct coordinate autodiff.

Metrics:

\[
\frac{\|y-y_{ref}\|_\infty}{\|y_{ref}\|_\infty+\epsilon}
\]

for both fp64 and fp32.

### 7.3 Value-mode runtime and memory

Benchmark value computation only.  Pattern grid:

- order 3: 111, 112, 123
- order 4: 1111, 1112, 1122, 1123, 1234
- order 6: 111111, 111122, 112233, 123456
- order 8: 11111111, 11111122, 11112222, 11223344, 12345678

### 7.4 Backward-mode runtime and memory

Benchmark

\[
L=\|\partial^\alpha u_\theta(x)\|_2^2
\]

and run `backward()`.  This is the training-relevant experiment.

### 7.5 Scaling studies

Vary:

- derivative order \(p\)
- rank/direction count
- batch size
- network width/depth
- dtype

### 7.6 Ablation studies

Required ablations:

1. direct Waring directions vs old Python merge path
2. complex Waring vs real polarization for same rank patterns
3. auto selection threshold \(\tau\)
4. square-free failure case

### 7.7 Summary tables

Main paper should include compact tables only.  Full CSV-style results go to appendix or repository.

---

## 8. Discussion

### 8.1 When the method is preferable

Repeated-index high-order derivatives where complex rank is significantly below polarization count.

### 8.2 When it is not preferable

Square-free patterns and workloads where complex arithmetic overhead dominates direction-count savings.

### 8.3 Limitations

- current Taylor rules support `Linear/Tanh` only
- complex activations require analytic primitives
- real Waring formulas are future work
- not intended for differential-operator contractions

### 8.4 Broader relevance

Potential use in PINNs, neural PDE solvers, derivative-regularized models, sensitivity analysis, and verification of high-order derivative code.

---

## 9. Conclusion

Restate:

1. single mixed partials can be computed as coefficient extraction from directional Taylor coefficients;
2. monomial Waring rank yields rank-optimal complex schedules;
3. Taylor-mode AD turns these schedules into efficient exact neural-network derivatives;
4. experiments characterize both wins and limitations.

---

## Appendix plan

### Appendix A: Algebraic proofs

- coefficient extraction identity
- roots-of-unity filtering proof
- relation to monomial Waring rank and apolarity

### Appendix B: Taylor jet rules

- Linear
- Tanh
- discussion of analytic activations

### Appendix C: Implementation details

- caching
- complex dtype handling
- timing methodology

### Appendix D: Extended numerical tables

- full pattern sweep
- fp32/fp64
- value/backward
- memory
