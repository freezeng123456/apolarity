# Numerical experiment plan

This plan is limited to **single monomial partial derivatives** and PDE/PINN examples whose residual contains such terms.

---

## 1. Experiment goals

1. Test whether Waring directional schedules compute single high-order mixed partials faster than nested coordinate AD.
2. Compare deterministic exact computation against stochastic baselines such as STDE-style estimators.
3. Test whether exact derivative computation improves PINN training accuracy or accuracy-cost tradeoffs for PDEs with single monomial derivative terms.
4. Identify regimes where the method should not be used, especially square-free high-order patterns.

---

## 2. Baselines

### 2.1 Exact deterministic baselines

#### Direct nested coordinate AD

Reference method.  Computes \(\partial^\alpha u\) by repeated coordinate `autograd.grad`.

Use for:

- exact value reference;
- exact gradient reference;
- runtime/memory baseline.

#### Real polarization + Taylor jet

Exact real-direction baseline.  Uses polarization identity and Taylor-mode AD.

Use for:

- fair deterministic directional baseline;
- square-free patterns where complex Waring has no rank advantage.

#### Nested scalar reverse / JVP

Optional low-level baseline for \(T_p(x;v)\), not necessary for every main table.

---

### 2.2 Approximate deterministic baselines

#### Finite difference directional derivative

Use central finite-difference stencils along polarization/Waring directions.

Purpose:

- speed/accuracy tradeoff baseline;
- show exact methods avoid truncation error.

Not a primary method.

---

### 2.3 Stochastic baselines

#### STDE-style single-monomial estimator

For one multi-index \(\alpha\), use the Gaussian Stein/Hermite identity:

\[
\partial^\alpha u(x)
=\sigma^{-p}\mathbb E\left[H_\alpha(Z)u(x+\sigma Z)\right]
\]

or its centered/symmetric/Taylor-corrected variants when applicable.

This provides a stochastic single-term analogue of STDE/randomized derivative estimation.

Compare against Waring/Taylor exact methods in terms of:

- bias vs \(\sigma\);
- variance vs sample count \(K\);
- wall-clock time;
- peak memory;
- value relative error;
- gradient noise.

#### Official STDE baseline where feasible

If the official STDE implementation can express the desired single monomial operator as a contraction, include it as a baseline.

Important distinction:

- STDE is designed for arbitrary differential-operator contractions and stochastic amortization.
- Our method targets exact deterministic computation of one single monomial partial.

Therefore comparisons should be framed as accuracy-cost tradeoffs, not as identical problem classes.

---

## 3. Direct derivative benchmark suite

### 3.1 Pattern grid

Use expanded one-based alpha strings.

Order 3:

```text
111, 112, 123
```

Order 4:

```text
1111, 1112, 1122, 1123, 1234
```

Order 6:

```text
111111, 111122, 112233, 123456
```

Order 8:

```text
11111111, 11111122, 11112222, 11223344, 12345678
```

### 3.2 Methods

- `direct_autodiff`
- `polarization_jet`
- `waring_complex_jet`
- `auto`
- `finite_difference_directional` optional
- `stde_single_monomial` stochastic baseline

### 3.3 Metrics

Value mode:

- relative error against direct AD;
- median evaluation time;
- peak memory;
- direction count;
- stochastic estimator mean/std if applicable.

Backward mode:

- relative error of parameter gradients against direct AD;
- median time for `loss = partial.square().mean(); loss.backward()`;
- peak memory;
- gradient variance for stochastic baselines.

### 3.4 Scaling axes

- derivative order \(p\);
- active exponent pattern;
- input dimension \(d\);
- batch size;
- network width/depth;
- dtype fp64/fp32;
- STDE sample count \(K\);
- smoothing radius \(\sigma\) for stochastic estimators.

---

## 4. STDE-style single-monomial derivative experiments

### 4.1 Test cases

Use patterns where exact direct AD is still available as reference:

```text
112, 1122, 112233, 123456, 11112222, 11223344
```

### 4.2 STDE estimator variants

1. Raw Hermite/Stein estimator.
2. Symmetric centered estimator for even order.
3. Taylor-corrected estimator if implementation time permits.
4. Official STDE implementation if it supports the operator cleanly.

### 4.3 Expected comparison

Waring/Taylor exact methods should have:

- no sampling variance;
- no \(\sigma\)-bias;
- stable gradients;
- possibly higher per-sample deterministic cost than one stochastic sample, but better accuracy-cost at target tolerances.

STDE-style methods may win when:

- dimension is extremely high;
- approximate derivative is sufficient;
- one evaluates broad operator contractions rather than one coordinate partial.

---

## 5. PINN case studies

We should include PINN experiments only after direct derivative benchmarks are stable.

### 5.1 PDE design principles

Use manufactured solutions and residuals with a single dominant high-order monomial derivative term:

\[
\partial^\alpha u + \text{lower-order terms} = f.
\]

The residual should isolate the derivative-computation method rather than introduce unrelated PDE complications.

### 5.2 Candidate PDE families

#### High-order 1D/low-support equation

Example residual contains:

```text
u_xxxxxx or u_xxxxxxxx
```

Good for pure-power patterns.

#### Mixed derivative PDE

Example residual contains:

```text
u_xxyy, u_xxyyzz, u_xxxxyy
```

Good for repeated-index mixed patterns.

#### Square-free stress test

Example residual contains:

```text
u_xyztuv or u_abcdefgh
```

Used to demonstrate limitation and backend selection.

### 5.3 PINN methods to compare

- exact residual with direct AD;
- exact residual with Waring/Taylor backend;
- exact residual with polarization/Taylor backend;
- stochastic residual with STDE-style estimator;
- optional finite-difference residual.

### 5.4 PINN metrics

- relative solution error;
- residual loss;
- wall-clock to target error;
- final error under fixed time budget;
- training stability across seeds;
- GPU peak memory;
- gradient variance for stochastic methods.

---

## 6. Reporting plan

Main paper should include only compact tables and figures:

1. Direction-count table by pattern.
2. Value-mode runtime/error table.
3. Backward-mode runtime/error table.
4. STDE accuracy-cost curves for selected single monomial patterns.
5. PINN error-vs-wall-clock curves.
6. Backend-selection success/failure table.

Extended raw grids should go to appendix and CSV files in the repository.
