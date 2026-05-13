# Numerical experiment plan

All experiments serve one goal: evaluate exact deterministic computation of **one single monomial partial derivative** and its use in PINN residuals.

---

## 1. Baselines

### Exact deterministic baselines

1. `direct_autodiff`  
   Nested coordinate automatic differentiation.  This is the exact reference.

2. `polarization_jet`  
   Real polarization directions evaluated by Taylor jets.  This is the exact real directional baseline.

3. `waring_complex_jet`  
   Complex monomial Waring directions evaluated by Taylor jets.  This is the proposed rank-aware exact method.

4. `auto`  
   Pattern-based selection between `waring_complex_jet` and `polarization_jet`.

### Approximate deterministic baseline

5. `finite_difference_directional`  
   Central finite-difference stencils along directional schedules.  Optional speed/accuracy baseline.

### Monte Carlo baseline

6. `gaussian_hermite_mc`  
   A direct Gaussian-Hermite estimator for one multi-index:

\[
\partial^\alpha u(x)\approx \sigma^{-p}\frac1K\sum_{k=1}^K H_\alpha(Z_k)u(x+\sigma Z_k).
\]

This baseline is included to compare exact deterministic schedules against a simple randomized derivative estimator in accuracy-cost terms.

---

## 2. Direct derivative benchmark suite

### Pattern grid

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

### Metrics

Value mode:

- relative error against direct AD;
- median evaluation time;
- peak memory;
- direction count;
- Monte Carlo mean/std across repeated samples when applicable.

Backward mode:

- parameter-gradient relative error against direct AD;
- median time for `loss = partial.real.square().mean(); loss.backward()`;
- peak memory;
- gradient variance for Monte Carlo baseline.

### Scaling axes

- derivative order \(p\);
- exponent pattern;
- input dimension \(d\);
- batch size;
- network width/depth;
- dtype fp64/fp32;
- Monte Carlo sample count \(K\);
- smoothing radius \(\sigma\).

---

## 3. Monte Carlo Hermite accuracy-cost experiments

### Test patterns

```text
112, 1122, 112233, 123456, 11112222, 11223344
```

### Sweep variables

- \(K\in\{16,64,256,1024,4096\}\);
- \(\sigma\in\{0.02,0.05,0.1,0.2\}\);
- dtype fp32/fp64;
- value and backward mode.

### Expected plots

- error vs wall-clock time;
- error vs sample count;
- gradient variance vs sample count;
- bias vs \(\sigma\).

---

## 4. PINN case studies

PINN experiments should be added after direct derivative benchmarks are stable.

### PDE design principle

Use manufactured PDEs of the form

\[
\partial^\alpha u + \text{lower-order terms}=f
\]

where the residual contains one dominant high-order monomial partial.

### Candidate PDE families

1. Pure-power derivative:

```text
u_xxxxxx, u_xxxxxxxx
```

2. Repeated mixed derivative:

```text
u_xxyy, u_xxyyzz, u_xxxxyy
```

3. Square-free stress test:

```text
u_xyztuv, u_abcdefgh
```

### PINN methods

- residual with direct AD;
- residual with Waring/Taylor backend;
- residual with polarization/Taylor backend;
- residual with Gaussian-Hermite Monte Carlo estimator;
- optional finite-difference residual.

### Metrics

- relative solution error;
- residual loss;
- wall-clock to target error;
- fixed-time final error;
- training stability across seeds;
- GPU peak memory;
- gradient variance for randomized residuals.

---

## 5. Reporting plan

Main paper should report compact tables/figures:

1. Direction-count table by pattern.
2. Value-mode runtime/error table.
3. Backward-mode runtime/error table.
4. Monte Carlo Hermite accuracy-cost curves.
5. PINN error-vs-wall-clock curves.
6. Backend-selection success/failure table.

Full raw grids should be kept as CSV/JSON in the repository.
