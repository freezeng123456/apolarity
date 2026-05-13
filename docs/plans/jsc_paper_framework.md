# JSC paper outline plan

Target journal: **Journal of Scientific Computing**.

This document fixes the paper-level section titles before writing details.  The structure follows a typical JSC numerical-method article: introduction, mathematical formulation, numerical method, numerical experiments, discussion, conclusion.

---

## Working title

**Exact Single-Monomial High-Order Partial Derivatives via Waring Directional Schedules and Taylor-Mode Automatic Differentiation**

---

## Main section titles

### 1. Introduction

Role:

- motivate exact high-order single mixed partials in scientific computing and PINNs;
- explain why a single monomial partial is different from a contractable operator sum;
- position the work relative to nested AD, Taylor-mode AD, polarization, Monte Carlo Hermite estimators, and operator-level acceleration;
- state contributions.

### 2. Single-Monomial Partials as Coefficient Extraction

Role:

- define directional Taylor coefficients

\[
T_p(x;v)=\frac{1}{p!}D^pu(x)[v,\ldots,v];
\]

- show that \(\partial^\alpha u\) is a coefficient of the homogeneous polynomial \(T_p(x;\sum_i z_i e_i)\);
- define a directional schedule for one multi-index;
- introduce real polarization as the basic exact directional schedule.

### 3. Waring Directional Schedules

Role:

- present the roots-of-unity Waring schedule for one multi-index;
- prove coefficient filtering in derivative language;
- state complex-rank optimality through monomial Waring rank / apolarity;
- classify exponent patterns: pure, repeated binary, repeated multi-support, square-free.

### 4. Taylor-Mode Evaluation

Role:

- describe Taylor-jet evaluation of \(T_p(x;v)\);
- explain complex directions and real-valued final outputs;
- explain parameter-gradient computation;
- define practical backends:
  - direct coordinate AD;
  - real polarization + Taylor jet;
  - complex Waring + Taylor jet;
  - automatic pattern selection.

### 5. Numerical Experiments

Role:

All numerical results should be in this one section.  Subsections:

#### 5.1 Experimental Setup

Hardware, software, network architecture, dtype, timing protocol, memory measurement, random seeds.

#### 5.2 Direct Derivative Benchmarks

Value and backward-mode benchmarks for single monomial partials.  Compare exact deterministic methods and Monte Carlo Hermite baselines.

#### 5.3 PINN Case Studies

Manufactured PDEs whose residuals contain one dominant high-order monomial partial.  Compare training accuracy, wall-clock time, and memory.

#### 5.4 Ablation and Scaling

Derivative order, exponent pattern, Waring rank, polarization direction count, batch size, width/depth, dtype, and automatic selection threshold.

### 6. Discussion and Limitations

Role:

- summarize regimes where Waring schedules win;
- explain square-free limitations;
- discuss complex arithmetic overhead;
- discuss Monte Carlo Hermite baselines as accuracy-cost competitors, not as identical methods;
- discuss future real Waring schedules and broader activation support.

### 7. Conclusion

Role:

- concise restatement of the method and empirical findings;
- no new theory or experiments.

---

## Appendix titles

### Appendix A. Proofs of Directional Schedule Identities

Coefficient extraction, roots-of-unity filtering, monomial Waring rank connection.

### Appendix B. Taylor-Jet Recurrences

Primitive rules and implementation notes.

### Appendix C. Extended Numerical Results

Full pattern grids, value/backward, fp32/fp64, memory.

### Appendix D. Baseline and Hyperparameter Details

Monte Carlo Hermite estimator settings, finite-difference settings, seeds, hardware.

---

## Planned main tables and figures

1. Pattern taxonomy and direction counts.
2. Direct derivative benchmark, value mode.
3. Direct derivative benchmark, backward mode.
4. Speedup vs derivative order and exponent pattern.
5. Runtime vs Waring rank / direction count.
6. Monte Carlo Hermite accuracy-cost curves for selected single monomial partials.
7. PINN error vs wall-clock time.

---

## Boundary of the paper

Do not expand into:

- Laplacian powers;
- trace contractions;
- general operator compression;
- general real Waring rank theory beyond necessary context.
