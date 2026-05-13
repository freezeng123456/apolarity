# JSC paper outline plan

Target journal: **Journal of Scientific Computing**.

This is an outline plan only.  It fixes the paper-level section titles and the role of each section before writing details.

The structure follows typical JSC numerical-method papers: a short introduction, mathematical formulation, numerical method/algorithm, numerical experiments, discussion, conclusion.  Related work is folded into the Introduction unless it becomes too long.

---

## Working title

**Exact Single-Monomial High-Order Partial Derivatives via Waring Directional Schedules and Taylor-Mode Automatic Differentiation**

---

## Proposed main section titles

### 1. Introduction

Purpose:

- motivate high-order single mixed partial derivatives in scientific ML/PINNs;
- distinguish single monomial partials from contractable differential-operator sums;
- summarize limitations of nested AD, polarization, stochastic estimators, and operator-level methods;
- state contributions.

The related-work discussion should be placed here in compact paragraphs:

- higher-order AD and Taylor-mode AD;
- PINN derivative bottlenecks;
- STDE and stochastic contraction estimators;
- DOF/forward differential-operator methods;
- monomial Waring rank theory.

### 2. Single-Monomial Partials as Directional Coefficient Extraction

Purpose:

- define \(T_p(x;v)=p!^{-1}D^p u(x)[v,\ldots,v]\);
- show \(\partial^\alpha u\) is a coefficient of the homogeneous polynomial \(T_p(x;\sum z_i e_i)\);
- define a directional schedule for one multi-index;
- introduce polarization as the basic real schedule.

### 3. Waring Directional Schedules

Purpose:

- present the roots-of-unity construction for one multi-index;
- prove coefficient filtering in derivative language;
- state complex-rank optimality via monomial Waring rank/apolarity;
- give the pattern taxonomy: pure, repeated binary, repeated multi-support, square-free.

### 4. Taylor-Mode Evaluation and Backend Selection

Purpose:

- describe Taylor-jet evaluation for directional probes;
- describe complex-valued directions and real final outputs;
- explain value and parameter-gradient computation;
- define the practical backends:
  - `direct_autodiff`,
  - `polarization_jet`,
  - `waring_complex_jet`,
  - `auto`;
- give the rank-threshold selection rule.

### 5. Numerical Experiments

Purpose:

- verify value and gradient accuracy;
- compare runtime and memory in value mode;
- compare runtime and memory in backward/training mode;
- compare against STDE-style baselines where applicable;
- include a small PINN case study for PDEs whose residual contains single monomial partials.

Subsections should be limited to:

#### 5.1 Experimental Setup

Hardware, implementation, architecture, dtype, timing, memory measurement.

#### 5.2 Direct Derivative Benchmarks

Single-monomial derivative value/backward benchmarks against exact and stochastic baselines.

#### 5.3 PINN Case Studies

Training comparisons on PDE examples containing high-order single monomial partials.

#### 5.4 Ablation and Scaling

Rank, order, batch, network size, dtype, and backend-selection threshold.

### 6. Discussion and Limitations

Purpose:

- summarize when Waring schedules win;
- explain square-free limitations;
- discuss complex arithmetic overhead;
- discuss relation to STDE and why exact single partials are a different target;
- discuss future real Waring schedules and broader primitive support.

### 7. Conclusion

Purpose:

- concise restatement of the method and empirical conclusion;
- no new experiments or new theory.

---

## Appendix titles

Appendices should hold technical material, not main narrative.

### Appendix A. Proofs of Directional Schedule Identities

Coefficient extraction, roots-of-unity filtering, relation to monomial Waring rank.

### Appendix B. Taylor-Jet Recurrences

Primitive rules and implementation notes.

### Appendix C. Extended Experimental Tables

Full pattern grids, value/backward, fp32/fp64, memory.

### Appendix D. Additional Baseline Details

STDE estimator setup, finite-difference settings, random seeds, hyperparameters.

---

## Preferred final table/figure list

1. Table: pattern taxonomy and direction counts.
2. Table: direct derivative benchmark, value mode.
3. Table: direct derivative benchmark, backward mode.
4. Figure: speedup vs order/pattern.
5. Figure: runtime vs direction count/rank.
6. Table/Figure: STDE accuracy-cost tradeoff on single monomial derivatives.
7. Table/Figure: PINN case-study error vs wall-clock time.

---

## Boundary of the paper

Keep the paper focused.  Do not expand into:

- Laplacian powers;
- trace contractions;
- general differential-operator compression;
- general real Waring rank theory beyond what is needed for context.
