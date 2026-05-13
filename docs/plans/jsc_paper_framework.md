# JSC paper framework plan

Working target: Journal of Scientific Computing.

This is a framework only; not a draft.

## Tentative title

Exact Single-Monomial High-Order Partial Derivatives via Waring Directional Schedules and Taylor-Mode Automatic Differentiation

## Core claim

We repurpose monomial Waring decompositions as deterministic directional-derivative schedules for exact computation of single mixed partial derivatives of neural networks.

The method targets one multi-index derivative \(\partial^\alpha u\) at a time, not contractable differential-operator sums.

## Proposed structure

### 1. Introduction

- High-order mixed partials in PINNs and scientific ML
- Cost of nested coordinate autodiff
- Distinction between single monomial partials and differential-operator contractions
- Summary of Waring directional schedule + Taylor-mode implementation

### 2. Background

- Symmetric tensors and high-order directional derivatives
- Taylor-mode AD / jets
- Monomial Waring rank over complex and real fields
- Polarization as a real baseline

### 3. Single-monomial derivative as coefficient extraction

- Define \(T_p(x;v)\)
- Show \(\partial^\alpha u = \alpha![z^\alpha]T_p(x;\sum z_i e_i)\)
- Explain direction schedules as linear coefficient extraction formulas

### 4. Complex Waring directional schedules

- Roots-of-unity construction
- Minimality over complex directions via monomial Waring rank / apolarity
- Direction count formula and pattern examples
- Numerical conditioning notes

### 5. Taylor-jet implementation

- Jet rules for `Linear/Tanh`
- Complex-valued directions and real-valued final output
- Value and parameter-gradient computation
- Backend selection and caching

### 6. Backend selection

- Direct autodiff reference
- Real polarization
- Complex Waring
- Optional real Waring / realification future work
- Proposed rank-threshold selection rule

### 7. Numerical experiments

- Accuracy vs direct autodiff
- Value runtime and memory
- Backward runtime and memory
- Pattern sweep by active exponent structure
- Scaling in order, batch, depth, width, dtype

### 8. Discussion

- When Waring schedules win
- Square-free limitations
- Complex arithmetic overhead
- Relation to STDE, DOF, Taylor-mode neural operators, hyper-dual methods

### 9. Conclusion

- Exact deterministic single-monomial backend
- Future: real Waring formulas, broader primitive support, integration into PINN pipelines

## Differentiation from existing work

- Waring theory exists, but is not originally a derivative-computation method.
- Taylor-mode AD exists, but does not choose monomial-rank-optimal directions.
- STDE handles stochastic contractions, not exact single partials.
- DOF focuses on differential operators, mainly second-order/forward propagation.

## Key tables/figures

1. Pattern table: alpha pattern, complex rank, polarization count, auto choice
2. Runtime vs derivative order
3. Runtime vs rank/direction count
4. Backward memory vs method
5. Square-free failure/limitation plot
6. Accuracy histogram across patterns
