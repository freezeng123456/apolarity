# aploarity

`aploarity` studies exact deterministic computation of a **single monomial partial derivative**

\[
\partial^\alpha u(x)
\]

for neural networks, using monomial Waring directions and Taylor-mode automatic differentiation.

Scope:

- single multi-index / single mixed partial only;
- no Laplacian powers, trace contractions, or contractable operator sums;
- no stochastic derivative estimation;
- value and parameter-gradient computation via Taylor jets.

## Core idea

For \(p=|\alpha|\), define

\[
T_p(x;v)=\frac{1}{p!}D^p u(x)[v,\ldots,v].
\]

A monomial Waring decomposition gives directions and weights such that

\[
\partial^\alpha u(x)=\sum_r c_r T_p(x;v_r).
\]

`aploarity` computes the directional Taylor coefficients with a Taylor-jet forward pass for `Linear/Tanh` MLPs.

## Repository layout

```text
src/aploarity/
  waring.py          # complex monomial Waring directions
  real_waring.py     # real-direction fallback/generators
  taylor_jet.py      # Taylor-mode AD for Linear/Tanh MLPs
  operators.py       # single-monomial partial APIs
experiments/
  benchmark_single_monomial.py
docs/
  theory/
  plans/
tests/
```

## Quick benchmark

```bash
python experiments/benchmark_single_monomial.py \
  --device auto --dtype float64 --d 8 --batch 8 \
  --hidden 64 --depth 3 --alphas '111111;111122;112233;12345678'
```

## Current research target

The likely JSC contribution is an exact, deterministic backend for single high-order mixed partials:

```text
monomial Waring schedule + Taylor-mode AD + automatic pattern-based backend selection
```

The method is expected to be strongest for high-order repeated-index multi-indices where the complex Waring rank is much smaller than polarization direction count.
