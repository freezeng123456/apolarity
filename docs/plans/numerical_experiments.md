# Numerical experiment design

## Goal

Evaluate exact deterministic computation of a single monomial partial derivative

\[
\partial^\alpha u_\theta(x)
\]

using Waring directional schedules and Taylor-mode AD.

No experiment in this project should benchmark contractable operator sums as the main object.

## Methods

1. `direct_autodiff`
   - nested coordinate `torch.autograd.grad`
   - correctness reference
2. `polarization_jet`
   - real polarization directions
   - Taylor jet evaluation
3. `waring_complex_jet`
   - complex monomial Waring directions
   - Taylor jet evaluation
4. `auto`
   - select `waring_complex_jet` when complex rank is clearly below polarization direction count
   - otherwise select `polarization_jet`
5. Future: `waring_real_jet`
   - real Waring / realification schedules

## Primary metrics

- value relative error against `direct_autodiff`
- gradient relative error against `direct_autodiff` for `loss = partial.square().mean()`
- median wall-clock time
- CUDA peak allocated memory
- direction count
- coefficient norm and cancellation ratio

## Core pattern sweep

Use expanded one-based alpha strings.

### Order 3

```text
111, 112, 123
```

### Order 4

```text
1111, 1112, 1122, 1123, 1234
```

### Order 6

```text
111111, 111122, 112233, 123456
```

### Order 8

```text
11111111, 11111122, 11112222, 11223344, 12345678
```

## Scaling axes

1. derivative order: `p in {3,4,6,8}`
2. exponent pattern: pure, repeated binary, repeated multi-support, square-free
3. network size:
   - small: `hidden=16, depth=2`
   - medium: `hidden=64, depth=3`
   - large: `hidden=128, depth=4`
4. batch size: `B in {1,8,64,256}`
5. dtype: `float64/complex128`, `float32/complex64`
6. mode: `value`, `backward`

## Hypotheses

1. Waring schedules are strongest for repeated-index high-order patterns.
2. Square-free patterns have no rank advantage and should prefer real polarization.
3. Taylor jet plus Waring directions remains much faster than direct autodiff at p >= 6.
4. Complex backward has a constant-factor penalty relative to real polarization, so backend selection should account for direction count ratio.

## Initial auto rule

For backward-mode training use:

```text
if complex_rank <= 0.7 * polarization_dirs:
    use waring_complex_jet
else:
    use polarization_jet
```

This threshold should be tuned from benchmark grids.

## Reproducibility

Every benchmark should write CSV and JSON outputs under `results/` and record:

- git commit
- torch version
- GPU name
- dtype
- method
- alpha pattern
- rank/direction count
- random seed
