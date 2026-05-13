# Complex Waring step profile

Date: 2026-05-14

Goal: identify which part of `waring_complex_jet` is expensive.

## Setup

```text
alpha=11223344
order=8
complex rank=27
d=8
B=4
hidden=64
depth=3
activation=tanh
dtype=float64 / complex128
GPU=T4
```

Script:

```text
experiments/profile_complex_waring_steps.py
```

Raw outputs:

```text
results/profile_11223344_tanh_value.csv
results/profile_11223344_tanh_backward.csv
```

## Value mode

Total: `10.844 ms`

| step | ms | percent |
|---|---:|---:|
| layer 1 tanh | 2.203 | 20.31% |
| layer 3 tanh | 2.167 | 19.98% |
| layer 5 tanh | 2.165 | 19.97% |
| layer 4 linear 64x64 | 1.421 | 13.10% |
| layer 2 linear 64x64 | 1.421 | 13.10% |
| layer 0 linear 8x64 | 0.548 | 5.06% |
| layer 6 linear 64x1 | 0.520 | 4.80% |
| direction generation | 0.228 | 2.10% |
| prepare terms | 0.108 | 0.99% |
| weighted sum | 0.064 | 0.59% |

Value-mode interpretation:

- Almost all time is Taylor-jet forward evaluation.
- Tanh layers dominate the forward pass for this architecture.
- Direction generation, tensor preparation, and weighted summation are negligible.

## Backward mode

Total: `33.181 ms`

| step | ms | percent |
|---|---:|---:|
| backward | 22.014 | 66.34% |
| layer 1 tanh | 2.266 | 6.83% |
| layer 3 tanh | 2.234 | 6.73% |
| layer 5 tanh | 2.218 | 6.68% |
| layer 4 linear 64x64 | 1.466 | 4.42% |
| layer 2 linear 64x64 | 1.465 | 4.42% |
| layer 0 linear 8x64 | 0.545 | 1.64% |
| layer 6 linear 64x1 | 0.523 | 1.58% |
| direction generation | 0.275 | 0.83% |
| prepare terms | 0.112 | 0.34% |
| weighted sum | 0.064 | 0.19% |

Backward-mode interpretation:

- The dominant cost is not direction generation or Waring summation.
- The dominant cost is PyTorch autograd through the complex Taylor-jet graph.
- In value mode, activation recurrences are the largest forward components.
- In backward mode, the backward pass through all complex operations accounts for about two thirds of runtime.

## Optimization implications

1. Caching Waring directions is useful for cleanliness but will not materially improve runtime.
2. Optimizing weighted sum will not matter.
3. The main value-mode target is fused/compiled Taylor-jet forward, especially activation recurrences.
4. The main backward-mode target is reducing complex autograd graph cost.
5. Potential optimizations:
   - `torch.compile` the Taylor-jet graph for fixed shapes;
   - custom fused activation-jet kernels;
   - custom backward for Taylor-jet primitives;
   - avoid complex backend in backward mode unless rank reduction is large;
   - explore real Waring formulas for repeated patterns to keep low rank without complex autograd.
