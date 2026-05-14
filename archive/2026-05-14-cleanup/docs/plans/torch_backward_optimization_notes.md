# PyTorch backward analysis and first optimization

Date: 2026-05-14

## Question

Can we optimize `waring_complex_jet` backward by understanding PyTorch's execution bottlenecks?

## Profiler setup

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
measure=backward
GPU=T4
```

## Torch profiler observation

Top CUDA operators before targeted optimization:

- `aten::mm` dominates CUDA time.
- `MmBackward0` dominates matmul backward time.
- many `MulBackward0` / `aten::mul` calls come from activation Taylor recurrences.
- many `aten::to` / `aten::_to_copy` calls came from repeatedly casting real Linear parameters to complex.

Representative profiler table showed:

```text
aten::mm                  ~59% self CUDA
volta_zgemm_*             dominant GEMM kernels
MulBackward0 / aten::mul  many small elementwise kernels
aten::to/_to_copy         repeated real->complex casts
```

## Source-level cause

The Taylor-jet Linear rule originally cast real weights to complex once per jet order.  For order `p`, every Linear layer has `p+1` jet terms, so this created repeated casts and repeated `ToCopyBackward` graph nodes.

This is unnecessary because all jet terms share the same dtype.

## Optimization implemented

Cast Linear weights and biases once per layer, then reuse them for all jet orders.

Files changed:

```text
src/apolarity/taylor_jet.py
experiments/profile_complex_waring_steps.py
```

## Effect

For `alpha=11223344`, order 8, rank 27, backward mode:

Before:

```text
total   33.181 ms
backward 22.014 ms
```

After cast-once optimization:

```text
total   32.000 ms
backward 21.439 ms
```

The improvement is modest, around `3.6%` total runtime reduction.  The linear layer times decreased, especially small first/last Linear layers, but the main bottleneck remains complex autograd and GEMM/elementwise backward.

## Current bottleneck after optimization

Step profile after optimization:

| step | ms | percent |
|---|---:|---:|
| backward | 21.439 | 67.00% |
| tanh layer 1 | 2.316 | 7.24% |
| tanh layer 3 | 2.279 | 7.12% |
| tanh layer 5 | 2.272 | 7.10% |
| linear 64x64 | 1.330 | 4.16% |
| linear 64x64 | 1.330 | 4.16% |
| direction generation | 0.302 | 0.94% |
| weighted sum | 0.065 | 0.20% |

## Interpretation

The slowdown is not primarily caused by direction generation, Waring summation, or repeated parameter casting.  Those are small.

The dominant cost is still PyTorch autograd through a complex Taylor-jet graph:

1. complex GEMM backward;
2. many small complex elementwise backward kernels from activation recurrences;
3. graph-level overhead from many Taylor recurrence operations.

## Next optimization candidates

1. `torch.compile` for shape-stable Taylor jet.
2. Fused activation-jet kernels for `tanh/sigmoid/sin`.
3. Custom `autograd.Function` for jet activation rules to reduce saved tensors and backward graph size.
4. Real Waring formulas for important repeated patterns to avoid complex backward entirely.
5. Mode-aware backend selection: complex for value mode, real polarization or real Waring for backward unless rank reduction is large.
