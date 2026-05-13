# Matrix multiplication optimization analysis

Date: 2026-05-14

## Question

After custom activation VJP, should we optimize matrix multiplication in Taylor-jet Linear layers?

## Idea

A Linear layer on a Taylor jet originally computed one matrix multiplication per Taylor coefficient: `y_k = W x_k`, for `k=0,...,p`.  This creates `p+1` GEMM nodes per Linear layer.

Since all Taylor coefficients have the same shape, we concatenate the Taylor-order dimension into the batch dimension and use one larger GEMM per Linear layer:

```python
flat = torch.cat(jet.terms, dim=0)
yflat = flat @ W.T
out = yflat.split(n, dim=0)
```

## Implemented

Files:

```text
src/apolarity/taylor_jet.py
experiments/profile_complex_waring_steps.py
```

All tests pass:

```text
5 passed
```

## Backward benchmark after merged GEMM

Setup:

```text
d=8, B=4, hidden=64, depth=3, activation=tanh, dtype=float64, measure=backward
```

| alpha | pol dirs | complex dirs | polarization ms | complex Waring ms | complex/pol |
|---|---:|---:|---:|---:|---:|
| `111111` | 3 | 1 | 6.99 | 8.28 | 1.19 |
| `111122` | 7 | 5 | 6.86 | 8.43 | 1.23 |
| `112233` | 13 | 9 | 6.94 | 8.86 | 1.28 |
| `123456` | 32 | 32 | 7.60 | 9.85 | 1.30 |
| `11111111` | 4 | 1 | 10.37 | 11.76 | 1.13 |
| `11112222` | 12 | 5 | 10.43 | 11.99 | 1.15 |
| `11223344` | 40 | 27 | 11.12 | 13.17 | 1.19 |
| `12345678` | 128 | 128 | 12.91 | 22.26 | 1.72 |

Both real polarization and complex Waring become faster after merged GEMMs.  Real polarization benefits strongly because the merged GEMMs are real-valued.

## Step profile after merged GEMM

Representative case:

```text
alpha=11223344, order=8, rank=27, d=8, B=4, hidden=64, depth=3, activation=tanh, measure=backward
```

Total: `27.055 ms`

| step | ms | percent |
|---|---:|---:|
| backward | 18.676 | 69.03% |
| tanh layer 1 | 2.311 | 8.54% |
| tanh layer 3 | 2.295 | 8.48% |
| tanh layer 5 | 2.275 | 8.41% |
| linear 64x64 | 0.373 | 1.38% |
| linear 64x64 | 0.362 | 1.34% |
| direction generation | 0.266 | 0.98% |
| weighted sum | 0.062 | 0.23% |

Linear forward components are no longer large.

## Torch profiler after merged GEMM

For `alpha=11223344`, profiler confirms the change:

```text
aten::mm       11 calls
MmBackward0     4 calls
```

Before merging, the profiler showed many more matmul nodes:

```text
aten::mm       83 calls
MmBackward0    28 calls
```

So the matrix multiplication optimization is effective.

## Remaining bottleneck

After merged GEMMs, the remaining cost is mostly:

1. complex GEMM backward for the merged Linear layers;
2. custom activation VJP elementwise convolutions;
3. complex tensor memory traffic.

Direction generation and weighted summation remain negligible.

## Conclusion

Matrix multiplication optimization was necessary and has been implemented.  It reduces GEMM node count dramatically and improves runtime, but it does not make complex Waring universally faster than real polarization in backward mode.  The next likely targets are fused activation VJP kernels, mode-aware backend selection, and real Waring schedules for important repeated patterns.
