# Custom autograd VJP analysis for Taylor-jet activations

Date: 2026-05-14

## Question

Can optimization candidate C — a custom `torch.autograd.Function` for Taylor-jet primitives — be implemented, and does it help complex Waring backward?

## Key observation

For an activation

\[
y(t)=\phi(x(t)),
\qquad x(t)=\sum_{k=0}^p x_k t^k,
\]

let

\[
q(t)=\phi'(x(t))=\sum_{k=0}^p q_k t^k.
\]

A perturbation satisfies

\[
\delta y(t)=q(t)\delta x(t).
\]

Therefore

\[
\delta y_k=\sum_{j=0}^k q_j\delta x_{k-j}.
\]

Given output adjoints \(g^y_k\), the VJP is

\[
g^x_m=\sum_{j=0}^{p-m}\overline{q_j}\,g^y_{m+j}.
\]

The conjugate is required for complex PyTorch gradients.

This makes custom activation VJP feasible and simple: save the derivative-jet coefficients \(q_j\) during forward, and compute the above convolution in backward.

## Implemented

File:

```text
src/apolarity/taylor_jet.py
```

Implemented custom VJP for:

- `tanh`
- `sigmoid`
- `sin`

Tests:

```text
tests/test_activation_custom_vjp.py
```

The tests compare custom VJP gradients against nested scalar autograd and pass at fp64 tolerance.

## Benchmark effect

Setup:

```text
d=8, B=4, hidden=64, depth=3, activation=tanh, dtype=float64, measure=backward
```

Selected results after custom VJP:

| alpha | complex rank | complex Waring ms | polarization ms |
|---|---:|---:|---:|
| `111122` | 5 | 10.04 | 9.37 |
| `112233` | 9 | 10.77 | 10.28 |
| `11223344` | 27 | 18.79 | 17.41 |
| `12345678` | 128 | 20.14 | 17.27 |

Before custom VJP, the same tanh backward workloads were roughly:

| alpha | complex Waring ms before |
|---|---:|
| `111122` | ~19.9 |
| `112233` | ~19.8 |
| `11223344` | ~33.6 |
| `12345678` | ~37.4 |

Thus custom activation VJP gives about `1.8x--2.0x` speedup for complex Waring backward.

## Profiler after custom VJP

For `alpha=11223344`, rank 27:

Top CUDA cost after custom VJP:

```text
aten::mm / zgemm      dominant
_TanhJetFunction      custom forward/backward visible as fused autograd node
aten::mul/add         still present, but far fewer backward nodes
```

The previous hundreds of `MulBackward0` nodes are replaced by three activation-level custom backward nodes, one per activation layer.

## Remaining bottleneck

After custom VJP, GEMM dominates:

- complex `aten::mm` and `MmBackward0` take most CUDA time;
- activation graph overhead is significantly reduced;
- direction generation and weighted sum remain negligible.

## Conclusion

Candidate C is implementable and effective.

It does not eliminate the complex arithmetic overhead, but it substantially reduces PyTorch autograd graph overhead for Taylor-jet activation recurrences.  This makes complex Waring more viable in backward/PINN workloads.

## Next possible custom-autograd steps

1. Custom Linear jet VJP to reduce Python/autograd overhead around multiple GEMMs.
2. Fused CUDA/Triton activation-jet kernels for value and backward.
3. Mode-aware backend selection after re-benchmarking with custom VJP.
