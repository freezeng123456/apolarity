# Engineering optimization summary for Waring/Taylor single-partial derivatives

Date: 2026-05-14

This document consolidates the engineering optimization path and results for the `waring_complex_jet` backend.  It is intended as a source for the implementation and numerical-method discussion in the paper.

---

## 1. Optimization target

The method computes one single-monomial partial derivative via

\[
\partial^\alpha u(x)=\sum_{r=1}^{R}c_rT_p(x;v_r),
\qquad
T_p(x;v)=\frac{1}{p!}D^pu(x)[v,\ldots,v].
\]

For complex Waring schedules, \(v_r,c_r\in\mathbb C\), and the directional Taylor coefficients are evaluated by Taylor-mode automatic differentiation.

The engineering question is: where is the runtime spent, and how can we reduce the cost of value and backward computations?

---

## 2. Initial bottleneck analysis

### 2.1 Step-level profile

Representative case:

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

Before targeted optimizations, step-level profiling gave:

### Value mode

Total: `10.844 ms`

| component | ms | percent |
|---|---:|---:|
| activation jet layers | ~6.54 | ~60% |
| hidden Linear layers | ~2.84 | ~26% |
| first/last Linear layers | ~1.07 | ~10% |
| direction generation | 0.228 | 2.10% |
| prepare terms | 0.108 | 0.99% |
| weighted sum | 0.064 | 0.59% |

### Backward mode

Total: `33.181 ms`

| component | ms | percent |
|---|---:|---:|
| PyTorch backward through complex Taylor-jet graph | 22.014 | 66.34% |
| activation jet forward layers | ~6.72 | ~20% |
| Linear jet forward layers | ~4.00 | ~12% |
| direction generation | 0.275 | 0.83% |
| weighted sum | 0.064 | 0.19% |

### 2.2 Interpretation

The bottleneck is not the Waring decomposition itself:

- direction generation is less than 1--2%;
- coefficient summation is less than 1%;
- tensor preparation is negligible.

The real bottlenecks are:

1. Taylor-jet forward evaluation in value mode;
2. PyTorch autograd through the complex Taylor-jet graph in backward mode.

---

## 3. Activation-function check

We tested three smooth activations:

- `tanh`
- `sigmoid`
- `sin`

Setup:

```text
d=8, B=4, hidden=64, depth=3, dtype=float64, measure=backward, GPU=T4
```

Summary:

| activation | representative complex / polarization slowdown |
|---|---:|
| tanh | 1.20x--1.37x |
| sigmoid | 1.17x--1.39x |
| sin | 1.17x--1.36x |

Conclusion:

The backward slowdown is not specific to the `tanh` recurrence.  It is primarily caused by complex-valued autograd, complex GEMM, complex elementwise kernels, and graph overhead.

---

## 4. Optimization 1: cast real Linear weights once per layer

### 4.1 Problem

For complex directions through a real-valued model, Linear weights must be cast from real to complex.  Initially, the implementation cast weights separately for every jet coefficient.  For order \(p\), this created \(p+1\) casts per Linear layer.

This produced extra `aten::to`, `aten::_to_copy`, and `ToCopyBackward` graph nodes.

### 4.2 Change

Cast each Linear layer's weight and bias once per layer and reuse them for all jet coefficients.

### 4.3 Effect

Representative backward case `alpha=11223344`, order 8, rank 27:

| version | total ms | backward ms |
|---|---:|---:|
| before | 33.181 | 22.014 |
| cast once | 32.000 | 21.439 |

This gives a modest `~3.6%` total runtime reduction.  It removes an unnecessary overhead but does not change the main bottleneck.

---

## 5. Optimization 2: custom VJP for Taylor-jet activations

### 5.1 Mathematical observation

For

\[
y(t)=\phi(x(t)),
\qquad
x(t)=\sum_{k=0}^{p}x_kt^k,
\]

let

\[
q(t)=\phi'(x(t))=\sum_{k=0}^{p}q_kt^k.
\]

A perturbation satisfies

\[
\delta y(t)=q(t)\delta x(t).
\]

Therefore

\[
\delta y_k=\sum_{j=0}^{k}q_j\delta x_{k-j}.
\]

Given output adjoints \(g^y_k\), the vector-Jacobian product is

\[
g^x_m=\sum_{j=0}^{p-m}\overline{q_j}\,g^y_{m+j}.
\]

The conjugate is required for complex gradients.

This means the activation Taylor-jet backward pass can be implemented as a short convolution using saved derivative-jet coefficients \(q_j\), rather than letting PyTorch trace the entire recurrence graph.

### 5.2 Implemented custom VJP

Implemented custom `torch.autograd.Function` for:

- `tanh`
- `sigmoid`
- `sin`

Files:

```text
src/apolarity/taylor_jet.py
tests/test_activation_custom_vjp.py
```

The tests compare custom VJP gradients against nested scalar autograd and pass at fp64 tolerance.

### 5.3 Runtime effect

Setup:

```text
d=8, B=4, hidden=64, depth=3, activation=tanh, dtype=float64, measure=backward
```

| alpha | complex rank | complex Waring before | complex Waring after | speedup |
|---|---:|---:|---:|---:|
| `111122` | 5 | ~19.9 | 10.04 | ~2.0x |
| `112233` | 9 | ~19.8 | 10.77 | ~1.8x |
| `11223344` | 27 | ~33.6 | 18.79 | ~1.8x |
| `12345678` | 128 | ~37.4 | 20.14 | ~1.9x |

### 5.4 Profiler effect

Before custom VJP, the PyTorch graph contained hundreds of elementwise backward nodes such as:

- `MulBackward0`
- `SubBackward0`
- `NegBackward0`

After custom VJP, these are replaced by one activation-level custom backward node per activation layer, e.g.

```text
_TanhJetFunctionBackward
```

The main remaining CUDA cost becomes complex GEMM:

```text
aten::mm / zgemm
MmBackward0
```

---

## 6. Current bottleneck after optimizations

After cast-once and custom activation VJP, the main bottleneck is no longer activation graph overhead.  It is mostly:

1. complex matrix multiplication;
2. complex matrix-multiplication backward;
3. remaining elementwise operations inside custom activation VJP;
4. memory traffic from complex tensors.

Direction generation and weighted summation remain negligible.

---

## 7. Implications for paper and implementation

### 7.1 Paper claims supported

The implementation is not merely a direct application of PyTorch autograd.  It uses method-specific engineering:

1. Waring direction schedules reduce the number of directional probes.
2. Taylor-mode AD avoids nested coordinate derivative graphs.
3. Custom activation VJPs reduce the backward graph for Taylor-jet recurrences.

### 7.2 When complex Waring should be used

- In value mode, complex Waring is often favorable when it reduces rank.
- In backward/PINN mode, complex Waring now becomes much more competitive after custom VJP, but complex GEMM overhead still matters.
- Square-free patterns still prefer real polarization because complex rank gives no direction-count advantage.

### 7.3 Next engineering steps

1. Custom Linear-jet VJP to reduce multi-GEMM autograd overhead.
2. Triton/CUDA fused kernels for activation-jet value and backward passes.
3. Mode-aware backend selection using measured cost models, not rank alone.
4. Real Waring schedules for repeated patterns to keep low rank while avoiding complex GEMM.
5. Optional `torch.compile` for fixed-shape value-mode workloads.

---

## 8. Recommended wording for the paper

A concise implementation paragraph could say:

> We implement directional Taylor probes using a forward Taylor-jet propagation through the network.  A naive PyTorch implementation of the activation recurrences creates a large number of small elementwise backward nodes, especially for complex Waring directions.  We therefore implement custom vector-Jacobian products for the Taylor-jet activation rules.  The custom VJP follows from the identity \(\delta \phi(x(t))=\phi'(x(t))\delta x(t)\), yielding a convolution between the derivative jet and the output adjoints.  This reduces the backward graph from hundreds of scalar elementwise nodes to one activation-level node per layer and improves complex Waring backward time by approximately a factor of two in our benchmarks.
