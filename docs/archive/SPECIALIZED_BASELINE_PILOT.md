# Specialized baseline pilot

Status: diagnostic only. These methods are not part of the frozen `jsc_v2`
protocol and their pilot outputs must not be merged into formal tables.

The accompanying strict Vanilla PINN control is a four-hidden-layer tanh MLP
using ordinary nested coordinate autodiff. This is kept distinct from the
existing auxiliary `tanh` architecture, which uses the project's common Waring
Taylor-jet backend to isolate architecture effects.

The pilot uses one representative setting per formal PDE family, seed 0, literal
input width 128, the existing paired collocation convention, and a 180-second
training budget:

| task | diagnostic baseline | reason |
|---|---|---|
| `poly_d2_o4` | one-network MIM-p-style mixed residual | lower the largest derivative from order four to order two |
| `chirp_a2` | WIRE-PINN | localized complex Gabor representation for a spatially varying frequency |
| `maxwell_a4` | PWNN | trainable plane-wave basis for a Helmholtz/Maxwell field |

## Poly: mixed residual

The implementation follows the partial mixed formulation in Lyu et al.,
*MIM: A deep mixed residual method for solving high-order partial differential
equations*, JCP 452 (2022), 110930:
<https://arxiv.org/abs/2006.04146>.

For `order=4`, one tanh network outputs `u` and
`v = Delta(u)/(-S)`. The normalized residuals are

```text
Delta(u) / S + v = 0,
Delta(v) / S + f / S^2 = 0,
```

with `u=v=0` on the boundary. The original authors do not provide an official
source repository, so this is a paper-derived implementation rather than an
upstream-code port.

## Chirp: WIRE

Upstream repository: <https://github.com/vishwa91/wire>

Pinned commit: `bf95232e0f60434bcbd9b4398ef4c11490832526` (MIT).

The local adaptation preserves the upstream complex Gabor activation and its
`width / sqrt(2)` capacity adjustment. It changes `float32/complex64` to
`float64/complex128`, sets four total hidden Gabor layers, and evaluates the
Chirp PINN loss with direct coordinate autodiff. These are precision and PDE
interface adaptations; the upstream experiments are image/INR tasks, not PINNs.

## Maxwell: PWNN

The implementation follows Wang, Cui, and Xiang, *A Neural Network with Plane
Wave Activation for Helmholtz Equation*:
<https://arxiv.org/abs/2012.13870>.

It represents the complex field as

```text
E(x) = sum_r c_r exp(i w_r^T x),
```

with trainable complex amplitudes and real wave vectors. Its Laplacian is
evaluated analytically. The original authors do not provide an official source
repository, so this is a paper-derived implementation rather than an
upstream-code port.

## Acceptance checks

- direct Laplacian of a quadratic polynomial;
- WIRE output shape and float64 public output;
- PWNN analytic Laplacian against nested coordinate autodiff;
- one-second smoke run for each baseline before the timed pilots.

## Seed-0, 180-second pilot results

Lower relative held-out L2 error is better. `Tanh + shared jet` is retained as
an architecture-only control; `Vanilla direct AD` is the strict conventional
PINN comparison.

| task | Tanh + shared jet | Vanilla direct AD | Complex Sinh | structured baseline |
|---|---:|---:|---:|---:|
| `poly_d2_o4` | 0.999848 | 0.999251 | **0.037898** | MIM-p 0.618935 |
| `chirp_a2` | 2.436501 | 1.779957 | **0.311201** | WIRE 1.518233 |
| `maxwell_a4` | 3.129766 | 2.216312 | **0.371363** | PWNN 1.127652 |

These are one-seed diagnostics, not paper evidence. They support proceeding to
a preregistered multi-seed comparison, but they do not establish statistical
superiority. In particular, the structured baselines use their own architecture
and derivative implementations, so step counts and parameter counts must be
reported alongside equal-wall-clock accuracy.
