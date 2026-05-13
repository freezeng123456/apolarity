# Initial numerical experiment results

Date: 2026-05-14

These are preliminary results for single-monomial derivative benchmarks and a small PINN case study. Raw CSV/JSON files are stored locally under `results/` and are intentionally not committed.

## Setup

- GPU: NVIDIA T4
- Network: `Linear/Tanh` MLP
- Main value benchmark: `d=8`, `B=8`, `hidden=64`, `depth=3`, `dtype=float64`
- Main backward benchmark: `d=8`, `B=4`, `hidden=64`, `depth=3`, `dtype=float64`

## Value-mode summary, order 6 and 8

| alpha | complex rank | rel err | direct AD ms | Waring complex ms | speedup |
|---|---:|---:|---:|---:|---:|
| `111111` | 1 | 1.14e-15 | 34.40 | 5.62 | 6.1x |
| `111122` | 5 | 4.14e-16 | 34.34 | 5.65 | 6.1x |
| `112233` | 9 | 8.71e-16 | 34.31 | 5.79 | 5.9x |
| `123456` | 32 | 7.09e-15 | 34.37 | 8.22 | 4.2x |
| `11111111` | 1 | 1.10e-15 | 277.38 | 9.03 | 30.7x |
| `11112222` | 5 | 1.21e-15 | 279.45 | 8.95 | 31.2x |
| `11223344` | 27 | 2.27e-15 | 276.43 | 10.45 | 26.5x |
| `12345678` | 128 | 3.23e-14 | 276.18 | 21.55 | 12.8x |

## Backward-mode summary, order 6 and 8

| alpha | complex rank | rel err | direct AD ms | Waring complex ms | speedup |
|---|---:|---:|---:|---:|---:|
| `111111` | 1 | 8.12e-16 | 101.14 | 18.28 | 5.5x |
| `111122` | 5 | 5.82e-16 | 101.16 | 18.57 | 5.4x |
| `112233` | 9 | 1.05e-15 | 100.32 | 18.97 | 5.3x |
| `123456` | 32 | 8.19e-15 | 100.58 | 21.68 | 4.6x |
| `11111111` | 1 | 1.20e-15 | 834.96 | 28.84 | 29.0x |
| `11112222` | 5 | 1.24e-15 | 833.72 | 28.69 | 29.1x |
| `11223344` | 27 | 2.27e-15 | 824.85 | 33.14 | 24.9x |
| `12345678` | 128 | 5.29e-14 | 831.73 | 37.75 | 22.0x |

## New value-mode benchmark, order 10

Setup: `d=10`, `B=2`, `hidden=64`, `depth=3`, `dtype=float64`.

| alpha | pattern | complex rank | rel err | direct AD ms | Waring complex ms | speedup |
|---|---|---:|---:|---:|---:|---:|
| `1111111111` | `(10,)` | 1 | 5.19e-15 | 2486.08 | 13.98 | 177.8x |
| `1111111122` | `(8,2)` | 9 | 3.97e-15 | 2474.15 | 13.96 | 177.2x |
| `1111112222` | `(6,4)` | 7 | 2.43e-15 | 2477.32 | 14.14 | 175.2x |
| `1111222333` | `(4,3,3)` | 20 | 3.59e-16 | 2476.16 | 14.41 | 171.9x |
| `1122334455` | `(2,2,2,2,2)` | 81 | 1.19e-15 | 2475.76 | 16.60 | 149.1x |
| `12345678910` | square-free | 512 | 3.62e-14 | 2484.31 | 31.82 | 78.1x |

Order-10 value-mode confirms that complex Waring schedules remain efficient for substantially more complex derivatives.  Even the square-free case is much faster than direct AD, but it has no rank advantage and uses much more memory.

## New backward-mode benchmark, order 10

Setup: `d=10`, `B=2`, `hidden=64`, `depth=3`, `dtype=float64`.

| alpha | pattern | complex rank | rel err | polarization ms | Waring complex ms |
|---|---|---:|---:|---:|---:|
| `1111111111` | `(10,)` | 1 | 5.19e-15 | 34.14 | 43.21 |
| `1111112222` | `(6,4)` | 7 | 2.43e-15 | 35.21 | 43.31 |
| `1122334455` | `(2,2,2,2,2)` | 81 | 1.19e-15 | 40.74 | 48.15 |
| `12345678910` | square-free | 512 | 3.62e-14 | 48.56 | 66.59 |

Order-10 backward-mode shows an important limitation: complex arithmetic overhead can dominate in training/backward mode, even when the complex rank is lower.  The auto-selection rule should be made mode-aware; the current rank-only threshold is too aggressive for backward workloads.

## Gaussian-Hermite Monte Carlo baseline

Setup: `K=1024`, `sigma=0.1`, `d=8`, `B=4`, `hidden=64`, `depth=3`, `dtype=float64`.

| alpha | Waring complex rel err | Waring complex ms | Gaussian-Hermite rel err | Gaussian-Hermite ms |
|---|---:|---:|---:|---:|
| `112` | 8.03e-16 | 2.82 | 1.23e+03 | 1.43 |
| `1122` | 1.12e-15 | 3.74 | 1.06e+04 | 1.44 |
| `112233` | 1.05e-15 | 5.97 | 4.15e+05 | 1.45 |
| `123456` | 8.19e-15 | 6.51 | 7.89e+06 | 1.43 |

The raw Gaussian-Hermite estimator is fast per evaluation but has prohibitive variance/bias at these settings.  It is useful as an accuracy-cost baseline, but not competitive as an exact derivative method.

## PINN case study: alpha `111122`

Manufactured PDE residual with one dominant monomial partial.  Setup: `d=8`, `steps=80`, `hidden=32`, `depth=2`, `dtype=float32`.

| method | final loss | val rel L2 | median logged backward ms | wall time | peak MB |
|---|---:|---:|---:|---:|---:|
| `direct_autodiff` | 7.539e-4 | 5.895 | 30.59 | 4.26s | 20.31 |
| `polarization_jet` | 7.539e-4 | 5.895 | 6.40 | 1.04s | 21.62 |
| `waring_complex_jet` | 7.539e-4 | 5.895 | 9.51 | 1.36s | 23.60 |
| `auto` | 7.539e-4 | 5.895 | 6.57 | 1.11s | 21.62 |

All exact derivative backends train to the same loss and validation error.  Taylor-jet backends reduce wall-clock time substantially relative to direct AD.

## Immediate interpretation

1. Waring complex schedules are highly effective for repeated-index high-order patterns in value mode.
2. Backward/training workloads need mode-aware backend selection because complex arithmetic has a larger constant factor.
3. Square-free patterns should prefer real polarization.
4. Raw Gaussian-Hermite Monte Carlo is not competitive as an exact derivative method without variance reduction.
5. PINN training confirms that exact Taylor-jet derivative backends can replace direct AD without changing the optimization trajectory in the tested manufactured problem.
