# Initial numerical experiment results

Date: 2026-05-14

These are preliminary results for the single-monomial derivative benchmark. Raw CSV/JSON files are stored locally under `results/` and are intentionally not committed.

## Setup

- GPU: NVIDIA T4
- Network: `Linear/Tanh` MLP
- Main value benchmark: `d=8`, `B=8`, `hidden=64`, `depth=3`, `dtype=float64`
- Main backward benchmark: `d=8`, `B=4`, `hidden=64`, `depth=3`, `dtype=float64`

## Value-mode summary

| alpha | complex rank | exact complex rel err | direct AD ms | Waring complex ms | speedup |
|---|---:|---:|---:|---:|---:|
| `111111` | 1 | 1.14e-15 | 34.40 | 5.62 | 6.1x |
| `111122` | 5 | 4.14e-16 | 34.34 | 5.65 | 6.1x |
| `112233` | 9 | 8.71e-16 | 34.31 | 5.79 | 5.9x |
| `123456` | 32 | 7.09e-15 | 34.37 | 8.22 | 4.2x |
| `11111111` | 1 | 1.10e-15 | 277.38 | 9.03 | 30.7x |
| `11112222` | 5 | 1.21e-15 | 279.45 | 8.95 | 31.2x |
| `11223344` | 27 | 2.27e-15 | 276.43 | 10.45 | 26.5x |
| `12345678` | 128 | 3.23e-14 | 276.18 | 21.55 | 12.8x |

## Backward-mode summary

| alpha | complex rank | exact complex rel err | direct AD ms | Waring complex ms | speedup |
|---|---:|---:|---:|---:|---:|
| `111111` | 1 | 8.12e-16 | 101.14 | 18.28 | 5.5x |
| `111122` | 5 | 5.82e-16 | 101.16 | 18.57 | 5.4x |
| `112233` | 9 | 1.05e-15 | 100.32 | 18.97 | 5.3x |
| `123456` | 32 | 8.19e-15 | 100.58 | 21.68 | 4.6x |
| `11111111` | 1 | 1.20e-15 | 834.96 | 28.84 | 29.0x |
| `11112222` | 5 | 1.24e-15 | 833.72 | 28.69 | 29.1x |
| `11223344` | 27 | 2.27e-15 | 824.85 | 33.14 | 24.9x |
| `12345678` | 128 | 5.29e-14 | 831.73 | 37.75 | 22.0x |

## Gaussian-Hermite Monte Carlo baseline

Setup: `K=1024`, `sigma=0.1`, `d=8`, `B=4`, `hidden=64`, `depth=3`, `dtype=float64`.

| alpha | Waring complex rel err | Waring complex ms | Gaussian-Hermite rel err | Gaussian-Hermite ms |
|---|---:|---:|---:|---:|
| `112` | 8.03e-16 | 2.82 | 1.23e+03 | 1.43 |
| `1122` | 1.12e-15 | 3.74 | 1.06e+04 | 1.44 |
| `112233` | 1.05e-15 | 5.97 | 4.15e+05 | 1.45 |
| `123456` | 8.19e-15 | 6.51 | 7.89e+06 | 1.43 |

The raw Gaussian-Hermite estimator is fast per evaluation but has prohibitive variance/bias at these settings.  It is useful as an accuracy-cost baseline, but not competitive as an exact derivative method.

## Immediate interpretation

1. Waring complex schedules are highly effective for repeated-index high-order patterns.
2. Square-free patterns remain expensive because complex rank equals polarization count.
3. Backward-mode experiments confirm that the method can support training-relevant gradients.
4. The raw Gaussian-Hermite Monte Carlo baseline needs variance reduction to be useful, which strengthens the exact deterministic positioning of the method.
