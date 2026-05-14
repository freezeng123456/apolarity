# Current performance evaluation

Date: 2026-05-14

This document summarizes the current performance of the single-monomial derivative backends after the engineering optimizations:

1. cast-once Linear parameters;
2. custom activation VJP;
3. merged Taylor-order GEMMs in Linear layers;
4. mode-aware `auto` backend selection.

## Setup

Unless noted otherwise:

```text
GPU: NVIDIA T4
network: Linear/Tanh MLP
hidden=64, depth=3
dtype=float64
```

Raw local result files:

```text
results/0514_final_value_compare.csv
results/0514_final_backward_compare.csv
results/0514_final_value_order10.csv
results/0514_final_backward_order10.csv
results/0514_gaussian_hermite_baseline.csv
results/pinn_111122_*.csv
```

## Accuracy

For deterministic exact backends, relative errors against direct coordinate AD are at floating-point roundoff level:

- fp64 order 6/8: about `1e-15` to `1e-14`;
- square-free high-rank cases can reach about `5e-14` due to larger cancellation;
- no systematic bias observed.

## Value mode, order 6 and 8

Setup: `d=8`, `B=4`, `hidden=64`, `depth=3`, `measure=value`.

| alpha | pattern | complex rank | polarization dirs | direct AD ms | complex Waring ms | auto ms | complex speedup | auto backend |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `111111` | `(6,)` | 1 | 3 | 34.44 | 4.73 | 4.72 | 7.3x | complex |
| `111122` | `(4,2)` | 5 | 7 | 34.57 | 4.80 | 4.80 | 7.2x | complex |
| `112233` | `(2,2,2)` | 9 | 13 | 35.52 | 4.84 | 4.94 | 7.3x | complex |
| `123456` | square-free | 32 | 32 | 34.74 | 4.86 | 4.80 | 7.2x | polarization |
| `11111111` | `(8,)` | 1 | 4 | 279.57 | 6.89 | 6.89 | 40.6x | complex |
| `11112222` | `(4,4)` | 5 | 12 | 279.37 | 6.98 | 6.98 | 40.0x | complex |
| `11223344` | `(2,2,2,2)` | 27 | 40 | 281.13 | 7.09 | 7.08 | 39.7x | complex |
| `12345678` | square-free | 128 | 128 | 277.75 | 11.97 | 7.75 | 23.2x | polarization |

Interpretation:

- Value mode is highly favorable for Waring/Taylor backends.
- Repeated-index order-8 derivatives achieve about `40x` speedup over direct AD.
- Square-free patterns have no rank advantage; `auto` correctly prefers polarization.

## Backward mode, order 6 and 8

Setup: `d=8`, `B=4`, `hidden=64`, `depth=3`, `measure=backward`.

| alpha | pattern | complex rank | polarization dirs | direct AD ms | complex Waring ms | auto ms | complex speedup | auto speedup | auto backend |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `111111` | `(6,)` | 1 | 3 | 103.95 | 8.69 | 7.76 | 12.0x | 13.4x | polarization |
| `111122` | `(4,2)` | 5 | 7 | 102.74 | 8.87 | 7.45 | 11.6x | 13.8x | polarization |
| `112233` | `(2,2,2)` | 9 | 13 | 103.86 | 9.06 | 7.76 | 11.5x | 13.4x | polarization |
| `123456` | square-free | 32 | 32 | 104.16 | 10.08 | 7.76 | 10.3x | 13.4x | polarization |
| `11111111` | `(8,)` | 1 | 4 | 853.27 | 12.37 | 10.59 | 69.0x | 80.5x | polarization |
| `11112222` | `(4,4)` | 5 | 12 | 846.14 | 12.55 | 10.69 | 67.4x | 79.1x | polarization |
| `11223344` | `(2,2,2,2)` | 27 | 40 | 843.55 | 12.97 | 11.23 | 65.1x | 75.1x | polarization |
| `12345678` | square-free | 128 | 128 | 857.87 | 21.52 | 12.62 | 39.9x | 68.0x | polarization |

Interpretation:

- Backward mode is still much faster than direct AD: about `10x--80x` speedup.
- Real polarization is currently the better default for backward/PINN workloads, even when complex rank is smaller, because complex autograd has a larger constant factor.
- `auto` is therefore conservative in backward mode and currently selects polarization.

## Order-10 stress test, value mode

Setup: `d=10`, `B=2`, `hidden=64`, `depth=3`, `measure=value`.

| alpha | pattern | complex rank | direct AD ms | complex Waring ms | speedup |
|---|---|---:|---:|---:|---:|
| `1111111111` | `(10,)` | 1 | 2469.40 | 9.35 | 264.2x |
| `1111111122` | `(8,2)` | 9 | 2474.91 | 9.47 | 261.5x |
| `1111112222` | `(6,4)` | 7 | 2487.06 | 9.43 | 263.7x |
| `1111222333` | `(4,3,3)` | 20 | 2480.03 | 9.46 | 262.0x |
| `1122334455` | `(2,2,2,2,2)` | 81 | 2466.01 | 9.69 | 254.6x |
| `12345678910` | square-free | 512 | 2462.54 | 27.07 | 91.0x |

Interpretation:

- For high order, the method is dramatically faster than nested coordinate AD.
- Repeated-index order-10 cases reach over `250x` value-mode speedup.
- Square-free order-10 remains much more expensive, but still much faster than direct AD.

## Order-10 stress test, backward mode

Setup: `d=10`, `B=2`, `hidden=64`, `depth=3`, `measure=backward`.

| alpha | pattern | complex rank | polarization ms | complex Waring ms | auto ms |
|---|---|---:|---:|---:|---:|
| `1111111111` | `(10,)` | 1 | 16.31 | 16.61 | 16.27 |
| `1111112222` | `(6,4)` | 7 | 16.41 | 16.52 | 16.35 |
| `1122334455` | `(2,2,2,2,2)` | 81 | 21.53 | 19.21 | 18.05 |
| `12345678910` | square-free | 512 | 24.57 | 43.70 | 21.25 |

Interpretation:

- Backward-mode order-10 remains feasible.
- Complex Waring can win when rank reduction is substantial and direction count is high, e.g. `1122334455`.
- Square-free cases should avoid complex Waring.

## Gaussian-Hermite Monte Carlo baseline

Setup: `K=1024`, `sigma=0.1`, `d=8`, `B=4`, `hidden=64`, `depth=3`, `dtype=float64`.

| alpha | Waring complex rel err | Waring complex ms | Gaussian-Hermite rel err | Gaussian-Hermite ms |
|---|---:|---:|---:|---:|
| `112` | 8.03e-16 | 2.82 | 1.23e+03 | 1.43 |
| `1122` | 1.12e-15 | 3.74 | 1.06e+04 | 1.44 |
| `112233` | 1.05e-15 | 5.97 | 4.15e+05 | 1.45 |
| `123456` | 8.19e-15 | 6.51 | 7.89e+06 | 1.43 |

Interpretation:

- The raw Gaussian-Hermite estimator is fast per call but has prohibitive variance/bias at these settings.
- It is useful as an approximate stochastic baseline, but not competitive as an exact derivative method without variance reduction.

## PINN case study

Manufactured PDE residual with one dominant monomial partial `alpha=111122`.

Setup: `d=8`, `steps=80`, `hidden=32`, `depth=2`, `dtype=float32`.

| method | final loss | val rel L2 | median logged backward ms | wall time | peak MB |
|---|---:|---:|---:|---:|---:|
| `direct_autodiff` | 7.539e-4 | 5.895 | 30.59 | 4.26s | 20.31 |
| `polarization_jet` | 7.539e-4 | 5.895 | 6.40 | 1.04s | 21.62 |
| `waring_complex_jet` | 7.539e-4 | 5.895 | 9.51 | 1.36s | 23.60 |
| `auto` | 7.539e-4 | 5.895 | 6.57 | 1.11s | 21.62 |

Interpretation:

- All exact derivative backends train to the same loss and validation error in this small manufactured problem.
- Taylor-jet backends substantially reduce training wall time relative to direct AD.
- `auto` behaves similarly to polarization in backward mode, which is currently the right choice for this pattern.

## Engineering impact

The current results include the following optimizations:

1. cast real Linear parameters to complex once per layer;
2. custom activation VJP for `tanh`, `sigmoid`, and `sin`;
3. merged Taylor-order GEMMs in Linear layers;
4. mode-aware `auto` backend selection.

Effect of the main optimizations on the representative complex backward case `alpha=11223344`:

| version | time |
|---|---:|
| naive complex Taylor jet | ~33 ms |
| custom activation VJP | ~19 ms |
| merged Linear GEMM | ~13 ms |

## Overall conclusion

1. The method is accurate to roundoff for deterministic exact backends.
2. Value-mode computation is the strongest setting for complex Waring schedules.
3. Backward/PINN workloads benefit strongly from Taylor jets but currently prefer real polarization unless complex rank reduction is large.
4. High-order repeated-index derivatives show the largest gains, reaching over `250x` value-mode speedup at order 10.
5. Square-free patterns are the main limitation because Waring rank equals polarization direction count.
