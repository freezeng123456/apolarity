# Activation effect analysis

Date: 2026-05-14

Question: is the backward slowdown of `waring_complex_jet` caused by the `tanh` activation rule?

## Experiment

We added Taylor-jet support for three smooth activations:

- `tanh`
- `sigmoid`
- `sin`

Setup:

```text
d=8, B=4, hidden=64, depth=3, dtype=float64, measure=backward, GPU=T4
```

Compared methods:

- `polarization_jet`
- `waring_complex_jet`
- `auto`

Patterns:

```text
111111, 111122, 112233, 123456, 11111111, 11223344, 12345678
```

Raw CSV files:

```text
results/0514_activation_tanh_backward.csv
results/0514_activation_sigmoid_backward.csv
results/0514_activation_sin_backward.csv
```

## Summary

| activation | alpha | polarization ms | complex Waring ms | complex / polarization |
|---|---|---:|---:|---:|
| tanh | `111122` | 14.54 | 19.89 | 1.37 |
| sigmoid | `111122` | 14.20 | 19.71 | 1.39 |
| sin | `111122` | 14.80 | 19.49 | 1.32 |
| tanh | `112233` | 15.21 | 19.80 | 1.30 |
| sigmoid | `112233` | 17.94 | 20.95 | 1.17 |
| sin | `112233` | 17.07 | 19.90 | 1.17 |
| tanh | `11111111` | 22.79 | 30.57 | 1.34 |
| sigmoid | `11111111` | 22.12 | 29.97 | 1.35 |
| sin | `11111111` | 23.75 | 30.50 | 1.28 |
| tanh | `11223344` | 28.02 | 33.65 | 1.20 |
| sigmoid | `11223344` | 27.13 | 35.28 | 1.30 |
| sin | `11223344` | 28.80 | 35.75 | 1.24 |
| tanh | `12345678` | 28.40 | 37.40 | 1.32 |
| sigmoid | `12345678` | 26.78 | 36.91 | 1.38 |
| sin | `12345678` | 29.64 | 40.17 | 1.36 |

## Interpretation

The slowdown is **not specific to tanh**.

Across `tanh`, `sigmoid`, and `sin`, `waring_complex_jet` is consistently about `1.17x--1.39x` slower than real `polarization_jet` in backward mode.

Therefore the main cause is likely:

1. complex-valued autograd overhead;
2. complex matmul and complex elementwise backward constants;
3. larger complex activation/gradient storage;
4. not the specific `tanh` recurrence.

## Consequence for backend selection

The `auto` rule should be mode-aware:

- in value mode, complex Waring is often attractive when rank is lower;
- in backward/PINN mode, complex Waring should require a stronger rank advantage.

A preliminary rule:

```text
value mode:    use complex if R_complex <= 0.8 * R_polarization
backward mode: use complex if R_complex <= 0.5 * R_polarization
```

This should be tuned using a larger benchmark grid.
