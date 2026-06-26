# Core method experiments (paper Section 5)

The method-level experiments behind the complex-Waring + Taylor-jet derivative
backend, separate from the oscillatory-PDE suite.

| script | role |
|---|---|
| `benchmark_single_monomial.py` | microbenchmark: exact single-monomial high-order derivative backends (naive nested autodiff vs. Taylor-jet vs. complex-Waring). Timing + memory. |
| `profile_complex_waring_steps.py` | step-by-step profiler of complex-direction generation and Taylor-jet evaluation. |
| `train_pinn_monomial.py` | small PINN case study with a single high-order monomial partial in the residual (manufactured \(\partial^\alpha u = f_\alpha\)). |
| `train_pinn_ch_sixth_order.py` | the 4D Cahn-Hilliard-type **6th-order** PINN training run (paper Section 5.4). |
| `generate_paper_tables.py` | formats saved CSVs into LaTeX table rows (no recompute). |

## Outputs
CSV/JSON written to `data/`. Tables are produced by `generate_paper_tables.py`.

## Reproduce
```bash
python benchmark_single_monomial.py      # microbenchmark
python train_pinn_ch_sixth_order.py      # Section 5.4 PINN
python generate_paper_tables.py          # -> LaTeX rows
```
