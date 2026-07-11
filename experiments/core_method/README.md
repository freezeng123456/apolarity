# Core-method implementation diagnostics

These scripts diagnose the complex-Waring + Taylor-jet derivative backend.
They are separate from the formal `jsc_v2` evidence pipeline.

| script | role |
|---|---|
| `benchmark_single_monomial.py` | microbenchmark: exact single-monomial high-order derivative backends (naive nested autodiff vs. Taylor-jet vs. complex-Waring). Timing + memory. |
| `profile_complex_waring_steps.py` | step-by-step profiler of complex-direction generation and Taylor-jet evaluation. |
| `train_pinn_monomial.py` | small PINN case study with a single high-order monomial partial in the residual (manufactured \(\partial^\alpha u = f_\alpha\)). |
| `train_pinn_ch_sixth_order.py` | 4D Cahn–Hilliard-type 6th-order PINN diagnostic. |
| `generate_paper_tables.py` | formats diagnostic CSVs into LaTeX rows without recomputation. |

## Status

`data/` is empty. There are no formal results; any associated paper table or
figure is **TBD**. Outputs from these historical scripts are implementation
diagnostics and cannot be used as paper evidence.

The only formal methods are `complex_sinh`, SIREN, mFF-PINN, and
MscaleDNN-2-sin, and the only formal protocol is `jsc_v2`. Formal tasks are
limited to Poly, Chirp, and Maxwell and must use `scripts/run_jsc_main3.sh`.

## Diagnostic commands

```bash
python benchmark_single_monomial.py
python train_pinn_ch_sixth_order.py
python generate_paper_tables.py
```
