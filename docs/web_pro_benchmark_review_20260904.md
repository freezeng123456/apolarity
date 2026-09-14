# Pro-generated benchmark review — 2026-09-04

This note records the independent review of the artifact generated in the
separate ChatGPT Pro session.  The ZIP is preserved at
`tmp/apolarity_web_pro_result_20260904.zip` and has
SHA-256
`03275fa1f831e157f55e4a2a735c4150a2f3f612edbd156f48be74c80b1f9592`.

## What was imported

The Pro-generated Torch modules were copied into `src/apolarity/` as
`benchmark_multiindex.py`, `benchmark_derivatives.py`, `benchmark_models.py`,
and `benchmark_residuals.py`.  The two runners are in
`experiments/torch_benchmarks/` and `experiments/jax_benchmarks/`; the frozen
protocols are duplicated in `configs/` and in the JAX experiment directory.
The STDE source remains under `reference/stde_upstream/` and is not imported
by the benchmark implementation.

The existing manuscript changes and the earlier modular draft under
`experiments/benchmarks/` were preserved.  The new files therefore represent
the Pro-generated suite without deleting unrelated work.

The earlier STDE-based comparison harness and its T4 records were subsequently
copied out of the temporary review tree into
`experiments/stde_comparison/`. That directory is an isolated archive and is
not imported by the mainline runners or used in the revised manuscript.

## Review changes applied after generation

The generated implementation was kept as the baseline.  Two small, explicit
review changes were made:

1. PDE residual functions now accept `create_graph`; value-mode Torch timing
   passes `create_graph=False` to the nested baseline, while parameter-gradient
   mode retains `create_graph=True`.
2. The JAX runner and tests were mechanically formatted, and a dead
   `if False` model-evaluation branch plus an unused import were removed.

The first change makes the value/gradient boundary visible in the API and
removes an avoidable parameter graph from the value measurement.  It does not
change the mathematical residual or the rank-optimal schedule.

## Verification

The original ZIP passed `unzip -t`.  The integrated source passed Python
`compileall`.  On the T4 review host (`freezeng-2ho6ltsa0a`, Tesla T4,
15,360 MiB, driver 535.161.07, PyTorch 2.0.1+cu118), the patched source ran
the primary Torch configuration with 32/32 rows successful.  The result
bundle contains the JSON/CSV outputs, frozen YAML, provenance, log, exit code,
and the run card in
`output/web_pro_benchmark_review_20260904/t4_primary_graphfree_20260904/`.
The patched source tree hash recorded by the runner is
`b35cb1791b58a6b4248c1be0965a7a86e41b75fe6b132d42814ee361e3fac308`.

The integrated source also passed 13 active tests and 29 self-contained legacy
core tests on the same host, with non-fatal PyTorch/NumPy and complex-module
warnings.  The JAX code was first exercised on CPU and then launched in the
T4's separate JAX 0.4.30 CUDA environment.  A full primary JAX matrix did not
finish XLA compilation within the deliberately bounded 30-minute observation
window; its raw status and stop reason are preserved under
`output/web_pro_benchmark_review_20260904/t4_jax_mainline_20260904/`, and no
full-matrix JAX timing is claimed.

After the manuscript scope was reduced to the two mainline methods, the
current integrated tree passed 14 active/JAX tests and the same 29 core tests.
The benchmark plotting script generated the T4 runtime, speedup, and allocation figures
under `output/web_pro_benchmark_review_20260904/plots/`, and the paper's
numerical section was rebuilt without an STDE comparison.

The copied isolated STDE adapter was also run independently on the review host;
its eight schedule, residual, and derivative tests passed. This check is
separate from the mainline test counts above.

### JAX T4 execution records

The full `configs/main.yaml` JAX request used the same two methods and four
benchmark families as the Torch run, with `JAX_PLATFORMS=cuda` and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.  After approximately 30 minutes it was
still compiling the high-order MLP transforms (the process had no result JSON),
so it was stopped by the operator.  This is recorded as
`STOPPED_TIMEOUT` in the run card; the empty log and raw `RUNNING` status file
are retained rather than being rewritten as a successful or failed numerical
run.

To verify that the generated JAX path itself executes on the GPU, a separate,
explicitly labelled smoke configuration used one target (`u_{112233}`), a
width-8/depth-2 network, batch size four, and two timed repeats.  It completed
two rows (nested AD and rank-optimal) on `cuda:0`, with analytic errors below
`7e-6`.  These values are a code-path smoke check only and are not part of the
paper's benchmark evidence.

## T4 primary Torch results

The numbers below are median steady-state milliseconds over the ten repeats in
`configs/main.yaml`.  They are raw review measurements, not statistical claims
based on multiple machines or seeds.  `speedup` is
`nested_ad median / rank_optimal median`; `memory ratio` is the analogous ratio
of the recorded CUDA peak allocations.  A ratio below one means that the
rank-optimal path used more peak memory for that case.

### Value mode

| case | nested (ms) | rank-optimal (ms) | speedup | rank directions | memory ratio |
|---|---:|---:|---:|---:|---:|
| `u_112233` | 29.760 | 4.320 | 6.89 | 9 | 0.83 |
| `u_111222333` | 743.889 | 14.579 | 51.02 | 16 | 2.88 |
| `u_11223344` | 253.183 | 22.670 | 11.17 | 27 | 0.99 |
| `u_123456` | 29.639 | 6.462 | 4.59 | 32 | 0.55 |
| KdV2D | 10.148 | 9.236 | 1.10 | 12 | 0.75 |
| g-KdV (`case=4`, `eq=2`) | 15.729 | 12.550 | 1.25 | 12 | 0.93 |
| `\Delta^2` in 2D | 15.150 | 7.329 | 2.07 | 5 | 1.13 |
| `\Delta^3` in 2D | 129.240 | 15.691 | 8.24 | 12 | 2.27 |

### Parameter-gradient mode

| case | nested (ms) | rank-optimal (ms) | speedup | rank directions | memory ratio |
|---|---:|---:|---:|---:|---:|
| `u_112233` | 87.077 | 7.495 | 11.62 | 9 | 0.81 |
| `u_111222333` | 2207.201 | 24.977 | 88.37 | 16 | 2.91 |
| `u_11223344` | 735.726 | 39.309 | 18.72 | 27 | 0.95 |
| `u_123456` | 86.192 | 10.270 | 8.39 | 32 | 0.50 |
| KdV2D | 24.544 | 16.935 | 1.45 | 12 | 0.74 |
| g-KdV (`case=4`, `eq=2`) | 43.194 | 22.767 | 1.90 | 12 | 1.02 |
| `\Delta^2` in 2D | 39.061 | 12.241 | 3.19 | 5 | 1.36 |
| `\Delta^3` in 2D | 364.709 | 26.635 | 13.69 | 12 | 2.64 |

All analytic validation errors in this run were below `6e-7`; the recorded
parameter-gradient relative discrepancies versus nested AD were below
`3.2e-6`.  Repeated-index schedules use `complex64` directions, whereas the
square-free `u_123456` schedule uses real `float32` directions.

## Remaining limitations

- The T4 run is a single fixed-seed measurement.  It is suitable as a smoke
  and protocol check, not as the final multi-seed statistical table.
- The two-dimensional KdV and g-KdV residuals are assembled term by term, as
  specified by the supplied benchmark equations.  Their total direction
  counts are recorded, but the result should not be interpreted as an
  operator-level Waring decomposition.
- The supplied legacy test snapshot still has the missing historical modules
  documented in the Pro `TEST_REPORT.md`; those failures were not hidden or
  fabricated around.
