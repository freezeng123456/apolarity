# First group: derivative values without whole-function JIT

Protocol `jax_first_group_values_v2`, execution `no_outer_jit`, approved
2026-09-08. Earlier compiled results retain their original interpretation and
source under `results/historical/h20_first_group_20260908/` at repository root.

## Equal task and execution policy

Both methods evaluate one prescribed partial of the same real tanh MLP on
identical points. No loss, parameter backward or optimizer is included.

- `nested_jvp`: repeated official coordinate `jax.jvp`, batched over points;
  no full high-order tensor.
- `waring_batched`: rank-optimal roots/weights, official `jax.experimental.jet`,
  batched over directions and points, then weighted reduction.
- `waring_serial`: internal parallelism ablation with identical directions
  evaluated using a Python loop; no compiled scan body in this execution mode.

Neither primary evaluator receives outer whole-function JIT. Primitive-level
JAX compilation/caching still occurs. This is not compiler-free execution.
The optional `whole_function_jit` flag is for separately labelled ablations.
The previously observed JIT ranking reversal is preserved in the diagnostic
results; a longer compile alone does not prove special AD-only optimization.

This is not STDE code or a reproduction of its training-speed tables; it
isolates derivative values. STDE's main settings include parameter gradients:
https://proceedings.neurips.cc/paper_files/paper/2024/file/dd2eb5250696753ea37141bbd89bb569-Paper-Conference.pdf

## Frozen matrix

- Main B=100: 111111, 111222, 112233, 123456, 11223344, 111222333;
  direction counts 1,4,9,32,27,16; two methods, three seeds (36 rows).
- Serial/batched ablation: 123456 and 111222333 at B=100 (6 serial rows).
- Batch scaling: those two targets at B=400,1600, both methods (24 rows).
- Total 22 isolated cells, 66 seed-level rows; single H20, no overlapping GPU
  cells or direction chunking; one-hour per-cell timeout recorded as failure.

Input dimension 16; three width-128 tanh hidden layers plus scalar output.
Uniform weights/biases bounded by inverse sqrt(fan-in); input Gaussian standard
deviation 0.35. Seeds 20260908,20260909,20260910; model seed=seed+2, points=seed+1.
float32 real data; complex64 only for directions requiring complex roots.
Pure/square-free directions stay float32. JAX/JAXLIB 0.4.38; highest matmul
precision, no AMP; parameters and input points are dynamic arguments.

One cold call is separately recorded. Ten warmups and 30 synchronized wall-clock
calls per seed, data resident before timing. Report mean and sample standard
deviation of three per-seed medians and mean paired speedups. Basic output
consistency checks are outside timing, not an accuracy study.

## Reporting

Retain configs, exact source, all timing samples, values, status, failure logs
and checksums. Generate CSV/Markdown/LaTeX tables, not new plots.
No-outer-JIT whole-function compile time and executable-memory estimate are
unavailable, not zero. Allocator process-lifetime peak is explicitly labelled
and is not a runtime-only GPU peak.

Author approval of paragraph structure/tables precedes manuscript edits. The
matrix does not start the second PDE benchmark.
