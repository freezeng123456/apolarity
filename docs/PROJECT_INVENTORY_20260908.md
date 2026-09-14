# Project inventory and proposed cleanup — 2026-09-08

**Execution update:** the approved cleanup is now complete. The following
inventory remains a pre-cleanup record, not the current path map. See
`../archive/MOVES_20260908.json`, `../archive/README.md` and the top-level README
for verified destination paths and the active no-outer-JIT protocol.

Read-only inventory of existing project content. No existing files have been
moved or deleted, and the benchmark execution policy has not yet been changed.

## Current location and working-tree state

- Main checkout: `work/apolarity_pr10` relative to the desktop workspace.
- Branch: master; HEAD: c0587d7.
- Allocated disk use: approximately 198 MiB including 42 MiB of Git metadata.
- Before adding this inventory: 8492 tracked files; 877 untracked non-ignored
  files; 5 modified tracked files. Untracked does not mean disposable.
- Modified tracked files: README.md, experiments/README.md, pyproject.toml,
  src/apolarity/__init__.py, src/apolarity/operators.py.

## Latest experiment preference

The user prefers a main comparison without whole-function JIT. For a matched
execution policy, both compared evaluators should use no outer whole-function
JIT. Any asymmetric policy must be separately specified and labelled, not
silently introduced. JAX primitive-level compilation still exists in this mode.
The current first_group.py still compiles both evaluators; this inventory does
not change it or reinterpret completed results. Compilation duration by itself
does not establish that AD receives a special, method-specific optimization.
Retain the compiled results and the observed JIT ranking reversal as historical
and diagnostic evidence with their original configuration.

## Active material to preserve

| Path | Role |
|---|---|
| docs/paper/jsc_paper_main.tex | Current manuscript |
| docs/paper/jsc_paper_main.bib | Bibliography |
| docs/paper/jsc_paper_main.pdf | Current compiled manuscript |
| docs/paper/figures, tables, class/style files | Manuscript dependencies |
| experiments/jax_benchmarks/first_group.py | Current fixed-partial evaluator and measurement entry |
| experiments/jax_benchmarks/run_first_group_matrix.py | First-group matrix launcher |
| experiments/jax_benchmarks/summarize_first_group.py | Results summarizer |
| experiments/jax_benchmarks/plot_first_group.py | Result plots |
| experiments/jax_benchmarks/src/jax_apolarity_bench | Standalone JAX directions, jet, model and residual code |
| experiments/jax_benchmarks/tests | Standalone package tests |
| tests_active/test_jax_standalone.py | Additional JAX checks, currently outside package |
| experiments/jax_benchmarks/FIRST_GROUP.md | Existing compiled-run protocol; requires a new explicitly versioned non-outer-JIT protocol |

The standalone JAX first-group runner imports only its own directions,
derivatives and models modules, not the root Torch package. Its package
__init__.py also imports residuals; therefore residuals.py is not safe to delete
merely because first_group.py does not reference it directly.

## Grouped inventory

| Path | Approximate size | Interpretation / proposed treatment |
|---|---:|---|
| docs/ | 7.7 MiB | Keep current paper and dependencies; separate literature notes, slides and old previews |
| experiments/jax_benchmarks/ | 224 KiB | Keep as active JAX package; distinguish first_group.py from older all-family run.py |
| src/apolarity/ | 264 KiB | Older Torch-centred API plus multiple benchmark backends; archive only as a dependency-complete unit |
| experiments/torch_benchmarks/ | 84 KiB | Historical Torch runner and protocol |
| experiments/benchmarks/ | 132 KiB | Earlier modular benchmark implementation |
| experiments/stde_comparison/ | 368 KiB | Earlier STDE adapter/comparison; not active first-group code |
| experiments/archived/ | 19 MiB | Existing archive; preserve, avoid nesting it blindly into another archive |
| experiments/results/ | 25 MiB | Historical experiment bundles; archive with sources/protocols, do not discard |
| outputs/ | 48 MiB | Historical training/search results, many Git-tracked |
| output/ | 18 MiB | Recent H20 evidence, diagnostic traces, external-model packages and previews; classify per run |
| tmp/ | 36 MiB | Mixed previews and source/result handoff packages; not all are disposable |
| tests/, tests_active/, scripts/, configs/ | Under 1 MiB combined | Multiple generations of tests/launchers; migrate with their dependent implementation |

Recent runs that must retain source/config/raw data/checksums together:

- output/h20_first_group_20260908 — compiled formal matrix, approximately 1.2 MiB.
- output/h20_first_group_diagnosis_20260908 — affine and layout diagnosis,
  approximately 12 MiB including HLO and traces.
- output/h20_jit_diagnosis_20260908 — JIT on/off diagnosis, approximately 72 KiB.

## Dependency and documentation hazards

1. Root pyproject.toml still requires Torch and defaults pytest to tests_active.
   Most tests_active files import Torch or the root apolarity package. Moving
   src/apolarity alone would break the public package and these tests.
2. The standalone JAX package has a separate pyproject.toml and can remain
   runnable independently. Its older run.py uses nested grad, while the new
   first_group.py uses nested coordinate JVP; do not conflate their baselines.
3. README.md and experiments/README.md mix old four-family plans, historical
   training protocols and the new first group. The root introduction also
   describes older complex-sinh settings, not the current real tanh benchmark.
4. The current manuscript explicitly references fig_benchmark_t4_runtime and
   fig_benchmark_t4_memory. Do not remove these assets before intentionally
   replacing the manuscript references and recompiling.
5. tmp/ contains external-model requests, returned code and smoke records in
   addition to PDFs. Preserve these as provenance, even if they leave the
   active tree. A ZIP and similarly named extracted directory are not proven
   redundant until their contents are compared.
6. .gitignore ignores results/ and build artifacts. A file not reported by
   ordinary git status can still contain important result evidence.

## Proposed cleanup order (not yet executed)

1. Preserve a recoverable snapshot of the current dirty worktree, with an
   inventory and hashes, outside the archive target. Do not rely on HEAD alone.
2. Rewrite the top-level navigation to clearly identify paper, current JAX
   first group, pending PDE work and historical evidence. Version the intended
   non-outer-JIT protocol separately from completed compiled runs.
3. Move obsolete code as complete units: historical training family folders,
   their launchers/configs/tests, old Torch benchmark and earlier modular
   benchmark. Maintain an explicit old-path -> new-path manifest.
4. Group historical outputs and external-model handoffs by run. Keep original
   internal directory structure so configs/checksums remain interpretable.
5. Only remove regenerated caches/build intermediates or checksum-confirmed
   duplicates, preferably through recoverable trash. Never delete raw results,
   uncommitted source, .git or manuscript dependencies as cleanup shortcuts.
6. Verify the standalone JAX import/smoke tests and manuscript build after any
   path migration. An archive move is not complete merely because files moved.

## Outside the main checkout

The desktop workspace also contains work/stde_upstream (308 KiB),
work/stde_apolarity_t4_compare (2.6 MiB), work/tmp (2.5 MiB), top-level tmp/pdfs
(3 MiB) and other small review folders. These are reference/history candidates,
not proof of byte-identical duplication with the checkout's reference/ tree.
Do not treat the desktop workspace itself as the Git repository or delete it.
