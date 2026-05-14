# Cleanup archive: 2026-05-14

This archive collects files moved out of the active project tree during cleanup.

## Contents

- `results/`: raw benchmark, profiling, smoke-test, Gaussian-Hermite baseline, and PINN experiment outputs.
- `docs/plans/`: superseded or narrow-scope experiment notes whose main conclusions are covered by current summary documents.

## Removed instead of archived

- `.pytest_cache/`
- `__pycache__/`
- `*.pyc`

## Kept in active tree

- Core package code under `src/apolarity/`
- Main tests under `tests/`
- Main experiment entry points under `experiments/`
- Current paper/performance/theory documents under `docs/`
