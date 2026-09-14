# Cleanup validation

- Base before cleanup: c0587d7 (master), including the pre-existing dirty tree.
- Recoverable snapshot: ../snapshots/pre_cleanup_20260908.FhFtl1/worktree.tar.gz.
- 42 groups and 9625 files moved with per-file SHA-256 verification.
- No permanent deletion; no Git-history rewrite. Old implementation,
  configuration and test files retained together in archive/legacy_code/project.
- Active root is now the independent JAX package with no Torch dependency.
- `python -m pytest -q`: 7 passed, including current first-group direct-JVP,
  batched-jet and Python-serial equality checks and a guard against explicit JIT.
- `compileall` succeeded for the active JAX package and result-table generator.
- A tiny no-outer-JIT CPU CLI check is retained outside the repository in the
  snapshot directory. It is not H20 performance evidence.
- The existing manuscript compiled with Tectonic to the snapshot directory;
  no manuscript source or existing PDF was overwritten. Existing style-file
  encoding/underfull-box warnings remain; there was no build failure.
- New numerical evidence will be presented in tables. Author approval of
  paragraph structure is required before manuscript changes or new paper commit.
- No second-group GPU run was launched by this cleanup.
