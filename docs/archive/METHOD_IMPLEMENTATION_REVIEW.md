# Four-method implementation review

Status: frozen for pre-experiment readiness review.

The source-level upstream audit, pinned commits, formulas, local differences,
and adaptation decisions are maintained in
[`docs/BASELINE_IMPLEMENTATION_AUDIT.md`](docs/BASELINE_IMPLEMENTATION_AUDIT.md).
That document supersedes the pre-reset review, which described the old
single-scale Fourier model, folded-scale Mscale ensemble, and approximate SIREN
parameterization.

The formal method registry is exactly:

1. `complex_sinh`: this project's complex128 four-hidden-layer sinh MLP;
2. `siren`: the `vsitzmann/siren` explicit-omega parameterization;
3. `fourier`: MultiscalePINNs-style two-branch mFF-PINN with shared tanh trunk;
4. `mscale`: MscaleDNN-2-sin with explicit fixed input scales.

Auxiliary `tanh`, Cauchy, and `complex_sinh_noinit` code is not available to the
formal atomic runner. `real_sinh` is not supported.

All four formal methods use literal hidden width `H=128`; trainable real degrees
of freedom are not matched. For Maxwell, each external baseline uses two
`H=128` split-real component networks. Actual trainable real DOF is recorded
for transparency, with complex scalars counted twice and frozen Fourier maps
excluded.

Executable contracts are in:

- `tests/test_architecture_fidelity.py`;
- `tests/test_backend_equivalence.py`;
- `tests/test_jsc_protocol.py`.

No experiment result is evidence for implementation fidelity. Formal
measurements remain blocked until the audit, tests, protocol table, and atomic
runner pass readiness review.
