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

Capacity is matched by trainable real degrees of freedom. The reference is
native-complex `complex_sinh` at `H=128`; external baselines receive the closest
integer width within 5%. For Maxwell, the two split-real component networks are
counted together. Frozen Fourier maps are excluded from trainable DOF.

Executable contracts are in:

- `tests/test_architecture_fidelity.py`;
- `tests/test_backend_equivalence.py`;
- `tests/test_jsc_protocol.py`.

No experiment result is evidence for implementation fidelity. Formal
measurements remain blocked until the audit, tests, protocol table, and atomic
runner pass readiness review.
