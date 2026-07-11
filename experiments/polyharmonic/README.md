# Polyharmonic order sweep

**Problem.** Controlled order axis at fixed frequency.
- \(d\)D: \(\Delta^m u=(-d\pi^2)^m u\) on \((-1,1)^d\),
  \(u=\prod_{i=1}^d\sin(\pi x_i)\).
- 1D: \(d^{2m}u/dx^{2m}=(-\pi^2)^m u\) on \((-1,1)\), \(u=\sin\pi x\).
Navier (simply-supported) BCs: \(\Delta^j u=0\), \(j=0..m-1\). Only the operator
**order** changes across the sweep — no frequency confound.

**Source.** Vahab 2022 (high-order generalization of the biharmonic benchmark).

**Formal `jsc_v2` grid.** \(d\in\{2,3\}\), order
\(2m\in\{2,4,6\}\). The \(d=3\), order-6 case is required, not an optional
follow-up. The frozen initialization is \(\omega_0=2\pi\) and
\(\sigma=\pi\). An order-\(m\) operator amplifies initialization frequency like
\(\omega^m\), so the initialization must remain part of the protocol.

## Formal comparison

The only formal methods are `complex_sinh`, SIREN, mFF-PINN, and
MscaleDNN-2-sin under `protocol_id=jsc_v2`. Complex Sinh \(H=128\) defines the
true trainable real-parameter budget. External baselines receive automatically
matched integer widths with at most \(5\%\) mismatch. \(H=64\) is not run or
discussed.

## Current outputs

`data/` is empty. No formal Poly result exists, and all Poly paper figures and
tables are **TBD**.

## Launch one formal setting

```bash
bash scripts/run_jsc_main3.sh poly --dim 3 --order 6
python scripts/validate_jsc_results.py \
  experiments/results/jsc_v2/poly_d3_o6
```

Choose exactly one allowed `--dim` and one allowed `--order` per launch. The
family-local `run.sh`, historical 1D/width-study commands, and archived runners
are implementation diagnostics only; their outputs cannot be used as paper
evidence.
