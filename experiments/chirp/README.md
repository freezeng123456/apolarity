# Non-separable radial chirp

**Problem.** \(-\Delta u + u = f\) on \((-1,1)^2\), manufactured radial chirp
\(u=\sin(\tfrac{a\pi}{2}(x^2+y^2))\) whose local frequency \(|\nabla\phi|=a\pi r\)
grows with radius — so \(u\) is **not** a single Fourier mode. Dirichlet \(=u_\star\).

**Source.** Chirp / space-varying-frequency expressivity test (Tancik 2020,
Liu 2020). Removes the "separable sine" confound present in the other families.

**Sweep.** chirp rate \(a\in\{1,2,3\}\). Init \(\omega_0=\max(10,2\pi a)\),
\(\sigma=\max(2,\pi a)\).

## Formal comparison

The three sweep values are formal `jsc_v3` settings with frozen
`pow10_reasonable_v1` scalar weights in `experiments/common/boundary_weights.py`.
The only formal methods
are `complex_sinh`, SIREN, mFF-PINN, and MscaleDNN-2-sin. Complex Sinh
and all three baselines use literal hidden width \(H=128\). Trainable real
parameter counts are reported separately and are not used to rescale widths.

## Current outputs

`data/` is empty. The completed fixed-`bc_weight=100` `jsc_v2` bundle is kept as
historical evidence under `experiments/results/jsc_v2/`; no `jsc_v3` Chirp result
has been launched yet, so the new Chirp paper figure and table are **TBD**.

## Launch one formal setting

```bash
bash scripts/run_jsc_main3.sh chirp --sweep 2
python scripts/validate_jsc_results.py \
  experiments/results/jsc_v3/chirp_a2
```

Choose exactly one allowed `--sweep` value per launch. The family-local
`run.sh` and archived runners are implementation diagnostics only; their
outputs cannot be used as paper evidence.
