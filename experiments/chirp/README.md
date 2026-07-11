# Non-separable radial chirp

**Problem.** \(-\Delta u + u = f\) on \((-1,1)^2\), manufactured radial chirp
\(u=\sin(\tfrac{a\pi}{2}(x^2+y^2))\) whose local frequency \(|\nabla\phi|=a\pi r\)
grows with radius — so \(u\) is **not** a single Fourier mode. Dirichlet \(=u_\star\).

**Source.** Chirp / space-varying-frequency expressivity test (Tancik 2020,
Liu 2020). Removes the "separable sine" confound present in the other families.

**Sweep.** chirp rate \(a\in\{1,2,3\}\). Init \(\omega_0=\max(10,2\pi a)\),
\(\sigma=\max(2,\pi a)\).

## Formal comparison

The three sweep values are formal `jsc_v2` settings. The only formal methods
are `complex_sinh`, SIREN, mFF-PINN, and MscaleDNN-2-sin. Complex Sinh
\(H=128\) defines the true trainable real-parameter budget; external baselines
receive automatically matched integer widths with at most \(5\%\) mismatch.
\(H=64\) is not run or discussed.

## Current outputs

`data/` is empty. No formal Chirp result exists, and the Chirp paper figure and
table are **TBD**.

## Launch one formal setting

```bash
bash scripts/run_jsc_main3.sh chirp --sweep 2
python scripts/validate_jsc_results.py \
  experiments/results/jsc_v2/chirp_a2
```

Choose exactly one allowed `--sweep` value per launch. The family-local
`run.sh` and archived runners are implementation diagnostics only; their
outputs cannot be used as paper evidence.
