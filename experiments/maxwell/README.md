# Time-harmonic Maxwell, lossy medium (complex-valued)

**Problem.** TM-mode reduction to a complex Helmholtz equation
\(\Delta E + \kappa^2 E = f\) on \((-1,1)^2\), \(\kappa^2=(a\pi)^2(1+i\beta)\),
\(\beta=0.2\) (loss tangent → complex permittivity → genuinely complex \(E\)).
Manufactured plane wave \(E=e^{i a\pi(x+y)}\), Dirichlet \(=E_\star\). Real baselines
use a split-real (Re/Im) pair (RVPINN).

**Source.** Jiang 2024 (lossy TM variant).

**Sweep.** wavenumber \(a\in\{2,4,6\}\). Init \(\omega_0=\max(10,2\pi a)\),
\(\sigma=\max(2,\pi a)\).

## Formal comparison

The three sweep values are formal `jsc_v2` settings. The only formal methods
are `complex_sinh`, SIREN, mFF-PINN, and MscaleDNN-2-sin. Complex Sinh
\(H=128\) uses the native complex representation and defines the true trainable
real-parameter budget. Each external baseline uses a split-real representation
and an automatically matched integer width; both real networks count toward
the parameter budget, whose mismatch must not exceed \(5\%\). \(H=64\) is not
run or discussed.

## Current outputs

`data/` is empty. No formal Maxwell result exists, and the Maxwell paper figure
and table are **TBD**.

## Launch one formal setting

```bash
bash scripts/run_jsc_main3.sh maxwell --sweep 4
python scripts/validate_jsc_results.py \
  experiments/results/jsc_v2/maxwell_a4
```

Choose exactly one allowed `--sweep` value per launch. The family-local
`run.sh`, historical completion/merge scripts, and archived runners are
implementation diagnostics only; their outputs cannot be used as paper
evidence.
