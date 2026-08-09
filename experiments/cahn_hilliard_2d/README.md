# Two-dimensional Cahn--Hilliard

This is the active Cahn--Hilliard experiment.  Here "2D" means two spatial
coordinates: the physical network input is `(x, y, t)`.

## Problem

On `(x,y) in (0,pi)^2`, `t in [0,1]`, solve

```text
u_t - Delta(u^3-u) + eta_q Delta^q u = f,
```

with:

- CH4: `q=2`, `eta_q=+1e-2`;
- CH6: `q=3`, `eta_q=-1e-2`.

The signs give a dissipative leading symbol in both cases.  The manufactured
solution is

```text
exp(-t) * [
  0.50 cos(x) cos(y)
  + 0.25 cos(2x) cos(y)
  + 0.25 cos(x) cos(2y)
].
```

Cosines are used only to define the analytic source, initial condition, and
held-out truth.  They are not network features.

## Boundary and initial conditions

Natural homogeneous no-flux conditions are used:

- CH4: `d_n u = d_n Delta u = 0`;
- CH6: `d_n u = d_n Delta u = d_n Delta^2 u = 0`.

The initial condition is the manufactured solution at `t=0`.  Mass conservation
is logged as a diagnostic and is not added to the optimisation loss.

## Fair comparison

Both methods use:

- affine-normalised raw `(x,y,t)` input, with no Fourier embedding;
- four hidden layers of width 128;
- `sinh` activation;
- the same variance-matched Xavier initialisation family;
- the same resampled collocation stream, losses, and wall-clock budget.

Only these properties differ:

- `war`: native `complex64` parameters and Waring/Taylor-jet derivatives;
- `real_sinh_autodiff`: real `float32` parameters and direct nested autodiff.

This is a literal layer-shape match, as specified by the protocol, rather than
a trainable-real-parameter-count match.  Both networks contain 50,177 scalar
parameter elements; because every native complex element has two real degrees
of freedom, the logged counts are 100,354 for WAR and 50,177 for real AD.
Both counts are stored in every result and in the search manifest.

## Weight search

After smoke validation, search the shared vector
`[lambda_ic, lambda_bc]`, with each value in
`{1e-3,1e-2,1e-1,1,1e1,1e2,1e3}`.  Each task therefore has 49 candidates.

```bash
python scripts/run_cahn2d_weight_search.py smoke --seconds 3 \
  --ephemeral-conclusion /absolute/path/cahn2d_smoke_conclusion.json
python scripts/run_cahn2d_weight_search.py orchestrate --seconds 60 --resume
python scripts/run_cahn2d_weight_search.py summarize
```

In ephemeral mode, the runner builds its JSON/log/manifest bundle inside a
system temporary directory, writes one compact conclusion, and then deletes the
entire raw bundle even when a cell fails.  The reviewed conclusion is copied to
`docs/archive/SMOKE_CONCLUSIONS_zh.md`; it is never used as paper evidence.
