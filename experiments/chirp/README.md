# Non-separable radial chirp

**Problem.** Helmholtz-type equation whose manufactured solution is a radial chirp
\(u=\sin(a\pi r^2)\), \(r^2=x^2+y^2\): the local frequency grows with radius, so the
field is **not** a sum of global plane waves. Sweeps the chirp rate \(a\).

**Source.** Chirp / space-varying-frequency expressivity test (graded-frequency
generalization of the Helmholtz benchmark).

**Why it matters.** Stresses methods that rely on a fixed global frequency basis:
Fourier features must cover the whole band everywhere, whereas the complex `sinh`
phase adapts locally.

## Outputs (`data/`)
- `chirp_v3.csv` -- accuracy vs chirp rate.
- `chirp_history.json` -- traces at rate 4.

## Reproduce
```bash
bash run.sh
```
Figure: `fig_chirp.pdf`.
