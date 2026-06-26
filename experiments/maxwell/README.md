# Time-harmonic Maxwell, lossy medium (complex-valued)

**Problem.** 2D TM-mode time-harmonic Maxwell / Helmholtz with complex
permittivity (a lossy medium), so the field \(E_z\) is genuinely **complex**:
\(\Delta E_z + \omega^2(\varepsilon' + i\varepsilon'')E_z = 0\). Sweeps frequency.

**Source.** Time-harmonic Maxwell PINN benchmark (e.g. Jiang et al. 2024); lossy
(complex-permittivity) variant.

**Why it matters.** Second genuinely complex-valued case, linear this time, with a
complex coefficient (loss). Together with NLS it covers complex fields from both
linear and nonlinear physics.

## Outputs (`data/`)
- `maxwell_v3.csv` -- accuracy vs frequency.
- `maxwell_history.json` -- traces at sweep 4.

## Reproduce
```bash
bash run.sh
```
Figure: `fig_maxwell.pdf`.
