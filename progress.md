# Project progress

Last updated: 2026-07-14

## Current state

The legacy experiment repository was reset before the `jsc_v2` campaign. New
formal bundles now exist for Poly `d=2`, orders 2/4/6; Poly `d=3`, order 6;
and Chirp `a=1`. Each completed bundle contains 20 method/seed rows and carries
a `VALIDATED` marker. Manuscript and slide result sections remain `TBD` until
the preregistered campaign is complete and the validated data are reviewed.

No formal experiment is active at this update. New data is admissible only
when it carries `protocol_id=jsc_v2` and passes
`scripts/validate_jsc_results.py`.

The three representative smoke tasks have now completed successfully. Their
temporary result bundles are not scientific results and are deleted before
Git upload.

## Formal methods

The formal registry contains exactly four methods:

1. **Complex Sinh**: the paper method; complex128, four trainable sinh hidden
   layers, and frequency-rich complex first-layer initialization.
2. **SIREN**: explicit `sin(omega * Linear(x))` parameterization, upstream
   weight initialization, and default first/hidden omega of 30.
3. **mFF-PINN**: two frozen Fourier branches with scales `(1, sigma)`, a shared
   four-layer tanh trunk, branch concatenation, and a linear output.
4. **MscaleDNN-2-sin**: independent sine subnets evaluated at fixed input
   scales `(1, 2, 4)`, initialized with the original Gaussian rule and summed.

`real_sinh` is removed. `tanh`, Cauchy, and `complex_sinh_noinit` are auxiliary
implementation checks and are not visible to the formal runner.

The complete evidence chain is in
`docs/BASELINE_IMPLEMENTATION_AUDIT.md`. Pinned upstream commits are:

- SIREN: `vsitzmann/siren@4df34baee3f0f9c8f351630992c1fe1f69114b5f`;
- MultiscalePINNs: `PredictiveIntelligenceLab/MultiscalePINNs@ba7d6bb8af6cabe348def80bed72110f5f0e3621`;
- general Fourier-feature cross-reference:
  `tancik/fourier-feature-networks@9c110c31ce3794222fff408ac27bbf74d8fe8993`;
- original MscaleDNN:
  `xuzhiqin1990/mscalednn@1c6c6f69e9ad586ccaea90a8e8fa0d07313460b2`;
- PyTorch MscaleDNN cross-reference:
  `Blue-Giant/MscaleDNN_torch@b63796dd42a2020a0c2b241b1c824cc0405fad91`.

The GitHub CLI is installed locally, but this host has no authenticated GitHub
token. Immutable public API/raw source URLs were therefore used for the audit;
all evidence links are commit-pinned.

## Literal-width fairness definition

Fairness was corrected on 2026-07-11: all four formal methods use literal
hidden width `H=128`. Trainable real degrees of freedom are disclosed but do
not determine widths. A complex trainable scalar counts as two real degrees of
freedom, frozen Fourier maps do not count, and Maxwell counts both split-real
component networks.

The resulting parameter tables are:

- scalar 2D, all `H=128`: Complex Sinh `100098 DOF`; SIREN `50049`;
  mFF-PINN `66305`; MscaleDNN-2-sin `150147`;
- scalar 3D, all `H=128`: Complex Sinh `100354 DOF`; SIREN `50177`;
  mFF-PINN `66305`; MscaleDNN-2-sin `150531`;
- Maxwell, all component networks `H=128`: Complex Sinh `100098 DOF`;
  split-real SIREN `100098`; split-real mFF-PINN `132610`; split-real
  MscaleDNN-2-sin `300294`.

This is literal-width plus equal-wall-clock control, not parameter-count
matching. The protocol and validator reject every formal width other than 128.

## Frozen protocol

`experiments/common/protocol.py` is the single source of truth:

- protocol: `jsc_v2`;
- literal hidden width: `H=128` for every formal method and component network;
- wall-clock: 1200 seconds per method and seed;
- seeds: `0..4`;
- depth: four trainable hidden layers;
- collocation: 4096 interior and 512 boundary points;
- optimizer schedule: Adam, cosine learning-rate decay;
- paired collocation: `paired_seed_v1`;
- fixed held-out evaluation: `fixed_seed_12345_n8192_v1`.

Preregistered settings:

- Poly: `d={2,3}` and `order={2,4,6}`, including `d=3, order=6`;
- Chirp: `a={1,2,3}`;
- Maxwell: `a={2,4,6}` with split-real external baselines.

There are 12 atomic tasks. Each task contains
`1 setting x 4 methods x 5 seeds = 20` runs, or 24000 seconds of nominal
training time. Tasks are not chained or scheduled by the repository.

## Experiment todo tracker

Status meanings:

- `READY`: protocol, parameter table, runner, and smoke path are ready, but no
  formal run has been authorized;
- `RUNNING`: one detached formal task is active;
- `VALIDATING`: training finished and the canonical bundle is under review;
- `DONE`: the 20-row bundle passed validation and manual log review;
- `BLOCKED`: a concrete implementation, data, or hardware issue must be fixed.
- `QUEUED`: user-authorized task waiting in the detached serial queue; this is
  an operator state rather than part of the formal protocol.

| Task ID | Setting | Status | Formal output |
|---|---|---|---|
| `poly_d2_o2` | Poly `d=2`, order 2 | DONE | `experiments/results/jsc_v2/poly_d2_o2` |
| `poly_d2_o4` | Poly `d=2`, order 4 | DONE | `experiments/results/jsc_v2/poly_d2_o4` |
| `poly_d2_o6` | Poly `d=2`, order 6| DONE | `experiments/results/jsc_v2/poly_d2_o6` |
| `poly_d3_o2` | Poly `d=3`, order 2 | DONE | `experiments/results/jsc_v2/poly_d3_o2` |
| `poly_d3_o4` | Poly `d=3`, order 4 | QUEUED | none |
| `poly_d3_o6` | Poly `d=3`, order 6 | DONE | `experiments/results/jsc_v2/poly_d3_o6` |
| `chirp_a1` | Chirp `a=1` | DONE | `experiments/results/jsc_v2/chirp_a1` |
| `chirp_a2` | Chirp `a=2` | QUEUED | none |
| `chirp_a3` | Chirp `a=3` | QUEUED | none |
| `maxwell_a2` | Maxwell `a=2` | READY | none |
| `maxwell_a4` | Maxwell `a=4` | READY | none |
| `maxwell_a6` | Maxwell `a=6` | READY | none |

Queued serial execution plan (2026-07-14):
1. `poly_d3_o2`
2. `poly_d3_o4`
3. `chirp_a2`
4. `chirp_a3`

Total nominal training time: ~26 hours 40 minutes (4 tasks × 6 hours
40 minutes), plus validation and Git upload overhead.

The queue stops immediately if training, validation, commit, or push fails.
Each successor starts only after its predecessor has been validated and pushed.

## Runner and result contract

`scripts/run_jsc_main3.sh` launches exactly one detached atomic task with
`setsid + nohup`. It never commits or pushes. Examples:

```bash
bash scripts/run_jsc_main3.sh --dry-run poly --dim 3 --order 6
bash scripts/run_jsc_main3.sh poly --dim 3 --order 6
bash scripts/run_jsc_main3.sh chirp --sweep 2
bash scripts/run_jsc_main3.sh maxwell --sweep 4
```

Formal execution requires a clean Git worktree so the recorded SHA identifies
the code. Smoke output is isolated under `experiments/results/_smoke/`.

Every canonical row records the protocol and task IDs, Git SHA/dirty flag,
problem setting, actual width, real DOF, representation, seed, collocation and
evaluation protocols, all method frequency/scale settings, budget, optimizer,
steps, and hardware. The validator requires 20 unique method/seed rows, all
five seeds, finite metrics, nonempty monotone histories, a consistent Complex
Sinh DOF reference, and the exact protocol metadata.

The plot and table tools read only task directories carrying a `VALIDATED`
marker and canonical jsc_v2 files. Validated data now exist, but manuscript
generation remains deferred until the selected Poly and Chirp tasks complete.

## Verification

- Full test suite: **68 passed**.
- Expected warnings: 13 PyTorch warnings about complex modules.
- Architecture tests cover upstream SIREN initialization/omega placement,
  mFF branch sharing/mapping, Mscale explicit scaling/initialization, Complex
  Sinh initialization, output shapes, literal width, and parameter reporting.
- Jet-vs-direct tests cover low/high-order input derivatives and parameter
  gradients through scaled sine, Fourier branches, and Mscale subnets.
- Existing backend, complex-gradient, 3D Delta-cubed, collocation RNG,
  Gaussian-Hermite, and PDE formula tests remain passing.
- Modified Python files have no IDE linter diagnostics.
- Python compilation, shell syntax, and `git diff --check` pass.
- The atomic `poly d=3, order=6` dry run emits a 20-run manifest and a valid
  four-method parameter table.

## Smoke status

The first quick smoke phase used superseded automatically selected widths and
was discarded. A corrected smoke then ran every method at literal `H=128`
using one seed, 32 interior points, 16 boundary points, and one second per
method:

- `poly_d3_o6`: all four H=128 methods wrote histories and passed validation;
- `chirp_a2`: all four H=128 methods wrote histories and passed validation;
- `maxwell_a4`: native-complex and all three split-real H=128 baselines wrote
  histories and passed validation.

All corrected smoke metrics were finite and no run reported NaN. Temporary
bundles are not retained or interpreted as accuracy measurements.

## Readiness review

Implementation readiness checks pass:

- source-pinned baseline audit: complete;
- four formal method specifications: complete;
- unexplained architecture branches: none in the formal registry;
- literal `H=128` enforcement: implemented and smoke-validated;
- protocol and 12-setting grid: frozen;
- runner/validator/plot/table chain: implemented and tested;
- old results and hard-coded empirical conclusions: removed;
- completed formal bundles: five; queued formal bundles: four.

The implementation, audit, parameter tables, and smoke paths are ready. The
user has authorized the four-task serial queue listed above. Paper figures,
tables, timings, and accuracy conclusions remain `TBD` until the selected
campaign is complete and reviewed.
