# Historical experiment ledger

This document records the experiment families that were explored before the
repository was reduced to the examples used by the current WDD manuscript. It
preserves the scientific history and the status of each line of work without
retaining the large checkpoints, predictions, per-step logs, or superseded
result bundles.

The descriptions below are historical records. Only the experiments listed in
the **Current retained evidence** section are part of the present paper and its
reproducibility package. A historical experiment should not be cited as a
current numerical result unless its original data are recovered and audited
independently.

## Current retained evidence

The repository now retains four examples used in the manuscript.

| Paper section | Example | Comparison | Retained evidence |
|---|---|---|---|
| 4.1 | Fixed high-order partial derivatives with several orders, multi-index patterns, and input batch sizes | Minimum-direction complex Waring evaluation versus nested coordinate JVP | Audited summary table, provenance, environment lock, and the current reconstruction script |
| 4.2 | One-dimensional KdV equation with a gradient-enhanced objective | Real-direction WDD versus nested coordinate JVP | Five paired seeds, per-run configurations and summaries, and measured error curves |
| 4.3 | Two-dimensional KdV equation with mixed derivatives | Real-direction WDD versus nested coordinate JVP | Five paired seeds, per-run configurations and summaries, and measured error curves |
| 4.4 | Fourth-order Cahn--Hilliard equation | Real-direction WDD versus nested coordinate JVP | Five paired seeds, per-run configurations and summaries, and measured error curves |

For the PDE examples, both interior points and initial/boundary training points
were resampled at every update. The retained dataset contains 30 completed runs
and 17,998 measured error observations. Aggregate metrics, paired comparisons,
environment records, source provenance, and the frozen training source are also
kept. Large recovery artifacts were removed after the manuscript tables,
figures, endpoint values, and reconstruction identities had been checked.

## Stage 1: derivative-backend and implementation diagnostics

The project began with exact high-order derivative experiments rather than the
current end-to-end PDE comparison. The explored implementations included
nested automatic differentiation, direct Taylor propagation, complex Waring
representations, and polarization formulas. The work included:

- single-monomial derivative timing and memory diagnostics;
- activation-function experiments, especially the behavior of complex
  directions with `tanh` and complex `sinh` networks;
- profiling direction generation, Taylor propagation, repeated complex casts,
  and linear matrix operations;
- experiments that merged repeated Taylor linear operations;
- small manufactured PINN residuals containing one prescribed high-order
  monomial derivative;
- a sixth-order Cahn--Hilliard-type implementation diagnostic.

These runs were useful for understanding the implementation, but most were not
run under the final paired protocol. They were therefore never evidence for the
numerical claims of the current paper. The earlier automatic selector between
Waring and polarization was later removed: the current code uses the complex
roots-of-unity Waring formula for an isolated partial derivative and prescribed
real power-sum identities for the PDE examples.

## Stage 2: architecture, activation, and PDE-family exploration

An exploratory width-study phase compared complex-activation networks with
several real-network baselines, usually at widths 64 and 128 and wall-time
budgets of 600 or 1200 seconds. The families considered were:

| Family | Historical problem or sweep | Historical status |
|---|---|---|
| Polyharmonic | Two- and three-dimensional problems at derivative orders 2, 4, and 6 | Several completed width studies and later formal grids; superseded by the derivative-focused paper |
| Chirp | Oscillatory problems with frequency/amplitude indices 1, 2, and 3 | Completed width studies and formal JSC grids; not retained in the current paper |
| Maxwell | Parameter indices 2, 4, and 6 | Completed width studies and formal JSC grids; not retained in the current paper |
| Helmholtz | Isotropic frequency sweep and the Wang-type modes `(1,1)`, `(1,2)`, and `(1,4)` | Width-study results existed; later archived outside the formal grid |
| Variable-coefficient Helmholtz | Manufactured heterogeneous-medium problems | Implementation and planned sweep; no accepted formal result |
| Nonlinear Schrödinger | Split-real representation of a complex soliton problem | Width-study exploration; not part of the final evidence |
| Plate and beam | Kirchhoff plate, Euler--Bernoulli beam, and mixed plate modes | Width-study and branch-level formal work; not part of the final evidence |
| KdV | Initially a linearized third-order dispersive problem, later nonlinear one- and two-dimensional KdV objectives | Early family study was superseded by the paired KdV examples in Sections 4.2 and 4.3 |
| Cahn--Hilliard | Fourth- and sixth-order manufactured problems | Early diagnostic and branch-level studies were superseded by the paired fourth-order example in Section 4.4 |

Some archived family descriptions contained an empty `data/` directory and
explicitly marked their tables and figures as **TBD**. Those entries record an
implemented or proposed experiment, not a completed result.

## Stage 3: JSC v2 and JSC v3 formal grids

The `jsc_v2` campaign narrowed the active formal grid to three task families:

- Poly problems in dimensions 2 and 3 at orders 2, 4, and 6;
- Chirp problems with indices 1, 2, and 3;
- Maxwell problems with indices 2, 4, and 6.

The compared network families included complex `sinh`, SIREN, multiscale
Fourier-feature PINNs, and MscaleDNN variants. The repository once contained
validated CSV/JSON summaries, detailed histories, manifests, and per-method
subruns for these grids.

This phase also produced several supporting studies:

- a five-scale MscaleDNN sensitivity study over nine two-dimensional tasks;
- short vanilla and specialized-baseline pilots;
- PDE-loss alignment ablations and loss-weight searches;
- shared-boundary-weight studies for the Poly problems;
- Cartesian and power-of-ten weight grids, including longer confirmation runs;
- studies in dimensions 2 and 3 at orders 2, 4, and 6.

The later `jsc_v3` campaign compared compact Taylor-based evaluation with an
automatic-differentiation baseline on Poly, Chirp, and Maxwell tasks. It
included compact benchmarks, three-seed 1000-second runs, fixed-weight runs,
and an extensive real-autodiff weight search. These experiments belonged to an
earlier paper framing based on network families and tuned loss weights. They
were removed when the paper was rewritten around exact derivative evaluation
and a single nested-JVP baseline.

## Stage 4: transition to derivative and matched-wall-time comparisons

The project then moved from architecture comparisons to matched derivative
backends. This transition included:

- a JAX no-outer-JIT fixed-partial protocol and an H20 benchmark;
- shared real-direction Taylor evaluators for PDE residuals;
- a migration of the active implementation to eager PyTorch;
- paired 1000-step PINN pilots;
- paired 600-second wall-time experiments;
- a 10,000-step linear-decay protocol;
- fixed-partial T4 timing matrices over derivative patterns and batch sizes;
- a two-dimensional KdV prescribed-direction protocol;
- gradient-enhanced one-dimensional KdV timing and training experiments;
- three-PDE wall-time runs including Cahn--Hilliard;
- learning-rate sweeps and denser error-trajectory recording.

Several intermediate datasets used fixed initial and boundary training points,
fixed update counts, different learning rates, or three rather than five seeds.
They were useful in selecting and debugging the final protocol, but they were
superseded. The current paper reports only the later five-seed, equal-wall-time
runs with fresh interior and constraint samples at every update.

## Stage 5: current resampled-constraint WDD protocol

The final campaign froze the training source, resampled initial and boundary
points at every update, and ran three problems by two derivative backends by
five paired seeds. It supplied the data now used in Sections 4.2--4.4. The
fixed-partial experiment in Section 4.1 uses the retained batched Waring and
nested-JVP measurements; the historical serial-Taylor measurements are not
part of the paper.

The final method terminology is:

- **WDD method** for the complete Waring decomposition of derivatives method;
- **complex Waring representation** for the isolated mixed-partial formula;
- **real power-sum identities** for the three PDE residual constructions;
- **nested coordinate JVP** for the sole comparison baseline.

## Data removed during repository compaction

The following categories were deliberately removed from the active Git tree:

- obsolete JSC v2/v3 result bundles and their long training histories;
- architecture-width studies and non-paper PDE families;
- baseline pilots, weight searches, loss ablations, and fixed-step runs;
- superseded fixed-boundary and earlier equal-time datasets;
- checkpoints and model states;
- stored predictions and evaluation arrays;
- per-update JSONL/CSV histories when a verified curve or summary was retained;
- duplicated batch and constraint-sampling hashes;
- scheduler, launcher, recovery, and transfer logs;
- old paper previews, external review bundles, and duplicate source exports.

This deletion reduces repository storage; it does not retract the fact that the
experiments were attempted or completed at the status recorded above. It does
mean that exact numerical reconstruction of a removed campaign requires an
independent external backup. The historical Git identifiers immediately before
the history rewrite were `faa2158` for the finalized WDD version and `8331b3a`
for the compact current-result tree. These identifiers are recorded for
provenance but are not expected to resolve after the remote history is rewritten.

## Historical branches removed during compaction

The remote also contained agent and editorial branches for overnight results,
high-order candidate pilots, hyperbolic/nonlinear PDE candidates, plate and
Cahn--Hilliard runs, manuscript polishing, operator-symbol exposition, and
monomial-Waring proof revisions. Their useful conclusions are represented in
the stages above or in the current manuscript. The branch refs were removed so
that they would not keep the deleted binary and result history reachable.

## Interpretation rule

Use this ledger to understand what was tried and why the final protocol was
chosen. Use the current manuscript dataset for numerical claims. A label such
as “implemented,” “pilot,” “planned,” “TBD,” “completed,” or “superseded” is
part of the result and must not be upgraded to a stronger status without the
underlying data and a new verification.
