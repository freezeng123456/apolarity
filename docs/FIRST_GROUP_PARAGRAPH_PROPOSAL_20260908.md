# First-group numerical section: structure for author approval

Status: approved and applied on 2026-09-08. The text below preserves the
proposal reviewed by the author. The manuscript now contains four paragraphs
and two tables; old dependent experimental claims have been synchronized.

Proposal only. No manuscript text or result table has been inserted. The v2
run is now complete and verified (22 cells, 66 seed records, 1980 timed calls).
The evidence is in ../results/first_group/h20_no_outer_jit_20260908/report/REPORT.md.

## Proposed compact structure

Use a short general experimental-setting paragraph, followed by a subsection
on fixed high-order partial derivatives. Do not retain separate long sections
for profiling details, gradient verification or memory claims for this group.

1. **Setting and comparison boundary.** State the H20/JAX environment, real
   tanh network, same inputs/parameters, float32/complex64 policy, value-only
   scope, and no outer whole-function JIT for either method. Summarize warmup,
   synchronized timing and the three-seed statistic. Identify nested coordinate
   JVP precisely; do not describe it as full-tensor AD or STDE. Use prose and
   existing notation rather than introducing another set of variables.
2. **Targets and purpose.** Introduce the six derivative patterns. Explain
   that the sixth-order patterns hold order fixed while changing exponent
   structure, and the eighth/ninth-order patterns test higher orders. List the
   theoretical direction counts alongside measured times in the main table.
3. **Main observations.** Discuss only trends actually supported by the main
   table. Distinguish the fixed-order comparisons from the higher-order ones;
   do not infer universal monotonicity or runtime optimality from Waring rank.
   A short basic output-consistency statement suffices; no accuracy-study or
   parameter-gradient claims are supported by this run.
4. **Parallel execution and batch size.** Explain the same-direction serial
   comparison and the 100/400/1600-point checks. Report verified ratios, then
   state that timing conclusions apply to the stated execution policy. Keep
   the earlier JIT reversal and compiled source/results accessible as an
   implementation ablation, not silently discarded data.

## Table plan

- Main table: six derivatives; order; direction count; nested-JVP time;
  proposed-method time; paired speedup. Three-seed mean ± sample standard
  deviation of per-seed medians, times in milliseconds.
- Supplementary table: batch-size comparisons for 123456 and 111222333, with
  a small second panel for the serial/batched direction comparison if useful.
- No figures. No memory-saving table until an appropriate runtime-memory
  measurement is defined; allocator lifetime peak alone is not sufficient.

## Boundaries for the eventual edit

Replace old first-group T4/Torch timings and related plots only after approval.
Do not mix old parameter-backward or PDE rows into the new H20 value-only table.
The current PDE benchmark paragraphs/tables also need an explicit historical
or pending status when the numerical section is rewritten; the second group
must be designed and verified separately rather than relabelled as completed.
Preserve mathematical sections and existing theorem/equation labels.
The conclusion also currently refers to T4 timings and parameter-gradient
accuracy. When the approved numerical edit is made, reconcile those dependent
claims with the new value-only experiment rather than leaving contradictory
hardware or scope descriptions. This is a consistency edit, not new evidence.

Math-paper-writing guidance informed the setting -> target -> measured result
-> scoped interpretation sequence. This structure still requires user approval.

## Verified observations available for the proposed paragraphs

- Main table at 100 points: sixth-order speedups 2.15–2.26x, eighth-order
  5.71x, ninth-order 11.33x. No claim of universal monotone rank/runtime scaling.
- The 123456 speedups at 100/400/1600 points are 2.18/2.23/2.25x;
  the 111222333 speedups are 11.33/10.64/9.45x.
- Same-direction serial/batched ratios at 100 points: 29.69x for 123456,
  14.77x for 111222333. This is an implementation-level batching ablation,
  including Python dispatch savings, not an isolated hardware-efficiency claim.
- All paired value consistency checks passed; no gradient or precision study
  was performed by this matrix. Earlier compiled comparisons remain preserved.
