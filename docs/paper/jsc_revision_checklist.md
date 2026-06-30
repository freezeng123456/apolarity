# JSC Revision Checklist

## Completed in this revision

- Reframed the paper as a scientific-computing method for one fixed high-order single-monomial partial derivative.
- Removed unsupported claims that the experiments benchmark against STDE; STDE is now discussed as related complementary work.
- Replaced the strong C2 wall-clock claim with the supported direction-count/complexity claim and measured timing caveat.
- Aligned the microbenchmark table with `results/quick_compare_T4_20260602_0303.txt`.
- Aligned the PINN experiment description with `results/pinn_5min_20260602_0436.txt` (`hidden=32`, `depth=4`, `complex128`).
- Corrected the repeated-index polarization count for `(2,2,2)` from the raw sign-count interpretation to the coalesced count used by the implementation.
- Added a related-work positioning paragraph, cost discussion, and numerical-stability caveat.
- Added `tests/test_paper_patterns.py` to lock the pattern table counts.
- Added `experiments/generate_paper_tables.py` for CSV-to-LaTeX table snippets.
- **Width protocol (2026-06):** removed outdated “√2 × H parameter matching” language everywhere in the active codebase and paper. The official design is: literal widths; real baselines at `hidden=128`; complex sinh at `64` and `128` (bracketing study). See `experiments/README.md`, `osc_experiments.tex` §\ref{sec:bm:protocol}, and `docs/apolarity_benchmarks.html`.

## Remaining risks before external submission

- The PINN table currently reports one seed; add multiple seeds and uncertainty if time allows.
- The saved quick benchmark is a text log. For final submission, regenerate it as CSV/JSON with `benchmark_single_monomial.py` and generate table rows from that file.
- No STDE baseline is implemented in this repository. Do not claim STDE benchmark results unless a separate implementation and saved result are added.
- `waring_real_jet` appears only in historical logs and is not part of the current public backend API.
- Consider adding one additional manufactured PDE/operator pattern to broaden the JSC numerical evidence.

---

## Paper structure: why it feels “not JSC” and how to fix it

### Diagnosis (current `jsc_paper_main.tex`)

The manuscript currently reads as **two papers glued together**:

| Block | Content | Typical JSC role |
|-------|---------|------------------|
| §1–4 | Waring ↔ directional schedule, Taylor jet, backends | **Core method** (good) |
| §5 | Microbenchmark + one 4D sixth-order PINN (60 s, 1 seed, backend comparison) | **Method validation** (good, but thin) |
| §6 (`osc_experiments.tex`) | Nine PDE families, 45 instances, 4 architectures, figures + long narrative | **Separate application / companion study** (too large for an appendix to §5) |

Problems for a JSC reader:

1. **Dual narrative in abstract and contributions** — theory + backend speed + oscillatory PINN sweep compete for attention; the abstract over-promises “competitive-to-superior on the majority of instances.”
2. **Two experiment sections with different hardware/budgets** — §5 uses T4 / 60 s / backend ablation; §6 uses H20 / 600 s / architecture shootout. JSC expects one coherent experimental protocol per claim.
3. **§6 is JCP-scale** — per-family PDE + table + 3-panel figure + 1–2 pages of discussion × 9 families ≈ 15–20 pages of results alone; JSC numerical sections are usually compact and claim-driven.
4. **Claims ladder is unclear** — C1–C3 (method) vs “34/45 wins” (ansatz marketing) are different epistemic levels mixed in Discussion/Conclusion.
5. **Article class** — `article` + very long appendix-style oscillatory dump; JSC papers are typically tighter, with experiments directly supporting the algorithm section.

`osc_experiments.tex` itself is **internally well written** (protocol paragraph, width bracketing, honest summary). The issue is **placement and length**, not the per-family write-ups.

### Recommended target structure (single JSC paper)

**Option A — Method-first (recommended for JSC)**

```
1. Introduction
   - Problem: one fixed ∂^α u for high-order PINN residuals
   - One sentence on downstream use (complex sinh ansatz), not a second paper

2. Background / related work (short)

3.–4. Theory (keep current §2–3, tighten examples)

5. Algorithm & implementation (current §4)
   - One algorithm box, complexity remark, software note

6. Numerical experiments  ← ONE section only
   6.1 Verification: microbenchmark (accuracy + timing, Table micro)
   6.2 Integration: 4D manufactured sixth-order PINN (backend ablation, multi-seed)
   6.3 Expressivity snapshot (NOT full 45-instance dump):
       - Design: width protocol (128 real vs complex 64/128), 600 s, 2 seeds
       - Table 1: suite overview (current Table suite)
       - Table 2: **aggregate** summary only (current Table res-summary)
       - Figure 1: 2×2 panel — polyharmonic 2D order, Helmholtz k, chirp a, Maxwell a
       - 1 page prose: where it wins / where it does not (honest, from §6.9 summary)

7. Discussion & limitations (short)

8. Conclusion (3 paragraphs max)

Appendix A: Proofs
Appendix B (optional): Per-family tables/figures moved from main text
Supplementary material / separate arXiv note: full 45-instance CSV + all fig_*.pdf
```

**Option B — Split into two submissions**

- **JSC:** Waring + Taylor jet + microbenchmark + 4D PINN (+ maybe 1 compact expressivity table).
- **JCP or CMAME:** Full oscillatory suite as “Complex sinh PINNs for high-order oscillatory PDEs” citing the JSC backend.

Option B is cleaner if the oscillatory suite is a main selling point for the group.

### Concrete edits to `jsc_paper_main.tex`

1. **Abstract** — Lead with the derivative backend; demote oscillatory suite to “we additionally report a width-robustness study on nine manufactured families (summary Table X)” without win counts.
2. **Contributions (item 5)** — Rename to “Expressivity snapshot” not a sixth contribution equal to the theorem; or move to Discussion as “additional evidence.”
3. **Merge §5 and §6 headers** — Replace `\input{osc_experiments}` opening `\section{...}` with `\subsection{Oscillatory PINN suite (summary)}` and `\input{osc_experiments_summary.tex}` (new shortened file); move full `osc_experiments.tex` to `docs/paper/supplement_oscillatory.tex` or appendix.
4. **Harmonize hardware** — Either rerun §5 PINN on H20 for one paragraph of consistency, or state explicitly “§5 validates backends; §6 validates ansatz under a different budget.”
5. **Tone** — Replace “34/45 wins” in abstract/conclusion with “advantage concentrates on high-order balanced operators and complex-valued fields; not universal.” (Already in §6.9 — promote that honesty to the front.)
6. **Bibliography** — JSC wants lean related work; move long PINN literature drops to oscillatory supplement.
7. **Class** — Consider `elsarticle` / JSC template if targeting the journal formally.

### What is already correct (do not rewrite)

- `osc_experiments.tex` §\ref{sec:bm:protocol} width-robustness paragraph (lines 80–96).
- Auto-generated `w_*.tex` table captions (“real @128; complex @128 and @64”).
- `experiments/README.md` width study description.
- `docs/apolarity_benchmarks.html` (after 2026-06 protocol fix).

### Suggested writing order

1. Freeze Option A vs B with co-authors.
2. Cut §6 main text to summary + 4 figures (1–2 days).
3. Add 3–5 seeds to §5 PINN table (1 day compute).
4. Shorten abstract + contributions to match cut structure.
5. External read: “Can a JSC referee find the algorithm and its validation in < 25 pages?”
