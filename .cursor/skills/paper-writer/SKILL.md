---
name: paper-writer
description: Abstract and introduction story template for this manuscript. Distilled from Martinis/von Delft (sci-brain paper-writer) and Tao/Knuth/Pak (write-math). Use before drafting or rewriting the abstract or introduction.
paths:
  - "**/*.tex"
  - "docs/paper/**"
---

# Paper writer (this manuscript)

Read this skill before drafting or rewriting the abstract. It is the
story template. Voice and TeX wrap stay in
`.cursor/skills/english-paper-writing/SKILL.md`. Abstract-versus-
introduction architecture stays in
`.cursor/skills/jcp-paper-structure/SKILL.md`.

Sources:

- QuantumBFS/sci-brain `skills/paper-writer` (Martinis, von Delft)
- cboone/agent-harness-plugins `write-math` (Tao, Knuth, Pak, Halmos)
- Thirty JCP abstracts plus STDE (Shi–Hu–Lin–Kawaguchi, NeurIPS 2024)

Do not draft a new abstract until the user has accepted the sentence
plan below. Do not jump from a complaint about one sentence to a
whole-paragraph rewrite.

## Iron rules that govern the abstract

1. **One concept per sentence.** If two ideas must sit together, both
   are already familiar. Break any sentence with three new ideas.
2. **One paragraph, 5–10 lines.** Roughly one sentence per body
   section, not a compressed literature review.
3. **Results before a method tour.** Write-math: *We consider
   [problem]. We show that [result]. The construction uses [method].*
   History and competitor catalogs belong in the introduction.
4. **At most one default method.** Name the evaluation this paper
   replaces or times against (here: nested automatic differentiation).
   Do not list finite differences, randomized estimators, stochastic
   contractions, or STDE in the abstract. Those are introduction jobs.
5. **No citations, no `$`, no experimental numbers, no marketing.**
6. **Do not advertise an absence.** Exactness is a property of the
   formula, not a list of things the formula does not need.

## What other abstracts actually do

They do **not** survey the field. They name, at most, the class they
sit in or the one default they replace.

| Paper | What the abstract names besides itself |
|---|---|
| Raissi–Perdikaris–Karniadakis (PINNs) | Almost nothing. Introduce the object, then what it does. |
| Chan 2018 (entropy-stable DG) | The class it sits in (diagonal-norm SBP), then the construction. |
| Babbar et al. 2026 (AD for Lax–Wendroff) | The one procedure it replaces (approximate Lax–Wendroff). |
| STDE (NeurIPS 2024) | Back-propagation as the cost model; two *problem scalings* already treated; then their construction. Not a catalog of FD / Hutchinson / smoothing. |

Finite differences, sampling, and neighboring estimators appear in
those papers' **introductions**. Putting them in the abstract is how
the abstract starts to read like related work.

## Abstract template for this paper

Five or six short sentences, one job each. Map them to the body.

| # | Sentence job | Body section it reports |
|---|---|---|
| 1 | The evaluation: one high-order mixed partial of a smooth map. A short *where it arises* clause is optional, not a tourist list. | Introduction / problem |
| 2 | The computational primitive we sit on: Taylor-mode returns the directional coefficients of that order. | Preliminaries |
| 3 | The reduction: any such partial is a linear combination of those coefficients. | Preliminaries |
| 4 | The main result: the shortest combination is the Waring decomposition of the corresponding monomial. | Method / analysis |
| 5 | The construction: a roots-of-unity formula attains that length. | Method |
| 6 | What was checked: the formula is exact, and it is substantially faster than nested automatic differentiation. | Experiments |

That is the whole abstract. Nested AD appears once, in sentence 6, as
the timing baseline. Taylor mode appears once, in sentence 2, as the
substrate, not as a competitor.

Forbidden in this template:

- A sentence that names two or more alternative methods.
- Opening on PINNs, residuals, or collocation.
- A first sentence that is only an application list
  (polyharmonic / plate / Helmholtz / Maxwell) with no evaluation.
- Restating STDE, Hutchinson, or finite differences.

## Thinking framework (use this before writing)

1. **What is the object?** One mixed partial, not a PINN, not an
   operator family.
2. **What is the result?** The shortest directional combination equals
   the Waring rank of the monomial, and a roots-of-unity schedule
   attains it.
3. **What is the one default?** Nested AD. Everything else waits for
   the introduction.
4. **Write the six jobs as six short sentences.** Then check: one
   concept each; no `$`; no catalog.
5. **Only after the user accepts the plan**, write it into the TeX.

## Introduction (do not write it while the abstract is open)

Martinis four beats, after the abstract is frozen:

(a) field-level question and why the evaluation matters,
(b) prior methods by what they *do* (nested AD, FD, randomized
    contractions including STDE, Taylor mode as a primitive),
(c) what this paper constructs, more specific than the abstract,
(d) where the main results live.

The introduction is allowed the competitor survey. The abstract is not.
