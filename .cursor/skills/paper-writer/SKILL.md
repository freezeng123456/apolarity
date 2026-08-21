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

The abstract is frozen to the user-locked wording below. Do not
rewrite it, tighten it, or expand it. Grammar-only fixes already
applied: plural *neural networks*, *Experiments show*.

```
Computing high-order derivatives with neural networks is often
computationally and memory intensive, primarily due to the huge
computational graphs caused by automatic differentiation. In this
work, we decompose the derivative into directional Taylor-mode
automatic differentiation. We establish a fundamental connection
between derivative extraction and polynomial algebra, proving that
the minimal number of directional evaluations required to recover a
target partial derivative equals the Waring rank of its corresponding
monomial. We present a roots-of-unity formula that strictly attains
this theoretical lower bound. Experiments show that our method
significantly outperforms traditional automatic differentiation in
speed and memory efficiency.
```

Five sentences, one job each:

| # | Sentence job |
|---|---|
| 1 | Neural networks, cost, and the AD graph. |
| 2 | Decompose the derivative into directional Taylor-mode. |
| 3 | Derivative extraction equals monomial Waring rank. |
| 4 | Roots-of-unity attains the lower bound. |
| 5 | Experiments: faster and more memory-efficient than traditional AD. |

PINNs, residuals, and collocation still do not belong in the abstract.

Forbidden:

- A competing-method catalog (finite differences, STDE, Hutchinson).
- A first sentence that is only an application list
  (polyharmonic / plate / Helmholtz / Maxwell).
- Another structural rewrite of this paragraph.

## Thinking framework (use this before writing)

1. **What is the object?** A high-order mixed partial of a neural
   network (a composed map with a computational graph). Not a PINN
   residual and not an operator family.
2. **What is the result?** The shortest directional combination equals
   the Waring rank of the monomial, and a roots-of-unity schedule
   attains it.
3. **What is the one default?** Nested AD. Everything else waits for
   the introduction.
4. **Do not rewrite the frozen abstract.** If the user has locked
   the paragraph, only grammar they explicitly request.
5. **Only after the user accepts the plan**, write it into the TeX.

## Introduction (do not write it while the abstract is open)

Martinis four beats, after the abstract is frozen, in this order:

(a) one mixed partial, with the defining equation, then where such
    derivatives appear (polyharmonic, plate, Helmholtz, Maxwell as
    examples, not as the opening),
(b) two ways to compute it: exact repeated AD, with the $O(d^{p})$ and
    $O(2^{p-1}L)$ scalings cited, then estimators (FD, Hutchinson, STDE),
(c) this paper's construction, more specific than the abstract,
(d) where the main results live.

Do not open the introduction with a stack of AD jargon. Bettencourt
and STDE say *applying automatic differentiation repeatedly* and
*backpropagation*. The scaling of repeated reverse AD is cited from
STDE, not described as ``the graph grows.''
