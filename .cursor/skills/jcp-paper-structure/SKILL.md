---
name: jcp-paper-structure
description: JCP manuscript architecture learned from 30 Journal of Computational Physics papers. Use when writing or rewriting the abstract, introduction, related work, section outline, or when the introduction is restating the abstract.
paths:
  - "**/*.tex"
  - "docs/paper/**"
---

# JCP paper structure

Read this skill before drafting or rewriting an abstract or introduction
for a Journal of Computational Physics manuscript. It complements
`.cursor/skills/english-paper-writing/SKILL.md` (voice and TeX) and
`.cursor/skills/paper-writer/SKILL.md` (abstract sentence plan). This
file is about **what each front-matter block is for**. The voice skill
does not authorize an introduction that is the abstract with citations.
The paper-writer skill does not authorize an abstract that surveys
other methods.

The rules below were read off the abstracts and introductions of 30 JCP
papers (classics, traditional discretizations, and SciML method papers).
The corpus list is `references/corpus.md`.

## The abstract and the introduction are different documents

The abstract is a stand-alone report of the finished paper. The
introduction is a new argument that a reader who skipped the abstract
still needs. If the introduction can be recovered by adding citations,
displays, or names to the abstract, it is wrong.

| | Abstract | Introduction |
|---|---|---|
| Reader | Someone who will not open the paper | Someone who will read the method |
| Job | Report the object, the result, the construction, and the one check | Motivate the evaluation, survey how it is done, then state the construction with enough mechanism to start §2 |
| Citations | None | Named methods, with what each one does |
| Opening sentence | The object and where it is used | One level more general **or** more specific than the abstract. Never the same sentence |
| Equations | None (no `$` in this manuscript's abstract) | Allowed. Display an identity only when the introduction is stating that construction, not when repeating the abstract |
| Length | One paragraph, 120–220 words | Typically 1.5–3 pages; 4–7 **long** paragraphs, not seven three-sentence clones of the abstract |

Test: delete every citation and displayed equation from the introduction.
If what remains is the abstract, rewrite the introduction.

## How JCP introductions actually open

They do **not** open by reciting the abstract's application list. They
open on a method class, a computational property, or a problem class
that the paper then specializes.

- Chan 2018 (entropy-stable DG): high-order accuracy per degree of
  freedom, then DG on unstructured meshes. The abstract already named
  entropy conservation; the intro does not start there.
- Ranocha–Glaubitz 2024 (upwind SBP): “Stability and robustness are
  crucial properties of numerical methods for conservation laws.”
- Adjoint constraints 2024: “Numerous problems … are cast in terms of
  optimization of systems described by PDEs.”
- HDG transmission 2025: “DG methods are widely used … because …”
- Babbar et al. 2026 (AD for Lax–Wendroff): “This work is a contribution
  to Lax–Wendroff methods …” then the Cauchy–Kovalevskaya flux.
- Jiang–Shu 1996 (WENO): continues Liu–Osher–Chan; the intro is a
  history of ENO/WENO stencils, not a restatement of the abstract.
- Raissi–Perdikaris–Karniadakis 2019: ML and scarce data, then prior
  physics as a regularizer. The abstract already defined PINNs.
- McClenny–Braga-Neto 2023: SciML / PINNs as an alternative to
  time-stepping, then where the baseline PINN fails, then SA-PINNs.
- Hu–Shi–Karniadakis–Kawaguchi 2025: PINNs in high dimension, then
  randomized smoothing and its bias, then the analysis. The abstract
  reported the bias–variance result; the intro does not start with it.
- Hermite optimization 2026: quantum optimal control as a field, then
  open-loop versus closed-loop, then the discrete adjoint.

Pattern: the first paragraph **widens or specializes**. It does not
translate the abstract.

For a derivative-evaluation paper this means: open on the evaluation
of high-order partials of a smooth map (nested AD, the stored graph),
not on “high-order derivatives arise in polyharmonic, plate, Helmholtz,
and Maxwell models.” That sentence belongs to the abstract. The
operator families may appear later as examples of maps that request
those partials, or in the experiment paragraph.

## Paragraph jobs in a JCP method introduction

Use **long** paragraphs (eight to twenty source lines), one job each.
Do not emit a sequence of three-sentence snippets that retell the
abstract.

Typical order for a numerical-method paper (Chan, Jiang–Shu, HDG,
Ranocha, Babbar, adjoint):

1. **Situation.** Why this class of computation exists. Mechanism of
   the default method, and why its cost or stability matters.
2. **Literature as methods, not a name list.** Each cited work gets
   what it *does* (ENO chooses one stencil; WENO weights all of them;
   Hutchinson estimates a trace; STDE estimates a contraction with
   Taylor-mode tangents). This survey is an introduction job. The
   abstract names at most the one default it times against.
3. **The strain those methods leave.** Stated as a fact about the
   methods, not as “the remaining question is” and not as a list of
   things this paper will not do.
4. **This construction.** First appearance, in the body, of what the
   paper builds. More specific than the abstract: an identity, an
   assumption, a length, a schedule. “In this work we …” is allowed
   **here**, after the literature, not as sentence two of the paper.
   Do not use “in this paper we present” as a stock opener.
5. **What is verified.** The comparison protocol in words. No wall of
   experimental numbers unless the user asks.
6. **Organization.** One short paragraph. Traditional JCP puts related
   work in the introduction (Jiang–Shu, Chan, HDG). Some SciML JCP
   papers add a separate Related-work section (Hu 2025). This
   manuscript keeps related work in the introduction.

SciML JCP papers (Raissi, McClenny, Hu, Wang NTK) follow the same
division of labor. They are longer on motivation and shorter on
displayed equations. Traditional JCP often displays the governing
equation in paragraph 1 or 2. Display an equation in the introduction
only when stating the construction, not to decorate a recap.

## Whole-paper skeleton (JCP method paper)

The 30-paper corpus, including the classics, is almost always:

1. Abstract (one paragraph)
2. Keywords
3. Introduction (literature included)
4. Problem statement / preliminaries / discretization
5. The method and its analysis
6. Numerical experiments
7. Conclusions
8. Appendix if a proof or schedule table would break the method
   section

There is usually **no** stand-alone Related-work section. There is
usually **no** contribution bullet list in traditional JCP; SciML JCP
sometimes uses bullets (McClenny, Wang NTK). This manuscript states
the construction in prose, then organizes the sections.

The conclusion restates the result in one paragraph, then the
numerical finding, then a short continuation. It is not a third copy
of the abstract.

## What this manuscript must not do

- Copy the abstract's first sentence into the introduction.
- Walk the abstract's plot (need → nested AD → finite differences /
  sampling → Taylor mode → Waring → faster) as the introduction's
  only spine, with citations taped on.
- Open the introduction on PINNs, residuals, or collocation. Those
  are the verification workload. They appear when the comparison is
  described.
- Open a paragraph with “in this paper we present.”
- Use “the remaining question is,” “the effectiveness of the proposed
  method is demonstrated,” or a list of non-contributions.
- Put formula symbols in the abstract.
- Mix a full identity into a running sentence in the introduction.
  If the identity is stated, display it.

## Working method

1. Write or freeze the abstract as a stand-alone report.
2. Outline the introduction as jobs 1–6 above. Check that job 1 is
   not the abstract's first sentence and that job 4 is more specific
   than the abstract's “we show.”
3. Draft long paragraphs. Cite methods by what they do.
4. Apply `.cursor/skills/english-paper-writing/SKILL.md` for voice
   and TeX.
5. Run the delete-citations test. Rewrite if the introduction
   collapses to the abstract.
