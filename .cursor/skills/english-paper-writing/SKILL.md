---
name: english-paper-writing
description: Write and edit English academic LaTeX manuscripts (JCP/SIAM, docs/paper/*.tex). Use for 写论文, 改 tex, 摘要, 引言, 排版, equation/label/ref, source wrapping, and when removing apologetic or AI-flavoured prose. Enforces TeX source conventions and a non-apologetic scientific voice.
paths:
  - "**/*.tex"
  - "docs/paper/**"
---

# English academic paper writing

Read this skill before drafting or editing manuscript TeX. It has two jobs:
source formatting, and scientific voice. Do not treat either as optional.

For the abstract sentence plan, also read
`.cursor/skills/paper-writer/SKILL.md`. For abstract-versus-introduction
architecture, section order, and how JCP method papers open, also read
`.cursor/skills/jcp-paper-structure/SKILL.md`. An introduction that
restates the abstract with citations is wrong even if the voice and the
TeX wrap are clean.

The TeX rules below are the manuscript formatter. The voice rules are
mandatory for abstracts, introductions, conclusions, and contribution claims.

## When to use

- Writing or rewriting abstract, introduction, method, experiments, or conclusion.
- Local TeX formatting: equations, theorems, labels, cross-references, line breaks.
- The user asks to 写论文, 改论文, 排版, 换行, 去 AI 味, or stop sounding apologetic.

Do not turn a local formatting request into a whole-manuscript rewrite unless
the user asks for that pass.

## Scientific voice: do not confess

A paper states what it does and what it obtains. It does not pledge loyalty
to a hypothetical reviewer by listing what it does not do.

### Banned in abstract and introduction

Do not write any of the following, or close paraphrases:

- Advertising an absence: "requires no linear solve", "without a linear
  solve", "no step size", "no sampling variance" used as a selling point
  rather than a contrast with a named competing method in one sentence.
- Disclaiming a claim nobody made: "no claim of joint optimality", "we do
  not claim", "makes no claim", "local to one multi-index" as a caveat.
- A list of non-contributions: "X is not a contribution", "Taylor mode is
  not", "PINN training is not", "the Waring rank formula is not".
- Scope groveling: "the question is narrower", "we only treat", "this paper
  merely", "serves only to check".
- Reviewer-facing limitation oaths in the front matter. If a method applies
  to one mixed partial, state the problem as that evaluation. Do not add
  "and we do not treat the sum as jointly optimal".

### Where a restriction may appear

A usage fact belongs in the method, as an instruction, not as an apology.
Allowed: "A differential operator that is a linear combination of mixed
partials is evaluated by applying the schedule to each monomial."
Forbidden: "we make no claim of joint optimality for the sum."

A genuine experimental finding may appear in the experiments or conclusion
("at order two the cheaper derivative does not change the optimizer"). That
is a result, not a confession.

Related work names other methods and what they do. It does not apologize
for not being those methods.

### How to state a result

Write the result in the affirmative, then stop.

- Yes: "We construct a directional Taylor-jet schedule whose length equals
  the Waring rank of the associated monomial, and realize it by a
  roots-of-unity formula."
- No: "The construction is local to one multi-index: it is a minimum for
  that derivative and makes no claim for a sum of several."

Cite prior tools (Taylor mode, Waring rank, PINN residuals) as tools. Do
not spend front-matter sentences explaining that you did not invent them.

### Diction

Prefer plain, specific words. Do not use fashionable ML padding or AI-typical
academic filler.

Do not use: surrogate, stencil (say network, finite differences, formula),
delve, leverage, utilize, tapestry, landscape, pivotal, underscore, foster,
holistic, multifaceted, seamless, meticulous, elucidate, "in this paper we
present" as a stock opener, "the remaining question is", "the effectiveness
of the proposed X is demonstrated", "it is worth noting", "not merely X but
Y", "primary concern rather than an implementation detail".

Do not use `---` em dashes in prose. Prefer commas, parentheses, or a new
sentence.

US spelling. No displayed mathematics in the abstract. No experimental
numbers in the abstract unless the user asks for them.

### Abstract shape

One paragraph, 5–10 lines. One concept per sentence. Factual.
Stand-alone. No citations. Uncommon terms defined once if they must
appear. Follow the four-sentence plan in
`.cursor/skills/paper-writer/SKILL.md`.

Order:

1. Neural networks and why nested AD is expensive (the graph).
2. Directional Taylor-mode, and the Waring-rank lower bound.
3. The roots-of-unity formula that attains the bound.
4. Exact, and faster and more memory-efficient than nested AD.

Do not use the abstract as a method catalog. Finite differences,
randomized estimators, stochastic contractions, and STDE belong in the
introduction. Do not list limitations, non-claims, or future work.

The introduction is not a second abstract. Its first sentence, its
paragraph jobs, and its literature block are specified in
`.cursor/skills/jcp-paper-structure/SKILL.md`.

## TeX source formatting

Preserve mathematical meaning, notation, macros, labels, numbering, and
authorial intent. Do not invent assumptions, lemmas, propositions, theorem
statements, proof steps, citations, bibliography keys, or mathematical
claims. Do not silently change notation, theorem numbering, section
structure, or citation style.

### Line breaks

- Wrap prose to a consistent source width of 80 columns so the TeX is easy
  to read in an editor. Break at phrase boundaries (commas, conjunctions,
  prepositional phrases), not in the middle of a word, a citation, or a
  math token.
- Do not require one sentence per source line. A long sentence may occupy
  several wrapped lines; a short sentence may share the wrap with the next
  only if the width stays even. Prefer filling to the same column rather
  than leaving ragged one-word lines.
- Do not insert a blank line inside a paragraph (that starts a new
  paragraph in LaTeX).
- Avoid line breaks in the middle of displayed equations.

### Environments, math, and references

- In theorem, lemma, proposition, assumption, remark, equation, and similar
  environments, indent the content by 4 spaces relative to the environment
  tags:

  ```tex
  \begin{theorem}
      Content of the theorem.
  \end{theorem}
  ```

- In displayed and inline math, add source-code spaces that reflect
  mathematical structure when practical: after commas in argument lists and
  around `=`, `+`, `-`, and other binary operators.
- Prefer `f(x, \mu, u) = ...` over `f(x,\mu,u)=...`.
- When a superscript or subscript has more than one source character or
  token, wrap it in braces: `m_{\mu}`, `x_{ij}`, `a^{n+1}`.
- Prefer `\begin{equation}...\end{equation}` or `equation*` for displayed
  equations. Avoid `\(...\)`, `\[...\]`, `$$...$$`, and `\displaymath`.
- Use only `$...$` for inline math.
- For a short standalone definition in `equation` or `equation*`, keep it
  on one source line. Do not split it just to isolate `=`, `\quad`, commas,
  or a short trailing term.
- In `align`, `align*`, `aligned`, `split`, `gathered`, `cases`, and similar
  environments, keep each equation row on one source line. Break only after
  `\\`.
- For multi-line derivations, put `aligned`, `split`, or `cases` inside
  `equation` or `equation*` when that matches the surrounding manuscript.
- Place `\label{...}` immediately after `\begin{equation}`. In `align`,
  place `\label{...}` at the end of the relevant equation line.
- Replace "the previous theorem", "the above equation", and similar vague
  pointers with `Theorem~\ref{...}`, `Lemma~\ref{...}`,
  `Proposition~\ref{...}`, or `\eqref{...}`. If the manuscript already uses
  `\cref`/`\Cref`, keep that style.
- If a cross-reference is needed and the object has no label, add a local
  `\label{...}` in the same edit.
- Write sequences with parentheses, not curly braces:
  $(a_i)_{i \ge 0}$, not $\{a_i\}_{i \ge 0}$.
- Do not use `\qquad`; prefer `\quad`.

### Formatter examples

Short standalone `equation*` stays one source line:

```tex
\begin{equation*}
    \Vcal(t, v, c; \mu) = \sup_{u \in \mathcal{U}} J(u, \mu).
\end{equation*}
```

In `aligned`, one row per source line, break only after `\\`:

```tex
\begin{equation*}
\begin{aligned}
    T = 1, \quad \mu_g = 0.4, \\
    a = 2.
\end{aligned}
\end{equation*}
```

Readable spacing and braced multi-character subscripts, displayed and inline:

```tex
\begin{equation}\label{eq_sr_cost}
    f(x, \mu, u) = \frac{1}{2}u^2 - q\,u\bigl(m_{\mu}(t) - z\bigr) + \frac{\epsilon}{2}\bigl(m_{\mu}(t) - z\bigr)^2,
    \quad g(x, \mu) = \frac{c}{2}\bigl(m_{\mu}(T) - z\bigr)^2,
\end{equation}
```

```tex
The drift is $b_z(x, \mu, u) = a\bigl(m_{\mu}(t) - z\bigr) + u$.
```

## Editing discipline

- Read the relevant paragraph or subsection before changing it. Do not
  globally search-and-replace manuscript prose.
- Judge every sentence in its local mathematical and rhetorical context.
- Keep claims aligned with the implemented backends and saved experiment
  data. Do not invent speedups, ranks, or comparisons.
- Preserve established terminology in background, citations, and related
  work, even when similar wording is changed elsewhere.
- Recheck equations, cross-references, captions, and numerical claims after
  editing.
- If a formatting change could alter numbering, references, or meaning
  beyond the local target, stop and say so instead of rewriting globally.
- After a voice rewrite, grep the edited region for: `no claim`, `not a
  contribution`, `linear solve`, `narrower`, `serves only`, `it is worth`,
  `proposed method`, `surrogate`, `stencil`.

## Working method

1. Identify the TeX anchor: file, section, paragraph, theorem, equation, or
   label.
2. Read nearby macros, labels, and local style.
3. Apply voice rules if the text is prose; apply formatter rules always.
4. Make the smallest reviewable edit.
5. If labels, numbering, or equation environments changed, compile or inspect
   the log; otherwise skip a full build unless asked.
