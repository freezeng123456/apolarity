# Multi-index expansion of a repeated directional derivative

## Question and recommendation

The line in Section 3.1 should not explain the coefficient by saying that a
particular basis vector “occurs” a prescribed number of times.  The clean
mathematical route is to identify the repeated Fréchet derivative with a
constant-coefficient differential operator and then apply the multinomial
theorem:

```tex
Since
\[
    D^p u(\bx)[\bv,\ldots,\bv]
    = \left(\sum_{j=0}^{n}v_j\partial_j\right)^p u(\bx),
\]
the multinomial theorem gives
\begin{equation}\label{eq_directional_expansion}
    \Tp(\bx;\bv)
    =
    \sum_{|\beta|=p}
    \frac{\bv^\beta}{\beta!}\,\partial^\beta u(\bx).
\end{equation}
```

Here `|beta|=p` is the existing multi-index convention, with the indices
restricted to the active coordinates `0,...,n`; there is no need to repeat
`\beta\in\N_0^{n+1}` below the summation.  The equality follows from

\[
\frac{1}{p!}\binom{p}{\beta}
 = \frac{1}{\beta!},
\qquad
\binom{p}{\beta}=\frac{p!}{\beta!},
\]

and all notation (`|\beta|`, `\beta!`, `\bv^\beta`, and `\partial^\beta`) is
already introduced in Eq. (7), up to the section's relabeling of the active
coordinates.  If one wants to make that relabeling completely explicit, use
the short parenthetical “where the sum is over multi-indices in the active
coordinates” in the preceding sentence, rather than adding a second set
qualification to the display.

## Primary/high-trust sources checked

### Loomis and Sternberg, *Advanced Calculus*

The most direct standard reference located is L. H. Loomis and S. Sternberg,
*Advanced Calculus*, revised edition (Jones and Bartlett, 1990; author-hosted
copy at Harvard).  Section 3.17, “The Taylor Formula,” first writes the
directional Taylor term as

\[
\frac{1}{m!}\left(\sum_i y_i\frac{\partial}{\partial x_i}\right)^mF(a)
 = \frac{1}{m!}\sum_{i_1,\ldots,i_m}
 y_{i_1}\cdots y_{i_m}
 \frac{\partial^mF}{\partial x_{i_1}\cdots\partial x_{i_m}}(a),
\]

and then rewrites the same term in multi-index notation as

\[
\frac{1}{m!}\sum_{|\mathbf{k}|=m}
 \binom{m}{\mathbf{k}}D^{\mathbf{k}}F(a)\,\mathbf{x}^{\mathbf{k}}.
\]

This is precisely the desired operator-to-multi-index transition.  It uses
the multinomial coefficient explicitly and avoids any informal counting
language.  The Harvard author page identifies the book and authors, and the
PDF is the revised edition:

- [Harvard author page](https://people.math.harvard.edu/~shlomo/)
- [Loomis--Sternberg, *Advanced Calculus*, author-hosted PDF](https://people.math.harvard.edu/~shlomo/docs/Advanced_Calculus.pdf)
- [University of Pennsylvania catalog record](https://onlinebooks.library.upenn.edu/webbin/book/lookupid?key=olbp53325)

This is the recommended citation for the displayed identity if the manuscript
needs a standard mathematical source.  A suitable BibTeX key would be
`LoomisSternberg1990`; that key is not currently present in
`work/apolarity_pr10/docs/paper/jsc_paper_main.bib`.

### Griewank and Walther (2008)

The official SIAM page confirms that Chapter 13 is “Taylor and Tensor
Coefficients,” in *Evaluating Derivatives: Principles and Techniques of
Algorithmic Differentiation*, 2nd ed. (2008), pp. 299--334.  The chapter is a
good source for the computational Taylor-coefficient/jet context, and the
manuscript already has the key `Griewank2008`.

- [SIAM Chapter 13 page](https://epubs.siam.org/doi/10.1137/1.9780898717761.ch13)

The accessible official page exposes the chapter metadata and abstract but
not the chapter's full equations.  Thus it is appropriate to cite
`Griewank2008` for Taylor-mode AD and coefficient conventions, but it is not
the best auditable citation for the exact multinomial identity unless the
authors have access to the chapter and verify the relevant page/equation.

### Bettencourt, Johnson, and Duvenaud (2019)

The authors' workshop paper presents Taylor-mode automatic differentiation in
JAX and explicitly motivates it through propagation of higher-order Taylor
polynomials; it cites Griewank and Walther Chapter 13 as the prior AD source.
The paper is useful support for the manuscript's computational interpretation
of `T_p`, but its presentation is algorithmic/jet-oriented rather than a
standard statement of the multi-index identity above.

- [OpenReview paper PDF](https://openreview.net/pdf?id=SkxEF3FNPH)
- [NeurIPS workshop record](https://neurips.cc/virtual/2019/14782)

Accordingly, `Bettencourt2019` should remain attached to claims about
Taylor-mode AD, not be used as the sole citation after Eq. (8).

## Citation placement

The citation should follow the sentence introducing the standard operator
identity, or follow the display, for example:

```tex
Since $D^p u(\bx)[\bv,\ldots,\bv]
= (\sum_{j=0}^{n}v_j\partial_j)^p u(\bx)$, the multinomial theorem gives
\cite{LoomisSternberg1990}
\begin{equation} ... \end{equation}
```

If adding a new textbook reference is undesirable, leave the equation
uncited and retain `\cite{Griewank2008}` in the surrounding Taylor-mode AD
discussion; the proof is a one-line application of the multinomial theorem.
Do not cite `Bettencourt2019` as authority for the multi-index coefficient
unless its full text is checked and an exact page/equation is added.

## Source-to-claim status

| Claim | Best source | Status |
|---|---|---|
| Repeated directional derivative is `(v dot nabla)^p u` | Loomis--Sternberg, Sec. 3.17 | Directly supported |
| Expansion coefficient is `1/beta!` after dividing by `p!` | Loomis--Sternberg, Sec. 3.17 | Directly supported via `binom(p,beta)=p!/beta!` |
| Taylor jets are a computational AD representation | Griewank--Walther, Ch. 13; Bettencourt et al. (2019) | Directly supported at chapter/paper level |
| Exact Eq. (8) appears in Griewank--Walther (2008) | Official SIAM metadata only | Not verified; do not claim |

