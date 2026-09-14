# Active-coordinate notation in primary Waring references

## Evidence from the primary sources

- **Carlini--Catalisano--Geramita (2012).** In *The Solution to Waring's Problem for Monomials*, the authors first state a general reduction in **Remark 2.3, p. 4**: the rank of a form can be computed in the polynomial ring with the least number of variables containing the form. They then write a monomial directly in that smaller ring, `M=x_1^{b_1}\cdots x_n^{b_n}`, with `1\le b_1\le\cdots\le b_n`, and say that it is enough to work in `k[x_1,\ldots,x_n]`. In **Corollary 3.3, p. 5**, the monomial is again treated directly in the reduced ring; no coordinate subspace `span\{e_i\}` is introduced. The paper therefore removes variables with zero exponents by passing to the least polynomial ring containing the monomial, rather than by defining an ambient coordinate subspace. [arXiv:1110.0745](https://arxiv.org/abs/1110.0745), [Journal of Algebra version](https://doi.org/10.1016/j.jalgebra.2012.07.028).

- **Buczyńska--Buczyński--Teitler (2013).** In *Waring Decompositions of Monomials*, the authors make the same reduction explicitly in the introduction (**p. 2**): for a monomial `F=x_0^{d_0}\cdots x_n^{d_n}` with `0<d_0\le\cdots\le d_n`, they work in `\mathbb C[x_0,\ldots,x_n]`. Their subsequent **Remark on p. 3** says that if a polynomial depends only on the first `n+1` variables but is viewed in a larger polynomial ring, a minimal Waring decomposition still uses linear forms depending only on those variables. The explicit-expression section (**p. 3**) consequently uses only the variables occurring in the monomial. Again, the source does not introduce a span of coordinate vectors; it either works in the smallest polynomial ring or pads the ambient ring while observing that the extra variables do not occur in a minimal decomposition. [arXiv:1201.2922](https://arxiv.org/abs/1201.2922), [Journal of Algebra version](https://doi.org/10.1016/j.jalgebra.2012.12.011).

- **STDE++ (Shi--Hu--Lin--Kawaguchi, 2026).** The paper is an application/method paper rather than a Waring-rank reference. Its abstract and main presentation formulate Taylor-jet directions through input tangents and contractions; they do not motivate an auxiliary coordinate subspace for the nonzero entries of a multi-index. [JMLR article and primary PDF](https://www.jmlr.org/papers/v27/25-1474.html).

## Recommendation for Section 3.1

The current `\operatorname{span}\{e_0,\ldots,e_n\}` is mathematically valid but unnecessary for the exposition. It introduces a geometric object even though the proof only needs a list of coefficients and the associated padded direction vectors. Following the primary Waring references, use the reduced coordinates directly:

```latex
After permuting the coordinates, write
\[
    \nu=(\nu_0,\ldots,\nu_n,0,\ldots,0),
    \qquad 1\leq \nu_0\leq\cdots\leq\nu_n.
\]
For $\bz=(z_0,\ldots,z_n)$, set
\[
    \bz^\nu:=z_0^{\nu_0}\cdots z_n^{\nu_n}.
\]
```

When returning to the original `d`-dimensional input, define each direction by padding the reduced coefficient list with zeros, for example

```latex
\[
    \bv_\zeta=(1,\zeta_1,\ldots,\zeta_n,0,\ldots,0)\in\mathbb C^d.
\]
```

This says exactly what the construction uses: the first `n+1` entries carry the coefficients and all remaining entries are zero. It avoids defining a coordinate subspace, avoids an additional projection map, and is consistent with the cited Waring literature. The only caveat is that the text should state once that the coordinate permutation has been absorbed into the ordering of the nonzero entries; no further `span` notation is needed.
