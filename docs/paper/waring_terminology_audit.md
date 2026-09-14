# Waring terminology and Proposition 3.1

## Terminology in the literature

The literature uses “Waring decomposition” in two closely related ways, so
the manuscript should distinguish the existence of a sum-of-powers identity
from minimality.

Carlini, Catalisano, and Geramita formulate the Waring problem for a degree-
`d` form as finding the least `s` for which the form can be written as a sum
of powers of `s` linear forms. Their abstract explicitly describes the
problem as computing “the minimal number of linear forms” and separately
calls the resulting invariant the Waring rank. This supports reserving
“Waring rank” (or “minimal Waring decomposition”) for the minimum length,
while using “Waring decomposition” more generally for a sum-of-powers
representation. See [Carlini--Catalisano--Geramita, *The Solution to
Waring's Problem for Monomials*](https://arxiv.org/abs/1110.0745), especially
the abstract and Section 1.

The same non-minimal usage is explicit in Catalisano, Chiantini, Geramita,
and Oneto: “A Waring decomposition of `f` is a way to express `f` as a sum
of `d`th powers of linear forms.” Their definition does not impose a least
length; minimality enters when discussing rank. See [*Waring-like
decompositions of polynomials, 1*](https://www.sciencedirect.com/science/article/pii/S002437951730441X).

There is also a narrower convention in parts of the literature. The article
[*Waring decompositions of monomials*](https://www.sciencedirect.com/science/article/pii/S0021869312006308)
describes a Waring decomposition as a sum of powers “where the number of
summands is minimal possible.” Thus, the phrase is not completely uniform:
some authors use it for any decomposition, while others use it for a
minimal one. The manuscript should avoid relying on that ambiguity.

Comon, Golub, Lim, and Mourrain use the tensor language of a symmetric
decomposition and symmetric tensor rank. Their paper studies decompositions
into symmetric rank-one terms and treats rank as the minimum number of such
terms; it does not require every displayed decomposition to be minimal. See
[Comon--Golub--Lim--Mourrain, *Symmetric Tensors and Symmetric Tensor
Rank*](https://epubs.siam.org/doi/pdf/10.1137/060661569).

## Recommendation for the proposition title

The proposition does not yet identify a minimum, and it does not yet use the
Waring rank. It proves an equivalence between two representations of the
same coefficient data:

\[
\text{directional Taylor representation}
\quad\Longleftrightarrow\quad
\text{homogeneous sum-of-powers identity}.
\]

For that reason, `Waring characterization` is too strong at this point:
“characterization” suggests that the proposition characterizes the rank or
the minimum number of terms. `Waring relation` is possible, but it is less
standard and does not say what is being related. The clearest minimal-change
title is:

```tex
\begin{proposition}[Directional--Waring equivalence]\label{prop_bridge}
```

If a less specialized title is preferred, `Equivalence of directional and
Waring representations` is also precise. I recommend `Directional--Waring
equivalence`; it describes exactly what the proposition proves and leaves the
rank statement for the subsequent theorem/corollary.

## Wording for the proof

The phrase “degree-`p` polynomials centered at `\bx` can prescribe the
order-`p` derivatives independently” is understandable but informal. A
cleaner statement follows directly from the monomial basis and avoids
“prescribe”:

```tex
The monomials $(\bz-\bx)^\beta$ with $|\beta|=p$ are linearly independent,
and their order-$p$ derivatives at $\bx$ are independent. Comparing the
coefficients of these derivatives therefore gives
```

Because only the reduced coordinates are used in the current setting, an
even more direct version is:

```tex
Testing the identity on the monomials $(\bz-\bx)^\beta$ with $|\beta|=p$
shows that the coefficients of the order-$p$ derivatives must agree for
each multi-index $\beta$.
```

The second version is shorter, but the first makes the linear-independence
reason explicit. Either is preferable to “can prescribe,” provided the
notation `\bz` and `\bx` is interpreted consistently in the proof.

## Bottom line

Use “Waring decomposition” for the displayed sum-of-powers identity and
define Waring rank as its minimum length. Rename Proposition 3.1 to
`Directional--Waring equivalence` (or the fully descriptive
`Equivalence of directional and Waring representations`). Do not introduce
“minimal” in the proposition title, because minimality is not proved there.
