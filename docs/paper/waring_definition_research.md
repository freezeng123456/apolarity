# Waring decomposition terminology and the definition of a linear form

## Sources checked

1. E. Carlini, M. V. Catalisano, and A. V. Geramita, *The Solution to
   Waring's Problem for Monomials*, arXiv:1110.0745, Section 1, p. 1 of the
   manuscript. The authors take an algebraically closed field of
   characteristic zero and state that, for a degree-​`d` form `F`, the
   Waring problem asks for the least `s` such that there are linear forms
   `L_1,...,L_s` with `F = sum_i L_i^d`; this least `s` is called the Waring
   rank. The paper's abstract also describes the result as the minimum
   number of linear forms needed to write a monomial as a sum of powers of
   linear forms.
   Source: https://arxiv.org/abs/1110.0745
   Direct manuscript PDF: https://staff.math.su.se/shapiro/ProblemSolving/GeramitaCarlini.pdf

2. N. J. Wiebe, *Lecture Notes on Tensor Computation*, Section 2.5.3,
   Definition 2.5.14, p. 18 of the PDF. For a homogeneous polynomial
   `f in F[x_1,...,x_n]` of degree `m`, the notes define the Waring rank as
   the smallest `r` for which
   `f(x) = sum_{i=1}^r lambda_i (g_i^T x)^m`, with vectors `g_i in F^n` and
   scalars `lambda_i in F`. The equation is called a Waring decomposition.
   Source: https://mathweb.ucsd.edu/~njw/Teaching/Fa17_277TC/LecNote277A_chap02.pdf

3. P. Comon, G. Golub, L.-H. Lim, and B. Mourrain, *Symmetric Tensors and
   Symmetric Tensor Rank*, arXiv:0802.1681, abstract and introductory
   discussion. The paper uses
   the equivalent formulation “a homogeneous polynomial ... as a sum of
   powers of linear forms,” and relates it to symmetric tensor rank.
   Source: https://arxiv.org/abs/0802.1681

## What the standard definition makes explicit

The literature commonly introduces `L_i` first as linear forms, without
immediately expanding their coefficients. For a computational paper,
however, the phrase “linear form” should be made explicit because the
directions in the present manuscript are vectors. The standard coordinate
meaning is

```tex
L_r(\bz)=\bv_r\cdot\bz=\sum_{j=0}^{n}v_{r,j}z_j,
\qquad \bv_r=(v_{r,0},\ldots,v_{r,n})\in\C^{n+1}.
```

This is not a new mathematical restriction: every linear form in
`C[z_0,...,z_n]` has this representation, and the coefficient vector
`\bv_r` is exactly the direction already used in the directional Taylor
coefficient. Writing the definition this way also makes the later identity
`(\bv_r\cdot\bz)^p` transparent.

## Recommendation for the manuscript

A concise, textbook-style definition suitable for the current audience is:

```tex
For a homogeneous polynomial $F\in\C[z_0,\ldots,z_n]$ of degree $p$, a
Waring decomposition over $\C$ is an identity
\begin{equation*}
    F=\sum_{r=1}^{R}c_rL_r^p,
\end{equation*}
where $c_r\in\C$ and
\[
    L_r(\bz)=\bv_r\cdot\bz
    =\sum_{j=0}^{n}v_{r,j}z_j,
    \qquad \bv_r=(v_{r,0},\ldots,v_{r,n})\in\C^{n+1},
\]
are linear forms. The Waring rank $R(F)$ is the minimum value of $R$ among
all such decompositions over $\C$.
```

If minimizing notation is more important than giving the coefficient
expansion, the middle display can be shortened to “where `c_r in C` and
`L_r(bz)=v_r dot bz` for `v_r in C^{n+1}`.” The important point is to define
the linear form as a function of `bz` and to state its coefficient vector;
the bare phrase “each `L_r` is a linear form” is standard but vague in this
particular proof, where the vector-direction correspondence is central.

The original monomial-rank paper does not need to spell out the coefficient
vector because its algebraic setting already treats `L_i` as elements of the
degree-one component of the polynomial ring. The present manuscript should
be slightly more explicit because it immediately identifies `L_r` with the
directional vector `\bv_r`.
