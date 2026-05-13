# Monomial Waring decomposition in derivative language

Monomial Waring decomposition was not originally developed for derivative computation.  It is a statement about homogeneous polynomials and symmetric tensor rank.

This project uses the same tensor identity as a directional derivative schedule.

## Coefficient extraction view

Let \(p=|\alpha|\), and define

\[
T_p(x;v)=\frac{1}{p!}D^p u(x)[v,\ldots,v].
\]

For active coordinates \(i_0,\ldots,i_n\), set

\[
q(z)=T_p\left(x;\sum_j z_j e_{i_j}\right).
\]

Then

\[
q(z)=\sum_{|\beta|=p}\frac{\partial^\beta u(x)}{\beta!}z^\beta,
\]

so

\[
\partial^\alpha u(x)=\alpha![z^\alpha]q(z).
\]

Computing a single monomial partial is therefore equivalent to extracting one coefficient of a homogeneous polynomial from directional evaluations.

## Roots-of-unity schedule

Let active exponents be sorted as

\[
1\le a_0\le a_1\le\cdots\le a_n.
\]

Use the least exponent variable as base.  For \(j=1,\ldots,n\), let \(\zeta_j\) range over \((a_j+1)\)-th roots of unity and define

\[
v_{\zeta}=e_{i_0}+\sum_{j=1}^n\zeta_j e_{i_j}.
\]

Then

\[
\partial^\alpha u(x)
=
\frac{\alpha!}{\prod_{j=1}^{n}(a_j+1)}
\sum_{\zeta}
\left(\prod_{j=1}^{n}\zeta_j\right)T_p(x;v_{\zeta}).
\]

The roots-of-unity sums act as exact coefficient filters.

## Minimality over complex schedules

Any formula

\[
\partial^\alpha u(x)=\sum_{r=1}^R c_rT_p(x;v_r)
\]

that holds for every smooth \(u\) is a symmetric tensor decomposition of the monomial coefficient functional.  This is equivalent to a Waring decomposition of the monomial with active exponents \((a_0,\ldots,a_n)\).

By the monomial Waring rank theorem,

\[
R_\mathbb C=\prod_{j=1}^{n}(a_j+1)
=\frac{\prod_{j=0}^{n}(a_j+1)}{a_0+1}.
\]

Thus the roots-of-unity derivative schedule uses the minimal number of complex directional Taylor probes.
