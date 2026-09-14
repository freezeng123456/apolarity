# Prior-result attribution in papers built from existing identities

This note surveys papers in which a new computational result is obtained by
combining an established mathematical identity or theorem with a new bridge,
discretization, or implementation. The common practice is to identify the
external result before stating the paper's own consequence, then state clearly
what is new. Repeating a standard calculation is acceptable when it is
explicitly framed as a verification, adaptation, or self-contained derivation.

## Examples

### 1. Carlini--Catalisano--Geramita, *On the Waring's problem for monomials*

Primary source: [arXiv:1110.0745](https://arxiv.org/abs/1110.0745), and the
published [Journal of Algebra article](https://doi.org/10.1016/j.jalgebra.2012.07.028).

- **Prior result and claim.** This is the foundational monomial-specific Waring
  rank paper. Its Corollary 3.3 states the rank formula after sorting the
  exponents; Corollary 3.8 gives a roots-of-unity-supported decomposition.
  These are the results that must be treated as prior ingredients in the
  Apolarity manuscript, not as newly discovered polynomial facts.
- **Attribution practice.** The paper introduces the rank formula and then
  labels the displayed decomposition as a corollary of the preceding theory.
  The source of the result is therefore visible at the statement level rather
  than postponed to the end of a later proof.
- **Proof repetition.** The paper proves its own rank result because that is
  its mathematical contribution. For our manuscript, CCG should be cited for
  the monomial-rank formula and roots-of-unity support, while the derivative
  bridge should be identified as the new step.
- **Directly relevant detail.** CCG Remark 2.3 contains the projection
  observation used when a form depends on fewer variables than the ambient
  polynomial ring. That observation should be cited if the same argument is
  repeated in Section 3.2.

### 2. Buczyńska--Buczyński--Teitler, *Waring decompositions of monomials*

Primary source: [arXiv:1201.2922](https://arxiv.org/abs/1201.2922), and the
published [Journal of Algebra article](https://doi.org/10.1016/j.jalgebra.2012.12.011).

- **Prior result and claim.** Section 2, equation (8), writes the explicit
  coefficients for a roots-of-unity Waring decomposition of a monomial. After
  converting the multinomial coefficient, the coefficient is the same closed
  form used in the current Apolarity Section 3.2.
- **Attribution practice.** The formula is presented as an existing explicit
  construction in the Waring literature, with the paper's own discussion
  clearly separated from the formula it records. The reference belongs next
  to the formula, not only in a concluding sentence.
- **Proof repetition.** Section 2 verifies the identity by expanding the
  powers and applying roots-of-unity cancellation. Reusing this calculation
  is reasonable for a self-contained derivative paper, but it should be
  introduced as a verification of the cited decomposition in the present
  notation.
- **Implication for Apolarity.** The current theorem should say that the
  paper translates the cited polynomial decomposition through the directional
  Taylor bridge. It should not present the roots-of-unity coefficients as a
  new polynomial construction.

### 3. Donaldson--Elliott, *A unified approach to quadrature rules with
asymptotic estimates of their remainders*

Primary source: [SIAM Journal on Numerical Analysis,
10.1137/0709051](https://epubs.siam.org/doi/10.1137/0709051).

- **Prior result and claim.** The paper's abstract states that it starts from
  a theorem giving a contour-integral representation of quadrature remainders,
  derives many known quadrature rules from that theorem, and then gives new
  rules and asymptotic remainder estimates.
- **Attribution practice.** The opening sentence explicitly identifies the
  theorem as the starting point. The paper does not blur the distinction
  between the general representation and the quadrature rules or estimates
  obtained from it.
- **Proof repetition.** Standard quadrature consequences are derived from the
  theorem to make the framework usable; the novelty is assigned to the
  unified derivation, new rules, and error analysis. This is a close analogue
  of citing the Waring identity while claiming the derivative representation
  and computational realization.
- **Lesson.** State the external identity before introducing the paper's new
  consequence, and use the transition “Combining this representation with ...”
  to mark the contribution boundary.

### 4. Epstein--Greengard--O'Neil, *A fast algorithm for Quadrature by
Expansion in three dimensions*

Primary source: [Journal of Computational Physics article,
10.1016/j.jcp.2019.108976](https://doi.org/10.1016/j.jcp.2019.108976), with the
earlier two-dimensional algorithm discussed in the official JCP record for
[QBX I](https://www.sciencedirect.com/science/article/pii/S0021999117303418).

- **Prior result and claim.** The paper builds a fast three-dimensional
  algorithm from the existing Quadrature by Expansion machinery and the point
  FMM. The JCP record describes the earlier QBX method and the new work as an
  algorithmic coupling/extension, rather than claiming the underlying
  expansion identity as new.
- **Attribution practice.** The existing quadrature and FMM components are
  named before the new coupling is introduced. The contribution is framed as
  an efficient embedding and implementation with complexity and numerical
  demonstrations.
- **Proof repetition.** The paper explains enough of the existing machinery to
  establish the new coupling, but does not re-prove the foundational FMM or
  QBX theory. Routine background is cited; the new algorithmic interface and
  performance analysis are developed in the paper.
- **Lesson.** In Apolarity, the Waring identity can be summarized and cited,
  while the directional-Taylor interface, batching, VJP, and measured cost
  should remain in the paper.

### 5. Hu et al., *Fast adjoint algorithm for linear responses of hyperbolic
chaos*

Primary source: [SIAM Journal on Applied Dynamical Systems,
10.1137/22M1522383](https://epubs.siam.org/doi/full/10.1137/22M1522383).

- **Prior result and claim.** The paper explicitly says that ensemble and
  operator formulas for linear response are well-known and rigorously proved
  under hyperbolicity assumptions. It then develops an adjoint algorithm from
  an adjoint shadowing lemma and analyzes its cost.
- **Attribution practice.** The prior formulas and their equivalence are
  identified before the new algorithm is described. The paper distinguishes
  a known exact formula from the new computational organization that makes it
  efficient on one orbit.
- **Proof repetition.** Existing formulas are cited rather than reproved.
  The paper proves the new adjoint lemma/algorithmic result that is needed for
  its claimed cost advantage.
- **Lesson.** This is a particularly useful model for Apolarity: cite the
  established algebraic identity, prove the new bridge to the target
  derivative, and give implementation results for the new computational form.

## Practical recommendation for Section 3.2

The closest precedent is the combination of CCG and BBT. A safe opening is:

> Carlini, Catalisano, and Geramita established the complex Waring rank of a
> monomial and described a minimal decomposition supported on roots of unity.
> Buczyńska, Buczyński, and Teitler subsequently gave the coefficients of this
> decomposition in closed form. Combining these polynomial results with
> Proposition 3.1 gives the rank-optimal representation of the target partial
> derivative.

The subsequent coefficient calculation may remain in the main text, but it
should be introduced as a verification of the cited decomposition in the
notation of the present paper. The novelty claim should attach to the bridge
from polynomial Waring decomposition to directional Taylor coefficients, the
minimum-direction statement for partial derivatives, and the batched
Taylor-jet implementation—not to the monomial rank formula or the explicit
roots-of-unity coefficient identity themselves.

