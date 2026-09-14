# Section 2.2--3 terminology and setting research

## 1. Local point and complex-neighborhood assumptions

The standard multivariable Taylor statement is local: one fixes a point and expands in an increment. For this paper the clean order is to state the real domain, fix x in Omega and the target multi-index, assume that u has a holomorphic extension to a complex neighborhood of this x, and then define T_p(x;v) for v in C^d.

A global assumption that u is holomorphic on a complex neighborhood of all of Omega is stronger than needed for one evaluation point. The local formulation also handles boundary points: a boundary point need not have a real open ball contained in Omega, but the affine path x+tau v only requires a complex neighborhood of x for sufficiently small tau. The assumption is not automatic for every C^p(Omega;R) function, which need not be defined at complex arguments. It is automatic for networks formed from affine maps and entire activations such as sinh, and is natural for networks representing complex-valued solutions.

Recommended Section 2.2 wording: “For the remainder of this section, fix x in Omega and a multi-index nu of order p. Assume that u admits a holomorphic extension to a complex neighborhood of x. For v in C^d, define ...” This keeps the general notation in Section 2.2 and avoids restating the full setting in Section 3.

## 2. Taylor-mode terminology

Griewank, Utke, and Walther, “Evaluating higher derivative tensors by forward propagation of univariate Taylor series,” Mathematics of Computation 69 (2000), 1117--1131, describe propagating univariate Taylor series and note that selected directions can be evaluated without forming the full derivative tensor. Official DOI: https://doi.org/10.1090/S0025-5718-00-01120-0.

Bettencourt, Johnson, and Duvenaud, “Taylor-mode automatic differentiation for higher-order derivatives in JAX,” NeurIPS 2019 workshop/preprint, uses Taylor-mode AD to propagate truncated Taylor information through a program. These sources support the hierarchy: a Taylor jet is the whole truncated collection through order p; T_p(x;v) is one directional Taylor coefficient, namely the coefficient of tau^p along v; and propagation of the whole collection through the network is Taylor-mode automatic differentiation. Therefore T_p should remain “directional Taylor coefficient,” while “Taylor jet” should be reserved for the propagated order-p object and the batched implementation.

The original STDE paper is Shi, Hu, Lin, and Kawaguchi, “Stochastic Taylor Derivative Estimator: Efficient amortization for arbitrary differential operators,” NeurIPS 2024: https://arxiv.org/abs/2412.00088 and https://papers.nips.cc/paper_files/paper/2024/file/dd2eb5250696753ea37141bbd89bb569-Paper-Conference.pdf. Its Introduction defines multi-index derivatives. Equations (11)--(13) identify the pushed-forward Taylor coefficient with a derivative-tensor contraction, state the unbiasedness condition, and state the expectation identity. The hierarchy is input tangent/input jet, Taylor-mode propagation, and output Taylor coefficient. STDE is stochastic and does not call one scalar output coefficient a jet.

STDE++, Shi et al., “STDE++: Polynomial-Time Amortization for Linear Differential Operators,” JMLR 27 (2026), 1--50: https://www.jmlr.org/beta/papers/v27/25-1474.html. Its abstract and Sections 2--3 retain the same hierarchy and add polynomial-time Taylor-jet constructions for mixed partial derivatives. This supports using “linear combinations of directional Taylor coefficients” for the mathematical representation and “Taylor jet” for the propagated truncated series.

## 3. Consequences for Sections 2.2 and 3

There is mild repetition if Section 3 again defines nu in N_0^d, the target derivative, e_0,...,e_n, and the general directional coefficient after Section 2.2 has already done so. Keep Section 2.2 global and begin Section 3 only with the support form nu=(a_0,...,a_n,0,...) and a=(a_0,...,a_n), followed by the associated monomial z^a=z_0^{a_0}...z_n^{a_n}. Define the coordinate vectors only when the explicit roots-of-unity directions are introduced.

The restriction to e_0,...,e_n is not needed for the final lower bound. It is the natural subspace for the explicit construction. For minimality over all of C^d, project every ambient linear form onto span{e_0,...,e_n}, equivalently set the unused variables to zero. This preserves the number of summands and produces a decomposition of the reduced monomial. This is exactly the argument in CCG, The Solution to Waring’s Problem for Monomials, Remark 2.3, PDF pp. 2--3: https://staff.math.su.se/shapiro/ProblemSolving/GeramitaCarlini.pdf.

## 4. Section 3 title suggestions

Suitable titles that do not emphasize “exact” are: (1) “Directional representations and Waring rank,” best for a section that first develops the polynomial bridge and then transfers it to derivatives; (2) “Rank-optimal directional representations,” best if minimum direction count is the main emphasis; and (3) “Waring decompositions and directional representations,” best if the polynomial-to-derivative organization is made explicit. Recommendation: use “Directional representations and Waring rank” for the section title and reserve “Rank-optimal directional representation” for the derivative theorem subsection.

## Bottom line

Use a local “fix x in Omega” convention for the complex Taylor argument; do not impose a global holomorphic neighborhood of all of Omega unless needed. Keep T_p(x;v) as a directional Taylor coefficient and use Taylor jet only for the propagated truncated series. In Section 3, avoid repeating the Section 2.2 setting, introduce e_0,...,e_n at the explicit construction, and handle arbitrary ambient directions only in the minimality argument by projection.
