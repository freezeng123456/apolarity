# Reference candidates for the JCP manuscript

Target: about 40 cited works in the published paper.
This file lists **100** candidates taken from the bibliographies of the
seed papers below (not from a generic web search). Pick from here; unused
entries stay out of `jsc_paper_main.bib`.

Seed papers (their reference lists were read):

- Shi, Hu, Lin, Kawaguchi, STDE, NeurIPS 2024, arXiv:2412.00088
- Hu, Shi, Karniadakis, Kawaguchi, HTE, CMAME 2024, arXiv:2312.14499
- Hu, Yang, Wang, Karniadakis, Kawaguchi, bias-variance / randomized
  smoothing, JCP 2025, arXiv:2311.15283
- He et al., PINN without stacked back-propagation, AISTATS 2023,
  arXiv:2202.09340
- Hu, Shukla, Karniadakis, Kawaguchi, SDGD, Neural Networks 2024,
  arXiv:2307.12306
- Li et al., Forward Laplacian, Nat. Mach. Intell. 2024, arXiv:2307.08214
- Griewank, Utke, Walther, univariate Taylor tensors, Math. Comp. 2000
- Baydin, Pearlmutter, Radul, Siskind, AD survey, JMLR 2018,
  arXiv:1502.05767
- Carlini, Catalisano, Geramita, monomial Waring, J. Algebra 2012
  (standard citations of that paper)
- Papers already in `jsc_paper_main.bib`

Priority for the eventual 40:

- **A**: almost certainly cite (method landscape or theorem)
- **B**: cite if the introduction names that method or PDE
- **C**: background only; easy to drop

Already in the `.bib` is marked `(in bib)`.

---

## A. Algorithmic / Taylor-mode differentiation (1–22)

| # | Work | Year | Venue | Seed | Pri. | Why |
|---|------|------|-------|------|------|-----|
| 1 | Griewank and Walther, *Evaluating Derivatives* (2nd ed.) `(in bib)` | 2008 | SIAM | STDE, HTE, Griewank 2000 | A | Nested reverse mode; Taylor arithmetic |
| 2 | Griewank, Utke, Walther, Evaluating higher derivative tensors by forward propagation of univariate Taylor series | 2000 | Math. Comp. | Griewank 2000 itself; JAX jet literature | A | Mixed partials from a family of univariate Taylor series |
| 3 | Bettencourt, Johnson, Duvenaud, Taylor-mode AD for higher-order derivatives in JAX `(in bib)` | 2019 | NeurIPS workshop | STDE, HTE, bias-var. | A | The jet we implement |
| 4 | Bischof, Corliss, Griewank, Structured second- and higher-order derivatives through univariate Taylor series | 1993 | Optim. Methods Softw. | Griewank 2000 | A | Same univariate-Taylor idea, earlier |
| 5 | Neidinger, An efficient method for the numerical evaluation of partial derivatives of arbitrary order | 1992 | ACM TOMS | Griewank 2000 | A | Multivariate Taylor / high-order partials |
| 6 | Berz, Differential algebraic description of beam dynamics to very high orders | 1989 | Part. Accel. | Griewank 2000 | B | DA / truncated Taylor polynomials |
| 7 | Griewank, Juedes, Utke, ADOL-C | 1996 | ACM TOMS | Griewank 2000 | B | Univariate Taylor driver in a production AD tool |
| 8 | Baydin, Pearlmutter, Radul, Siskind, Automatic differentiation in machine learning: a survey | 2018 | JMLR | Baydin survey | A | Standard AD survey (forward vs reverse) |
| 9 | Bendtsen and Stauning, TADIFF | 1997 | IMM report | STDE | B | Taylor-series AD package |
| 10 | Karczmarczuk, Functional differentiation of computer programs | 1998 | ICFP | STDE | C | Functional HOAD |
| 11 | Wang, High order reverse mode of automatic differentiation | 2017 | PhD thesis | STDE | B | Nested reverse-mode cost |
| 12 | Laurel et al., Abstract interpretation of higher-order AD | 2022 | OOPSLA | STDE | C | Semantics of HOAD |
| 13 | Li, Wang, Ye, He, Wang, DOF: accelerating high-order differential operators with forward propagation | 2024 | ICLR AI4DE workshop | STDE | A | Forward-mode operator evaluation, sibling of Forward Laplacian |
| 14 | Li et al., Forward Laplacian `(in bib)` | 2024 | Nat. Mach. Intell. | STDE, Forward Laplacian | A | Exact recurrence for one operator, not one monomial |
| 15 | Gao, Koehler, Foster, folx `(in bib)` | 2023 | software | our bib | B | JAX Forward Laplacian |
| 16 | Oktay et al., Randomized automatic differentiation | 2021 | ICLR | STDE, HTE | B | Randomizing AD graphs |
| 17 | Pearlmutter, Fast exact multiplication by the Hessian | 1994 | Neural Comp. | Baydin survey (standard) | B | HVP; what Hutchinson randomizes |
| 18 | Griewank, On automatic differentiation | 1989 | Math. Programming | Griewank 2000 | C | Early AD survey |
| 19 | Bischof et al., ADIFOR 2.0 | 1996 | IEEE Comp. Sci. Eng. | Baydin survey | C | Classic AD tool |
| 20 | Paszke et al., Automatic differentiation in PyTorch | 2017 | NIPS workshop | Forward Laplacian | C | Nested reverse backend we time against |
| 21 | Bradbury et al., JAX | 2018 | software | STDE, HTE | C | Jet host; cite only if the implementation paragraph needs it |
| 22 | Martens, Sutskever, Swersky, Estimating the Hessian by back-propagating curvature | 2012 | arXiv | Forward Laplacian | C | Nested second-order AD |

---

## B. Randomized / Hutchinson / STDE / smoothing (23–42)

These are the papers the introduction must name when it says “randomized estimators.”

| # | Work | Year | Venue | Seed | Pri. | Why |
|---|------|------|-------|------|------|-----|
| 23 | Shi, Hu, Lin, Kawaguchi, STDE `(in bib)` | 2024 | NeurIPS | STDE itself | A | Random contraction of the order-$p$ tensor via Taylor-mode tangents |
| 24 | Hu, Shi, Karniadakis, Kawaguchi, HTE `(in bib)` | 2024 | CMAME | STDE, HTE | A | Hutchinson on Hessian / TVP; Taylor-mode HVP |
| 25 | Hu, Yang, Wang, Karniadakis, Kawaguchi, Bias-variance trade-off with randomized smoothing `(in bib)` | 2025 | JCP | STDE, HTE, bias-var. | A | RS-PINN is biased; hybrid biased/unbiased |
| 26 | He et al., Learning PINNs without stacked back-propagation `(in bib)` | 2023 | AISTATS | STDE, HTE, bias-var. | A | Stein identity / Gaussian smoothing, no nested AD |
| 27 | Hu, Shukla, Karniadakis, Kawaguchi, SDGD `(in bib)` | 2024 | Neural Networks | STDE, HTE, bias-var. | A | Random coordinate subset of the residual |
| 28 | Hutchinson, A stochastic estimator of the trace | 1989 | Comm. Statist. Simul. | STDE, HTE | A | Trace estimator HTE rests on |
| 29 | Stein, Estimation of the mean of a multivariate normal `(in bib)` | 1981 | Ann. Statist. | STDE, He | A | Stein identity behind RS / He 2023 |
| 30 | Meyer, Musco, Musco, Woodruff, Hutch++ | 2021 | SOSA | HTE | B | Optimal stochastic trace estimation |
| 31 | Persson, Cortinovis, Kressner, Improved Hutch++ | 2022 | SIMAX | HTE | C | Trace-estimation variants |
| 32 | Roosta-Khorasani and Ascher, Sample size for implicit matrix trace estimators | 2015 | FoCM | HTE | B | Variance of Hutchinson |
| 33 | Skorski, Modern analysis of Hutchinson’s trace estimator | 2021 | CISS | STDE, HTE | C | Analysis of Hutchinson |
| 34 | Cohen, Rosenfeld, Kolter, Certified adversarial robustness via randomized smoothing | 2019 | ICML | bias-var., He | B | Original RS; not a PDE paper |
| 35 | Lecuyer et al., Certified robustness with differential privacy | 2019 | IEEE S&P | bias-var. | C | Related smoothing |
| 36 | Chiu et al., CAN-PINN `(in bib)` | 2022 | CMAME | bias-var., He | A | Coupled AD + finite differences |
| 37 | Yu, Lu, Meng, Karniadakis, Gradient-enhanced PINNs | 2022 | CMAME | STDE, HTE | B | Extra derivative tensors in the loss |
| 38 | Song, Garg, Shi, Ermon, Sliced score matching | 2019 | arXiv | STDE | C | Random projections of derivatives |
| 39 | Pang, Xu, Li, Song, Ermon, Zhu, Finite-difference score matching | 2020 | NeurIPS | STDE | C | FD instead of AD for scores |
| 40 | Martinsson and Tropp, Randomized numerical linear algebra | 2020 | Acta Numer. (survey arXiv) | STDE | C | Broader randomization |
| 41 | Hu, Zhang, Karniadakis, Kawaguchi, Score-based PINNs for Fokker–Planck | 2025 | SISC (arXiv:2402.07465) | STDE, HTE | B | High-dim second-order operators |
| 42 | Liu et al., Primer on zeroth-order optimization | 2020 | IEEE SPM | STDE | C | Derivative-free contrast |

---

## C. Collocation / PINN as a workload, not the subject (43–72)

Cite a short stack so the introduction can say “collocation residuals are one place these derivatives are requested,” without a PINN survey.

| # | Work | Year | Venue | Seed | Pri. | Why |
|---|------|------|-------|------|------|-----|
| 43 | Raissi, Perdikaris, Karniadakis, PINNs `(in bib)` | 2019 | JCP | all seeds | A | Collocation residual |
| 44 | Karniadakis et al., Physics-informed machine learning `(in bib)` | 2021 | Nat. Rev. Phys. | all seeds | B | Review pointer |
| 45 | Sirignano and Spiliopoulos, DGM | 2018 | JCP | STDE, HTE, bias-var. | B | Mesh-free collocation, high-dim |
| 46 | E and Yu, Deep Ritz | 2017 | Comm. Math. Stat. | STDE, HTE | B | Variational residual |
| 47 | Han, Jentzen, E, Deep BSDE `(in bib)` | 2018 | PNAS | STDE, HTE, bias-var. | B | High-dim PDE, not nested AD of a net |
| 48 | Han, Jentzen, Deep learning for parabolic PDEs / BSDEs | 2017 | Comm. Math. Stat. | HTE, bias-var. | C | Precursor of 47 |
| 49 | Zang, Bao, Ye, Zhou, Weak adversarial networks | 2020 | JCP | STDE, HTE | C | High-dim weak form |
| 50 | Lu et al., PINNs with hard constraints | 2021 | SISC | STDE, HTE, bias-var. | C | Architecture, not derivatives |
| 51 | Wang, Teng, Perdikaris, Gradient pathologies `(in bib)` | 2021 | SISC | our bib | C | Training, not derivative cost |
| 52 | Jagtap, Kawaguchi, Karniadakis, Adaptive activations `(in bib)` | 2020 | JCP | HTE, bias-var. | C | Activations |
| 53 | Jagtap and Karniadakis, XPINN | 2020 | Commun. Comput. Phys. | HTE | C | Domain decomposition |
| 54 | Hu, Jagtap, Karniadakis, Kawaguchi, When do XPINNs improve generalization? | 2022 | SISC | HTE, bias-var. | C | Same |
| 55 | Shin, Darbon, Karniadakis, Convergence of PINNs | 2020 | Commun. Comput. Phys. | HTE, bias-var. | C | Theory of collocation |
| 56 | Mishra and Molinaro, Generalization error of PINNs | 2022 | IMA J. Numer. Anal. (arXiv 2006.16144) | HTE, bias-var. | C | Same |
| 57 | Wang, Li, He, Wang, Is $L^2$ physics-informed loss always suitable? | 2022 | NeurIPS | HTE, bias-var., FwdLap | C | Loss, not derivative eval |
| 58 | Hao et al., PINNacle `(in bib)` | 2024 | NeurIPS D&B | our bib | C | Benchmark suite |
| 59 | Cai et al., PINNs for fluid mechanics: a review | 2021 | Acta Mech. Sin. | HTE | C | Application review |
| 60 | Jin, Cai, Li, Karniadakis, NSFnets | 2021 | JCP | HTE | C | Navier–Stokes PINN |
| 61 | Pang, Lu, Karniadakis, fPINNs | 2019 | SISC | bias-var. | C | Fractional operators |
| 62 | Cho et al., Separable PINN | 2022 | arXiv:2211.08761 | HTE, bias-var. | C | High-dim architecture |
| 63 | Raissi, Forward-backward stochastic neural networks | 2018 | arXiv:1804.07010 | STDE, HTE, bias-var. | B | High-dim PDE nets |
| 64 | Beck, Becker, Cheridito, Jentzen, Neufeld, Deep splitting | 2021 | SISC | STDE, HTE, bias-var. | B | High-dim parabolic, not PINN AD |
| 65 | Huré, Pham, Warin, Deep backward schemes | 2020 | Math. Comp. | HTE, bias-var. | B | High-dim nonlinear PDE |
| 66 | Hutzenthaler, Jentzen, Kruse, Nguyen, von Wurstemberger, Overcoming the curse of dimensionality (semilinear parabolic) | 2020 | Proc. R. Soc. A | STDE, HTE, bias-var. | C | Picard, not AD |
| 67 | Beck, E, Jentzen, ML approximation of fully nonlinear PDEs | 2019 | J. Nonlinear Sci. | HTE, bias-var. | C | High-dim PDE |
| 68 | Sitzmann et al., SIREN `(in bib)` | 2020 | NeurIPS | our bib | B | Derivatives of implicit nets; Helmholtz in that paper |
| 69 | Tancik et al., Fourier features `(in bib)` | 2020 | NeurIPS | our bib | C | Frequency, not derivatives |
| 70 | Chen, Lu, Karniadakis, Dal Negro, PINNs for inverse nano-optics | 2020 | Opt. Express | He 2023 | C | Helmholtz-type inverse |
| 71 | Haghighat et al., PINN inversion in solid mechanics | 2021 | CMAME | HTE | C | Elasticity / biharmonic-adjacent |
| 72 | Psaros, Kawaguchi, Karniadakis, Meta-learning PINN losses | 2022 | JCP | HTE | C | Training |

---

## D. Operators / models where high-order derivatives arise (73–82)

| # | Work | Year | Venue | Seed | Pri. | Why |
|---|------|------|-------|------|------|-----|
| 73 | Vahab, Haghighat, Khaleghi, Khalili, PINN for biharmonic elasticity `(in bib)` | 2022 | J. Eng. Mech. | our bib | B | Plate / biharmonic |
| 74 | Jiang et al., Complex DeepONet for 3D Maxwell `(in bib)` | 2024 | arXiv:2411.18733 | our bib | B | Maxwell / Helmholtz |
| 75 | Gazzola, Grunau, Sweers, *Polyharmonic Boundary Value Problems* | 2010 | Springer LNM | standard for $\Delta^m$ (via CCG-style analysis papers) | B | Classical polyharmonic |
| 76 | Ciarlet, *Mathematical Elasticity, Vol. II: Theory of Plates* | 1997 | North-Holland | plate literature cited from 73 | B | Plate operator |
| 77 | Evans, *Partial Differential Equations* | 1998 / 2010 | AMS | He 2023 | C | Textbook Helmholtz / Laplace |
| 78 | Darbon and Osher, Algorithms for HJ equations | 2016 | Res. Math. Sci. | HTE | C | High-dim first-order |
| 79 | Darbon, Langlois, Meng, Neural HJ | 2020 | Res. Math. Sci. | HTE | C | Same |
| 80 | Chan-Wai-Nam, Mikael, Warin, ML for semilinear PDEs | 2019 | J. Sci. Comput. | HTE, bias-var. | C | Semilinear collocation |
| 81 | Pfau et al., FermiNet | 2020 | Phys. Rev. Research | Forward Laplacian | C | Laplacian of a net in VMC |
| 82 | Hermann, Schätzle, Noé, PauliNet | 2020 | Nat. Chem. | Forward Laplacian | C | Same |

---

## E. Algebraic geometry: a small toolkit, not a survey (83–94)

The computational list above is large because the methods paragraph has to
name competing evaluators. Algebraic geometry is the opposite: the paper
uses **one** closed-form rank and a one-paragraph apolarity sketch. Cite
six to eight works, not twelve. CCG is a tool; the contribution is the
identification of a directional derivative schedule with a monomial Waring
decomposition, plus the jet comparison with nested AD.

### What the manuscript actually uses

- Waring rank of a homogeneous polynomial (Definition 3.1), allowing
  scalar coefficients so that the same definition works over \(\mathbb{R}\)
  and \(\mathbb{C}\).
- The identification: a directional schedule for \(\partial^\nu u\) is
  exactly a Waring decomposition of \(p!\,z^a\).
- Complex monomial rank \(R_{\mathbb{C}}(z^a)=\prod_{j=1}^n(a_j+1)\), the
  theorem of Carlini–Catalisano–Geramita.
- The explicit roots-of-unity schedule (CCG Corollary 3.8, with
  coefficients written out as in Buczyńska–Buczyński–Teitler).
- Appendix: the apolar ideal of a monomial is the complete intersection
  \((\partial_0^{a_0+1},\ldots,\partial_n^{a_n+1})\); a Hilbert-function
  lower bound; a reduced apolar scheme of that degree.
- Complex directions when some factor \(U_{m_j}\) with \(j\ge 1\) has
  \(m_j\ge 3\). That is a fact about **this construction**, not a theorem
  that the real Waring rank is strictly larger.

### What the manuscript does not use

Alexander–Hirschowitz generic rank; border rank; cactus versus Waring
rank; uniqueness of decompositions / \(\mathrm{VSP}\); Segre varieties;
the 1851 binary history; Veronese secants as a topic. Do not cite those
unless a later remark actually needs them.

A caution on Ranestad–Schreyer 2011: their Hilbert-type bound is for the
length of an **arbitrary** (not necessarily reduced) apolar scheme, and
for a monomial \(x_0^{a_0}\cdots x_n^{a_n}\) with \(a_0\le\cdots\le a_n\)
it omits the **largest** exponent. CCG’s Waring rank omits the
**smallest**. The two numbers agree only when all exponents are equal
(e.g.\ \(xyz^2\) has Waring rank 6 and cactus bound 4). Citing RS2011 in
the appendix without that distinction would make the lower bound look
wrong. Skip it unless the appendix is rewritten to say cactus versus
rank, which a JCP paper does not need.

### Where a cite may appear

- Abstract: the word “Waring”, no citation.
- Introduction: **at most two sentences**. Name the invariant, point to
  CCG for the length, and send the apolarity argument to the appendix.
  Do not open an algebraic-geometry paragraph, and do not mention
  Veronese varieties or generic rank.
- Section 3, definition: one SIAM-facing pointer that Waring rank is
  symmetric-tensor rank.
- Theorem 3.1(ii): CCG for the length; Buczyńska–Buczyński–Teitler as
  “see also”, not as the source of the formula.
- Appendix: Iarrobino–Kanev (and Landsberg) for the general apolar
  characterization; CCG for the monomial complete intersection.

### Pick list

Already in the `.bib`: 83–85. Add 89 and 92. The rest stay out unless
you promote them.

| # | Work | Year | Venue | Role | Pri. |
|---|------|------|-------|------|------|
| 83 | Carlini, Catalisano, Geramita, The solution to the Waring problem for monomials `(in bib)` | 2012 | J. Algebra | Length formula and the roots-of-unity points | **must** |
| 84 | Iarrobino and Kanev, *Power Sums, Gorenstein Algebras, and Determinantal Loci* `(in bib)` | 1999 | Springer LNM | Apolarity / inverse systems in the appendix | **must** |
| 85 | Landsberg, *Tensors: Geometry and Applications* `(in bib)` | 2012 | AMS GSM | Textbook name for Waring / symmetric rank | **must** |
| 89 | Comon, Golub, Lim, Mourrain, Symmetric tensors and symmetric tensor rank | 2008 | SIMAX | SIAM-facing definition at Def. 3.1 or in the intro clause | **add** |
| 92 | Buczyńska, Buczyński, Teitler, Waring decompositions of monomials | 2013 | J. Algebra | Sibling of CCG: every monomial decomposition comes from a complete intersection; they write the roots-of-unity coefficients. Cite as “see also”, never as the rank formula | **add** |
| 87 | Geramita, Inverse systems of fat points | 1996 | Queen’s Papers, vol. 102 | The path CCG names; IK already covers the same ground and is easier to cite in JCP | optional |
| 93 | Landsberg and Teitler, On the ranks and border ranks of symmetric tensors | 2010 | FoCM | A few monomial ranks before CCG (\(xyz\), \(xyzw\), \(xyz^2\)). Cite only if a historical “some cases were known” sentence is wanted | optional |
| 91 | Ranestad and Schreyer, On the rank of a symmetric form | 2011 | J. Algebra | Hilbert / cactus lower bound. Skip unless the appendix discusses cactus rank | skip by default |
| 86 | Alexander and Hirschowitz, Polynomial interpolation in several variables | 1995 | J. Algebraic Geom. | Generic Waring rank. One contrast sentence at most; default is omit | skip |
| 90 | Reznick, Sums of even powers of real linear forms | 1992 | Mem. AMS | Real Waring. Cite only if we compute or bound \(R_{\mathbb{R}}\). Complex roots of unity in the construction do not need this | skip |
| 94 | Sylvester 1851; Buczyńska–Buczyński 2014 catalecticants (not the monomial paper); Tevelev; Carlini 2006 essential variables; Landsberg–Manivel Segre; Mammana 1954; Arrondo–Bernardi 2011 | — | — | Adjacent to CCG’s bibliography, unused here | skip |

The 2014 Buczyńska–Buczyński paper on catalecticants is **not** the
monomial-decomposition paper; do not substitute it for 92.

### Proposed prose (for when the introduction is rewritten)

Introduction, one connected pair of sentences, not a new paragraph:

> The shortest combination is a Waring decomposition of the corresponding
> monomial~\cite{Comon2008,Landsberg2012}. Over \(\mathbb{C}\) that length
> is known in closed form~\cite{CCG2012}; the apolarity argument behind
> the lower bound is recalled in the appendix~\cite{IK1999}.

Theorem 3.1(ii), keep the present attribution and add the sibling:

> The value of the complex rank of a monomial is the theorem of Carlini,
> Catalisano, and Geramita~\cite{CCG2012}; see also
> Buczyńska, Buczyński, and Teitler~\cite{BBT2013} for the geometry of
> all such decompositions.

Appendix sentence, keep close to the present one:

> See~\cite{IK1999,Landsberg2012} for the general theory
> and~\cite{CCG2012} for the monomial case.

That is five cited AG works (83–85, 89, 92). Geramita 1996 or
Landsberg–Teitler 2010 can join if you want seven; do not grow the list
past that. Number 92 replaces the 2014 catalecticant paper that stood
here in the first draft of this file.

---

## F. Extra AD / interpolation classics from Griewank 2000 (95–100)

| # | Work | Year | Venue | Seed | Pri. | Why |
|---|------|------|-------|------|------|-----|
| 95 | Berz, Higher derivatives in many variables (in Griewank–Corliss AD volume) | 1991 | SIAM | Griewank 2000 | B | Multivariate DA |
| 96 | Griewank, Automatic evaluation of first- and higher-derivative vectors | 1991 | Birkhäuser | Griewank 2000 | B | Directional higher derivatives |
| 97 | Griewank and Reddien, Cusp singularities for operator equations | 1989 | J. Comput. Appl. Math. | Griewank 2000 | C | Need for selected higher derivatives |
| 98 | Beda, Korolev, Sukkikh, Frolova, Programs for AD on BESM | 1959 | tech. report | Baydin survey | C | Historical AD |
| 99 | Bauer, Computational graphs and rounding error | 1974 | SINUM | Baydin survey | C | Computational graph |
| 100 | Naumann, *The Art of Differentiating Computer Programs* | 2012 | SIAM | Baydin survey / AD books | B | Modern AD monograph beside Griewank–Walther |

---

## Suggested core of ~40 (starting point, not a lock)

If we had to freeze a list tomorrow, the **A** rows plus a few **B**s would be:

From A: 1–5, 8, 13–14.
From B: 23–29, 36.
From C: 43, 45–47, 64–65, 68.
From D: 73–76.
From E: 83–85, 89, 92 (CCG, Iarrobino–Kanev, Landsberg, Comon et al.,
Buczyńska–Buczyński–Teitler). Not Alexander–Hirschowitz.
From F: 2 is already in A; 100 if a second AD book is wanted.

That is about 35–40 after dropping duplicates. STDE (23) and the bias-variance paper (25) are in this core.

## How those two enter the introduction (when we write it)

Do not dump PINN training. One connected methods paragraph:

- Nested reverse mode grows with order (1).
- Finite differences / CAN-PINN replace nesting by a step (36).
- Hutchinson / HTE (24, 28), SDGD (27), Stein / randomized smoothing (26, 29), and the bias-variance analysis of RS-PINN (25) estimate a residual by sampling; they are unbiased or biased according to that analysis, and the variance is a budget.
- STDE (23) is the randomized method that already uses Taylor-mode tangents: it estimates an *arbitrary contraction* of the derivative tensor.
- This paper does the deterministic opposite for *one* monomial: shortest exact linear combination, Waring length.

Pick, cut, or promote rows; the next edit of the introduction will cite only the chosen ~40.
