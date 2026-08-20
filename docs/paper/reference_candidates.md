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

## E. Waring rank / apolarity (83–94)

| # | Work | Year | Venue | Seed | Pri. | Why |
|---|------|------|-------|------|------|-----|
| 83 | Carlini, Catalisano, Geramita, Waring problem for monomials `(in bib)` | 2012 | J. Algebra | our bib | A | Closed-form rank we use |
| 84 | Iarrobino and Kanev, *Power Sums, Gorenstein Algebras, and Determinantal Loci* `(in bib)` | 1999 | Springer LNM | CCG 2012 | A | Apolarity |
| 85 | Landsberg, *Tensors: Geometry and Applications* `(in bib)` | 2012 | AMS GSM | CCG 2012 | B | Waring / tensor rank |
| 86 | Alexander and Hirschowitz, Polynomial interpolation in several variables | 1995 | J. Algebraic Geom. | CCG 2012 | B | Generic Waring rank |
| 87 | Geramita, Inverse systems of fat points | 1996 | Queen’s Papers | CCG 2012 | B | Apolarity lemma |
| 88 | Reznick, Sums of even powers of real linear forms | 1992 | Mem. AMS | CCG 2012 | B | Real Waring |
| 89 | Comon, Golub, Lim, Mourrain, Symmetric tensors and symmetric tensor rank | 2008 | SIAM J. Matrix Anal. Appl. | Landsberg / CCG line | B | Computational Waring |
| 90 | Carlini, Reducing the number of variables of a polynomial | 2006 | Algebraic Geometry and Geometric Modeling | CCG 2012 | C | Essential variables |
| 91 | Ranestad and Schreyer, On the rank of a symmetric form | 2011 | J. Algebra | CCG 2012 | B | Rank vs border rank |
| 92 | Buczyńska and Buczyński, Secant varieties to high-degree Veronese reembeddings, catalecticant matrices and polarizations | 2014 | J. Symbolic Comput. | CCG line | C | Apolar scheme |
| 93 | Tevelev, Projective duality and homogeneous spaces | 2005 | Springer | CCG line | C | Duality / apolarity |
| 94 | Sylvester, On a remarkable discovery in the theory of canonical forms | 1851 | Phil. Mag. | CCG 2012 (historical) | C | Binary Waring |

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
From E: 83–86, 89.
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
