# Numerical experiments in STDE and STDE++

## Scope and source status

This note records numerical experiments from primary sources only. STDE is the
NeurIPS 2024 conference paper. STDE++ is the authors' 2026 JMLR journal version,
which explicitly identifies STDE as its conference version. The JMLR paper
retains essentially the entire PINN experiment suite of the NeurIPS paper and
adds the polynomial-time mixed-partial construction. Consequently, the shared
PDE tables should not be counted as independent experimental evidence from two
different methods.

Primary sources:

- STDE conference paper: [NeurIPS 2024 PDF](https://papers.nips.cc/paper_files/paper/2024/file/dd2eb5250696753ea37141bbd89bb569-Paper-Conference.pdf), 38 pages.
- STDE arXiv record: [arXiv:2412.00088](https://arxiv.org/abs/2412.00088).
- STDE++ journal paper: [JMLR 27 (2026), paper 114](https://www.jmlr.org/papers/v27/25-1474.html), [PDF](https://www.jmlr.org/papers/volume27/25-1474/25-1474.pdf), 50 pages.
- Official implementation: [sail-sg/stde](https://github.com/sail-sg/stde).

## STDE: scope of the numerical study

The main paper states that STDE is evaluated on PINNs for (i) inseparable and
effectively high-dimensional PDEs, (ii) semilinear parabolic PDEs, and (iii)
high-order PDEs of orders 3 and 4. It also reports a weight-sharing experiment
and a detailed ablation of the source of the speed and memory gains. The main
paper says that the full setup and hyperparameters are in Appendix H and the
full results are in Appendix I (p. 7, Sec. 5).

### 1. Main speed and memory ablation: two-body Allen--Cahn

- **Purpose.** Isolate the source of the performance gain by changing only
  the derivative-evaluation method; compare speed, memory, and solution error
  across input dimensions (p. 8, Sec. 5.2).
- **Equation/task.** A nonlinear, inseparable two-body Allen--Cahn PINN with
  Laplacian differential part, an exact manufactured solution, and zero
  boundary condition on a (d)-dimensional unit ball. The paper evaluates
  dimensions (d=100,1{,}000,10{,}000,100{,}000,1{,}000{,}000) (Table 1 and
  Table 2, p. 8).
- **Methods.** Backward-mode SDGD in PyTorch; stacked backward-mode SDGD in
  JAX; parallelized stacked backward-mode SDGD via HVP; forward-over-backward
  SDGD; Forward Laplacian; and STDE (Tables 1--2, p. 8). The standard STDE
  setting uses Taylor-mode AD and random jet sampling.
- **Metrics.** Training speed in iterations per second (Table 1), peak GPU
  memory in MB (Table 2), and relative (L^2) solution error (Table 3 in
  Appendix I.1; discussion p. 8).
- **Core observations.** STDE remains runnable at one million dimensions,
  whereas the listed baselines encounter OOM at smaller dimensions. The paper
  reports that STDE is up to about (10\times) faster than the best
  parallelized SDGD realization and uses at least (4\times) less memory (p.
  9). The exact Forward Laplacian is best at (d=100), but its (O(d))
  scaling becomes inferior at larger dimensions (p. 9).
- **Caveat.** The authors explicitly note that randomized methods have
  comparable relative (L^2) error in the high-dimensional regime, while the
  exact Forward Laplacian can be better at small dimension (p. 8).

### 2. Framework and parallelization ablations

- **JAX versus PyTorch.** The original SDGD implementation is in PyTorch. The
  authors reimplement stacked backward-mode SDGD in JAX and report roughly
  (15\times) speed-up and up to (4\times) memory reduction. The comparison
  is part of Tables 1--2 and the accompanying discussion (p. 8).
- **Loop versus parallelized SDGD.** Replacing the loop over sampled
  dimensions by a parallelized HVP implementation gives roughly (15\times)
  speed-up and lowers peak memory during JIT compilation (p. 8, Sec. 5.2).
- **Mixed-mode AD.** Forward-over-backward SDGD has approximately the same
  performance as parallelized stacked backward-mode SDGD (p. 8).
- **Interpretation.** These controls show that implementation framework and
  parallelization materially affect measured speed; a fair derivative-method
  comparison should therefore keep the framework and batching strategy fixed.

### 3. Randomization-batch-size ablations

- **Purpose.** Test the trade-off between the number of random jets per
  iteration and computational cost/variance.
- **Task.** The main ablation reduces the batch size from 100 to 16 on the
  two-body Allen--Cahn equation and reports approximately $2\times$ speed-up
  without degrading the reported error (Appendix I.1.1 and Table 3).
- **Extended experiment.** Appendix L and Figure 5 repeat all three
  inseparable equations at $d=100{,}000$ with batch sizes
  $1,4,16,64,100,256$. They report final relative $L^2$ error, iterations per
  second, iterations to convergence, and time to convergence over five seeds.
- **Interpretation.** Lower batch size reduces cost, but the paper warns in
  its conclusion that it increases estimator variance; the speed--variance
  trade-off requires further analysis (p. 9, Conclusion).

### 4. Inseparable and effectively high-dimensional PDE family

- **Purpose.** Test whether sparse STDE scales to large dimension when the
  solution is not separable.
- **Equations.** On the (d)-dimensional unit ball, the paper considers the
  Poisson equation, Allen--Cahn equation, and Sine--Gordon equation (Appendix
  I.1, pp. 20--21). The exact solutions are manufactured from two-body and
  three-body interactions; coefficients are sampled as (c_i\sim N(0,1)).
- **Solutions.** The two-body solution uses local interactions involving
  adjacent coordinates and trigonometric terms. The three-body solution uses
  local triple interactions of the form (exp(x_i x_{i+1}x_{i+2})), multiplied
  by a boundary factor vanishing on the unit sphere (Appendix I.1, p. 20).
- **Measurements.** Tables 3--6 in Appendix I.1 report speed, memory, and
  relative (L^2) error for the equations and dimensions; the main text
  summarizes the Allen--Cahn ablation in Tables 1--2 (pp. 8, 21).

### 5. Semilinear parabolic PDEs

- **Purpose.** Test high-dimensional time-dependent PDEs rather than only
  elliptic/unit-ball manufactured problems.
- **Tasks.** Semilinear heat, time-dependent Allen--Cahn, and time-dependent
  Sine--Gordon equations, all evaluated at $x_{\rm test}=0$ and $T=0.3$
  (Appendix I.2).
- **Dimensions.** $d=10,100,1{,}000,10{,}000$ (Tables 7--9 in STDE; Tables
  8--10 in STDE++).
- **Methods and metrics.** STDE is compared with backward-mode SDGD in JAX and
  with values reported by the original PyTorch SDGD study. The tables report
  iterations per second, peak memory, and relative error against the reference
  value produced by a multilevel Picard method.
- **Setup.** The semilinear experiments use a four-layer, width-1024 tanh MLP,
  10,000 Adam steps, 2,000 residual points, and 100 boundary/initial points per
  step. The width-128 configuration in Appendix H applies to the inseparable
  elliptic experiments, not to this semilinear suite.
- **Source.** See the paper's Appendix I.2 and the official repository's
  “Semilinear Parabolic PDEs” reproduction instructions.

### 6. Weight sharing in high dimension

- **Purpose.** Separate the derivative-estimator cost from the parameter
  memory bottleneck of a fully connected first layer.
- **Method.** A non-overlapping 1D convolution with block size (B) is applied
  before the MLP. The paper derives a reduction in first-layer parameters from
  (d\times h) to approximately ((d/B)\times h+B) (Appendix G, pp. 18--19).
- **Experiment.** The two-body Allen--Cahn problem is run at one million and
  five million dimensions with block sizes $B=1,10,50,100,500,1000$
  (Table 10 in STDE; Table 11 in STDE++).
- **Metrics.** Iterations per second, peak memory, number of parameters, and
  relative $L^2$ error. At five million dimensions the unshared model is OOM,
  whereas suitable sharing blocks remain trainable; overly large blocks can
  sharply degrade accuracy.

### 7. High-order, low-dimensional PDEs (orders 3 and 4)

- **Purpose.** Demonstrate that STDE handles derivative orders beyond the
  second order and operators containing irreducible mixed partials.
- **Tasks.** 2D Korteweg--de Vries, 2D Kadomtsev--Petviashvili, and a
  gradient-enhanced 1D KdV problem (Appendix I.4.1). These residuals contain
  third- and fourth-order pure or mixed derivatives.
- **Sweep.** The base MLP has depth four and width 128; the paper separately
  varies depth to 8 and 16 and width to 256, 512, and 1024.
- **Comparisons and metrics.** Repeated backward-mode AD is compared with STDE
  (and an alternative lower-order-jet scheme for 2D KdV). Table 11 in STDE,
  renumbered Table 12 in STDE++, reports training throughput in iterations per
  second. The paper reports roughly $2\times$ speed-ups across several network
  sizes, but this table does not independently report derivative error or peak
  memory.
- **Relevance.** This is the closest STDE experiment family to the present
  paper's high-order derivative setting, but STDE uses randomized estimators,
  whereas the present method targets exact directional representations.

### 8. Amortized gradient-enhanced PINNs

- **Purpose.** Test whether STDE can make gradient-enhanced PINNs feasible in
  high dimension.
- **Tasks.** Two-body Allen--Cahn and Sine--Gordon equations with a randomized
  gradient-of-residual penalty (Appendix I.4.2).
- **Dimensions.** $d=100,1{,}000,10{,}000,100{,}000$.
- **Comparison.** STDE-based amortized gPINN is compared with a parallelized
  JVP--HVP baseline and with training without the gPINN penalty.
- **Metrics.** Iterations per second and relative $L^2$ error after 20,000
  training steps (Table 12 in STDE; Table 13 in STDE++). The result mainly
  demonstrates feasibility and throughput; it does not show uniformly better
  solution error than the no-gPINN setting.

### 9. Dense versus sparse STDE / connection to HTE

- **Status.** The dense-versus-sparse discussion is primarily theoretical in
  Sec. 4.4 and Appendix J/K. The paper gives a dense-jet construction for
  second-order operators and a counterexample for the fourth-order diagonal
  operator \(\sum_i\partial_i^4\), but the main numerical study is centered on
  sparse STDE. Appendix J also records the HTE construction and its variance
  facts; it should not be described as a separate large benchmark unless a
  specific table/figure is cited.

## What is new experimentally in STDE++?

The 2026 JMLR paper is an extended journal version of STDE. Its PDE experiment
tables are not a new independent suite: STDE++ Tables 2--13 reproduce the same
tasks and numerical values as STDE Tables 1--12, with the numbering shifted by
the insertion of one new table. The corresponding experiment families are the
Allen--Cahn performance ablation, inseparable Poisson/Allen--Cahn/Sine--Gordon,
semilinear parabolic equations, weight sharing, low-dimensional high-order
PDEs, and amortized gPINNs described above.

The additional computational result specific to the new polynomial-time
construction is Table 1 in STDE++ (p. 14). The authors run rejection sampling
for derivative orders $k=3,4,5,6,7$, list the selected primes, compare the sum
of the prime-based jet orders with the earlier exponential construction, and
verify uniqueness/minimality by dynamic programming and a Diophantine solver.
This is a construction/precomputation check, not a PINN accuracy or runtime
benchmark. The paper reports that for the practical range $k\leq7$ the older
exponential construction still has a smaller total jet order because the
polynomial construction has a large prefactor; the crossover is expected only
at substantially larger $k$.

Therefore STDE++ does **not** contain a dedicated scaling experiment that
compares its new mixed-partial construction with nested AD, the earlier STDE
construction, or another exact directional formula while sweeping derivative
order. This gap is directly relevant to the present apolarity paper: a clean
order/pattern sweep would provide evidence that is not already supplied by the
STDE/STDE++ papers.

## Experiments most suitable for the present apolarity/JCP paper

The most transferable design is a derivative-evaluation benchmark rather than
a large PINN-training suite. The paper should first use analytic test
functions or manufactured PDE solutions so that the target mixed derivative is
known exactly. Then it should vary the derivative order and the exponent
pattern of the multi-index, because the proposed direction count is governed
by the Waring-rank formula. The key measurements should be:

- relative error of the recovered derivative against analytic differentiation;
- wall-clock cost as the derivative order and number of required directions
  vary;
- peak memory;
- comparison with nested/repeated AD and, where relevant, STDE/STDE++;
- a separate scaling plot against the theoretical direction count
  \(\prod_{j=1}^{n}(\nu_j+1)\).

The STDE ablations suggest two useful controls: keep the software backend and
batching protocol fixed across methods, and report the exact method separately
from randomized estimators. STDE's high-order PDE examples (orders 3--4) and
its Allen--Cahn dimension sweep are useful templates for choosing tasks, but
the present paper should emphasize exactness and direction-count optimality,
not stochastic variance or PINN optimizer behavior unless those are directly
studied.
