# 2D KdV candidate: exact-solution and boundary review

Date: 2026-09-09
Scope: prescribed-trace benchmark design; no training-code changes and no GPU run in this review.

## Executive conclusion

For

\[
 u_{ty}+u_{xxxy}+3(u_{xy}u_x+u_yu_{xx})-u_{xx}+2u_{yy}=0,
 \qquad (x,y)\in[-1,1]^2,
\]

the proposed field

\[
 u(x,y,t)=\tanh\!\left(\tfrac12x+\tfrac12y-t\right)
\]

is an exact solution. It is a **one-dimensional oblique travelling wave** (a function of
\(\tfrac12x+\tfrac12y-t\)), not evidence of general two-dimensional dynamics.

The proposed constraints are a reasonable **prescribed-trace benchmark** for
reproducing this known solution. They must not be called a proved well-posed finite-
domain IBVP. The safest paper wording is “an analytic oblique-wave benchmark with
boundary traces chosen from the exact solution.”

## Exact-solution check

For the general ansatz \(u=A\tanh z\), \(z=kx+ly+wt\), direct substitution can be
integrated once with respect to \(z\) to give

\[
 k^3lU'''+3k^2l(U')^2+(lw-k^2+2l^2)U'=0,
 \qquad U(z)=A\tanh z.
\]

Writing \(S=\operatorname{sech}^2z\),

\[
 U'=AS,\qquad U'''=4AS-6AS^2.
\]

The coefficient of \(S^2\) gives \(A=2k\) for a nonzero wave, while the coefficient
of \(S\) gives

\[
 w=\frac{k^2}{l}-2l-4k^3\quad(l\ne0).
\]

Thus the selected values \(k=l=1/2\), \(A=1=2k\), \(w=-1\) give exactly the
proposed field. The implementation should retain an automatic residual test for this
specific \((k,l,A,w)\) tuple as a regression check.

## Boundary structure and the proposed constraints

Writing \(v=u_y\) gives

\[
 v_t+v_{xxx}+3(vu_x)_x-u_{xx}+2v_y=0,
 \qquad u(x,y,t)=g(x,t)+\int_{-1}^{y}v(x,\eta,t)\,d\eta,
\]

where \(g(x,t)=u(x,-1,t)\). The term \(2v_y\) has positive transport speed in the
increasing-
\(y\) direction, so \(y=-1\) is the inflow side for the \(v\)-equation and \(y=+1\)
is its outflow side. Therefore prescribing \(v(x,-1,t)=u_y(x,-1,t)\) and not
prescribing an independent \(v\) value at \(y=+1\) is directionally sensible. It is
not, by itself, a theorem of well-posedness for the coupled third-order nonlinear
problem.

In the \(x\)-direction, the leading spatial operator is third order. Standard finite-
interval KdV IBVPs consequently prescribe three independent boundary traces (often
one trace at one endpoint and two at the other, depending on the operator and
orientation). For example, Capistrano-Filho, Sun, and Zhang formulate three general
linear boundary operators for a finite-interval KdV problem and prove local
well-posedness only under assumptions on those operators and data regularity
([arXiv:1703.08154](https://arxiv.org/abs/1703.08154)). A commonly studied admissible
pattern is \(u(0,t),u(L,t),u_x(L,t)\), but that result is for their one-dimensional
KdV equation, not this mixed two-dimensional equation.

Therefore the proposed x traces

\[
 u(-1,y,t),\quad u(1,y,t),\quad u_x(1,y,t)
\]

are a defensible numerical choice by analogy with the third-order x operator, and
they match the exact solution when their right-hand sides are exact traces. They are
**prescribed traces for this benchmark**, not an established well-posedness closure
for the present PDE.

The anchor \(u(x,-1,t)=g(x,t)\) is useful for the reconstruction formula. It fixes
the integration constant in y; it is not an additional condition justified by the
third-order x count. In a direct \(u\)-network PINN it is simply a bottom-boundary
Dirichlet trace. If \(v\) is introduced as an independent numerical unknown, the
identity \(v=u_y\), the reconstruction relation, and all corner compatibilities must
be enforced consistently.

## Compatibility requirements

At minimum, generate all data from the same exact field:

* initial data: \(u(x,y,0)\), and, in a \(v\)-formulation, \(v(x,y,0)=u_y(x,y,0)\);
* x traces: \(u(-1,y,t)\), \(u(1,y,t)\), and \(u_x(1,y,t)\);
* bottom traces: \(u(x,-1,t)\) and \(u_y(x,-1,t)\);
* corner equalities at \(t=0\), and equality of the traces where x and y boundaries
  meet.

Do not prescribe an independent \(u\) value at \(y=+1\) if the intended formulation
treats that side as the \(v\)-transport outflow. It may still be used as an
**evaluation-only** boundary check against the analytic trace. Likewise, do not
impose both arbitrary \(v\) data and arbitrary \(u\) data that violate \(v=u_y\).

## Safe implementation recommendation

For the first 2D experiment, retain a single direct \(u(x,y,t)\) network and use the
analytic traces as soft constraints. Include interior residual, initial condition,
the three x traces, bottom \(u\), and bottom \(u_y\). Evaluate the omitted top trace
and an independent dense-grid residual only for validation. This tests the shared-jet
implementation without pretending that a mixed \((u,v)\) evolution solver has been
closed.

If a future experiment uses the reconstructed \((g,v)\) system, document the extra
coupling and boundary operators explicitly and run a linearized energy/trace study
before using “well-posed.” The transport inflow/outflow rule supports the orientation
of the bottom \(v\) condition, but does not replace that analysis.

## Claim boundaries for the paper

Supported claims:

1. The displayed \(\tanh\) field exactly satisfies the stated PDE (up to numerical
   evaluation error).
2. The experiment measures residual/solution accuracy and shared-jet cost on a bounded
   oblique-wave benchmark with prescribed exact traces.
3. The bottom \(v\) trace is consistent with the positive-speed \(y\)-transport term
   in the auxiliary equation.

Unsupported without a separate analysis:

1. “The selected conditions make the 2D nonlinear IBVP well posed.”
2. “No top boundary condition is universally required.” The statement is only for the
   chosen auxiliary transport orientation and numerical closure.
3. “The method handles general 2D KdV dynamics.” The exact solution depends on one
   scalar phase and is therefore an oblique travelling wave.

## Source

The finite-interval KdV source used for the three-trace/conditional-well-posedness
discussion is R. A. Capistrano-Filho, S.-M. Sun, and B.-Y. Zhang, *General Boundary
Value Problems of the Korteweg-de Vries Equation on a Bounded Domain*, arXiv:1703.08154
(2017), [https://arxiv.org/abs/1703.08154](https://arxiv.org/abs/1703.08154).

Its role here is limited: it describes boundary operators for a one-dimensional
finite-interval KdV problem. It is not being used as a theorem for the present mixed-
derivative two-dimensional PDE.
