# Two-dimensional slope-selection MBE

This active benchmark solves

`h_t = div((|grad h|^2 - 1) grad h) - 0.05 Delta^2 h`

on the `2*pi` periodic two-torus for `t in [0,1]`, with initial height

`h_0 = 0.2 cos(x) cos(y) + 0.1 cos(2x) cos(y)`.

The comparison is deliberately architecture-transparent:

- WAR: native `complex64`, `sinh`, Waring Taylor jet;
- baseline: real `float32`, `tanh`, direct nested autodiff;
- both: hidden width 128, depth 4, common Xavier family, affine raw
  `(x,y,t)` coordinates only;
- forbidden for both: trigonometric input features, periodic embeddings, and
  frequency-aware initialization.

Opposite periodic faces explicitly match the value and normal derivatives of
orders 1, 2, and 3.  Training uses the direct fourth-order residual; it does
not introduce an auxiliary field to lower the PDE order.

The 3-second smoke uses the manufactured profile `exp(-t) h_0` and its
analytic source.  Weight search and later multi-seed stages use the unforced
equation and a separately generated, three-level convergence-checked ETDRK4
pseudospectral reference supplied by `APOLARITY_MBE_REFERENCE_PATH`.

The mathematical setting matches the periodic-torus model and global
well-posedness result in Li, Qiao, and Tang, *Gradient bounds for a thin film
epitaxy equation* (arXiv:1410.7572).
