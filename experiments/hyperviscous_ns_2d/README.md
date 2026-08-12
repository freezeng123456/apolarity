# HO-04: 2D fourth-order hyperviscous Navier--Stokes

This benchmark solves the incompressible periodic system

```text
u_t + (u . grad)u + grad p - nu Delta u + eta Delta^2 u = 0,
div u = 0,
```

on `[0, 2*pi]^2 x [0, 1]`, with `nu=0.05` and `eta=0.01`.  Its
mean-zero Taylor--Green solution is

```text
A(t) = exp(-(2*nu + 4*eta)t),
u =  A(t) sin(x) cos(y),
v = -A(t) cos(x) sin(y),
p =  A(t)^2/4 * (cos(2x) + cos(2y)).
```

The single network outputs `(u,v,p)`.  The primary accuracy metric is the
combined velocity relative L2 error; pressure error, divergence, pressure
gauge and kinetic-energy decay are reported separately.

Fairness contract:

- WAR: native `complex64`, `sinh`, Waring Taylor jet;
- baseline: `float32`, `tanh`, direct coordinate autodiff;
- hidden 128, depth 4, common Xavier and identical literal layer shapes;
- affine-normalized raw `(x,y,t)` only;
- no trigonometric input features, periodic embedding or frequency-aware
  initialization;
- velocity periodic traces are matched through normal derivative order three;
- pressure is periodic at order zero and has a batched per-time zero-mean
  gauge penalty.

The full workflow is driven by `scripts/run_hyperns_pipeline.py`.  It uses
formula/unit gates, two CUDA smokes, a three-point sentinel, a complete 7x7
shared `(lambda_ic,lambda_bc)` search, a 3-seed pilot and, only after the
pre-registered accuracy and physics gates pass, a 5-seed formal run.

