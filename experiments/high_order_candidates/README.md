# High-order PDE candidate screen

This active screening family evaluates four mathematically well-posed candidate
benchmarks before any one of them is promoted to a formal paper experiment:

| task | spatial dimension | highest order | type |
|---|---:|---:|---|
| `zk_2d_o3` | 2 | 3 | nonlinear dispersive IVP |
| `zk_3d_o3` | 3 | 3 | nonlinear dispersive IVP |
| `dynamic_plate_2d_o4` | 2 | 4 | damped hyperbolic plate |
| `swift_hohenberg_2d_o4` | 2 | 4 | coercive nonlinear elliptic |

All tasks use smooth manufactured forcing so that solution error can be
measured directly.  This does not alter uniqueness: the initial/boundary value
problem is fixed before training, and the source is simply the forcing for that
problem.

The comparison is frozen to:

- WAR: native `complex64`, `sinh`, Waring/Taylor jet;
- baseline: real `float32`, `tanh`, direct nested autodiff;
- literal hidden width 128 and depth 4 for both;
- common Xavier-class initialization;
- affine-normalized raw coordinates only;
- no Fourier/periodic embedding and no frequency-aware initialization.

Periodic conditions are enforced by paired trace losses rather than by putting
`sin` or `cos` in the network input.  Candidate results are screening evidence,
not paper evidence, until a task is selected and rerun under a frozen five-seed
formal protocol.
