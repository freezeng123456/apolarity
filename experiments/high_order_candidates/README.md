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
`sin` or `cos` in the network input.

The completed 600-second pilot is stored under
`outputs/search/high-order-candidate-pilot-v1/`.  The 2D ZK and dynamic-plate
tasks passed the frozen trainability gate; 3D ZK and Swift–Hohenberg converged
to relative error near one under both methods.  The predeclared preference for
a trainable ZK problem distinct from the existing Poly/CH families selected
`zk_2d_o3` for the independent five-seed, 1200-second formal rerun.

The completed formal bundle is
`outputs/current/high-order-zk2d-formal-v1/`.  WAR obtained lower final relative
error on all five paired seeds (mean 0.0150 versus 0.0293 for real AD).  The
bundle contains raw JSON/log/history, checksums, aggregate CSVs, and figures
generated on the H20 server by the committed analysis script.
