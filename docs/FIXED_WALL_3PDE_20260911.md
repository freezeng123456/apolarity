# Frozen single-seed, equal-wall-time T4 pilot

Authorized: one paired seed (20260918) for 1D KdV gPINN, 2D KdV, and forced standard fourth-order 2D Cahn–Hilliard. Each method receives 600 seconds, six cells sequentially on one T4. Methods are nested_jvp and shared_jet_linear. No extra seeds or automatic retries.

Common model: real float32 tanh MLP, three hidden layers of width 128, scalar output. Eager PyTorch; TF32 disabled. Adam, constant learning rate 1e-4, beta=(0.9,0.999), eps=1e-8. Interior batch 400, independent uniform spacetime sampling each step. Identical initialization and batch prefix within each pair. Primary endpoint is completed optimizer-step ratio (shared/nested). Record actual wall time and possible final-step overrun.

1D KdV uses the existing manufactured exact solution, PDE and gradient-residual loss (weight .001), and three prescribed boundary traces with IC/BC weights 10. 2D KdV uses the existing manufactured solution and prescribed-trace benchmark, without gradient-residual augmentation; each IC/BC term weight 10. Existing trace sets are preserved and are not a well-posedness theorem.

Cahn–Hilliard: domain [0,pi]^2, t in [0,1]; u=exp(-t)[.5 cos(x)cos(y)+.25 cos(2x)cos(y)+.25 cos(x)cos(2y)]. Equation u_t=Delta(u^3-u)-.01 Delta^2 u+f with analytic manufactured source. Homogeneous normal derivative of u and normal derivative of Delta u on all four faces. Loss is interior residual MSE +10 IC MSE +10 mean(face normal-u MSE) +10 mean(face normal-Laplacian MSE). IC and each face use 16x16 points. No mass penalty or gradient-residual term. Exact biharmonic rank-three real directions and all shared coefficients are tested independently.

Timer includes sampling, differentiation, backward, Adam, finite-gradient checks, step logs, batch hashes, and periodic checkpoint IO. Warmup without optimizer updates and initial checkpoint precede timer. Terminal checkpoint and diagnostics follow timer. Checkpoints approximately every 10 seconds; full fixed-point loss is recomputed offline with the coordinate-JVP reference on common 400 fixed uniform points plus the same IC/BC points. Loss against wall time is the only process curve.

Final errors: relative L2 and Linf on 10000 fixed uniform random spacetime points, and separately 10000 uniform spatial points at physical t=1. A disjoint 20000-point spacetime estimate quantifies Monte Carlo sensitivity. Evaluation coordinates and predictions are saved. No regular-grid error evaluation.

Launch gates: tests and two full-width, batch-400 optimizer steps per method/case on the actual T4 interpreter; matching initialization and closely matching paired final full loss. These short validation runs are outside the six 600-second budgets. Final audit checks all six statuses, changed weights, matching training-batch prefix and evaluation sets, SHA256 manifests, and step/error endpoints. A single seed is a pilot, not a multi-seed statistical claim.
