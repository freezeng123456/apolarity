# Frozen g-KdV linear reconstruction comparison

Only reconstruct the original five shared real order-four jets more directly.
No change to directions, model, equation, JVP baseline, or original evaluator.
No outer JIT, parameter backward pass, chunking, complex roots, or activation changes.

Directions in (x,t): (1,0), (1,1), (1,-1), (1,2), (1,-2).
Let C_k(s) be the raw k-th directional derivative along (1,s), without k! normalization.
Use fixed matrix contractions for (u_x,u_t), (u_xx,u_tx,u_tt), and (u_xxxx,u_txxx).
In particular u_txxx = (8 C_4(1) - 8 C_4(-1) - C_4(2) + C_4(-2))/48.
Take u and u_xxx directly from the pure-x jet. This is an implementation of the
same shared-direction operator schedule, not a new rank or direction-count claim.

The timed output remains the complete gradient-enhanced scalar objective:
f = u_t + u u_x + 0.0025 u_xxx;
J = mean(f^2) + 0.001 [mean(f_x^2) + mean(f_t^2)].
It is not parameter-gradient computation or training throughput.

One Tesla T4, JAX/jaxlib 0.4.38, float32, highest matmul precision.
MLP width 128, depth 4 (three hidden layers), original tanh and initialization.
The original space-time sampler uses uniform radius and time in [0,1], with a
normalized Gaussian spatial direction. Parameters and inputs stay dynamic.

Frozen matrix: batches 100, 400, 1600; methods nested_jvp, shared_jet,
shared_jet_linear; seeds 20260908, 20260909, 20260910. Nine isolated sequential
processes, 27 seed records, 10 synchronized warmups and 30 synchronized timings
per record (810 raw timed samples). Cold first calls are recorded separately.
All methods are remeasured in this run; old runs are never pooled.

Before timing: exact rational reconstruction tests and full-width B=3 GPU
verification for all three methods and all three seeds (nine records).
Reference uses independent per-term coordinate JVP and differentiates unexpanded
f to check f_x and f_t. After timing, compare every full-batch derivative and
objective to the matched JVP result, and require identical input/parameter hashes.
All comparisons use rtol=1e-3, atol=1e-6 and finite float32 arrays.

Predeclared material-benefit gate: every one of the nine matched seed/batch pairs
must be >=1.10 times faster than the original shared jet and faster than JVP.
Otherwise retain the original implementation; no tuning sweep in this protocol.
Correct completion and performance qualification are separate statuses.
Display medians of the three per-seed medians; preserve all timings and failures.
Freeze source in Git, use a detached remote worktree, checksum and recover the
complete raw result tree before reporting completion. Do not edit the paper or push.
