# KP-II: five-seed mean curves on V100

The paper reports seeds **20260936–20260940**, with WDD and STDE++ each
completing 20,000 Adam updates in float32 on Tesla V100-SXM2-32GB GPUs.
The learning rate is 1e-4. Both methods use the same four-hidden-layer,
width-128 tanh network, 400 resampled interior points, and 256 resampled
points on each initial or lateral boundary face. Paired initial-parameter,
training-sample, and independent-test-set hashes match for every seed.

| Method | Training time (s), mean ± sample SD | Final RE, mean ± sample SD |
|---|---:|---:|
| WDD | 913.1 ± 37.3 | 0.03726 ± 0.00603 |
| STDE++ | 1205.6 ± 29.5 | 0.04052 ± 0.00705 |

The mean of the five paired STDE++/WDD time ratios is 1.3214 ± 0.0375.
At each recorded update index, the figure plots arithmetic mean RE against
arithmetic mean synchronized training time. Shading extends from the mean
to the mean plus one sample standard deviation. Every recorded point is
used; no smoothing, interpolation, extrapolation, or time cutoff is applied.
Time excludes setup, warm-up, test evaluation, and curve writing. RE uses
10,000 fixed independent uniform space-time test points.

## Cohort history

The initial observation cohort was 20260934–20260938 (job array 2391929).
After inspecting pronounced transients in the first two seed pairs, the user
requested two additional pairs, 20260939 and 20260940 (job array 2392431),
and explicitly chose 20260936–20260940 for the paper's five-seed average.
This is a selected descriptive cohort. All **seven** pairs, including the
two pilot pairs omitted from that average, remain under `cells/` and in
`all_seed_metrics.csv`. Original run plans retain their original cohort
roles and policies; this README records the later selection.

## Construction and reproducibility

The problem and line soliton follow Pu and Chen (2024), Eqs. (3.7)–(3.9),
<https://arxiv.org/html/2401.04982v1>. The Adam/resampling protocol is the
present paper's protocol, rather than Pu and Chen's L-BFGS experiment.

The STDE++ derivative logic is ported from `KdV2d_res`, case 6, at upstream
commit `fae88663b1d2f1b0666d4d250c38a3c34b852ac0`. It recovers `u_tx`
from an order-five nonlinear-curve expansion with raw second tangent e_x
and third tangent e_t, alongside an order-four x expansion and the two
spatial order-two coordinate expansions. The upstream Hessian-return
unpacking typo is corrected to use the returned function value.
WDD uses an order-four x expansion and three order-two directions:
(1,0,1), (1,0,-1), and (0,1,0). Both use the frozen common Taylor arithmetic.
No native JAX timing or newer prepared-coefficient optimization enters these
measurements. `source/` is the exact source inventory verified against all
recorded configuration hashes, with fingerprint
`5363b5b123d0309617d69dc0ff6bae7626a612f3bee7aa64d23818b9076df2e4`.

From the repository root, rebuild the mean curve, table, and numeric summaries:

```bash
python docs/paper/build_kp_experiment_assets.py
python docs/paper/build_kp_experiment_assets.py --check
```

To rerun on a V100 allocation with PyTorch 2.5.1+cu121, NumPy 1.26.4,
Python 3.11.7, 10 allocated CPU cores, and one Torch/OMP thread:

```bash
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python results/kp_fp32_v100_20261021/rerun.py plan --out /absolute/new/kp-run
python results/kp_fp32_v100_20261021/rerun.py preflight --out /absolute/new/kp-run
python results/kp_fp32_v100_20261021/rerun.py lane --out /absolute/new/kp-run --lane 0
```

Run lanes 0–4 on the same GPU/CPU configuration. Within each lane, the two
methods run sequentially, with their order alternated between lanes.
`validation/PASS.json` retains the original objective/component/gradient
checks. `campaigns/` retains scheduler logs, run plans, launchers, and
transfer checksums for both recovered campaigns. Each cell has its original
`SHA256SUMS`. Only numeric curves and metadata are retained; there are no
model or optimizer checkpoints.
