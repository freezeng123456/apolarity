# `jsc_v3` 权重冻结与运行时间预估

本文件记录下一轮正式训练使用的 boundary-loss profile。它只固化 archived
搜索中已经出现过的合理量级，并全部 round 到 (10^k)，不宣称这些值是全局
最优。v3 只比较 Complex Sinh 的 jet 后端和同结构的 direct-autodiff 后端。

## 冻结的权重

Poly 的向量顺序是 `[u, Delta u, Delta^2 u, ...]`；Chirp/Maxwell 只有一个
Dirichlet 分量，因此是单元素向量。

| family | task | `pow10_reasonable_v1` |
|---|---|---|
| Polyharmonic | d2/o2 | `[0.1]` |
| Polyharmonic | d2/o4 | `[0.1, 10.0]` |
| Polyharmonic | d2/o6 | `[0.01, 1.0, 10.0]` |
| Polyharmonic | d3/o2 | `[0.1]` |
| Polyharmonic | d3/o4 | `[0.1, 1.0]` |
| Polyharmonic | d3/o6 | `[0.1, 0.1, 1.0]` |
| Chirp | a1 | `[1.0]` |
| Chirp | a2 | `[0.1]` |
| Chirp | a3 | `[0.01]` |
| Maxwell | a2 | `[0.1]` |
| Maxwell | a4 | `[0.1]` |
| Maxwell | a6 | `[0.01]` |

这些值来自：

- `experiments/archived/results/poly_shared_weight_tuning/`、
  `poly_shared_weight_power_grid/`、`poly_shared_weight_cartesian/`；
- `experiments/archived/results/osc_shared_weight_grid/`；
- `experiments/archived/results/osc_shared_weight_full20m/selected_weights.json`；
- `experiments/archived/results/poly_power_grid_full20m/d3/selected_weights.json`。

旧的 active `jsc_v2` 结果仍保留在 `experiments/results/jsc_v2/`，其 loss 是
固定 scalar `bc_weight=100`；新 profile 使用新协议 `jsc_v3`，输出根目录为
`experiments/results/jsc_v3/`，避免把不同 loss 定义的 rows 混在一起。

## v3 运行协议

正式方法只有：

- `complex_sinh`：Complex Sinh + complex-Waring/Taylor-jet 导数；
- `complex_sinh_autodiff`：完全相同的网络、初始化和 loss，只把坐标导数
  换成 direct nested autodiff。

两条线都使用 (H=128)、3 个 seed（0/1/2）和每 run 1000 秒。每个
history 点记录 `[elapsed_seconds, rel_error, loss, L_int]`；结果行同时记录
`loss_last`、`L_int_last`、`L2_err` 和 `rel_error`。

## 单个 task 的运行量

每个 task 的正式比较是 2 methods × 3 seeds = 6 runs；每 run 的 training
budget 是 1000 秒。因此单个 task 的纯训练时间下界为：

```text
6 × 1000 s = 6,000 s = 1 h 40 min
```

这个估计不包括进程启动、模型初始化、周期性 evaluation、写盘和 validator。
根据已完成 v2 bundle 的最后 history timestamp，实际平均只比 1200 秒略高；高阶
Poly 的单步 overshoot 最大约 5 秒。

## 按 family 的总量（单 GPU 串行）

| family | task 数 | runs | 纯训练下界 | 按 v2 history endpoint 的近似 |
|---|---:|---:|---:|---:|
| Polyharmonic | 3 | 18 | 5 h 00 min | — |
| Chirp | 3 | 18 | 5 h 00 min | — |
| Maxwell | 3 | 18 | 5 h 00 min | — |
| 全部三组 | 9 | 54 | 15 h 00 min | — |

因此，如果仍按当前 `run_jsc_atomic.py` 的单 task、method/seed 串行方式使用
一张 H20，应该按 **15 小时纯训练，预留约 16–17 小时 wall-clock** 规划。若有
两张互不干扰的 H20 并且按 task 做并行调度，理论 run-level 下界约 7.5 小时；
按完整 task 分成两路则是 5 波、约 8 小时 20 分纯训练，实际建议按 8.5–9.5
小时规划。

## 当前代码状态

- 权重表：`experiments/common/boundary_weights.py`
- 新协议：`jsc_v3`，见 `experiments/common/protocol.py`
- 新结果根目录：`experiments/results/jsc_v3/`
- runner 会把 `boundary_profile_id` 和 `boundary_weights` 写入 manifest、rows 和
  history，实时曲线可以复核自己使用的权重
- 日志和结果行会显式打印/记录 `loss`、`L_int` 和 `rel_error`
- `scripts/run_jsc_main3.sh` 的日志前缀已切换为 `jsc_v3_`
- 目前没有启动训练，也没有生成任何 `jsc_v3` 结果目录
