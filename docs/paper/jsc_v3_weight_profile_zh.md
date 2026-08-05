# `jsc_v3` 权重冻结与运行时间预估

本文件记录下一轮正式训练使用的 boundary-loss profile。它只固化 archived
搜索中已经出现过的合理量级，并全部 round 到 (10^k)，不宣称这些值是全局
最优。当前只改配置和 runner，**尚未启动任何 `jsc_v3` 训练**。

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

## 单个 task 的运行量

每个 task 的正式比较是 4 methods × 5 seeds = 20 runs；每 run 的 training
budget 是 1200 秒。因此单个 task 的纯训练时间下界为：

```text
20 × 1200 s = 24,000 s = 6 h 40 min
```

这个估计不包括进程启动、模型初始化、周期性 evaluation、写盘和 validator。
根据已完成 v2 bundle 的最后 history timestamp，实际平均只比 1200 秒略高；高阶
Poly 的单步 overshoot 最大约 5 秒。

## 按 family 的总量（单 GPU 串行）

| family | task 数 | runs | 纯训练下界 | 按 v2 history endpoint 的近似 |
|---|---:|---:|---:|---:|
| Polyharmonic | 6 | 120 | 40 h 00 min | 约 40.03 h |
| Chirp | 3 | 60 | 20 h 00 min | 约 20.00 h |
| Maxwell | 3 | 60 | 20 h 00 min | 约 20.00 h |
| 全部三组 | 12 | 240 | 80 h 00 min | 约 80.04 h |

因此，如果仍按当前 `run_jsc_atomic.py` 的单 task、method/seed 串行方式使用
一张 H20，应该按 **约 80 小时纯训练，预留 81–83 小时 wall-clock** 规划，而
不是按一个夜晚规划。若有两张互不干扰的 H20 并且按 task 做并行调度，理想下界
约为 40 小时；这需要单独的并行调度计划，当前 runner 默认不会自动并行。

## 当前代码状态

- 权重表：`experiments/common/boundary_weights.py`
- 新协议：`jsc_v3`，见 `experiments/common/protocol.py`
- 新结果根目录：`experiments/results/jsc_v3/`
- runner 会把 `boundary_profile_id` 和 `boundary_weights` 写入 manifest、rows 和
  history，实时曲线可以复核自己使用的权重
- `scripts/run_jsc_main3.sh` 的日志前缀已切换为 `jsc_v3_`
- 目前没有启动训练，也没有生成任何 `jsc_v3` 结果目录
