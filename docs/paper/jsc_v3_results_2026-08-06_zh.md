# jsc_v3 正式结果（2026-08-06）

本次正式运行固定为 9 个任务、两条方法线、3 个 seed、每个 seed 1000 秒。`complex_sinh` 是 jet 后端，`complex_sinh_autodiff` 是同结构的直接 autodiff 基线。每条 history 使用

```text
[elapsed_seconds, rel_error, loss, L_int]
```

所有 9 个 task 均已生成 `VALIDATED`，共 54 个正式 run。下表的均值和标准差是在三个 seed 的最终 `rel_error` 上计算的。

| task | Jet mean ± std | Direct autodiff mean ± std | Jet seed values | Direct autodiff seed values |
|---|---:|---:|---:|---:|
| `poly_d2_o2` | 3.2468e-4 ± 1.1471e-4 | 5.4199e-4 ± 8.7753e-5 | 1.9238e-4, 3.9645e-4, 3.8519e-4 | 5.6185e-4, 4.4601e-4, 6.1811e-4 |
| `poly_d2_o4` | 6.5539e-4 ± 7.1195e-5 | 1.4274e-3 ± 1.0186e-4 | 5.9246e-4, 7.3266e-4, 6.4104e-4 | 1.3824e-3, 1.5440e-3, 1.3558e-3 |
| `poly_d2_o6` | 3.3587e-2 ± 1.8068e-2 | 9.1781e-2 ± 3.6188e-2 | 2.3453e-2, 5.4447e-2, 2.2861e-2 | 8.7101e-2, 1.3008e-1, 5.8161e-2 |
| `chirp_a1` | 3.8396e-4 ± 3.6751e-4 | 7.0457e-4 ± 3.5455e-4 | 3.3101e-5, 7.6611e-4, 3.5266e-4 | 3.5748e-4, 6.9011e-4, 1.0661e-3 |
| `chirp_a2` | 8.9304e-4 ± 1.0972e-3 | 8.3599e-4 ± 4.5624e-4 | 1.3912e-4, 2.1518e-3, 3.8822e-4 | 1.1971e-3, 9.8760e-4, 3.2325e-4 |
| `chirp_a3` | 1.8623e-4 ± 2.3483e-5 | 3.3843e-4 ± 1.2507e-4 | 1.7635e-4, 1.6930e-4, 2.1304e-4 | 2.0065e-4, 4.4482e-4, 3.6982e-4 |
| `maxwell_a2` | 1.2386e-3 ± 4.4156e-4 | 1.6064e-3 ± 7.1777e-4 | 1.0042e-3, 9.6373e-4, 1.7480e-3 | 7.8456e-4, 2.1101e-3, 1.9246e-3 |
| `maxwell_a4` | 1.2375e-3 ± 3.9620e-4 | 1.4776e-3 ± 6.7497e-4 | 1.2390e-3, 8.4054e-4, 1.6329e-3 | 1.3301e-3, 8.8851e-4, 2.2141e-3 |
| `maxwell_a6` | 9.2722e-3 ± 1.5196e-3 | 9.4163e-3 ± 1.4937e-3 | 8.9577e-3, 1.0924e-2, 7.9344e-3 | 9.8135e-3, 1.0671e-2, 7.7642e-3 |

## 主要观察

- 9 个 task 的 task-level 平均 `rel_error` 全部低于 `1e-1`。
- Jet 在 `poly_d2_o2`、`poly_d2_o4`、`poly_d2_o6`、`chirp_a1`、`chirp_a3`、`maxwell_a2`、`maxwell_a4` 上的均值低于直接 autodiff；在 `chirp_a2` 上两者接近，`maxwell_a6` 上基本持平。
- 最难的是 `poly_d2_o6`：Jet 均值约 `3.36e-2`，直接 autodiff 均值约 `9.18e-2`，且直接 autodiff 的一个 seed 为 `1.30e-1`。这组需要在论文中报告 seed 方差，不能只报一个最优 seed。
- `maxwell_a6` 是 Maxwell 中最难的频率设置，但两种后端仍维持约 `9e-3` 的平均相对误差。
- 运行日志逐条打印了最终 `loss`、`L_int` 和 `rel_error`；完整逐步数据在每个 task 的 `_history.json` 中。

## 产物

- 全量原始结果、canonical CSV/JSON、history、task manifest：`experiments/results/jsc_v3/`
- 汇总表：`experiments/results/jsc_v3/summary.csv` 和 `summary.json`
- 结果清单与校验和：`experiments/results/jsc_v3/RESULT_MANIFEST.json`、`SHA256SUMS`
- 运行日志及日志校验和：`experiments/logs/jsc_v3_*.log`、`experiments/logs/jsc_v3_SHA256SUMS`
- 论文用实时相对误差图（Poly d2/o4、Chirp a2、Maxwell a4，中位数曲线与 seed IQR）：`docs/paper/figures/jsc_v3_realtime_accuracy.svg`
