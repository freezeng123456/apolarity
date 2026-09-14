# T4 / PyTorch 第一部分结果

本目录为指定高阶混合偏导 value-only 矩阵的完整回收副本。它与第二部分共享 T4-B 与 PyTorch 2.0.1+cu118 的环境口径，但不重跑、修改或混合第二部分 PINN 实验。

- `results/`：逐单元配置、每次同步计时、三种子输出数组、输入/参数/输出哈希及原始校验。
- `REPORT.md`：正式中文矩阵和解释边界。
- `summary.csv`、`direction_parallelism_ablation.csv`：保留三种子中位数与配对比值的机器可读表。
- `audit_summary.json`、`output_validation.json`：独立审计；`SHA256SUMS` 是整个回收目录的传输完整性清单。

该工作负载只测导数求值吞吐，明确不含参数 backward，也不是 PINN 训练吞吐。
