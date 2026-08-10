# 2D third-order ZK formal result

独立正式协议：`zk_2d_o3`，WAR 与实数 AD 各 seeds 0–4、每 cell 1200 秒，
10/10 完成，0 失败、0 重试。正式结果没有混入 600 秒 pilot。

| 方法 | Mean rel_error | Std | Median | 胜出 seeds |
|---|---:|---:|---:|---:|
| WAR（complex64, sinh） | 0.0150048 | 0.00343113 | 0.0162814 | 5/5 |
| Real AD（float32, tanh） | 0.0292899 | 0.00368806 | 0.0301802 | 0/5 |

两种方法使用共同 Xavier、hidden=128、depth=4、相同墙钟与采样规模，只输入
仿射归一化 `(x,y,t)`，没有三角特征或频率初始化。周期性通过 trace loss
约束。固定 loss 权重为 `lambda_ic=lambda_bc=10`。

- `zk_2d_o3/seed_*/`：原始 JSON、文本日志、内嵌逐点 history 与配置；
- `analysis/formal_realtime_history.csv`：610 个原始正式 history 点；
- `analysis/formal_realtime_curves.csv`：五 seed 中位数与四分位带；
- `analysis/realtime_rel_error.{png,pdf}`：服务器生成的实时 accuracy 图；
- `analysis/REPORT_zh.md`：完整 pilot/formal 表、统计口径与公平性边界；
- `analysis/generate_analysis.py`：实际在 H20 Python 环境执行的生成脚本。

`SHA256SUMS` 覆盖 raw formal，`analysis/SHA256SUMS` 覆盖全部分析产物，
`DELIVERY_SHA256SUMS` 覆盖整个交付目录。论文图不是由 Codex 内置图片能力生成。
