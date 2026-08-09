# Polyharmonic：common-Xavier 低精度正式结果

这是当前 Polyharmonic 论文实验的完整交付包，不是 smoke 或历史 double
结果。三组任务均以单张 H20 严格串行运行，每个方法/seed 的训练预算为
1200 秒：

| task | 边界分量权重 | seeds | 方法 |
|---|---|---:|---|
| `poly_d2_o2` | `[1]` | 0–4 | WAR / real tanh AD |
| `poly_d2_o4` | `[1, 1]` | 0–4 | WAR / real tanh AD |
| `poly_d2_o6` | `[10, 1, 1]` | 0–4 | WAR / real tanh AD |

WAR 是 `complex64 + sinh + Waring/Taylor jet`；实数基线固定为
`float32 + tanh + direct autodiff`。两者都是四层、hidden=128、共同 Xavier
类初始化，不使用三角输入、频率匹配或任务感知频率初始化。

## 完整性

- `FORMAL_COMPLETE`、`summary.json`、`progress.json` 均记录 30/30 完成；
- 每个 task/seed 下都有 `config.json`、两份方法 JSON 和两份逐次文本日志；
- 每份方法 JSON 的 `history` 字段含 241 个实时打点，总计 7230 点；
- 原始 `SHA256SUMS` 覆盖服务器返回的 113 个数据文件且保持不变；
- `DELIVERY_SHA256SUMS` 额外覆盖本地审计、来源快照与本说明；
- `analysis/audit.json` 的状态为 `passed`。

## 结果入口

- `analysis/REPORT_zh.md`：验收、五 seed 汇总及结论；
- `analysis/final_metrics.csv`：30 个 cell 的最终 loss、rel_error、速度和显存；
- `analysis/paired_comparison.csv`：同 seed 的 WAR/AD 配对比较；
- `<task>/seed_<NNN>/<method>.json`：完整结果与实时 history；
- `<task>/seed_<NNN>/<method>.log`：逐次日志，末行同时含 loss 与 rel_error；
- `provenance/`：实际运行源文件快照及来源边界。

重要结论：o2、o4 均学到低误差解；o6 的两种方法最终相对误差都约为 1，
因此 o6 在当前无频率初始化协议下是失败算例，不能被汇总数值掩盖。
