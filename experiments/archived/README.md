# Archived experiments

这里保存已经退出当前论文口径的实验材料。它们可用于追溯实现与历史搜参，
但不能与 `outputs/current/` 的 float32/complex64 结果拼接。

| 路径 | 内容 |
|---|---|
| `other_families/chirp/` | 旧 radial-chirp 问题与 launcher |
| `other_families/maxwell/` | 旧 Maxwell 问题与 launcher |
| `other_families/cahn_hilliard/` | 旧一维/历史 CH family |
| `other_families/*` | Helmholtz、KdV、NLS、plate/beam 等历史 family |
| `results/double_protocols/jsc_v2/` | 旧 JSC v2 完整结果 |
| `results/double_protocols/jsc_v3/` | 已明确退出当前范围的 JSC v3 结果 |
| `results/*` | 旧 baseline、敏感性、权重搜索和控制实验 |
| `logs/jsc_v3/` | JSC v3 历史日志 |
| `scripts/` | 与上述旧实验配套的 runner、validator 与分析脚本 |
| `tools/jsc/` | 旧 JSC 图表/LaTeX 工具 |
| `jsc_v3/` | 旧 JSC v3 task registry、边界权重、Poly 入口与协议测试 |

旧脚本按“可追溯快照”维护，不保证继续作为当前入口运行。新的正式实验必须
从根目录 `scripts/` 启动；新的结果必须写入 `outputs/current/` 或
`outputs/search/`。Smoke 原始结果已经清理，只保留
`docs/archive/SMOKE_CONCLUSIONS_zh.md` 中的门禁结论。
