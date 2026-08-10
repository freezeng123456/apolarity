# 二维 Cahn–Hilliard 固定权重正式结果

这是当前 CH2D 的 20/20 正式交付包：CH4/CH6，WAR 与
`real_sinh_autodiff`，seeds 0–4，每 cell 1200 秒，固定
`(lambda_ic,lambda_bc)=(1,10)`。

- raw 完整性由根目录 `SHA256SUMS` 的 99 项校验；
- 每个 task/seed 保留 config、JSON、日志、history 与 DONE marker；
- `analysis/REPORT_zh.md` 给出五 seed 汇总、公平性边界和配对比较；
- `analysis/realtime_curves.csv` 是曲线事实来源；现有 PNG/PDF 来自服务器分析
  环境，不是 Codex 工作区内置图片生成结果；
- smoke 只保留 `SMOKE_CONCLUSION.json`，不保留 raw smoke bundle。

WAR 在 CH4、CH6 都是 5/5 seed 更低，但比较是相同字面层形状/墙钟预算，
不是等实自由度。
