# Current paper workspace

当前论文正文尚未从旧 JSC 草稿中重建。本目录只作为新论文入口，避免旧图表
被误认为当前证据。

允许引用的数据：

- `outputs/current/polyharmonic-common-xavier-fp32-formal-v1/`；
- `outputs/current/cahn-hilliard-2d-fixed-1-10-formal-v1/`；
- 参数选择说明可引用 `outputs/search/cahn-hilliard-2d-weight-search-v1/`，但必须
  明确其为 60 秒搜参，不是正式精度结果。

旧 JSC 论文、表格与图片位于 `docs/archive/paper_jsc_legacy/`，不得直接复制其
经验结论。新增图应在开发服务器或 T4 环境从 raw JSON/history 生成，并同时
保存曲线 CSV、参数和 manifest。
