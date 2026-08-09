# Result inventory

结果按科学角色而不是生成日期分层：

```text
outputs/
  current/   当前正式、可进入论文主结果的完整证据
  search/    当前协议的搜参与候选排序，不等同于正式精度结论
  archive/   历史协议，只用于追溯
```

## Current

| 目录 | 协议 | 状态 |
|---|---|---|
| `current/polyharmonic-common-xavier-fp32-formal-v1/` | Poly，WAR complex64+sinh vs real float32+tanh AD | 30/30，已审计 |
| `current/cahn-hilliard-2d-fixed-1-10-formal-v1/` | CH2D，权重 `(1,10)`，WAR vs real sinh AD | 20/20，已审计 |

正式包保留原始 JSON、日志、history、配置、summary、manifest 和 checksum。
分析文件是可再生派生物，不能替代 raw 数据。

## Search

`search/cahn-hilliard-2d-weight-search-v1/` 保存 CH4/CH6 的完整二维 60 秒权重
网格：98 个候选、196 个方法 cell。它用于选择权重与分析敏感性，不用于替代
1200 秒五 seed 正式实验。

## Archive

`archive/double/` 保存旧 float64/complex128 fixed-weight 和完整网格。其 raw
smoke 已删除；删除前 checksum 清单保存在相应 `metadata/`，归档根目录的新
`SHA256SUMS` 覆盖剩余内容。历史结果不能与 current 合并统计。
