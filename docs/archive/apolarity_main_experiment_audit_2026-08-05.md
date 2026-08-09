# P0-A / P0-B / P0-C 主实验核查

本核查明确排除 `auto` 选择器。完整的本地审计记录在
`outputs/apolarity_overnight_live/analysis/main_experiments_audit_zh.md`；这里保留
可随仓库复现的结论摘要。

## 三个问题的直接答案

### Boundary 是否全部搜过？

没有。

- P0-A 是 derivative microbenchmark，没有 boundary。
- P0-B 固定 `n_bc=64`、`bc_weight=100`、随机 face sampling；本轮没有搜索
  `bc_weight`、`n_bc`、face 比例或采样策略。
- P0-C 固定 `n_bc=512`；历史上搜过若干代表任务的标量 `bc_weight`，但没有
  搜 `n_bc`/采样策略/逐 boundary-component 权重。Maxwell `a=4` 的历史协议还
  存在 `0.03`（per-method tuning）和 `0.1`（shared Vanilla/Sinh grid）两套
  候选，夜间正式复核使用 `0.03`，需要统一后再封版。

### 每个主实验是否有 autodiff baseline？

不完全是。

- P0-A 有 direct nested AD，作为 value 和 parameter-gradient reference；
- P0-B 有正式 direct-AD 端到端 PINN，对应固定步数控制和 5-seed × 1200 s；
- P0-C 的 Chirp 有正式 5-seed vanilla direct-AD，但 Maxwell 正式 5-seed 组
  没有 direct-AD 训练基线，只有早期单 seed vanilla/direct-Laplacian pilot。

### 日志和数据打点是否足够？

- P0-A：1260 rows，包含 repeats、分位数计时、峰值显存和数值误差；没有
  training history，因为它不是训练实验。
- P0-B：每个 fixed-time run 有约 119–120 个 probe accuracy 点，并保留
  step、wall/training time、LR、`L_int`、`L_bc`、`L_im`、L2/Linf、PDE/BC RMS、
  checkpoint 和显存，粒度最完整。
- P0-C：20 个 CSV/JSON/history/manifest 全部存在；每条 history 有 83–248 个
  `(time, relative L2, L_int)` 点，但没有逐时间 `L_bc`、LR、显存，run log 主要
  是最终摘要。

## 结果摘要

| 实验 | 关键结果 |
|---|---|
| P0-A | 1260/1260 通过；最大 value abs error `6.883e-15`，最大 parameter-gradient relative L2 `1.058e-13`；complex128/B8 `(4,2)` cached jet 约 `26.9x` 快于 direct，`(8)` 约 `168x`。 |
| P0-B | 5-seed median final L2：direct `3.721e-3`、polarization `8.183e-4`、Waring `9.324e-4`；median ms/step：`151.537/8.065/7.231`。 |
| P0-C | Chirp a2：Complex Sinh `1.285e-4` vs vanilla `1.755e-3`；Maxwell a4：PWNN `1.276e-4` vs Complex Sinh `1.360e-3`。20/20 完成且 `nan=false`。 |

## 论文图

历史 mixed-scope 图的向量版、PNG 预览和 CSV 已移动到
`docs/archive/figures/mixed_scope/`；生成脚本已移动到
`experiments/archived/scripts/plot_paper_accuracy_mixed_scope.py`。这些文件混合了
已归档的 P0/backend/risk-baseline 结果，只保留作 provenance，不能作为当前三主实验
图。当前 active 图和新的边界/autodiff/logging 核查见
`docs/paper/active_three_main_experiments_audit_zh.md`。

在补齐 boundary protocol、Maxwell formal direct-AD baseline 和 P0-C 逐时
boundary logging 之前，不应把结果写成“boundary 已充分调优”或“所有主实验均
有完整 direct-AD baseline”。
