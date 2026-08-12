# HO-04 二维四阶超黏性 Navier--Stokes 结果交付清单

## 协议

- 任务：`hyperviscous_ns_2d_o4`
- 固定代码提交：`8118889c9f4ff8cca8dc02b94059d9502e2ce5a6`
- 结果协议：`hyperns_ho04_gated_pipeline_v1`
- 区域：`[0, 2pi]^2 x [0, 1]`；`nu=0.05`，`eta=0.01`
- 解：无外力解析 Taylor--Green；网络直接输出 `(u,v,p)`
- 输入/网络：raw affine `(x,y,t)`，共同 Xavier，hidden128，depth4；无三角输入、无周期嵌入、无频率初始化
- WAR：`complex64 + sinh + Waring jet`
- 对照：`float32 + tanh + direct autodiff`
- 正式固定权重：`lambda_ic=1e2, lambda_bc=1e1`
- 正式协议：seeds `0..4`，每方法每 seed `1200 s`，单张 H20 严格串行；采样 `n_int=2048, n_ic=512, n_bc=1024, n_eval=16384, history_eval_n=2048`

## 完整性

- `PIPELINE_COMPLETE`、`SEARCH_COMPLETE`、`PILOT_COMPLETE`、`FORMAL_COMPLETE` 均存在。
- 搜参：`98/98` method runs，`49/49` paired candidates，无失败。
- Pilot：`6/6` method runs，`3/3` paired seeds，无失败。
- Formal：`10/10` method runs，`5/5` paired seeds，无失败。
- formal 每个 run 均有 `history_points=121`，且末值同时含有限 `loss` 与 `rel_error`。
- GPU 在收尾验收时无计算进程，未残留 worker。

## Formal 汇总（5 seeds）

| 指标 | WAR | real-tanh autodiff |
|---|---:|---:|
| velocity relative L2 中位数（均值） | 0.005467 (0.005691) | 0.027698 (0.029214) |
| pressure relative L2 中位数（均值） | 0.043437 (0.043900) | 0.186795 (0.188816) |
| divergence RMS 中位数（均值） | 0.003720 (0.003507) | 0.019685 (0.020733) |
| energy relative RMSE 中位数（均值） | 0.006966 (0.007309) | 0.021116 (0.023586) |
| 每步耗时中位数（ms） | 56.25 | 224.32 |
| 峰值显存（MB） | 866.05 | 2521.88 |

WAR 在 `5/5` 个 paired seeds 的 velocity relative L2 上获胜。AD/WAR 中位每步耗时比为
`3.979`，峰值显存比为 `2.912`。共享几何平均误差中位数为 `0.012508`，共享
minimax 中位数为 `0.027698`。

## 文件与图

- `smoke/`、`sentinel/`、`search/`、`pilot/`、`formal/`：原始 JSON、逐点日志、history、配置、marker、rankings、runs 与阶段校验和。
- `analysis/final_metrics.csv`：formal 逐 seed 最终指标。
- `analysis/realtime_accuracy.csv`：服务器端实时 accuracy 数据。
- `analysis/realtime_velocity_rel_error.png`、`analysis/realtime_velocity_rel_error.pdf`：服务器端生成的曲线；本地未使用 Codex 内置图片生成。
- `run.log`、`search.log`、`pilot.log`、`formal.log`、`plot.log`、`matplotlib_install.log`：阶段与环境证据。

## 校验和说明

`analysis/SHA256SUMS`、`search/SHA256SUMS`、`pilot/SHA256SUMS` 和 `formal/SHA256SUMS` 在本地复核全部通过。根目录 `SHA256SUMS` 中除 `pipeline_status.json` 外的条目均通过；该文件的记录哈希为流水线收尾前的旧值，而当前 `pipeline_status.json` 内容为最终 `stage=complete` 状态，因此原始根清单出现一个可解释的陈旧条目。原始 `SHA256SUMS` 保持不改写；按当前文件生成的 `SHA256SUMS.rechecked` 已复核 `363/363` 条目通过，当前差异已记录在本清单，便于审计。
