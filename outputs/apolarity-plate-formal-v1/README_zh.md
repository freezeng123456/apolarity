# 动态板四阶 / 应变梯度板六阶正式结果

本目录收录服务器端完成的两组固定协议正式实验。原始 JSON、逐步 history、逐点日志、配置、汇总、运行矩阵、完成标记和 SHA-256 清单均原样保留；JSON 中的逐点记录包含训练 history，因此没有另行改写或压缩 history。

## 实验协议

- 代码提交：`ebd94ba725e796268a522dea0e7fb2dd7fa9776b`
- WAR：`complex64 + sinh + Waring jet`
- 实数基线：`float32 + tanh + direct autodiff`
- 两方法共用 Xavier 初始化、hidden=128、depth=4、raw affine `(x,y,t)` 输入；没有三角输入，也没有频率匹配初始化。
- `n_int=2048`、`n_ic=512`、`n_bc=1024`、`n_eval=16384`、`history_eval_n=2048`
- 单张 H20 串行；每个 cell 训练 1200 秒。

## 完成情况

| task | 固定权重 | seed 数 | 方法 cell 数 | 状态 | 失败 |
|---|---:|---:|---:|---|---:|
| `dynamic_plate_2d_o4` | `(lambda_ic, lambda_bc)=(0.1, 1.0)` | 5 | 10 | `FORMAL_COMPLETE` | 0 |
| `strain_gradient_plate_2d_o6` | `(lambda_ic, lambda_bc)=(10.0, 10.0)` | 5 | 10 | `FORMAL_COMPLETE` | 0 |

## 正式结果摘要

下表是从随附的 `runs.csv`/`paired.csv` 重新计算的 seed 级均值 ± 样本标准差；完整逐 seed 数值以 CSV、JSON 和日志为准。

| task / method | rel_error | loss | ms/step | peak memory |
|---|---:|---:|---:|---:|
| dynamic o4 / WAR | `0.001674 ± 0.000533` | `7.632e-7 ± 3.726e-7` | `28.84 ± 0.61` | `599.8 MB` |
| dynamic o4 / real AD | `0.011657 ± 0.002406` | `9.064e-6 ± 1.426e-6` | `50.56 ± 2.57` | `738.7 MB` |
| strain-gradient o6 / WAR | `0.005523 ± 0.002794` | `5.590e-5 ± 5.102e-5` | `69.89 ± 0.21` | `1972.5 MB` |
| strain-gradient o6 / real AD | `0.311209 ± 0.061776` | `3.045e-2 ± 1.125e-2` | `396.51 ± 2.43` | `4984.7 MB` |

WAR 在动态板四阶的 5/5 个 seed、应变梯度板六阶的 5/5 个 seed 上均取得较低的最终相对误差。六阶任务补齐 seed 3、4 后，WAR 仍在 5/5 个 seed 上胜过实数 AD；逐 seed 的胜负、几何平均误差和最大误差见各 task 目录下的 `paired.csv` 与 `summary.json`。

## 五 seed 结论

在这两个固定权重、固定 1200 秒预算的正式实验中，WAR 的结论是一致的：动态板四阶和应变梯度板六阶均为 5/5 seed 胜出。按中位数比较，WAR 的 `rel_error` 分别为 `1.597e-3` 和 `4.956e-3`，实数 AD 分别为 `1.219e-2` 和 `3.031e-1`；因此六阶任务中的差距尤其明显。WAR 的单步时间约为实数 AD 的 0.57 倍（动态四阶）和 0.18 倍（六阶），峰值显存也更低。这个结论支持“在相同网络、采样、初始化、输入表示和时间预算下，WAR 在这两个高阶板 PDE 上更稳定、更快”的实验性表述，但仍应明确它对应当前两个已选定权重，不等同于对所有权重的全局最优性结论。

## 完整性复核

- 两个 task 的远端 `SHA256SUMS` 均在本地逐项通过。
- `dynamic_plate_2d_o4`：10/10 cell 完成，22 个 JSON，10 个方法日志。
- `strain_gradient_plate_2d_o6`：10/10 cell 完成，22 个 JSON，10 个方法日志。
- 所有 JSON 数值递归检查为有限值；每个方法日志末行同时包含 `loss` 和 `rel_error`。
- `smoke/SMOKE_CONCLUSION.json` 保留服务器端 4/4 smoke 通过结论；smoke 原始小文件不参与正式统计。
- 服务器归档及其哈希记录见 `provenance/REMOTE_ARCHIVES.md`；对应归档副本也保留在 `provenance/`，便于复核原始目录结构。

## 目录说明

- `dynamic_plate_2d_o4/`：四阶动态板完整正式结果。
- `strain_gradient_plate_2d_o6/`：六阶应变梯度板完整正式结果。
- `smoke/`：正式实验前的服务器 smoke 结论和主日志。
- `provenance/`：服务器生成归档、远端/本地 SHA-256 记录。
