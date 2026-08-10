# 高阶 PDE 候选筛选与二维 ZK 正式实验

## 验收

- Pilot：`passed`，24/24 cell，744 个 history 点。
- Formal：`passed`，10/10 cell，610 个 history 点。
- 原始校验：pilot 92 项、formal 41 项全部通过。
- 两阶段均为 0 失败、0 重试；正式统计没有混入 600 秒 pilot。
- 固定协议：共同 Xavier，hidden=128，depth=4，原始仿射坐标，无三角输入和频率初始化。
- WAR 使用 complex64+sinh+Waring jet；基线使用 float32+tanh+direct autodiff。

## Pilot（3 seeds × 600 秒）

| 候选 | WAR median rel_error | Real AD median rel_error | WAR 胜出 | 通过门槛 |
|---|---:|---:|---:|:---:|
| ZK-2D (order 3) | 0.0417496 | 0.0512001 | 2/3 | 是 |
| ZK-3D (order 3) | 1.00001 | 0.999997 | 1/3 | 否 |
| Dynamic plate (order 4) | 0.00351619 | 0.0395286 | 3/3 | 是 |
| Swift–Hohenberg (order 4) | 0.999999 | 1.00082 | 2/3 | 否 |

二维 ZK 与动态板通过门槛。动态板的筛选误差更低，但按预先冻结的选择规则，优先选择与现有 Polyharmonic/Cahn–Hilliard 不同的可训练 ZK，因此正式实验选择 `zk_2d_o3`。三维 ZK 与 Swift–Hohenberg 在当前公平协议下均退化到相对误差约 1。

## 二维三阶 ZK 正式结果（5 seeds × 1200 秒）

| 方法 | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| WAR (complex64, sinh) | 0.0150048 | 0.00343113 | 0.0162814 | 0.0093166 | 0.0180882 |
| Real AD (float32, tanh) | 0.0292899 | 0.00368806 | 0.0301802 | 0.0248771 | 0.0338619 |

WAR 在 5/5 个配对 seed 上误差更低；Real AD/WAR 的平均误差比为 1.952×，配对比的几何均值为 1.989×。
5/5 同向的精确 sign test：单侧 p=0.03125，双侧 p=0.06250。样本数只有 5，因此这是强而一致的描述性证据，不应写成双侧 5% 水平显著。

## 曲线与口径

- `formal_realtime_history.csv`：所有原始逐点 history。
- `formal_realtime_curves.csv`：在 0–1200 秒的 20 秒公共网格上，对正值指标作 log-linear 插值，再计算五个 seed 的中位数与四分位带。
- `realtime_rel_error`：固定 2048 点 history 评估集上的实时相对误差。
- `final_rel_error_by_seed`：固定 16384 点最终评估集上的配对结果。

## 公平性边界

两种方法具有相同的字面层形状、墙钟预算、训练点和初始化类型，但激活函数按已冻结协议分别为 sinh 与 tanh；complex64 WAR 的复参数也对应约两倍实标量自由度。结论应表述为该固定方法实现与网络形状下的墙钟效率/精度比较，而不是严格等激活或等实参数量比较。
