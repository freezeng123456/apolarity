# 三个 active 主实验核查（`jsc_v2`，1200 秒）

本报告只核查当前 active 的三个主实验：Polyharmonic、Chirp、Maxwell。`auto`
以及 `experiments/archived/` 下的历史实验、边界权重搜索和单 seed 控制均不计入
正式证据；它们只用于回答“以前是否做过类似事情”。

## 先给结论

| 核查项 | 结论 |
|---|---|
| 是否做过“找一个合理权重”的历史搜索 | **三组都做过**。archive 对 Poly 的 d=2/3、order=2/4/6，Chirp 的 a=1/2/3，Maxwell 的 a=2/4/6 都能找到实际 grid/ranking/selected-weight 文件；但这不是 active `jsc_v2` 的 1200 秒、5-seed exhaustive sweep。|
| active formal 是否已经采用这些搜索结果 | **还没有**。当前 active `jsc_v2` 仍固定 `bc_weight=100`；Poly archive 多数是逐 boundary-component 向量，不能直接等价替换 active 的 scalar weight。Chirp/Maxwell 的历史 scalar weights 可以直接作为合理候选。|
| 每个主实验是否有 autodiff 基线 | **没有正式的**。active 的 240 条 canonical rows 全部是 `backend=jet`；三个 family 都没有纳入 active `jsc_v2` 的 direct-autodiff Vanilla 5-seed 对照。archive 中存在 Poly/Chirp/Maxwell 的单 seed direct-AD 控制，但不能替代正式基线。|
| 日志/数据打点是否完整 | **history 足够画实时曲线，但日志不完整**。active 有 240 条 history traces 和最终指标；没有随结果提交的独立 runner `.log`，history 也没有逐时 `L_bc`、学习率、显存或梯度范数。|

因此，如果问题是“有没有做过足以找合理权重的历史搜索”，答案是 **有，三组都有**；
如果问题是“active 正式 1200 秒、5-seed protocol 是否已经用统一规则重新确认过这些权重”，
答案仍是 **没有**。这两个结论需要区分。

## 1. boundary 参数核对

active `jsc_v2` 的固定 setting 是：

| 项目 | Polyharmonic | Chirp | Maxwell |
|---|---:|---:|---:|
| training budget | 1200 s | 1200 s | 1200 s |
| interior points | 4096 | 4096 | 4096 |
| boundary points | 512 | 512 | 512 |
| boundary weight | 100 | 100 | 100 |
| boundary sampling | `sample_boundary`：face 均匀、sign 均匀 | 同左 | Maxwell 自己的均匀 face sampler |
| boundary components | (u)，以及 Navier 的 (Delta^j u)（按 (S^j) 归一化） | Dirichlet (u) | Dirichlet (E) |

代码证据是 `experiments/common/osc_common.py` 中
`LinearProblem.bc_weight=100.0`、`loss=L_int + bc_weight*L_bc`，以及
`experiments/common/protocol.py` 中 `N_BOUNDARY=512`。Maxwell 的训练函数也直接写死
`loss = L_int + 100.0 * L_bc`。active CSV/JSON 只记录 `n_bc=512`，没有 boundary
weight 或 face 采样策略字段，说明这些量没有作为正式实验轴保存。

archive 中与“合理权重”直接相关的证据如下。

### Polyharmonic：三维度/阶数轴都有搜索记录

- `poly_shared_weight_tuning/`：d=2 的 order 2/4/6。order 2 做了 scalar grid；
  order 4 做了 fixed-sum component profiles；order 6 做了多组 profile、90 秒确认
  和 total-scale endpoint。
- `poly_shared_weight_power_grid/`：d=2 的 order 4/6 做了 (7^m) component grid；
  d=3 的 order 2/4/6 分别做了 7、49、343 个组合，grid 为
  `[0.001, 0.01, 0.1, 1, 10, 100, 1000]`，并保存了 ranking/selected weights。
- `poly_shared_weight_cartesian/`：d=2 order 4/6 又做了更窄的 6 点 Cartesian grid，
  order 6 还对 top-5 做了 90 秒 confirmation。
- `poly_power_grid_full20m/d3/`：d=3 order 2/4/6 使用 selected component weights
  做过 seed-0、1200 秒 Vanilla/direct-AD vs Complex-Sinh 控制。

典型 selected component weights 包括：

| setting | archived reasonable weights |
|---|---|
| Poly d2/o2 | `[0.3]`（scalar tuning） |
| Poly d2/o4 | `[0.02, 0.58]`；另有 power-grid `[0.1, 10.0]`、Cartesian `[0.3, 3.0]` |
| Poly d2/o6 | `[0.02, 0.1, 0.78]`；另有 power-grid `[0.01, 1.0, 10.0]` |
| Poly d3/o2 | `[0.1]` |
| Poly d3/o4 | `[0.1, 1.0]` |
| Poly d3/o6 | `[0.1, 0.1, 1.0]` |

### Chirp：a=1/2/3 都做过 scalar grid

`osc_shared_weight_grid/chirp_a{1,2,3}/ranking.json` 都存在，grid 是
`[0.001, 0.01, 0.1, 1, 10, 100, 1000]`，每个候选同时比较 Vanilla 和 Sinh，
按 geometric mean / max error 排序。随后 `osc_shared_weight_full20m/` 对三个 sweep
做了 seed-0、1200 秒控制，selected scalar weights 是：

```text
chirp_a1 = 1.0,  chirp_a2 = 0.1,  chirp_a3 = 0.01
```

### Maxwell：a=2/4/6 都做过 scalar grid

`osc_shared_weight_grid/maxwell_a{2,4,6}/ranking.json` 同样全部存在，使用同一个
7 点 scalar grid，并有 1200 秒 seed-0 control。selected scalar weights 是：

```text
maxwell_a2 = 0.1,  maxwell_a4 = 0.1,  maxwell_a6 = 0.01
```

需要保留一个历史差异说明：更早的 `pde_weight_tuning/manifest.json` 对代表性
`maxwell_a4` 给过 `0.03` 的 method-specific 候选；后来的 shared Vanilla/Sinh grid
选择了 `0.1`。这不是“没有搜过”，而是不同旧 protocol/目标函数得到的合理候选不同。

因此，按你现在的目标（找一个合理、可复用的权重，而不是声称全局最优），三组都可以
判定为 **已经搜过参**。剩余缺口是：这些历史 selected weights 还没有被统一映射进
active `jsc_v2`（尤其 Poly 的 component vector 与当前 scalar `bc_weight=100`
接口不一致），并且没有做 active 5-seed formal confirmation；`n_bc`、face 比例和
采样方式也没有搜索。

## 2. direct-autodiff 基线核对

active `jsc_v2` 的正式方法只有：

`complex_sinh`、`siren`、`fourier`（mFF-PINN）和 `mscale`（MscaleDNN-2-sin）。

对 active 的 12 个 task（Poly 6 + Chirp 3 + Maxwell 3）逐目录核验得到：

- 12 个 `VALIDATED` marker；
- 240 条 canonical rows = 12 tasks × 4 methods × 5 seeds；
- 所有 row 的 `backend` 都是 `jet`，没有 `vanilla_tanh_direct_ad` 或
  `backend=direct_autodiff`；
- `experiments/common/osc_common.py::deriv_alpha` 的 direct-AD 分支只给辅助
  `CauchyNet` 使用，不在 active `FORMAL_METHODS` 中。

archive 的 direct-AD 证据只能算“已有先导控制”：

| family | archive 中的 direct-AD 控制 | 是否等价于正式基线 |
|---|---|---|
| Polyharmonic | `poly_power_grid_full20m/d3` 的 `o2/o4/o6`，`vanilla_tanh_direct_ad`，seed 0，1200 s；另有 d=2 的 180 s 控制 | 否：只覆盖部分 setting，单 seed，未进入 active canonical bundle |
| Chirp | `osc_shared_weight_full20m/chirp_a1/a2/a3` 的 `vanilla`，seed 0，1200 s；其 archived runner 使用 direct autodiff | 否：单 seed、旧 weight protocol |
| Maxwell | `osc_shared_weight_full20m/maxwell_a2/a4/a6` 的 `vanilla`，seed 0，1200 s；其 archived runner 使用 direct autodiff | 否：单 seed、旧 weight protocol |

所以三个主实验逐一的答案都是：**active formal direct-AD baseline = 没有；archive
里有单 seed 控制 = 有**。

## 3. 日志、数据打点和结果完整性

### active 结果包完整性

每个 task 目录都有 `manifest.json`、canonical CSV/JSON、history JSON 和
`VALIDATED`。所有正式 rows 都满足：

- `budget_seconds=1200`、`n_int=4096`、`n_bc=512`；
- seed 为 0–4，四个正式 method 各 5 条；
- `lr=1e-3`、cosine schedule、`depth=4`、literal hidden width `H=128`；
- `nan=false`，最终 `L2_err`、`L_int_last`、`ms_per_step` 为有限值。

### history 能记录什么

每条 history 点是三元组：`[wall_seconds, heldout_relative_L2, L_int]`。因此可以
画实时 held-out relative (L^2) 曲线，而且是 paired fixed evaluation protocol
（`fixed_seed_12345_n8192_v1`）下的可比曲线。各 family 的 history 点数范围为：

| family | task 数 | traces | 单条 history 点数范围 |
|---|---:|---:|---:|
| Polyharmonic | 6 | 120 | 7–1023 |
| Chirp | 3 | 60 | 329–972 |
| Maxwell | 3 | 60 | 163–850 |

### 目前缺什么

active 结果目录中没有独立的 runner `.log`；仓库里可见的 `.log` 都属于 archive。
history 也没有逐时记录：

- `L_bc` 或各 boundary component loss；
- 当前 learning rate；
- GPU memory、梯度范数、step-level throughput；
- collocation 点或 face sampling 的统计。

最终 row 仍保留 `steps`、`ms_per_step`、`peak_mb`、`L_int_last`、`L2_err` 和
`nan`，所以最终性能和基本运行健康度可以复核，但不能从 active bundle 复原完整
的训练过程诊断。

另外，runner 的 1200 秒定义是**训练时间预算**，周期性/最终 evaluation 时间从
预算中扣除；单个 optimizer step 本身不能被截断，所以高阶 Poly 的最后一个 history
timestamp 可能略大于 1200 s（例如 `poly_d3_o6` 约到 1205 s）。图中保留原始
timestamp，并在 manifest 中注明这一点。

## 4. 现有 1200 秒结果的 final median（5 seeds）

以下数值来自 active canonical JSON，指标是 held-out relative (L^2)，越低越好；
不是 archive 结果，也没有混入 `auto`。

| task | Complex Sinh | SIREN | mFF-PINN | MscaleDNN-2-sin |
|---|---:|---:|---:|---:|
| Poly d2/o2 | 3.409e-4 | 1.000 | 3.795e-3 | 1.338e-2 |
| Poly d2/o4 | 7.844e-3 | 1.000 | 8.061e-1 | 5.630e-1 |
| Poly d2/o6 | 9.227e-2 | 1.000 | 1.002 | 9.379e-1 |
| Poly d3/o2 | 3.338e-3 | 1.000 | 4.735e-1 | 2.450e-1 |
| Poly d3/o4 | 1.503e-1 | 1.000 | 9.940e-1 | 9.336e-1 |
| Poly d3/o6 | 9.799e-1 | 1.000 | 1.003 | 1.061 |
| Chirp a1 | 5.858e-4 | 4.243e-1 | 2.096e-3 | 1.218e-2 |
| Chirp a2 | 2.637e-3 | 1.520 | 9.819e-1 | 7.669e-1 |
| Chirp a3 | 1.014e-1 | 1.060 | 1.403 | 8.892e-1 |
| Maxwell a2 | 3.470e-3 | 1.000 | 6.997e-1 | 1.086 |
| Maxwell a4 | 4.283e-2 | 1.000 | 1.123 | 1.291 |
| Maxwell a6 | 1.219e-1 | 1.001 | 1.156 | 1.346 |

描述性地看，Complex Sinh 在 Chirp/Maxwell 三个 sweep 和低/中阶 Poly 上明显下降；
Poly d3/o6 目前几乎没有收敛，不能用较低阶 Poly 的结果替代它。SIREN 在这套固定
1200 秒协议下多数 task 仍接近 1。现在可以说“历史上三组都做过合理权重搜索”，但
active 这批曲线仍使用固定 `bc_weight=100`，所以不能把当前曲线表述成“已经采用了
archive selected weights”，也不能声称相对 direct AD 的优势，因为 direct-AD 5-seed
baseline 尚未补齐。

## 5. 论文实时图

新图只读取 `experiments/results/jsc_v2`，默认选择三个 family 的中心 setting：

- Polyharmonic `d=2, order=4`；
- Chirp `a=2`；
- Maxwell `a=4`。

每个 panel 画四个 active formal method：实线是 5-seed raw median，阴影是 seed
IQR，点划/虚线是 median best-so-far，`×` 是最终 checkpoint median。纵轴明确写
成 held-out relative (L^2)（lower is better），避免把 error 误称成未经定义的
“accuracy”。

生成脚本：`experiments/tools/plot_jsc_realtime_accuracy.py`。

产物：

- `docs/paper/figures/jsc_realtime_accuracy.pdf`：论文向量版；
- `docs/paper/figures/jsc_realtime_accuracy.png`：300 dpi 预览；
- `docs/paper/figures/jsc_realtime_accuracy.csv`：插值后的曲线数据；
- `docs/paper/figures/jsc_realtime_accuracy_manifest.json`：来源、统计方式、task
  和每条曲线的 seed/点数/最终区间。

图和 CSV 都排除了 `auto` 与 `experiments/archived`。
