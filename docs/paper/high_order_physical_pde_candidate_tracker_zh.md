# 高阶物理 PDE 候选实验跟踪表

最后更新：2026-08-12

## 1. 文档目的

本文件跟踪下一阶段高阶物理 PDE 候选实验，目标是逐个判断 WAR 相对实数
direct-autodiff 基线在哪些问题上具有实际收益。候选方程必须同时满足：

1. 最高**空间**微分阶数不低于 4；
2. 有明确的物理建模背景；
3. 在实际采用的维数、初边值条件和时间区间下，有可引用的存在唯一性或适定性结果；
4. 能以直接高阶残差构造 PINN，不通过辅助变量把高阶方程降成多个二阶方程；
5. 能得到解析解、制造解或经过网格收敛验证的高精度参考解；
6. 能在现有 WAR/实数 AD 公平协议下记录实时 relative error、loss 和物理诊断量。

本文件只管理新候选。已经完成的 Polyharmonic、二维 Cahn--Hilliard、二维动态板和
六阶应变梯度板继续作为已覆盖参照，不在这里重复排队。Swift--Hohenberg 已做过
先导筛选且没有通过当前训练可行性门槛，也不重新排队。

## 2. 固定比较协议

除非后续在实验开始前明确修改并记录版本，所有候选统一采用以下协议。

| 项目 | WAR | 实数 autodiff 基线 |
|---|---|---|
| 数值精度 | `complex64` | `float32` |
| 激活函数 | `sinh` | `tanh` |
| 高阶导数 | Waring jet | direct autodiff |
| 初始化 | 共同 Xavier 类初始化 | 共同 Xavier 类初始化 |
| 网络 | hidden 128、4 个可训练隐藏层 | hidden 128、4 个可训练隐藏层 |
| 输入 | 仿射归一化的原始物理坐标 | 同左 |
| 禁止项 | 无三角输入、无周期嵌入、无频率匹配初始化 | 同左 |
| 训练预算 | 同一 wall-clock | 同左 |
| 采样与评估 | paired train seed、固定独立 eval set | 同左 |

补充约束：

- 周期问题使用相对边界上的显式 trace matching，不向网络输入添加
  `sin`/`cos` 特征。
- 方程以直接高阶残差实现。若为了参考求解器引入辅助变量，辅助变量只允许出现在
  reference solver，不能进入 WAR/AD 主比较网络。
- 多个初值或边界子项先按预注册的固定尺度归一化，再分别聚合成
  `L_ic` 和 `L_bc`；正式搜参仍只搜索聚合权重。
- 两种方法必须使用同一个 loss-weight 向量。共享权重按 shared minimax 或
  shared geometric mean 选择，不能给两种方法分别挑最优权重。
- smoke 原始 bundle 不进入论文结果；通过后删除 raw smoke，只在
  `docs/archive/SMOKE_CONCLUSIONS_zh.md` 保留结论。
- 所有图只能在实验服务器或 T4 环境由可审计脚本生成，不使用 Codex 内置图片生成。

## 3. 状态定义

| 状态 | 含义 |
|---|---|
| `RESEARCH` | 方程和文献已进入候选，但唯一性、边界条件或 reference 方案仍需核对 |
| `SPEC_READY` | 方程、维数、参数、初边值条件、网络和数据规模已冻结 |
| `IMPLEMENTED` | WAR、AD、reference、日志和恢复逻辑均已实现 |
| `SMOKE_PASSED` | 公式、导数、反向传播、CUDA、有限性和数据管线门禁通过 |
| `SEARCHED` | 完整共享权重搜索完成并验证 ranking 完整 |
| `PILOT_PASSED` | 3-seed、每方法 600 秒 pilot 通过进入正式实验的门槛 |
| `FORMAL_COMPLETE` | 5-seed、每方法 1200 秒正式实验完成并通过完整性审核 |
| `HOLD` | 候选有效，但优先级较低，等待前置候选结论 |
| `REJECTED` | 数学、训练、结果完整性或科学价值门槛未通过，不继续消耗 GPU |
| `TRAINING_FAILURE` | 数学/实现门禁通过，但完整权重搜索仍收敛到不可用假解；作为负结果保留，不再长跑 |

## 4. 总队列

队列顺序优先考虑：科学增量、对高阶微分方式的区分度、实现风险以及获取可靠
ground truth 的成本。

| 顺序 | ID | 候选 | 维数 | 最高空间阶数 | 当前状态 | 角色 | 是否允许正式长跑 |
|---:|---|---|---:|---:|---|---|---|
| — | `HO-01` | MBE slope-selection | 2D | 4 | `REJECTED / TRAINING_FAILURE` | 完整搜参负结果；49 个权重均为近零假解 | 否；停止 pilot/formal |
| 1 | `HO-04` | Hyperviscous Navier--Stokes | 2D | 4 | `IMPLEMENTED` | 当前第一优先级；跨物理体系、多输出 | 是；仅按本节预注册门禁自动放行 |
| 2 | `HO-02` | Modified Phase-Field Crystal | 2D | 6 | `RESEARCH` | 六阶主候选 | 否，等待 `HO-04` 结论 |
| 3 | `HO-03` | Kawahara / KdV--Kawahara | 1D | 5 | `RESEARCH` | 奇数阶诊断候选 | 否，先完成低成本 pilot |
| 4 | `HO-05` | Hyperviscous Navier--Stokes ABC flow | 3D | 4 | `HOLD` | 三维扩展 | 否，等待 `HO-04` |
| 5 | `HO-06` | generalized sixth-order Boussinesq | 1D | 6 | `RESEARCH` | 短时间高风险候选 | 否，只允许短时间 pilot |
| 6 | `HO-07` | Kuramoto--Sivashinsky | 1D | 4 | `HOLD` | 常见物理对照 | 否，等待前置候选结论 |
| 7 | `HO-08` | Functionalized Cahn--Hilliard | 2D | 6 | `HOLD` | 奇异势储备候选 | 否，等待 MPFC 结论 |

默认执行原则是：前一候选至少完成共享权重搜索和 3-seed pilot，并形成继续或停止的
书面结论后，再决定是否启动下一候选的正式阶段。2026-08-12 用户已对 `HO-04`
作一次性例外授权：两档 smoke、三点 sentinel、完整搜参、3-seed pilot 和 5-seed
formal 可以按本文件预注册门禁自动串行；任一门禁失败必须保留证据并停止，不能临时
改参、降规模或换精度。该授权不延伸到 `HO-02` 及后续候选。

## 5. 统一实验阶段与门槛

### 5.1 数学与公式门禁

每个候选首先完成：

- [ ] 方程、物理量、无量纲参数和最高空间阶数核对；
- [ ] 唯一性文献与实际维数、区域、边界条件、初值空间、时间区间逐项对应；
- [ ] 明确解析解、制造解或 reference solver 方案；
- [ ] 解析/符号 residual 与独立直接求导实现交叉校验；
- [ ] WAR 与 direct AD 的输入导数及参数梯度在小样本上交叉校验；
- [ ] 记录所有初值和边界分量，不允许只记录聚合 total loss。

如果文献只证明局部适定性，训练时间区间必须落在明确的短时间 setting 内，并在结果
说明中标记为 local-well-posed benchmark。

### 5.2 CUDA smoke

每种 task × method 运行 3 秒基础 smoke，再运行 3 秒正式采样规模 smoke。必须满足：

- [ ] 无 OOM；
- [ ] loss、relative error、全部 component loss 和梯度有限；
- [ ] history 单调记录 wall time；
- [ ] 日志末行同时含 final loss 与 final relative error；
- [ ] manifest 记录代码 SHA、dtype、网络结构、方程参数、采样规模和 eval seed；
- [ ] smoke 结束后 GPU 无残留 worker。

如果正式采样规模 smoke 不通过，保留失败证据并停止；不能自动缩小 batch、切换精度、
降低导数阶数或更换方程参数。

### 5.3 共享权重搜索

默认搜索：

\[
\lambda_{\mathrm{ic}},\lambda_{\mathrm{bc}}
\in
\{10^{-3},10^{-2},10^{-1},1,10,100,1000\}.
\]

对于只有单一初值或边界权重的任务，删除不存在的搜索轴。每候选、每方法、每权重向量
运行 60 秒，train seed 固定为 42，eval seed 固定并独立于 train seed。

搜索完成标准：

- [ ] 每个权重向量都有 WAR 与 AD 配对结果；
- [ ] JSON、日志和 history 原子写入且指标有限；
- [ ] WAR、AD、shared geomean、shared minimax 四类 ranking 完整；
- [ ] 选择的是共同权重，不采用 method-specific 最优权重；
- [ ] 给出 Top 10、稳定平台、方法冲突和推荐候选；
- [ ] 用户确认固定权重后才能进入 pilot。

若有两个权重轴，单个 task 的纯训练预算为：

\[
49\times 2\times60\text{ s}=5880\text{ s}\approx1.63\text{ h}.
\]

### 5.4 3-seed pilot

采用共享固定权重，运行 3 seeds × 2 methods × 600 秒。建议进入正式阶段的最低条件：

- [ ] 6/6 cells 完整，无 NaN/OOM；
- [ ] 至少一种方法的 median relative error 小于 0.2；
- [ ] 另一种方法的 median relative error 小于 0.75；
- [ ] 没有单个 seed 明显退化到恒等于零或相对误差约 1 的假解；
- [ ] 质量、能量、散度或其他物理诊断没有明显失真；
- [ ] 实时曲线显示训练仍有可解释的下降，而不是只比较偶然 final checkpoint；
- [ ] 根据 pilot 写出 `GO_FORMAL` 或 `STOP` 结论。

单个 task 的 pilot 纯训练预算为 1 小时。

### 5.5 5-seed 正式实验

正式协议固定为 5 seeds × 2 methods × 1200 秒，单个 task 纯训练预算约 3.33 小时。
必须交付：

- [ ] 10/10 cells；
- [ ] 每 cell 的原始 JSON、文本日志和逐点 history；
- [ ] 固定配置、manifest、环境信息、恢复 attempts、completion marker；
- [ ] SHA256 校验和及下载后复核；
- [ ] 5-seed mean、sample std、median、IQR；
- [ ] paired seed 的 error ratio 和胜负；
- [ ] 实时 relative-error 曲线及其绘图数据；
- [ ] total/component loss 曲线；
- [ ] steps、ms/step、time-to-accuracy 和 peak memory；
- [ ] 方程特有的质量、能量、散度或约束误差；
- [ ] 结果提交 GitHub 后的 commit/PR 记录。

## 6. “收益显著”的预注册判据

主比较使用同一 wall-clock、同一共享 loss 权重下的 5-seed median relative error：

\[
G_{\mathrm{error}}
=
\frac{\operatorname{median}(e_{\mathrm{AD}})}
     {\operatorname{median}(e_{\mathrm{WAR}})}.
\]

同时记录对数收益：

\[
S_{\mathrm{error}}=\log_{10}G_{\mathrm{error}}.
\]

解释规则：

| 结果 | 实际解释 |
|---|---|
| `G_error >= 2` 且 WAR 至少赢 4/5 paired seeds | 强收益候选 |
| `1.2 <= G_error < 2` 或 WAR 赢 3/5 seeds | 中等收益，需结合实时曲线和效率判断 |
| `0.8 < G_error < 1.2` | 基本持平 |
| `G_error <= 0.8` | WAR 在该问题上没有准确率收益 |

辅助指标：

- `G_TTA(tau) = TTA_AD(tau) / TTA_WAR(tau)`：达到同一误差阈值的时间比；
- `G_step = ms_per_step_AD / ms_per_step_WAR`：单步速度比；
- `G_mem = peak_MB_AD / peak_MB_WAR`：峰值显存比；
- 训练步数、失败率、seed 间离散程度和物理约束误差。

误差阈值 `tau` 只能从 `{0.5, 0.2, 0.1, 0.05, 0.02}` 中选择双方至少有一半 seed
能够达到的最小公共阈值。不得在看到单个方法曲线后单独挑选对它有利的阈值。

5 seeds 不足以支撑夸大的经典显著性检验结论，因此论文中以 paired effect size、
逐 seed 胜负和分布统计为主；“强收益”表示预注册的实际效果门槛，而不是声称
`p < 0.05`。

## 7. 候选实验卡

### HO-01：二维四阶 MBE slope-selection

方程：

\[
\partial_t h
=
\nabla\cdot\left[(|\nabla h|^2-1)\nabla h\right]
-\nu\Delta^2h.
\]

物理背景：分子束外延薄膜表面生长、Ehrlich--Schwoebel 效应和坡度选择。

唯一性依据：

- [Gradient bounds for a thin film epitaxy equation](https://arxiv.org/abs/1410.7572)
- [Well-posedness for a molecular beam epitaxy model](https://arxiv.org/abs/2311.16970)

建议首版 setting：

| 项目 | 固定值 |
|---|---|
| 区域 | `[0, 2pi]^2 x [0, 1]` |
| 参数 | `nu=0.05` |
| 初值 | `0.2 cos(x)cos(y) + 0.1 cos(2x)cos(y)` |
| 边界 | 显式周期匹配，空间法向导数 0--3 阶 |
| 输出 | 标量高度 `h` |
| 正式 ground truth | 无外力方程的收敛谱方法 reference |
| smoke ground truth | 同一初值模式的制造解与解析源项 |
| 特有指标 | 质量漂移、自由能、坡度统计、能量平衡 |

Reference solver 必须用至少三档空间/时间分辨率做收敛检查；制造解只用于公式和
smoke，不进入正式主表。

2026-08-12 门禁记录：ETDRK4 三档 `(N,dt)` 固定为
`(32,2e-3)/(64,1e-3)/(128,5e-4)`，32→64 与 64→128 的固定评估集相对差
分别为 `5.619516e-04` 与 `8.021451e-07`；质量漂移低于 `2e-18`，自由能
从 `0.2368129` 单调下降到 `0.1437878`。基础与正式搜参采样规模 CUDA
smoke 均为 2/2 cells 通过。

同日完整共享权重搜索已完成：49/49 个候选、98/98 个 60 秒 method cells，
无失败、无重试，四类 ranking 均为 49 行，服务器与下载后 264 项原始 SHA256
校验全部通过。WAR 最佳 relative error 为 `0.999881506`，实数 AD 最佳为
`0.996811509`；shared minimax 第一名为 `point_021=(1,1e-3)`，对应
WAR/AD 为 `0.999957383/0.999934435`。所有候选均未达到 `0.75`，预测的终态
slope RMS 也远低于谱 reference，说明两种网络都停在近零假解。HO-01 因此转为
`SEARCHED / HOLD`；`point_021` 只能作为用户若批准更长时间诊断 pilot 时的注册
共享权重，不能解释为已经找到有效权重。未经用户再次确认，不启动 3-seed pilot
或 5-seed formal。完整报告见
`outputs/apolarity-mbe-ho01-weight-search-v1/analysis/SEARCH_REPORT_zh.md`。

任务：

- [x] 核对周期适定性定理与实际参数；
- [x] 冻结 reference solver 分辨率和误差容限；
- [x] 实现直接四阶 WAR/AD residual；
- [x] 完成解析制造解交叉校验；
- [x] 完成两档 CUDA smoke；
- [x] 完成共享权重搜索；
- [x] 用户决定及时止损，不做诊断 pilot；
- [x] 写出 `STOP / TRAINING_FAILURE` 结论；
- [x] 保留完整 search 原始证据和报告；
- [ ] 不再启动 3-seed pilot 或 5-seed formal。

结果占位：

| 字段 | 值 |
|---|---|
| selected `(lambda_ic, lambda_bc)` | 不选择正式权重；`(1,1e-3)` 仅是失败搜索中的 shared-minimax 点 |
| pilot median WAR / AD | 不运行 |
| formal median WAR / AD | 不运行 |
| `G_error` | 不适用 |
| WAR paired wins | 不适用 |
| 结论 | `REJECTED / TRAINING_FAILURE`：49 个权重均为 `rel_error≈1` 的近零假解 |

### HO-02：二维六阶 Modified Phase-Field Crystal

方程：

\[
\beta\phi_{tt}+\phi_t=M\Delta\mu,
\qquad
\mu=\Delta^2\phi+2\Delta\phi+(1-\varepsilon)\phi+\phi^3.
\]

直接 residual 中必须显式保留 `Delta^3(phi)`，不能以独立 `mu` 网络降阶。

物理背景：原子长度尺度与扩散时间尺度上的晶体生长、缺陷演化，以及快速弹性弛豫和
慢速扩散的区分。

唯一性与物理依据：

- [Well-posedness and longtime behavior for the MPFC equation](https://arxiv.org/abs/1306.5857)
- [Phase-field-crystal models: an overview](https://arxiv.org/abs/1207.0257)

建议首版 setting：

| 项目 | 固定值 |
|---|---|
| 区域 | `[0, 2pi]^2 x [0, 1]` |
| 参数 | `M=1, beta=0.1, epsilon=0.25` |
| 初值 `phi` | `0.1 + 0.15 cos(x)cos(y) + 0.05 cos(2x)cos(y)` |
| 初值 `phi_t` | `0` |
| 边界 | 显式周期匹配，空间法向导数 0--5 阶 |
| 输出 | 标量原子密度相场 `phi` |
| 正式 ground truth | 无外力方程的收敛谱方法 reference |
| 特有指标 | 平均密度、pseudo-energy、质量演化、幅值范围 |

任务：

- [ ] 核对周期边界唯一性定理和能量空间；
- [ ] 推导并测试展开后的直接六阶 residual；
- [ ] 冻结 MPFC reference solver 和 convergence test；
- [ ] 完成制造解与初始速度交叉校验；
- [ ] 完成两档 CUDA smoke；
- [ ] 完成共享权重搜索；
- [ ] 用户确认固定权重；
- [ ] 完成 3-seed pilot；
- [ ] 写出 `GO_FORMAL`/`STOP` 结论；
- [ ] 完成 5-seed formal；
- [ ] 在服务器/T4 生成实时曲线；
- [ ] 提交 GitHub 并填写总结果表。

结果占位：

| 字段 | 值 |
|---|---|
| selected `(lambda_ic, lambda_bc)` | `TBD` |
| pilot median WAR / AD | `TBD / TBD` |
| formal median WAR / AD | `TBD / TBD` |
| `G_error` | `TBD` |
| WAR paired wins | `TBD / 5` |
| 结论 | `TBD` |

### HO-03：一维五阶 Kawahara / KdV--Kawahara

代表形式：

\[
u_t+\alpha uu_x+\beta u_{xxx}+\gamma u_{xxxxx}=0.
\]

物理背景：高阶色散波和浅水波；主要实验价值是检查 WAR 的收益是否能够从偶数阶
Laplacian 推广到奇数五阶导数。

依据：

- [Global well-posedness for the Kawahara equation](https://www.aimsciences.org/article/doi/10.3934/cpaa.2013.12.1321)
- [PINNs for nonlinear dispersive PDEs](https://arxiv.org/abs/2104.05584)

首版需要在实现前从论文实例中冻结具体系数、精确/参考解和边界 trace；不能仅凭方程
名称自行选择一个有利于某种方法的孤波参数。

任务：

- [ ] 从适定性和 PINN 文献中冻结一个共同的五阶 setting；
- [ ] 明确全空间截断或周期边界与唯一性定理的对应；
- [ ] 实现直接五阶 residual；
- [ ] 完成导数奇偶性和符号检查；
- [ ] 完成 CUDA smoke；
- [ ] 运行 3-seed × 600 秒低成本 pilot；
- [ ] pilot 有训练信号后再决定是否完整搜参；
- [ ] 通过后完成共享搜参和 5-seed formal；
- [ ] 填写总结果表。

结果占位：

| 字段 | 值 |
|---|---|
| frozen setting | `TBD` |
| selected loss weights | `TBD` |
| formal median WAR / AD | `TBD / TBD` |
| `G_error` | `TBD` |
| 结论 | `TBD` |

### HO-04：二维四阶 Hyperviscous Navier--Stokes

方程：

\[
\mathbf u_t+(\mathbf u\cdot\nabla)\mathbf u+\nabla p
-\nu\Delta\mathbf u+\eta\Delta^2\mathbf u=0,
\qquad \nabla\cdot\mathbf u=0.
\]

物理背景：超黏性流体、湍流数值模拟和气象中的高阶耗散。

依据：

- [Partial and full hyper-viscosity for Navier--Stokes](https://arxiv.org/abs/1809.03954)
- [Error estimates for PINNs approximating Navier--Stokes](https://arxiv.org/abs/2203.09346)

建议首版 setting：

| 项目 | 固定值 |
|---|---|
| 区域 | `[0, 2pi]^2 x [0, 1]` |
| 参数 | `nu=0.05, eta=0.01` |
| 解 | 超黏性 Taylor--Green vortex |
| 输出 | `u, v, p` |
| 边界 | 速度显式周期匹配至 3 阶；压力周期与零均值 gauge |
| ground truth | 无外力解析解 |
| 特有指标 | divergence、动能衰减、压力 gauge、分量误差 |

2026-08-12 冻结的解析 setting 为：

\[
A(t)=\exp[-(2\nu+4\eta)t]=\exp(-0.14t),
\]
\[
u=A\sin x\cos y,\qquad
v=-A\cos x\sin y,
\]
\[
p=\frac{A^2}{4}(\cos 2x+\cos 2y).
\]

它满足 `Delta(u,v)=-2(u,v)`、`Delta^2(u,v)=4(u,v)`；对流项恰由上述
压力梯度抵消，线性衰减率恰为 `2 nu + 4 eta`。压力的空间均值为零，因此解析解同时
固定了 pressure gauge。主误差定义为 `(u,v)` 联合 relative L2；压力误差不混入主
指标，但必须单独报告。

损失冻结为：

\[
L=L_{\rm momentum}+L_{\rm div}
+\lambda_{\rm ic}L_{\rm ic}
+\lambda_{\rm bc}L_{\rm bc}
+L_{\rm gauge}.
\]

其中两个动量分量等权；`L_div` 与 `L_gauge` 固定权重为 1；`L_ic` 只约束初始
速度；`L_bc` 等权聚合速度的 0--3 阶周期 trace 与压力 0 阶周期 trace。只搜索
`lambda_ic,lambda_bc`，不把不可压约束和 gauge 变成额外可调自由度。

自动流程冻结如下：

1. CPU 公式/多输出导数/参数梯度测试；
2. 每方法 3 秒基础 CUDA smoke；
3. 正式采样规模 `2048/512/1024/16384/2048` 的每方法 3 秒 smoke；
4. `(1,1)/(10,1)/(10,10)` 三个共享权重、每方法 180 秒 sentinel；
5. sentinel 至少一组满足“一法 `<0.2`、另一法 `<0.75`”及物理门禁后，运行
   完整 `7x7`、每方法 60 秒共享搜参；
6. 按 shared-minimax 第一名（`max_error<1.25`）固定共同权重；
7. 3 seeds × 2 methods × 600 秒 pilot；仅当 6/6 完整、误差门槛通过、每个 seed
   `<0.95`，且两方法 median 均满足 divergence `<0.25`、pressure mean `<0.25`、
   energy error `<0.5`、pressure error `<1.5` 时进入 formal；
8. 5 seeds × 2 methods × 1200 秒 formal，并在服务器环境生成曲线和 CSV。

该候选必须先证明多输出、压力和不可压约束不会使 smoke 显存或速度不可接受。结果分析
要区分“高阶导数收益”和“多输出优化差异”，不能只给一个聚合 relative error。

任务：

- [x] 写出并独立核对超黏性 Taylor--Green 解析解；
- [x] 核对速度、压力、衰减率和 pressure gauge；
- [x] 实现直接四阶动量残差和 divergence residual；
- [x] 扩展 WAR/direct-AD 后端支持逐通道多输出高阶导数；
- [ ] 完成多输出 CUDA smoke 与显存审核；
- [ ] 完成三点 sentinel；
- [ ] 完成共享权重搜索；
- [ ] 完成 3-seed pilot；
- [ ] 通过后完成 5-seed formal；
- [ ] 分别报告 velocity、pressure、divergence 和 efficiency；
- [ ] 填写总结果表。

### HO-05：三维四阶 Hyperviscous Navier--Stokes ABC flow

这是 `HO-04` 的条件扩展。只在二维版本满足以下条件后启动：

- [ ] 两种方法在二维都能达到可用误差；
- [ ] 压力和 divergence 约束稳定；
- [ ] 2D 单 cell 峰值显存为 3D 留出安全余量；
- [ ] 2D 结果显示该方程具有区分度，而不是两方法同时失败或完全饱和。

计划使用周期 ABC/Beltrami 流及其超黏性解析衰减。3D setting、采样规模和 evaluation
规模必须单独做资源预算，不能直接照搬 2D 数量。

### HO-06：一维广义六阶 Boussinesq

方程：

\[
u_{tt}-u_{xx}-u_{xxxx}-u_{xxxxxx}
-(u^2)_{xx}-(u^2)_{xxxx}-(uu_{xx})_{xx}-(u^3)_{xx}=0.
\]

物理背景：更高阶、长时间精度的单向浅水波模型。现有依据只给出局部适定性，因此首轮
只能使用短时间区间，不能直接进入长时间正式实验。

依据：

- [Local well-posedness for a generalized sixth-order Boussinesq equation](https://arxiv.org/abs/2403.04295)

任务：

- [ ] 从适定性结果确定允许的初值空间和短时间设置；
- [ ] 冻结解析/数值 reference 及截断边界；
- [ ] 实现直接六阶 residual 和双初值约束；
- [ ] 完成 smoke；
- [ ] 仅运行 3-seed × 600 秒短时间 pilot；
- [ ] pilot 后重新审核是否值得搜参；
- [ ] 未经再次确认不得进入 1200 秒 formal。

### HO-07：一维 Kuramoto--Sivashinsky

方程：

\[
u_t+uu_x+u_{xx}+u_{xxxx}=0.
\]

物理背景：反应扩散、火焰前沿和黏性流。它有唯一光滑解结果，但属于常见的一维四阶
PINN 问题，论文增量预计低于 MBE、MPFC 和超黏性流体。

依据：

- [The Well-Posedness of the Kuramoto--Sivashinsky Equation](https://epubs.siam.org/doi/10.1137/0517063)

任务：

- [ ] 等待 `HO-01`--`HO-06` 的收益图谱；
- [ ] 若需要一个标准化公共 benchmark，再冻结短时间非混沌 setting；
- [ ] 只在 reference、周期 trace 和初值与文献完全匹配后 smoke；
- [ ] 根据低成本 pilot 决定是否继续。

### HO-08：二维六阶 Functionalized Cahn--Hilliard

物理背景：两亲分子混合物、各向异性晶体和外延生长。该类六阶方程存在全局弱解
唯一性结果，但对数奇异势会显著增加实现和训练风险，并且与现有 CH4/CH6 重合较多。

依据：

- [Sixth-order Cahn--Hilliard equations with logarithmic potential](https://arxiv.org/abs/1909.01816)

任务：

- [ ] 等待 `HO-02` MPFC 结论；
- [ ] 只有当 MPFC 显示六阶材料模型收益显著时，才继续核对具体 FCH 形式；
- [ ] 冻结保证相场留在物理区间的参数和初值；
- [ ] 单独审核对数势、有限性和 reference solver；
- [ ] smoke 未通过时不得通过裁剪相场或修改势函数掩盖失败。

## 8. 总结果表

每个候选只有在同一固定协议下形成完整 5-seed 正式结果后才能填写 formal 列。Pilot
结果不得混入 formal 聚合。

| ID | task | order | selected weights | WAR median | AD median | `G_error` | WAR wins | `G_TTA` | `G_step` | peak MB WAR/AD | 物理诊断 | 最终决定 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `HO-01` | MBE 2D | 4 | 不进入正式实验 | `N/A` | `N/A` | `N/A` | `N/A` | `N/A` | `1.592`（60 秒搜索） | `957.74/965.29` | 近零坡度、能量约 0.25 | `TRAINING_FAILURE / STOP` |
| `HO-02` | MPFC 2D | 6 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD/5` | `TBD` | `TBD` | `TBD/TBD` | `TBD` | `TBD` |
| `HO-03` | Kawahara 1D | 5 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD/5` | `TBD` | `TBD` | `TBD/TBD` | `TBD` | `TBD` |
| `HO-04` | hyper-NS 2D | 4 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD/5` | `TBD` | `TBD` | `TBD/TBD` | `TBD` | `TBD` |
| `HO-05` | hyper-NS 3D | 4 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD/5` | `TBD` | `TBD` | `TBD/TBD` | `TBD` | `TBD` |
| `HO-06` | gSOBE 1D | 6 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD/5` | `TBD` | `TBD` | `TBD/TBD` | `TBD` | `TBD` |
| `HO-07` | KS 1D | 4 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD/5` | `TBD` | `TBD` | `TBD/TBD` | `TBD` | `TBD` |
| `HO-08` | FCH 2D | 6 | `TBD` | `TBD` | `TBD` | `TBD` | `TBD/5` | `TBD` | `TBD` | `TBD/TBD` | `TBD` | `TBD` |

## 9. GPU 时间预算

若一个标量候选完整经历 7×7 搜参、3-seed pilot 和 5-seed formal，纯训练时间约为：

\[
1.63+1.00+3.33=5.96\text{ h}.
\]

加上 smoke、reference 生成、周期 evaluation、写盘和恢复余量，建议按每候选
7--8 小时估计。向量流体和三维问题需要额外考虑 evaluation 与显存开销。

不建议八个候选全部直接进入 formal。正确用法是先让每个候选通过低成本门禁，只有出现
可训练且有区分度的候选才投入 5-seed 长跑。当前执行顺序调整为：

1. `HO-01` MBE：已经形成完整负结果，定性为 `TRAINING_FAILURE` 并停止；
2. `HO-04` 二维超黏性 Navier--Stokes：立即执行上述自动门禁、搜参、pilot 与条件 formal；
3. `HO-02` MPFC：等待 `HO-04` 结论后再开发；
4. `HO-03` Kawahara：保留为后续奇数阶低成本诊断。

`HO-04` 从 sentinel 到 formal 的纯训练上限为
`18 min + 1.63 h + 1 h + 3.33 h = 6.26 h`；加上两档 smoke、最终高阶导数
evaluation、写盘与绘图，按约 7--8 小时 wall time 预留。

## 10. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-12 | 用户决定停止 MBE：HO-01 定位为 `REJECTED / TRAINING_FAILURE`，不再 pilot/formal；HO-04 提升到 HO-02 MPFC 前，冻结 Taylor--Green setting、损失、物理门禁和自动长跑流程，并完成多输出实现。 |
| 2026-08-12 | HO-01 完成 49 个共享权重、98 个 method cells 的完整搜索；结果完整但全部饱和在近零假解，状态更新为 `SEARCHED / HOLD`，未启动 pilot/formal。 |
| 2026-08-11 | 建立候选队列、统一协议、收益判据、实验门槛和结果占位表；尚未启动新实验。 |
