# 二维六阶应变梯度动态板：实现审计与 120 秒校准

日期：2026-08-10

## 结论

二维四阶动态板此前没有做 loss-weight 网格搜索，只在
`(lambda_ic, lambda_bc)=(10,10)` 下完成了 3 seeds、每方法 600 秒的
pilot。该固定点上 WAR 三个 seed 全胜，但它只能证明这个点可训练，不能证明
它是最优或权重不敏感。

新增的二维六阶应变梯度动态板已经通过数学一致性测试、WAR/直接 AD 六阶
导数一致性测试、全规模 CUDA smoke，以及一个 seed、每方法 120 秒的配对
校准。120 秒结果为：WAR `rel_error=8.8696e-3`，实数 AD
`rel_error=4.2203e-1`。这是可行性结果，不是正式多 seed 统计。

## 四阶动态板现状

- 方程：`u_tt + 0.1*u_t + Delta_xy^2*u = f`。
- 空间域与时间区间：`[-1,1]^2 x [0,1]`。
- 固定权重：`lambda_ic=10, lambda_bc=10`。
- 协议：3 seeds、每方法 600 秒、共同 Xavier、raw affine coordinates；WAR
  为 complex64+sinh+Waring jet，基线为 float32+tanh+direct AD。
- WAR rel_error：`[3.5162e-3, 4.3102e-3, 2.9816e-3]`，均值
  `3.6026e-3`。
- 实数 AD rel_error：`[3.1899e-2, 3.9529e-2, 5.2517e-2]`，均值
  `4.1315e-2`。
- WAR 逐 seed 胜负：`3/3`。

没有发现任何针对该任务的 49 点或其他权重搜索结果。

## 新六阶问题

采用二维应变梯度 Kirchhoff 板：

```text
u_tt + 0.1*u_t + Delta_xy^2*u - 0.25*Delta_xy^3*u = f,
(x,y,t) in (-1,1)^2 x (0,1).
```

制造解为：

```text
u(x,y,t) = (1-x^2)^3 (1-y^2)^3 cos(pi*t).
```

边界条件为每条空间边上的 `u=0`、一阶法向导数为零、二阶法向导数为零；
初值给定位移和速度。差解满足齐次数据，其 H3 应变梯度板能量具有强制性，
配合阻尼可由标准能量估计得到唯一性。该模型属于六阶梯度弹性板问题；相关
数学背景可参见：

- https://academic.oup.com/jom/article/doi/10.1093/jom/ufac017/6593409
- https://www.sciencedirect.com/science/article/pii/S0997753816302066

## 固定网络与数据协议

- 网络：hidden 128，depth 4；两方法层数、宽度和激活层位置一致。
- WAR：complex64，sinh，Waring/Taylor jet。
- 基线：float32，tanh，直接坐标自动微分。
- 初始化：共同 Xavier 类；没有频率匹配初始化。
- 输入：仅 affine-normalized raw `(x,y,t)`；没有 sin/cos 输入特征。
- 暂定权重：`lambda_ic=10, lambda_bc=10`。
- PDE residual scale：500。
- `n_int=2048, n_ic=512, n_bc=1024, n_eval=16384, history_eval_n=2048`。
- 单 GPU 严格串行、相同 wall-clock budget。

## 验证结果

代码测试覆盖以下内容：

1. 独立自动微分复算制造解的六阶 PDE 源项；
2. 验证 `u`、一阶法向导数、二阶法向导数三类边界迹；
3. 验证纯六阶和两类混合六阶偏导的 Waring jet 与直接 AD 一致；
4. 验证两种方法的 tiny loss 和参数梯度均有限；
5. 单独验证 Poly-o6 的三重 Laplacian、三层 Navier 边界迹和任务级六阶
   Waring/AD 导数。

远端相关测试全部通过；PyTorch 只报告了其标准的 complex-module 实验性警告。

全规模 3 秒 CUDA smoke：

| 方法 | steps | ms/step | peak MB | final loss | rel_error |
|---|---:|---:|---:|---:|---:|
| WAR | 36 | 84.18 | 1972.51 | 1.3226 | 0.9441 |
| real AD | 6 | 504.90 | 4984.68 | 1.3127 | 0.9267 |

smoke 仅用于启动、有限性、显存和数据管线门禁，不作为精度证据。

一个 seed、每方法 120 秒校准：

| 方法 | initial rel_error | final rel_error | final loss | steps | ms/step | peak MB |
|---|---:|---:|---:|---:|---:|---:|
| WAR | 1.3395 | 8.8696e-3 | 6.5433e-4 | 1636 | 73.38 | 1972.51 |
| real AD | 1.2578 | 4.2203e-1 | 6.1398e-1 | 298 | 403.22 | 4984.68 |

WAR 的 history rel_error 在约 0/20/40/60/80/100/120 秒为
`1.3395/0.5316/0.03267/0.01682/0.01298/0.00982/0.00898`。实数 AD
在同样时间点为
`1.2578/0.8149/0.5286/0.3664/0.3846/0.4168/0.4148`。

## Poly-o6 校验

当前 Poly-o6 方程、制造解和边界条件在数学上相容：

```text
Delta^3 u = (-2*pi^2)^3 u,
u = Delta u = Delta^2 u = 0 on the boundary,
u = sin(pi*x) sin(pi*y).
```

任务级导数测试通过，因此当前失败不是“六阶 Waring 算错”。3 seeds 的初始
梯度审计显示，在固定权重 `(10,1,1)` 和共同 Xavier 初始化下，边界值项梯度
相对 PDE 梯度的中位比为：

- WAR：约 `6.29e6`；
- 实数 AD：约 `2.67e4`。

这与 1200 秒历史轨迹一致：优化首先把齐次边界压到近零，而 PDE 项几乎不动，
最终停在 rel_error 约 1 的近零解。历史 double+频率初始化的 WAR 曾达到五 seed
均值 `8.28e-2`，说明方程本身可训练；但该历史协议不能混入当前
float32/complex64、无频率初始化的公平正式统计。

## 建议的下一步

正式多 seed 之前，对四阶和六阶动态板使用完全共享的权重网格：

```text
(lambda_ic, lambda_bc) in
{1e-3,1e-2,1e-1,1,1e1,1e2,1e3}^2.
```

每个权重向量以同一 seed、两种方法各跑 60 秒，分别按 WAR、实数 AD、共享
几何平均和共享 minimax 排名。每个任务 49 个向量、98 个 method-cell；两个
任务共 196 个 cell，名义训练时间约 3 小时 16 分，含启动和评估开销预计约
3.7--4.2 小时。确定共享权重后，再单独启动 5 seeds、每方法 1200 秒的正式
实验；名义训练时间 3 小时 20 分，含开销预计约 3.6--3.9 小时。

若 8 小时窗口必须严格保证不超时，建议本轮先完成两个任务的权重搜索并交付
Top 10，待确认固定权重后再开正式实验，不把搜索 seed 混入正式统计。
