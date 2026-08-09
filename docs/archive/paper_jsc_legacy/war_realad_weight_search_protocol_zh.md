# WAR / 实数 autodiff 全量 loss-weight 搜索协议

协议标识：`war_realad_weight_grid_v1`。

本轮只做 B/C 五个算例的 loss 权重搜索，不包含 1200 秒、5 seeds 的正式训练。
历史/archived 搜索不参与选择，只用于诊断。

## 方法与固定训练配置

- `war`：native-complex、sinh、Waring/Taylor-jet；
- `real_tanh_autodiff`：real-float64、tanh、direct nested coordinate autodiff；
- 两者均为 4 个隐藏层、literal hidden width `H=128`；
- tuning seed `42`，held-out evaluation seed `54321`；
- Adam，学习率按 wall clock 从 `1e-3` cosine 衰减到 `1e-4`；
- 每个 `task × weight vector × method` 获得 60 秒训练预算；
- history 每约 5 个训练秒记录 `rel_error`、总 loss、PDE loss、全部原始及加权
  constraint-loss 分量。

## 权重网格和组合数

每个可调权重独立取

```text
1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3
```

并执行完整的有序笛卡尔积。

| task | 搜索向量 | 候选数 | 两方法 runs | 纯训练时间 |
|---|---|---:|---:|---:|
| `poly_d2_o2` | `[w_u]` | 7 | 14 | 14 min |
| `poly_d2_o4` | `[w_u,w_Delta_u]` | 49 | 98 | 98 min |
| `poly_d2_o6` | `[w_u,w_Delta_u,w_Delta2_u]` | 343 | 686 | 686 min |
| `cahn_hilliard_o4` | `[lambda_ic,mu_mean]` | 49 | 98 | 98 min |
| `cahn_hilliard_o6` | `[lambda_ic,mu_mean]` | 49 | 98 | 98 min |
| 合计 | — | 497 | 994 | 994 min = 16 h 34 min |

Poly 和 Cahn--Hilliard 的 PDE/bulk loss 系数均固定为 1。搜索 bulk 的共同倍数会
制造等价的相对 loss 比例，因此不进入网格。

## Cahn--Hilliard setting

使用周期初值问题

\[
u_t-\gamma_2\partial_x^2(u^3-u)+\gamma_1\partial_x^{2q}u=f,
\]

其中 CH4 为 `q=2, gamma1=+1e-2`，CH6 为
`q=3, gamma1=-1e-2`，两者 `gamma2=1`。制造解为

\[
u^\star(x,t)=e^{-t}\cos(2x),\qquad x\in[0,2\pi),\ t\in[0,1].
\]

网络输入采用 `(cos(x), sin(x), t)`，从结构上严格满足空间周期性；因此不加入软
periodic-boundary 权重。可调约束是初值权重 `lambda_ic` 与质量守恒权重
`mu_mean`。

CH4 的高阶项采用正的 `gamma1`，CH6 采用负的 `gamma1`，使傅里叶最高阶符号
均为耗散的正向抛物型符号。`gamma1/gamma2` 是 PDE 定义，不参与 loss 搜参。

## 排名与停止条件

每个 task 同时生成：

1. WAR 单方法排名；
2. real-autodiff 单方法排名；
3. shared geometric-mean 排名；
4. shared minimax 排名，主分数为两个方法 terminal held-out relative L2 的最大值。

网格完成后停止，不自动启动正式实验。正式阶段选择方法独立权重还是共享权重，需
根据完整排名和 Pareto 前沿另行决定。

## 执行入口

```bash
python scripts/run_weight_search.py smoke --seconds 5
python scripts/run_weight_search.py orchestrate --seconds 60 --resume
python scripts/run_weight_search.py summarize
```

runner 逐 cell 原子写入结果并支持 resume。失败必须保留明确状态，不能用
best-so-far 或缺失值替代。完整结束时生成 ranking CSV/JSON、run matrix、manifest、
progress、完成标记和 `SHA256SUMS`。
