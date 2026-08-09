# Two-dimensional Cahn–Hilliard

这里的“2D”指两个空间坐标，网络的物理输入固定为 `(x,y,t)`，不是旧一维
周期 CH 算例。

## 方程与唯一性设置

在 `(x,y) in (0,pi)^2`、`t in [0,1]` 上求解

```text
u_t - Delta(u^3-u) + eta_q Delta^q u = f,
```

其中：

- CH4：`q=2`, `eta_q=+1e-2`；
- CH6：`q=3`, `eta_q=-1e-2`。

这两个符号使最高阶 Fourier symbol 都是耗散的。制造解为

```text
exp(-t) * [
  0.50 cos(x) cos(y)
  + 0.25 cos(2x) cos(y)
  + 0.25 cos(x) cos(2y)
].
```

Cosine 只用于构造解析源、初值和 held-out truth，不作为网络输入特征。

## 初边值条件

- 初值：`u(x,y,0)` 等于制造解；
- CH4：`d_n u = d_n Delta u = 0`；
- CH6：`d_n u = d_n Delta u = d_n Delta^2 u = 0`；
- 质量守恒作为诊断记录，不额外加入训练 loss。

四条边统一采样，自然边界条件保证与制造解一致。相关公式、源项和高阶边界
导数均有 float32 与 jet/direct-AD 回归测试。

## 公平比较

两方法共同使用：

- 仅仿射归一化的 raw `(x,y,t)`，无 Fourier、sin/cos 或周期嵌入；
- 四个隐藏层、hidden=128、`sinh` 激活；
- variance-matched common Xavier；
- 相同 collocation、loss、seed、学习率和墙钟预算。

差别只有：

- `war`：native complex64 + Waring/Taylor jet；
- `real_sinh_autodiff`：real float32 + direct nested autodiff。

这是等字面层形状而不是等实自由度比较：每个复参数有两个实自由度，结果中
同时记录 parameter elements 和 real DOF。

## 权重搜索

搜索共享向量 `[lambda_ic, lambda_bc]`，两个分量都取
`{1e-3,1e-2,1e-1,1,1e1,1e2,1e3}`。每个 task 49 个候选，WAR 与 real AD
共 196 个 60 秒 cell，结果位于：

```text
outputs/search/cahn-hilliard-2d-weight-search-v1/
```

CH4 的共同 geomean/minimax 最优候选是 `(1,10)`。用户随后统一指定 CH4/CH6
正式实验都用 `(1,10)`；这不是声称它对 CH6 的 60 秒搜索最优。

## 正式协议与结果

- `(lambda_ic,lambda_bc)=(1,10)`；
- seeds 0–4，每方法每 seed 1200 秒；
- `n_int=4096`, `n_ic=1024`, `n_bc=2048`；
- 最终 `n_eval=32768`, history `n_eval=4096`；
- 单张 H20 严格串行，20/20 cell 完成。

五 seed 最终相对误差均值：

| task | WAR | real sinh AD |
|---|---:|---:|
| CH4 | `2.224e-3` | `3.158e-2` |
| CH6 | `6.040e-3` | `5.112e-1` |

WAR 在两个任务上均为 5/5 seeds 更低。完整 raw、analysis 与服务器生成的曲线
位于 `outputs/current/cahn-hilliard-2d-fixed-1-10-formal-v1/`。

## 入口

```bash
python scripts/run_cahn2d_weight_search.py smoke --seconds 3 \
  --ephemeral-conclusion /absolute/path/cahn2d_smoke_conclusion.json
python scripts/run_cahn2d_weight_search.py orchestrate --seconds 60 --resume

python scripts/run_cahn2d_fixed_weight_formal.py smoke --seconds 3
python scripts/run_cahn2d_fixed_weight_formal.py orchestrate \
  --seconds 1200 --seeds 5 --resume
```

Smoke raw 只在临时目录存在，结论通过后删除；它不进入论文统计。
