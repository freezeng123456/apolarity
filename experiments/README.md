# Experiments

当前可进入论文主结果的实验只有二维 Polyharmonic 和二维 Cahn–Hilliard。
本目录存问题实现，正式 runner 在仓库根目录的 `scripts/`，结果统一进入
`outputs/`；不再把正式结果写回 family 的 `data/` 或 `experiments/results/`。

## 当前问题族

| 目录 | 物理输入 | 任务 | 当前正式协议 |
|---|---|---|---|
| `polyharmonic/` | `(x,y)` | order 2/4/6 | WAR complex64+sinh vs real float32+tanh AD；common Xavier；1200 s；5 seeds |
| `cahn_hilliard_2d/` | `(x,y,t)` | CH4/CH6 | WAR complex64+sinh vs real float32+sinh AD；仿射输入；1200 s；5 seeds |

Poly 与 CH 的实数激活函数是两条不同的冻结协议，不应在汇总时误写成相同
网络。两者都关闭任务感知频率初始化；CH 的解析解包含 cosine，但网络输入没有
sin/cos/Fourier 特征。

## 当前正式入口

```bash
# Poly：固定权重，5 seeds × 3 tasks × 2 methods
python scripts/run_poly_fixed_weight_formal.py orchestrate \
  --seconds 1200 --seeds 5 --resume

# 二维 CH：固定 (lambda_ic, lambda_bc)=(1,10)
python scripts/run_cahn2d_fixed_weight_formal.py orchestrate \
  --seconds 1200 --seeds 5 --resume
```

正式运行前的 smoke 必须通过，但 raw smoke 只存在于临时目录，退出后删除；
正式 JSON/log/history 不受影响。

## 代码分层

- `common/osc_common.py`：共享 MLP、采样、直接 AD 与 jet 入口；
- `common/weight_search.py`：当前 Poly common-Xavier 训练内核；其中保留的旧
  1D 周期 CH 分支只为历史结果和回归测试兼容，不属于当前二维 CH 证据；
- 旧 `exp_polyharmonic.py` 与 JSC task registry 已移到
  `archived/jsc_v3/`，不得用于重建当前结果；
- `cahn_hilliard_2d/problem.py`：二维 CH 制造解、源项、自然边界和 loss；
- `archived/`：旧 JSC、Chirp、Maxwell、其他 PDE、历史 runner 和结果。

## 归档边界

`experiments/archived/` 可以用于追溯旧设计和权重选择，但其中的 double 结果、
频率初始化结果、旧 baseline 套件和 JSC 图表不能与当前 float32/complex64
正式结果拼接。当前数据清单以 `outputs/README.md` 为准。
