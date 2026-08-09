# Polyharmonic order sweep

## 方程与解析解

在 `(-1,1)^2` 上使用

```text
u(x,y) = sin(pi*x) sin(pi*y),
Delta^m u = (-2*pi^2)^m u,
```

其中 `m=1,2,3`，对应 `poly_d2_o2/o4/o6`。Navier 边界分量为
`u, Delta u, ..., Delta^(m-1)u`；训练代码按解析特征值归一化 PDE 与各边界
分量，避免只因导数阶数增加而产生量纲爆炸。

## 冻结正式设置

| task | 边界分量 | 权重 |
|---|---|---|
| `poly_d2_o2` | `u` | `[1]` |
| `poly_d2_o4` | `u, Delta u` | `[1,1]` |
| `poly_d2_o6` | `u, Delta u, Delta^2 u` | `[10,1,1]` |

- WAR：native complex64、sinh、Waring/Taylor jet；
- 实数基线：float32、**tanh**、direct autodiff；
- 两者：hidden=128、四个隐藏层、common Xavier、原始 `(x,y)` 输入；
- 不使用 sin/cos/Fourier 输入，不使用频率匹配或任务感知频率初始化；
- `n_int=4096`、`n_boundary=512`、最终 `n_eval=8192`；
- seeds 0–4，每方法每 seed 1200 秒，单 GPU 严格串行；
- history 每约 5 秒记录 time、step、learning rate、loss 和 rel_error。

旧 `exp_polyharmonic.py` 已移到 `experiments/archived/jsc_v3/`；其中存在的旧
频率参数不是当前正式协议。当前结果必须由
`scripts/run_poly_fixed_weight_formal.py` 及其来源快照解释。

## 当前结果

完整 30/30 原始包位于
`outputs/current/polyharmonic-common-xavier-fp32-formal-v1/`。

五 seed 最终相对误差均值：

| task | WAR | real tanh AD |
|---|---:|---:|
| o2 | `1.512e-4` | `5.860e-4` |
| o4 | `2.972e-3` | `2.073e-3` |
| o6 | `1.0000` | `0.9966` |

o6 两条曲线都没有学到目标解，应作为当前设置的失败结果如实保留。

## 运行与复核

```bash
python scripts/run_poly_fixed_weight_formal.py smoke --seconds 5
python scripts/run_poly_fixed_weight_formal.py orchestrate \
  --seconds 1200 --seeds 5 --resume
python scripts/analyze_poly_fixed_weight_formal.py \
  outputs/current/polyharmonic-common-xavier-fp32-formal-v1
```

Smoke raw 使用临时目录，不进入 Git；正式结果通过原始与交付两层 SHA-256
校验。
