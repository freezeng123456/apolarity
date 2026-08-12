# HO-02 二维六阶 MPFC 共享权重搜索报告

## 结论先行

HO-02 的 7×7 全网格搜索已完整完成：49 个共享权重候选、WAR 与
real-tanh autodiff 各 49 个 60 秒 cell，共 98/98，失败 0。结果绑定独立
Fourier/IMEX reference（4096 个评估点，SHA256
`8d6a81b8d5a00cbff56287be16bdad273d81dabaf4b6faaa43f863c9b72a77e5`）。

在这个 60 秒预算下，WAR 的 median `rel_error` 为 `0.408454`，real-tanh
autodiff 的 median 为 `0.417970`；WAR 在 47/49 个配对候选中更低，但两种
方法都还没有达到可以直接放行 pilot 的高精度水平（最佳 WAR `0.408109`，
最佳 AD `0.408765`）。因此本搜索说明 WAR 在短预算下有稳定优势，但尚不能
据此宣称 MPFC 已经适合 3-seed/5-seed 正式实验。

按共享目标，建议唯一候选为：

`(lambda_ic, lambda_bc) = (1e-3, 1e-2)`（shared-minimax 与
shared-geomean 均排名第一）。该候选的 WAR/AD `rel_error` 分别为
`0.408109 / 0.408922`。这是搜参后的诊断候选，不是自动启动 formal 的授权。

## 完整性与协议

| 检查项 | 结果 |
|---|---|
| 搜索完成度 | 49/49 paired candidates；98/98 method runs |
| 失败/重试 | 0 failures；最终报告无失败 cell |
| 排名 | WAR、real-tanh AD、shared geomean、shared minimax 各 49 行 |
| 日志 | 98/98 末行同时包含 `loss` 与 `rel_error` |
| history | 98 个 JSON 的 history 全部有限 |
| 精度/网络 | WAR `complex64+sinh+Waring jet`；AD `float32+tanh+direct autodiff`；共同 Xavier、hidden128、depth4 |
| 输入/边界 | affine-normalized raw `(x,y,t)`；无三角输入、周期嵌入或频率初始化；周期 normal trace 0--5 阶 |
| 训练采样 | `n_int=2048,n_ic=512,n_bc=1024`；评估 `n_eval=4096,history_eval_n=1024`；seed `42/68421` |
| 校验和 | 远端与下载后的 `SHA256SUMS` 均通过 |

早期修正过的 2 个 cell 保存在远端 `...-prepatch-2cells` 证据目录，未进入
本最终报告；最终输出根目录从修正后的 manifest 重新从 0/98 完成。

## 四类 Top 10（前五）

### Shared minimax / shared geometric mean

两种共享 ranking 的第一名相同，前四名如下：

| rank | `(lambda_ic,lambda_bc)` | WAR | AD | geometric mean | max |
|---:|---|---:|---:|---:|---:|
| 1 | `(1e-3,1e-2)` | 0.408109 | 0.408922 | 0.408516 | 0.408922 |
| 2 | `(1e-2,1e2)` | 0.410076 | 0.408765 | 0.409420 | 0.410076 |
| 3 | `(1e-1,1e3)` | 0.410172 | 0.409923 | 0.410047 | 0.410172 |
| 4 | `(1e-3,1e-3)` | 0.408332 | 0.410621 | 0.409475 | 0.410621 |

### WAR ranking

WAR 的最优前五为 `(1e-3,1e-2)`, `(1e-2,1e-1)`, `(1e1,1e2)`,
`(1e3,1e3)`, `(1e3,1e1)`；对应 WAR error 为
`0.408109, 0.408120, 0.408129, 0.408229, 0.408242`。

### real-tanh autodiff ranking

AD 的最优前五为 `(1e-2,1e2)`, `(1e-3,1e-2)`, `(1e-1,1e3)`,
`(1e-3,1e-3)`, `(1e1,1e3)`；对应 AD error 为
`0.408765, 0.408922, 0.409923, 0.410621, 0.412625`。

完整 49 行排名保存在 `mpfc_2d_o6/rankings/`，逐点原始结果和 history 保存在
`mpfc_2d_o6/points/`。

## 权重敏感性与决定

- WAR 对权重相当平坦：全网格范围约 `0.408109--0.411193`，按
  `lambda_bc` 分组的均值在 `0.40848--0.40943`；方法优势在 49 个候选中有
  47 个配对胜出。
- AD 更依赖边界权重：`lambda_bc=1e2/1e3` 时均值升至约 `0.445/0.461`，
  并出现最高约 `0.643` 的候选；低到中等 `lambda_bc` 更稳定。
- 共享候选 `(1e-3,1e-2)` 是一个保守的共同平台点，不是“最优解”证明；
  60 秒结果整体仍在 `rel_error≈0.41`，应先做一个短 pilot 验证是否随时间
  下降，再决定是否投入 5-seed formal。

## 下一步建议

在用户确认前不启动 formal。若继续，先用 `(1e-3,1e-2)` 做预注册的
3-seed pilot（保持同一 precision、网络、采样与 reference），门槛至少要求
两种方法的 median error 明显低于 0.2/0.75 目标并且不出现近零/饱和假解；
否则将 MPFC 标记为“可训练但短预算精度不足”，保留完整搜索作为负/中性结果。
