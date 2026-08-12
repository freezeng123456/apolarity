# HO-02 MPFC 预 smoke 结论

执行环境：固定 H20，`/root/apolarity-venv/bin/python3`，PyTorch `2.11.0+cu128`，
CUDA 可用。该结果只验证方程残差、六阶导数后端、边界 0--5 阶 trace、参数梯度和
full-size 单步显存，不包含搜参或正式长跑。

## 基础 smoke

- 任务：`mpfc_2d_o6`
- 权重：`lambda_ic=1, lambda_bc=1`
- hidden=5、depth=1、`n_int=2,n_ic=2,n_bc=4`
- WAR（complex64/sinh/Waring jet）：finite，loss `0.2015468031`
- real-tanh autodiff（float32/tanh/direct AD）：finite，loss `0.0502625294`

## Full-size 单步 smoke

- hidden=128、depth=4、`n_int=512,n_ic=256,n_bc=1024`
- WAR：finite，loss `0.0315963440`，耗时 `0.299 s`，峰值显存 `953.25 MB`
- real-tanh autodiff：finite，loss `0.0750850514`，耗时 `1.031 s`，峰值显存 `2131.64 MB`

两个方法均通过：没有 NaN/Inf，所有参数梯度有限且非空；没有 OOM。该 smoke 只允许
进入 reference/convergence 和共享权重搜索阶段，不能据此宣称 MPFC 的训练收益或启动
5-seed formal。

