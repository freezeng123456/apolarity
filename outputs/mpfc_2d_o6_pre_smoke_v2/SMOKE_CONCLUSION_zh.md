# HO-02 MPFC（二维六阶）门禁结论

本目录记录在 H20 上按 `mpfc_2d_o6_common_xavier_fp32_v1` 执行的两档临时 smoke。两档均通过，未发现 OOM、NaN、Inf 或非有限梯度；结果只用于启动门禁，不计入论文统计。

| smoke | WAR loss / rel_error | real-tanh AD loss / rel_error | 峰值显存 |
|---|---:|---:|---:|
| 基础（3 s） | 1.0561e-2 / 0.7273 | 1.9161e-2 / 0.8545 | 87.9 / 120.4 MiB |
| 搜索规模（`n_int=512,n_ic=256,n_bc=1024,n_eval=4096,history_eval_n=1024`，3 s） | 1.2618e-2 / 0.7331 | 2.6266e-2 / 0.9208 | 956.9 / 2133.8 MiB |

两档各 2 个 method cell，合计 4/4 complete；H20、CUDA 12.8、PyTorch 2.11.0+cu128。基础 smoke 的评估目标是初值门禁；正式搜参将绑定独立 Fourier 参考解。
