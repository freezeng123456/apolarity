# Smoke 结论记录（原始结果已清理）

本文件只保留 smoke 的用途、协议和结论；对应的 JSON、CSV、PT、history、
日志、manifest、completion marker 和 checksum 原始文件已按项目要求删除。
Smoke 不是正式收敛结果，也不进入论文统计。

## 二维 Cahn--Hilliard：GPU smoke 前的数值门禁

协议：二维物理输入 `(x,y,t)`、仅仿射归一化、无 Fourier/周期输入嵌入、
共同 `sinh` 网络与 Xavier 类初始化；WAR=`complex64`，实数 autodiff=
`float32`。CH4/CH6 分别施加 1/3 阶与 1/3/5 阶齐次自然边界条件。

2026-08-09 在服务器隔离副本上以 CPU 单线程完成 7 项测试，结果为
`7 passed`：制造源与直接 float32 求导一致；解析解满足全部自然边界；
WAR 混合高阶导数与直接 autodiff 一致；两种方法的 loss、梯度与反向传播
均为有限值；输入维数固定为 3 且没有三角特征。

这只是公式、导数和数据管线的数值门禁。H20 CUDA smoke、峰值显存与每步
速度的实际审核结论见文末 2026-08-09 节；它们同样不构成正式实验结果。

## 当前 `complex64/float32` Poly smoke

协议：common Xavier、关闭频率匹配初始化、WAR=`complex64`、实数
autodiff=`float32`，每个 cell 5 秒，seed 0。

六个 cell 均完成，loss、relative error 和梯度均为有限值：

| task | WAR loss | WAR rel_error | real AD loss | real AD rel_error |
|---|---:|---:|---:|---:|
| `poly_d2_o2` | `2.909595e-03` | `2.006355e-01` | `1.526996e-03` | `8.447900e-02` |
| `poly_d2_o4` | `2.606929e-01` | `9.999882e-01` | `2.606945e-01` | `1.000115e+00` |
| `poly_d2_o6` | `2.606941e-01` | `9.999877e-01` | `2.880623e-01` | `1.008137e+00` |

结论：共同 Xavier、Poly 高阶导数路径、反向传播和低精度 dtype 均能正常
运行；这只是启动门禁，不代表 1200 秒正式结果。

## 已清理的历史 smoke

- 旧 fixed-weight common-Xavier smoke：Poly 三任务 × 两方法，全部完成；
  验证了模型构造、loss、导数和输出格式。
- 旧 497-point weight-grid smoke：marker 记录 `failures=0`；验证了搜参
  调度、原子写入、history 和 ranking 生成。
- 旧 fixed-weight formal smoke：五任务 × 两方法的短时完整性检查通过；
  它不是正式实验。
- overnight core smoke：记录的 benchmark rows 为 `status=ok`。
- overnight PINN/autodiff/jet smoke：记录的短任务为 `status=complete`。
- overnight risk smoke：保留的结论是短时 runner/日志/结果格式检查完成；
  其数值不作为正式对比证据。

以上结论只用于说明“入口和数据管线曾经通过短时检查”，不用于证明某个
方法优于另一个方法。

## 2026-08-09 二维 Cahn--Hilliard H20 队列 smoke 审核

本次队列在 Polyharmonic `FORMAL_COMPLETE` 且 GPU 连续三次空闲确认后，按批准
顺序完成了两阶段 CUDA smoke：

1. 基础规模：CH4/CH6 × WAR/real_sinh_autodiff，共 4 个 cell、每个 3 秒；
2. 搜索规模：`n_int=512, n_ic=256, n_bc=512, n_eval=4096,
   history_eval_n=1024`，同样 4 个 cell、每个 3 秒。

两阶段均为 `passed=true`、`failure_count=0`，四个 cell 的 loss、rel_error、
peak memory 和梯度路径均为有限值；搜索规模 smoke 的峰值显存最高约
1608 MiB（CH6 实数 autodiff），没有 OOM 或 NaN。原始 smoke 目录未保留，
交付包只保留 `basic_smoke_conclusion.json` 与
`search_sized_smoke_conclusion.json` 两份结论 JSON。两阶段 smoke 仍只是
CUDA 启动、有限性和数据管线门禁，不进入论文统计；随后运行的 196 个
60 秒搜参 cell 才是本次可分析的权重搜索结果。
