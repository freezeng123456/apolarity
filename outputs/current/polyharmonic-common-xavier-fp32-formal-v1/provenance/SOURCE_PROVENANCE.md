# Poly 正式实验来源说明

结果 `manifest.json` 记录的基准 Git 提交为
`00113c16a4596e41871da6f5b00c43e968d63b8f`，分支为
`agent/apolarity-overnight-results`，同时明确记录 `git.dirty=true`。

这意味着只引用基准提交不足以重建实际运行代码。为保留可复核链路，服务器
运行目录中真正参与本次实验的三个源文件已逐字保存到
`source_snapshot/`：

| 运行时路径 | SHA-256 |
|---|---|
| `experiments/common/weight_search.py` | `1da016aa5f917ce445f184d74a5b5d10fcb5b7e53e0cb7f846eb428b68289e64` |
| `scripts/run_weight_search.py` | `904e43a3c96fa85d51a21f578be35ce0996ffe17dae3ba0ecff7f6b862aa8d85` |
| `scripts/run_fixed_weight_formal.py` | `f8063744f48c1400ff7a3263feb1aec8f7f8d5f547abbab5680a7d64b2395247` |

`SOURCE_SHA256SUMS` 可直接校验这三份快照。仓库当前的
`scripts/run_poly_fixed_weight_formal.py` 从该正式 runner 整理而来；只改变了
文件名、默认输出目录、Poly-only 默认任务及 raw smoke 不落盘策略，没有改变
这批正式结果所使用的方法、网络、dtype、权重、seed 或训练预算。未经整理的
原始 runner 仍以原文件名保存在本目录，避免用事后代码冒充运行时代码。
