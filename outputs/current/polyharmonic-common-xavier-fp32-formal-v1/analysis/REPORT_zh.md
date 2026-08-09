# Polyharmonic common-Xavier 正式实验

## 验收

- 状态：`passed`；30/30 cell，5 seeds，三组 Poly task。
- 原始校验和：113 项通过。
- 实时 history：7230 个数据点，均嵌在方法 JSON 的 `history` 字段中。
- WAR：complex64、sinh、common Xavier、无频率初始化。
- 实数 AD：float32、tanh、common Xavier、无频率初始化。
- 每方法每 seed 1200 秒；网络宽度 128、深度 4。

## 最终相对误差（5 seeds）

| Task | 方法 | Mean | Std | Median | Min | Max | 胜出 seeds |
|---|---|---:|---:|---:|---:|---:|---:|
| poly_d2_o2 | WAR | 0.000151217 | 9.1965e-05 | 0.000148164 | 2.78305e-05 | 0.000285897 | 5/5 |
| poly_d2_o2 | Real tanh AD | 0.000586022 | 0.000536525 | 0.000481307 | 0.000165205 | 0.00149567 | 0/5 |
| poly_d2_o4 | WAR | 0.00297159 | 0.00185825 | 0.00230869 | 0.00103301 | 0.0050109 | 2/5 |
| poly_d2_o4 | Real tanh AD | 0.00207344 | 0.00132211 | 0.00201344 | 0.00056103 | 0.00348898 | 3/5 |
| poly_d2_o6 | WAR | 1 | 2.00115e-06 | 1 | 0.999999 | 1 | 0/5 |
| poly_d2_o6 | Real tanh AD | 0.996552 | 0.00745363 | 0.999948 | 0.983221 | 0.99999 | 5/5 |

## 解读

- `poly_d2_o2`：WAR 在 5/5 seeds 上更低，平均误差约为实数 AD 的四分之一。
- `poly_d2_o4`：实数 tanh AD 在 3/5 seeds 上更低；两种方法都达到 1e-2 以下。
- `poly_d2_o6`：两种方法的误差都约为 1，说明共同 Xavier、无频率初始化的当前配置没有学到六阶算例。该 task 必须如实报告为失败，不能与旧频率初始化结果混用。

## Provenance 边界

结果记录的基准提交为 `00113c16a4596e41871da6f5b00c43e968d63b8f`，并明确记录 `git dirty=true`。为保证可复核性，实际运行时使用的三个源文件已原样保存在 `provenance/source_snapshot/`，并由固定 SHA-256 验证。当前仓库中的正式 runner 在该快照基础上仅做了目录命名、Poly-only 默认任务和“不保留 raw smoke”的结构性整理；实验方法与结果未改写。

完整逐 seed 数值见 `final_metrics.csv` 和 `paired_comparison.csv`。
