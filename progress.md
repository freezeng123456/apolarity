# Project progress

Last updated: 2026-08-10

## 当前结论

- Poly common-Xavier 低精度正式实验已完成并验收：30/30 cell，5 seeds，
  3 tasks，2 methods，每 cell 1200 秒；完整原始包、日志、内嵌 history、
  summary、来源快照和双层校验和已经进入 `outputs/current/`。
- 二维 Cahn–Hilliard 固定权重正式实验已完成并验收：20/20 cell，5 seeds，
  CH4/CH6，2 methods，每 cell 1200 秒。
- 二维 CH 权重搜索已完成：196/196 method cells，两个任务各 49 个二维
  `(lambda_ic, lambda_bc)` 候选。
- 旧 JSC、Chirp、Maxwell、double 和其他 PDE 已归档；raw smoke 已清理。

## 当前结果表

### Polyharmonic 最终相对误差

| task | WAR mean ± std | real tanh AD mean ± std | seed 胜负 |
|---|---:|---:|---|
| `poly_d2_o2` | `1.512e-4 ± 9.197e-5` | `5.860e-4 ± 5.365e-4` | WAR 5/5 |
| `poly_d2_o4` | `2.972e-3 ± 1.858e-3` | `2.073e-3 ± 1.322e-3` | WAR 2/5，AD 3/5 |
| `poly_d2_o6` | `1.000001 ± 2.00e-6` | `0.996552 ± 7.45e-3` | AD 5/5，但两者都失败 |

### 二维 Cahn–Hilliard 最终相对误差

| task | WAR mean ± std | real sinh AD mean ± std | seed 胜负 |
|---|---:|---:|---|
| CH4 | `2.224e-3 ± 9.657e-4` | `3.158e-2 ± 8.589e-3` | WAR 5/5 |
| CH6 | `6.040e-3 ± 9.628e-4` | `5.112e-1 ± 1.481e-1` | WAR 5/5 |

## 方法冻结状态

- Poly：WAR complex64+sinh；实数 AD float32+**tanh**。本次不改 tanh。
- CH2D：WAR complex64+sinh；实数 AD float32+sinh。
- 两个协议都使用 common-Xavier 类初始化、相同字面 hidden=128/depth=4
  层形状、同 seed/采样/墙钟；都没有频率匹配输入。
- 这是等层形状/等墙钟而非等实自由度比较。

## 数据质量

- Poly 原始 `SHA256SUMS`：113 个服务器数据文件通过；
- Poly 交付 `DELIVERY_SHA256SUMS`：覆盖 raw、analysis、说明和运行源快照；
- Poly 30 份方法 JSON 均有 241 个有限 history 点，总计 7230；
- Poly 30 份日志非空，末行均同时含 `loss` 与 `rel_error`；
- Poly manifest 如实记录运行目录 `git dirty=true`；实际运行的三个源文件已
  原样快照并分别用 SHA-256 固定，未用事后代码替代来源；
- CH 正式包与 CH 搜参包都保留各自的 checksum、manifest 和审核结果。

## 本次结构整理验证

- 在原 H20 的既有 PyTorch 环境中对整理后源码运行完整测试：`93 passed`；
- 14 条 warning 均为 PyTorch 对 complex module 的既有实验性提示；
- Python 编译、shell 语法、2767 份 current/search/double JSON 解析通过；
- Poly raw/delivery/analysis/source、CH raw/analysis、CH search raw/delivery、
  两份 double archive checksum 全部通过；
- CH 搜参交付表中原先缺失的 196 份逐点日志和两份主日志已从原服务器只读
  补回并逐项命中既有 `DELIVERY_SHA256SUMS`，没有重跑或改写实验；
- 活跃与归档结果中没有 raw smoke 目录，也没有 GitHub token 字符串；
- 源码/测试/文档的 `git diff --check` 通过；服务器原始 CSV 的 CRLF 行尾按
  checksum 原样保留，不为格式美化改写 raw 数据。

## 下一步科学工作

1. 在不改变本批正式数据的前提下分析 Poly o6 的优化失败；如要修改网络或
   激活，必须注册为新协议，不能覆盖当前结果。
2. 论文主表可以直接使用当前 CSV；Poly 实时 accuracy 图尚未生成。
3. 新图只能在开发服务器或 T4 环境由原始 JSON/history 生成，再把图与生成
   manifest 一并提交。Codex 工作区内置图片生成不属于允许流程。
4. 当前 `docs/paper/` 只保留新论文入口；旧 JSC 论文与图位于
   `docs/archive/paper_jsc_legacy/`。
