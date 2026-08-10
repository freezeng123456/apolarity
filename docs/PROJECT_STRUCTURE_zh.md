# apolarity 项目结构与证据边界

## 1. 核心代码

`src/apolarity/` 是可复用方法实现，不绑定某个 PDE：

- `waring.py`：复 Waring 方向与系数；
- `polarization.py`：实 polarization 方向；
- `taylor_jet.py`：Taylor jet 与网络传播；
- `operators.py`：`single_monomial_partial` 对外入口。

核心库只解决“一个固定 multi-index 的偏导”。Laplacian 幂、PDE 多项式和
边界组合在实验层展开，不应悄悄塞进核心 API。

## 2. 当前实验代码

```text
experiments/common/osc_common.py
    共享 MLP、采样、Laplacian 展开、direct AD / jet 分派

experiments/common/weight_search.py
    Poly common-Xavier 内核；WAR=complex64+sinh，real AD=float32+tanh

experiments/polyharmonic/
    当前方程与冻结协议说明；正式训练实现由 common/weight_search.py 提供

experiments/cahn_hilliard_2d/problem.py
    二维 (x,y,t) CH4/CH6、制造源、自然无通量边界和 loss

experiments/high_order_candidates/problem.py
    二/三维 ZK、二维动态板与 Swift–Hohenberg；共同候选筛选内核
```

二维 CH 的当前实现独立于 `weight_search.py` 中保留的旧一维周期 CH 兼容
分支。看到后者的 `PeriodicEmbeddedMLP` 不代表当前二维 CH 使用周期嵌入；
当前二维 CH 只使用仿射归一化 raw `(x,y,t)`。

## 3. Runner 与分析

| 文件 | 作用 | 默认结果层 |
|---|---|---|
| `scripts/run_poly_fixed_weight_formal.py` | Poly 3 tasks × 2 methods × 5 seeds × 1200 s | `outputs/current/` |
| `scripts/analyze_poly_fixed_weight_formal.py` | 校验 raw/checksum/source/history 并生成 CSV/报告 | 同一结果包的 `analysis/` |
| `scripts/run_weight_search.py` | Poly loss-weight 网格工具 | `outputs/search/` |
| `scripts/run_cahn2d_weight_search.py` | 二维 CH 98 vectors × 2 methods × 60 s | `outputs/search/` |
| `scripts/run_cahn2d_fixed_weight_formal.py` | 二维 CH 2 tasks × 2 methods × 5 seeds × 1200 s | `outputs/current/` |
| `scripts/analyze_cahn2d_fixed_weight_formal.py` | CH 正式结果统计与图表数据 | 同一结果包的 `analysis/` |
| `scripts/run_high_order_candidate_screen.py` | 4 个高阶 PDE 的 pilot，以及选定 task 的 formal | `outputs/search/` / `outputs/current/` |
| `scripts/analyze_high_order_candidate_results.py` | 同时验收 pilot/formal，生成逐点 CSV、统计与服务器图 | ZK 正式包的 `analysis/` |

正式 runner 逐 cell 原子写 JSON，失败证据进入 `attempts/`，`--resume` 只跳过
协议匹配且完整的 cell。文本日志末行同时保留 loss 与 rel_error。Smoke 使用
系统临时目录，只复制结论，不保留 raw bundle。

## 4. 证据层

```text
outputs/current/   五 seed、1200 秒正式结果
outputs/search/    60 秒权重搜索与排名
outputs/archive/   历史 double 协议
```

当前正式证据有三包：

1. Poly 30/30：每个方法 JSON 内嵌 241 个 history 点；实数基线为 tanh；
2. CH2D 20/20：CH4/CH6 固定 `(lambda_ic,lambda_bc)=(1,10)`；实数基线为
   sinh；
3. ZK2D 10/10：三阶周期初值问题，WAR=complex64+sinh、实数 AD=float32+tanh，
   5 seeds × 1200 秒；WAR 5/5 seed 更优。

CH 权重搜索 196/196 与高阶 PDE pilot 24/24 位于 `outputs/search/`。后者仅作
候选选择；正式 ZK 统计完全来自独立的 1200 秒结果。旧 JSC（包括旧 Poly family
入口）、Chirp、Maxwell 和其他
PDE 结果位于 `experiments/archived/`。不同协议、dtype、输入表示或激活的结果
不能拼接成一个方法均值。

## 5. Poly 来源例外

Poly 结果 manifest 如实记录 `git dirty=true`。为了让这批数据仍可复核，
`provenance/source_snapshot/` 保留服务器真正使用的三个源文件，固定 SHA-256
并与当前整理后的 runner 区分。数据没有被重写；原始 `SHA256SUMS` 保持原样，
`DELIVERY_SHA256SUMS` 再覆盖分析和 provenance。

## 6. 文档与归档

- `docs/paper/`：当前论文入口，只引用 current/search 数据；
- `docs/beamer/apolarity_idea.*`：方法原理说明，不承载当前实验结论；
- `docs/archive/paper_jsc_legacy/`：旧 JSC 论文、表和图；
- `docs/archive/beamer_legacy/`、`canvas_legacy/`：旧展示材料；
- `docs/archive/SMOKE_CONCLUSIONS_zh.md`：删除 raw smoke 后保留的门禁结论；
- `docs/archive/*.md`：历史 baseline、权重与实现审计。

## 7. 图像生成约定

原始 JSON/history 是唯一事实来源。新增论文图必须在开发服务器或 T4 环境用
可审计脚本生成，并同时提交曲线 CSV 与生成 manifest。不得使用 Codex 工作区
内置图片生成能力。当前 Poly 整理只生成 JSON/CSV/Markdown；CH 与 ZK 图由
服务器 Python/Matplotlib 生成，并与逐点曲线 CSV、生成脚本和校验和一同提交。

## 8. 推荐阅读顺序

1. 根目录 `README.md`：项目与当前结论；
2. `experiments/<family>/README.md`：方程和冻结设置；
3. `outputs/README.md`：数据层级；
4. 当前结果包的 `README.md` 与 `analysis/REPORT_zh.md`；
5. 需要复现时再读对应 runner 与 manifest；
6. 只有在追溯旧决定时进入 archive。
