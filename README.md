# apolarity

`apolarity` 是一个面向神经网络高阶偏导的确定性后端：先用单项式的 Waring
分解得到复方向，再以 Taylor-mode 自动微分计算一个固定 multi-index 的精确
偏导。当前论文实验只保留两个问题族：二维 Polyharmonic 与二维
Cahn–Hilliard；旧 JSC、Chirp、Maxwell、double 协议和其他诊断实验全部进入
archive，不再与当前结果混用。

## 当前实验口径

| 问题族 | 任务 | WAR | 实数 autodiff | 初始化与输入 |
|---|---|---|---|---|
| Polyharmonic | `d2/o2`, `d2/o4`, `d2/o6` | complex64 + sinh + Waring jet | float32 + **tanh** + direct AD | common Xavier；原始 `(x,y)`；无频率初始化 |
| Cahn–Hilliard 2D | CH4, CH6 | complex64 + sinh + Waring jet | float32 + sinh + direct AD | common Xavier；仿射 `(x,y,t)`；无三角特征 |

两条协议的实数激活不同，这是各自已经运行并冻结的设置：Poly 的实数基线
继续使用 `tanh`，本次整理没有把它改成 `sinh`。两种方法按相同字面层形状
和相同墙钟预算比较；WAR 的复参数含两倍实自由度，因此不能表述成等参数量
对比。

## 已验收结果

| 结果包 | 完整性 | 主要结论 |
|---|---:|---|
| `outputs/current/polyharmonic-common-xavier-fp32-formal-v1/` | 30/30；5 seeds × 3 tasks × 2 methods | o2：WAR 5/5 seed 更优；o4：AD 3/5 更优；o6 两者均约 1，当前设置失败 |
| `outputs/current/cahn-hilliard-2d-fixed-1-10-formal-v1/` | 20/20；5 seeds × 2 tasks × 2 methods | WAR 在 CH4/CH6 均为 5/5 seed 更优 |
| `outputs/search/cahn-hilliard-2d-weight-search-v1/` | 196/196；98 vectors × 2 methods | 完整二维权重搜索与 Top-10 排名 |

Poly 完整包含 30 份原始方法 JSON、30 份日志、15 份 seed 配置，以及
`summary`、`manifest`、rankings 和校验和。实时 accuracy/loss history 直接嵌在
每份方法 JSON 的 `history` 字段中，共 7230 个数据点。详见
`outputs/current/polyharmonic-common-xavier-fp32-formal-v1/analysis/REPORT_zh.md`。

## 项目结构

```text
src/apolarity/                         核心 Waring / Taylor-jet 库
experiments/
  common/                              当前共享模型、导数与 Poly 训练内核
  polyharmonic/                        当前 Poly 问题说明；旧 family runner 仅作诊断
  cahn_hilliard_2d/                    当前二维 CH 方程、边界与 loss 实现
  archived/                            JSC、Chirp、Maxwell、历史 family 与 runner
scripts/
  run_poly_fixed_weight_formal.py      Poly 30-cell 正式 runner（实数基线为 tanh）
  analyze_poly_fixed_weight_formal.py  Poly 数据验收与文本/CSV 汇总
  run_cahn2d_weight_search.py          二维 CH 196-cell 搜参 runner
  run_cahn2d_fixed_weight_formal.py    二维 CH 20-cell 正式 runner
  analyze_cahn2d_fixed_weight_formal.py
outputs/
  current/                             当前正式证据
  search/                              当前搜参证据
  archive/double/                      历史 float64/complex128 结果
docs/
  PROJECT_STRUCTURE_zh.md              完整目录、数据流与归档边界
  paper/                               当前论文入口（尚未重建正文）
  archive/                             历史论文、图、审计与 smoke 结论
tests/                                 数学、模型、runner 与协议测试
```

完整的“什么是当前证据、什么只能作为历史参考”说明见
`docs/PROJECT_STRUCTURE_zh.md` 和 `outputs/README.md`。

## 结果验收

Poly 正式包可以在不依赖 PyTorch、也不生成图片的情况下复核：

```bash
python scripts/analyze_poly_fixed_weight_formal.py \
  outputs/current/polyharmonic-common-xavier-fp32-formal-v1

cd outputs/current/polyharmonic-common-xavier-fp32-formal-v1
shasum -a 256 -c SHA256SUMS
shasum -a 256 -c DELIVERY_SHA256SUMS
```

CH 正式包可在任意环境只读校验 raw；会生成图片的分析脚本仅允许在开发
服务器或 T4 环境执行：

```bash
cd outputs/current/cahn-hilliard-2d-fixed-1-10-formal-v1
shasum -a 256 -c SHA256SUMS

# 仅开发服务器/T4：
python /path/to/apolarity/scripts/analyze_cahn2d_fixed_weight_formal.py .
```

Smoke 只做 CUDA、有限值和数据管线门禁。新的 runner 使用临时目录并只保留
结论；raw smoke 不进入 Git。历史 smoke 的原始文件已删除，审计结论保存在
`docs/archive/SMOKE_CONCLUSIONS_zh.md`。

## Python API

```python
from apolarity import single_monomial_partial

deriv = single_monomial_partial(model, x, alpha, backend="auto")
```

`alpha` 是零基坐标的展开 multi-index，例如 `(0, 0, 1)` 表示对坐标 0
求两次、对坐标 1 求一次。输入形状为 `(batch, d)`，模型输出必须为
`(batch, 1)`。库本身支持相应实数/复数 dtype；当前论文实验固定使用表中的
float32/complex64 协议。

## 图与论文

仓库以原始 JSON/CSV/history 为事实来源。按项目约定，新增论文图只能在开发
服务器或 T4 环境生成，不能使用 Codex 工作区内置图片生成能力。当前 Poly
交付没有生成新图；CH 目录中的现有图来自服务器分析环境。旧论文和旧图已移到
`docs/archive/`，当前论文入口见 `docs/paper/README.md`。
