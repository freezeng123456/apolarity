# apolarity 夜间补实验结果（2026-08-05）

本报告记录补齐方法证据链的 P0-A/P0-B/P0-C 实验。实验源代码提交为
`e1bbf7e36892bd4efa317e95b250c540812fe00e`，运行环境为 Python 3.11.15、
PyTorch 2.5.1+cu121、NVIDIA H20。suite manifest 显示 42/42 个任务完成，
没有 OOM、NaN、异常退出或未完成任务。

## 主要结论

- 核心后端 1260 行 value/backward 基准全部通过；最大 value 绝对误差
  `6.883e-15`，最大 parameter-gradient 相对 L2 误差 `1.058e-13`。
- 同一 4D `(4,2)` 六阶 PINN、相同初值和 batch stream 的 500 步控制实验中，
  polarization-jet 和 Waring-jet 分别比 direct nested AD 快约 22.23× 和
  22.66×，参数轨迹差异处于机器精度量级。
- 1200 秒、5 seeds 的固定时长实验中，direct/polarization/Waring 的中位
  `ms/step` 为 151.537/8.065/7.231；中位 final L2 为
  `3.721e-3/8.183e-4/9.324e-4`。Waring 每步更快，但 final L2 不是每个 seed
  都最好，因此应同时报告 anytime/best-so-far 指标。
- P0-C 的 5-seed 强基线复核显示：Chirp `a=2` 上 complex-sinh 的中位 L2
  `1.285e-4`，比 vanilla tanh 的 `1.755e-3` 低约 13.7×；Maxwell `a=4`
  上 PWNN 的中位 L2 `1.276e-4`，比 complex-sinh 的 `1.360e-3` 低约
  10.7×。Complex-Sinh 不是所有结构化 PDE 的最优表示。
- 当前 `auto` 选择器在 180 个 cached 选择单元上的实测最快后端 mismatch 为
  49.4%，最大 regret 为 27.9%；float64 mismatch 为 80.3%，需要校准或改写。

## 实验设计与结果

### P0-A：核心后端

`d=8`、宽度 64、4 层 Sinh MLP，主精度 complex128，B=8/64，seeds 0–2，
比较 direct AD、polarization-jet、complex-Waring-jet 和 `auto`，同时测
derivative value 与 parameter backward；每个 cell 20 次 warmup、至少累计 3 s
或最多 100 次重复。覆盖 `(3)`、`(4)`、`(2,2)`、`(6)`、`(4,2)`、`(2,2,2)`、
`(8)`、`(4,4)`、`(2,2,2,2)` 等模式。

complex128、B=8、backward 的 cached median（ms）和 direct speedup：

| pattern | direct | polarization | Waring | direct/polar | direct/Waring |
|---|---:|---:|---:|---:|---:|
| `(3)` | 8.429 | 2.981 | 2.977 | 2.83× | 2.83× |
| `(4)` | 21.408 | 3.786 | 3.786 | 5.65× | 5.65× |
| `(6)` | 172.641 | 5.734 | 5.692 | 30.11× | 30.33× |
| `(4,2)` | 155.777 | 5.778 | 5.790 | 26.96× | 26.90× |
| `(8)` | 1423.982 | 8.472 | 8.451 | 168.09× | 168.50× |
| `(4,4)` | 1413.546 | 8.591 | 8.465 | 164.54× | 166.99× |

### P0-B：端到端 PINN

PDE 是 manufactured 的单项式 `∂x1^4∂x2^2u=f`，输入 4、complex128 Sinh MLP
宽度 32、4 层，`n_int=128`、`n_bc=64`，损失
`L_int+100L_bc+1e-6L_im`。三个后端使用完全相同的初始 state 和 batch stream。

固定步数（500 步，seed 0）：

| backend | ms/step | final L2 | peak alloc |
|---|---:|---:|---:|
| direct AD | 177.078 | 0.0192739 | 435.8 MiB |
| polarization-jet | 7.965 | 0.0192739 | 517.9 MiB |
| Waring-jet | 7.815 | 0.0192739 | 395.2 MiB |

后端间 step 500 参数快照最大绝对差为 `2.24e-16`，相对 L2 差为 `2.29e-16`。

固定时长（1200 s，5 seeds）的中位结果：

| backend | steps | ms/step | final L2 | PDE RMS | BC RMS |
|---|---:|---:|---:|---:|---:|
| direct AD | 7,893 | 151.537 | 3.721e-3 | 8.736e-3 | 1.476e-3 |
| polarization-jet | 123,559 | 8.065 | 8.183e-4 | 1.612e-3 | 2.666e-4 |
| Waring-jet | 133,420 | 7.231 | 9.324e-4 | 1.576e-3 | 4.024e-4 |

### P0-C：强基线

正式 20 个 run 使用 seeds 10–14、每 run 600 s、`n_int=4096`、`n_bc=512`、
独立 scrambled Sobol `2^16`，固定 paired collocation 和 cosine LR `1e-3→1e-4`。

| task/method | median L2 | mean ± sd | median peak |
|---|---:|---:|---:|
| Chirp a2 / vanilla tanh | 1.755e-3 | 1.649e-3 ± 1.431e-3 | 311.3 MiB |
| Chirp a2 / complex-sinh | 1.285e-4 | 3.445e-4 ± 3.792e-4 | 573.6 MiB |
| Maxwell a4 / PWNN | 1.276e-4 | 1.151e-4 ± 3.274e-5 | 105.5 MiB |
| Maxwell a4 / complex-sinh | 1.360e-3 | 1.658e-3 ± 6.114e-4 | 573.8 MiB |

## 仍需补的实验

1. **校准 `auto`：** 扩展 dtype `{float32,float64,complex64,complex128}`、
   batch `{1,8,64}`、hidden `{32,64,128}`、depth `{2,4,8}`、order `{2,4,6,8,10}`；
   每 cell 3 seeds、20 warmup、100 repeats/累计 3 s，拆分 direction construction、
   cached execution 和 backward。目标是将 mismatch 降到 5% 以下或明确给出 regret 上界。
2. **高阶端到端扩展：** 在 P0-B 继续加入 `(6)`、`(2,2,2)`、`(4,4)` 和两个
   mixed-derivative 的 PDE；`d=4/8`、H=32/64、depth=4、`n_int=128`、
   `n_bc=64`、5–10 seeds、1200 s，报告 time-to-`1e-2/1e-3` 与 anytime AUC。
3. **容量匹配：** 对 direct/polarization/Waring 使用同一 real/complex 表示，
   参数量匹配到 ±2%，seeds 0–9，600 s 和 1200 s 两个预算，避免把后端收益和
   表示/参数量收益混在一起。
4. **Poly 正式反例复核：** `poly_d2_o4`、`poly_d3_o4`，vanilla direct AD、
   Complex-Sinh Waring、polarization，预注册 boundary weights，seeds 10–19、
   `n_int=4096`、`n_bc=512`、1200 s，给出 paired effect size 和 bootstrap CI。
5. **外推 PDE：** variable-coefficient Helmholtz + 一个非线性高阶 PDE，加入
   Neumann/mixed boundary；每个问题 3 方法、5 seeds、H=64/depth=4、
   `n_int=8192`、`n_bc=1024`、1200 s。
6. **复现性：** 在第二张 GPU 或固定时钟配置上重跑 P0-A complex128 B8 和
   P0-B seed 0，同时记录 torch 2.4.1/2.5.1 差异。当前绝对耗时只在单台 H20 上验证。

原始归档、逐 run JSON/CSV/history、日志和 SHA-256 清单随本次实验交付；论文应将
“后端加速”“Complex-Sinh 表示优势”“结构化基线优势”分成三个独立结论。
