# Current search evidence

本层保存参数搜索与候选筛选，不是五 seed 正式精度结论：

- `cahn-hilliard-2d-weight-search-v1/`：二维 CH 的 49×2 候选网格；
- `high-order-candidate-pilot-v1/`：四个高阶 PDE、两种方法、3 seeds × 600 秒。

每个候选都同时运行 WAR 和实数 AD。高阶 pilot 选择出的二维三阶 ZK 已在
`outputs/current/high-order-zk2d-formal-v1/` 独立完成 5-seed 正式复跑；pilot
数据没有混入正式统计。
