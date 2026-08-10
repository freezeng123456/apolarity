# Current formal evidence

- `polyharmonic-common-xavier-fp32-formal-v1/`：30/30 Poly 正式结果；实数 AD
  固定为 tanh。
- `cahn-hilliard-2d-fixed-1-10-formal-v1/`：20/20 二维 CH 正式结果。
- `high-order-zk2d-formal-v1/`：10/10 二维三阶 ZK 正式结果；WAR 与实数 AD
  各 5 seeds × 1200 秒。

每个结果包内的 README/analysis 报告给出协议、完整性和统计口径。只有通过
checksum 与 audit 的 raw 数据可以作为论文数值来源。
