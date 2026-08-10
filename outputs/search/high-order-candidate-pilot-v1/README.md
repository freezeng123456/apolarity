# High-order PDE candidate pilot

这是候选筛选证据，不是五 seed 正式统计。冻结协议为 4 tasks × 2 methods ×
3 seeds × 600 秒，共 24/24 cell，单张 H20 严格串行，0 失败、0 重试。

- WAR：complex64、sinh、Waring jet；
- baseline：float32、tanh、direct autodiff；
- 两者 hidden=128、depth=4、共同 Xavier；
- 仅仿射归一化 raw coordinates，无三角输入和频率初始化；
- `n_int=2048,n_ic=512,n_bc=1024,n_eval=16384,history_eval_n=2048`。

二维 ZK 与二维动态板通过预设门槛；三维 ZK 与 Swift–Hohenberg 两种方法均约
为 `rel_error=1`。动态板筛选误差最低，但预先冻结的规则优先选择与已有
Poly/CH 不同的可训练 ZK，因此正式复跑选择 `zk_2d_o3`。

`SMOKE_CONCLUSION.json` 只保留门禁结论，raw smoke 已删除。`SHA256SUMS`
覆盖服务器 raw pilot；`DELIVERY_SHA256SUMS` 额外覆盖本 README 和 smoke 结论。
