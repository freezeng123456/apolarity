# HO-02 MPFC reference 结论

服务器端使用 Fourier pseudospectral + two-thirds dealiasing + linearly implicit
IMEX 更新，独立于 PINN 残差实现。三层嵌套分辨率为
`(N,dt)=(32,5e-4),(64,2.5e-4),(128,1.25e-4)`，固定 eval seed `68421`，采样
1024 个网格对齐时空点。

- coarse→medium relative difference：`7.13083e-4`
- medium→fine relative difference：`3.64676e-4`
- convergence gate：passed（阈值 `2e-3`）
- mass error：三层均 `0`
- pseudo-energy：三层 diagnostic times 上均单调不增
- fine level 运行时间：约 `1.14 s`（H20）

`reference.npz`、`reference_report.json` 和 `REFERENCE_SHA256` 是当前 reference
候选。它只放行共享权重搜索，不放行 3-seed pilot 或 5-seed formal；下一步仍需先
固定搜索协议并检查所有候选的 loss/rel_error/history 有限性。

