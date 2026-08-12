# HO-02 MPFC 参考解结论

参考解由独立的 Fourier 伪谱 / 2/3 去混叠 / 线性隐式 IMEX 求解器生成，嵌套层级为 `(32, dt=5e-4)`, `(64, dt=2.5e-4)`, `(128, dt=1.25e-4)`，固定 `eval_seed=68421`，最终绑定 4096 个评估点。

- coarse→medium 相对差：`7.2818069e-4`
- medium→fine 相对差：`3.7198135e-4`
- 收敛门限：`2e-3`，通过
- 质量误差最大绝对值：`0`
- 伪能量：各层诊断时间点单调不增
- `reference.npz` SHA256：`8d6a81b8d5a00cbff56287be16bdad273d81dabaf4b6faaa43f863c9b72a77e5`

该参考解仅作为搜参/正式实验的独立评估目标，不参与训练损失，也不改变网络输入（仍只有 affine-normalized raw `(x,y,t)`）。
