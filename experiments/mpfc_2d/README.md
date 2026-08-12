# HO-02：二维六阶 Modified Phase-Field Crystal

该目录实现 MPFC 的第一版数学与 PINN 门禁接口。方程保留直接六阶空间残差：

\[
\beta\phi_{tt}+\phi_t-M\Delta\left(\Delta^2\phi+2\Delta\phi+
(1-\varepsilon)\phi+\phi^3\right)=0.
\]

默认 `M=1`、`beta=0.1`、`epsilon=0.25`，周期区域为
`[0,2pi]^2 x [0,1]`，初值为
`0.1 + 0.15 cos(x)cos(y) + 0.05 cos(2x)cos(y)`，初始速度为零。

网络输入只有 affine-normalized raw `(x,y,t)`；WAR 使用 complex64/sinh/Waring
jet，实数基线使用 float32/tanh/direct autodiff，共同 Xavier、hidden=128、depth=4。
周期边界显式匹配法向导数 0--5 阶。当前提交只包含方程、残差和 finite-gradient
门禁；reference solver、权重搜索和长跑必须在门禁通过后单独冻结。

