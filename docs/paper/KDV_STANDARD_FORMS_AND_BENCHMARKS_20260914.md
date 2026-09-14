# 标准 KdV 写法及相关论文的算例设置

核对日期：2026-09-14。范围：直接核对原论文、作者网页及 STDE 固定版本源码，并与当前主稿比较。本次为调研记录，不修改论文、算例或实验。下面的缩放与残差检验为本次计算，不标称独立审稿。

## 一维标准形式与系数

经典归一化形式之一是

\[
u_t+6uu_x+u_{xxx}=0.
\]

京都大学 Takasaki 的作者教学页明确采用此形式；Pu–Chen 的数值研究也在式 (3.1) 中采用它。[Takasaki](https://www2.yukawa.kyoto-u.ac.jp/~kanehisa.takasaki/soliton-lab/gallery/solitons/kdv-e.html)，[Pu–Chen，式 (3.1)](https://arxiv.org/html/2401.04982v1#S3.SS1)。

PDE 文献也使用带常系数的归一化；Tao 的作者资料页将其写为 $u_t+u_{xxx}+P(u)_x=0$，并指出常数系数通常可通过缩放消去。因而非线性系数是否为 6 并不决定方程是否属于 KdV。[Tao](https://www.math.ucla.edu/~tao/Dispersive/kdv.html)。

| 文献 | 实际模型写法 | 本次确认的含义 |
|---|---|---|
| Pu–Chen, *Lax pairs informed neural networks solving integrable systems*, 2024 | $u_t+6uu_x+u_{xxx}=0$，式 (3.1) | 经典归一化的一维 KdV |
| Raissi–Perdikaris–Karniadakis, *Physics Informed Deep Learning (Part II)*, 2017 | $u_t+\lambda_1uu_x+\lambda_2u_{xxx}=0$，式 (20)；Fig. 5 的参考系数为 $(1,0.0025)$ | 与本稿 4.2 的 PDE 相同 |
| STDE, arXiv:2412.00088v2 | $u_t+uu_x+\alpha u_{xxx}=0$，式 (71) | 普通 KdV 加 gPINN 损失；源码采用 $\alpha=0.0025$ |

来源：[Pu–Chen](https://arxiv.org/html/2401.04982v1)，[Raissi 等原论文，式 (20) 与 Fig. 5](https://arxiv.org/pdf/1711.10566)，[STDE 附录 I.4.1](https://arxiv.org/html/2412.00088v2#A9.SS4.SSS1)，[STDE 固定版本 equations.py](https://raw.githubusercontent.com/sail-sg/stde/fae88663b1d2f1b0666d4d250c38a3c34b852ac0/stde/equations.py)。源码依据是 `highord1d_res` 中 `eq == 2` 的分支。

阅读 Raissi 预印本时需交叉核对：紧邻式 (20) 的 $\mathcal N$ 表达式将色散项写成负号，与式 (20) 和 Fig. 5 的参考 PDE 不一致。本记录依据模型式和参考 PDE，不将这处不一致另解释成一个不同的实验模型。该作者网页也保留了同一处符号不一致。[作者网页 KdV 小节](https://maziarraissi.github.io/PINNs/)。

## 本稿一维问题的直接核验

当前主稿式 (23) 是

\[
u_t+uu_x+0.0025u_{xxx}=0.
\]

它是标准一维 KdV 的常系数形式。令 $\epsilon=0.05$，并定义

\[
X=x/\epsilon,\qquad T=t/\epsilon,\qquad
v(X,T)=u(\epsilon X,\epsilon T)/6.
\]

代入后，三个项依次带系数 $6/\epsilon,36/\epsilon,6/\epsilon$；除去公因子，得到 $v_T+6vv_X+v_{XXX}=0$。这是本次直接推导，不依赖算例引用。区域、时间区间和初边值数据必须随同变换，不能只改 PDE 中的两个系数。

对当前解析解

\[
u_\star(x,t)=\tfrac34\operatorname{sech}^2\bigl(5(x+\tfrac12-\tfrac14t)\bigr),
\]

本次采用 $w=\tanh(5(x+\tfrac12-\tfrac14t))$ 将导数转为多项式计算，精确残差为零。这验证它与当前 PDE 匹配，不验证其他边界问题的适定性或训练性能。

## 文献如何选择初值和边界

Raissi 等用 $u(x,0)=\cos(\pi x)$ 及周期边界生成参考解，采用 Fourier 离散和时间积分，再从两个时间截面抽取数据进行参数识别。因此，这是一项周期参考数据上的逆问题，不能直接等同于本稿的前向孤立波 PINN 问题。[原论文式 (20) 后的设置，印刷页 15–17](https://arxiv.org/pdf/1711.10566)。

Pu–Chen 的一维 KdV 算例使用孤立波 $2\operatorname{sech}^2(x-4t)$，在有限时空区域上给定它的初值与左右端点值，见式 (3.3)。这说明解析孤立波配边界迹也是实际使用的 PINN 算例形式；该文具体列出的约束并不与本稿逐项相同。[Pu–Chen，式 (3.3)](https://arxiv.org/html/2401.04982v1#S3.SS1)。

有限区间 KdV 的边界分析研究中，常见的一组边界迹是 $u(0,t),u(L,t),u_x(L,t)$，即两个端点的值和右端的一阶导数。Capistrano-Filho–Sun–Zhang 的式 (1.7) 明确列出了这种形式；其 PDE 另有线性平流项 $u_x$。这里引用的是边界结构的文献先例，不能把该定理直接套到本稿二维方程上。[2018 年发表版，式 (1.7)](https://sites.ufpe.br/wp-content/uploads/sites/132/2022/03/GBVP_KdV_Final.pdf)。

因此，“标准 KdV”没有规定唯一的初值和边界。周期余弦数据、全空间孤立波及有限区间解析边界迹应当作为不同问题设置分别说明。若改成周期余弦算例，需要相应的参考解与训练结果；不能仅替换边界文字后保留原孤立波结果。

## 二维方程的来源与命名

本稿式 (28) 是

\[
u_{ty}+u_{xxxy}+3(u_yu_x)_x-u_{xx}+2u_{yy}=0.
\]

Wazwaz (2020) 的式 (8) 为

\[
u_{ty}+u_{xxxy}+\alpha(u_yu_x)_x+\beta u_{xx}+\gamma u_{yy}=0.
\]

取 $(\alpha,\beta,\gamma)=(3,-1,2)$ 即得到当前式 (28)。原文将它称为新的 $(2+1)$ 维 KdV 方程，并从 $v=u_y$ 的势变量关系引入。Pu–Chen (2024) 式 (3.10) 和 STDE 式 (20) 采用了当前的具体系数。[Wazwaz 原论文，式 (8)–(9)](https://s3.cern.ch/inspire-prod-files-1/136472d8763496230daa8b6b72fb219a)，[Pu–Chen 式 (3.10)](https://arxiv.org/html/2401.04982v1#S3.SS1)，[STDE 式 (20)](https://arxiv.org/html/2412.00088v2#S4.SS3.SSS2)。

因此，严谨的称呼是“一个 $(2+1)$ 维 KdV 扩展”或“所引用的二维 KdV 型方程”，不能只凭名称认定它是一维标准 KdV 的唯一二维版本。Pu–Chen 还单独列出 KP 方程作为另一类二维模型，式 (3.7)，它与式 (3.10) 不同。[Pu–Chen](https://arxiv.org/html/2401.04982v1#S3.SS1)。

## g-KdV 与 gKdV

STDE 的附录标题将 gradient-enhanced KdV 缩写为 g-KdV；增强的是训练损失，式 (71) 的非线性仍为 $uu_x$。在 PDE 文献中，gKdV 通常指 generalized KdV，例如 $u_t+u_{xxx}+\partial_x P(u)=0$ 的非线性推广。因此，本稿保持 “Gradient-enhanced PINN for the KdV equation” 这个标题更清楚。[STDE](https://arxiv.org/html/2412.00088v2#A9.SS4.SSS1)，[Tao](https://www.math.ucla.edu/~tao/Dispersive/kdv.html)。

## 对本稿的建议

1. 4.2 保留现有 PDE 和孤立波基准；色散系数 0.0025 有明确 PINN 与 STDE 依据，无需为了“标准化”改成 1。
2. 区分 PDE 系数的来源与本工作选择的解析解、初边值数据；不要把全部设置归因于 STDE 或 Raissi。
3. 4.3 的命名指向具体的 $(2+1)$ 维扩展。若以后修订引用，应优先补充 Wazwaz 原始模型或 Pu–Chen 对该模型的研究，STDE 用于说明导数计算的使用场景。
4. 本次未修改主稿、PDF、打包文件或训练实现；未启动实验。
