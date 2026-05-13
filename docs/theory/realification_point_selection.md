# 通用 realification 的理论方案：从 complex direction 到 real Taylor-jet directions

> 日期：2026-05-14  
> 目标：针对单项 operator \(\partial^\alpha u\)，研究如何把 complex Waring directions 转成全实方向，避免 complex PyTorch / complex Taylor-jet 的工程复杂性。  
> 范围：只讨论单项 mixed partial，不讨论可 contract 的算子求和。

---

## 1. 背景与问题

复数域 monomial Waring 给出单项导数的 rank-optimal complex 方向公式：

\[
\partial^\alpha u(x)=\sum_{r=1}^{R_\mathbb C}c_r T_p(x;v_r),
\qquad
T_p(x;v)=\frac{1}{p!}D^p u(x)[v,\ldots,v].
\]

其中 \(c_r,v_r\in\mathbb C\)。

数学上这是漂亮的：方向数达到 complex Waring rank。工程上问题是：

1. complex MLP / complex `tanh` 常数大；
2. PyTorch complex module 仍不够成熟；
3. complex backward 和 real-parameter gradient 更难维护；
4. 共轭项相消可能导致 conditioning 问题。

所以我们希望把每个 complex formula 转成实方向公式：

\[
\partial^\alpha u(x)=\sum_{j=1}^{M}\tilde c_j T_p(x;\tilde v_j),
\qquad
\tilde c_j,\tilde v_j\in\mathbb R.
\]

这个过程称为 realification。

---

## 2. 文献调研要点

### 2.1 Complex monomial Waring rank

Carlini--Catalisano--Geramita, *The solution to the Waring problem for monomials and the sum of coprime monomials*, J. Algebra, 2012.

对单项式

\[
M=x_0^{a_0}\cdots x_n^{a_n},
\qquad
1\le a_0\le\cdots\le a_n,
\]

复数域 rank 为

\[
R_\mathbb C(M)=\prod_{i=1}^{n}(a_i+1)
=\frac{\prod_i(a_i+1)}{a_0+1}.
\]

### 2.2 Real rank of monomials

Carlini--Kummer--Oneto--Ventura, *On the real rank of monomials*, Math. Z., 2017.

主要结论：

\[
R_\mathbb R(M)=R_\mathbb C(M)
\quad\Longleftrightarrow\quad
 a_0=1.
\]

并给出一般上界：

\[
R_\mathbb R(M)\le \prod_{i=1}^{n}(a_i+a_0).
\]

构造来自 apolar ideal 中实分裂多项式定义的实点集。

### 2.3 Han--Moon 新上界

Han--Moon, *A New Bound for the Waring Rank of Monomials*, SIAM J. Appl. Algebra Geom., 2022.

给出更好的 real/rational 上界：

\[
R_\mathbb R(M),R_\mathbb Q(M)
\le
(a_0+a_1)\prod_{i=2}^{n}(a_i+1).
\]

该界在二元情形和最小指数为 1 的情形下是 sharp。

### 2.4 插值选点文献

Chebyshev interpolation 的经典原则：对区间 \([-1,1]\)，选择 Chebyshev zeros

\[
t_j=\cos\left(\frac{2j+1}{2(n+1)}\pi\right),
\qquad j=0,\ldots,n,
\]

使节点多项式

\[
\omega(t)=\prod_{j=0}^{n}(t-t_j)
\]

在 \([-1,1]\) 上的最大范数最小。

这主要优化 interpolation 区间内误差；我们的 realification 是从实轴节点外推到虚点 \(i\)，属于 extrapolation，所以 Chebyshev 不是严格最优，但它比等距节点稳定，是默认候选。

---

## 3. 通用 realification 的基本定理

考虑一个 complex conjugate pair：

\[
cT_p(x;v)+\bar cT_p(x;\bar v),
\qquad
v=a+ib,
\quad a,b\in\mathbb R^d.
\]

因为 \(D^p u(x)\) 对实网络是实系数对称多线性型，所以

\[
T_p(x;\bar v)=\overline{T_p(x;v)}.
\]

因此 conjugate pair 的贡献是实数：

\[
cT_p(x;v)+\bar cT_p(x;\bar v)
=2\operatorname{Re}\{cT_p(x;a+ib)\}.
\]

定义一元实多项式

\[
h(t):=T_p(x;a+tb).
\]

由于 \(T_p\) 对方向是 p 次齐次多项式，\(h(t)\) 是 degree \(\le p\) 的实系数多项式。

我们需要的是

\[
h(i)=T_p(x;a+ib).
\]

取任意 \(p+1\) 个互异实节点 \(t_0,\ldots,t_p\)。令 \(\ell_j(z)\) 是 Lagrange basis：

\[
\ell_j(z)=\prod_{m\ne j}\frac{z-t_m}{t_j-t_m}.
\]

对任意 degree \(\le p\) 多项式都有精确插值：

\[
h(i)=\sum_{j=0}^{p}\ell_j(i)h(t_j).
\]

于是

\[
\boxed{
 c h(i)+\bar c h(-i)
=\sum_{j=0}^{p} 2\operatorname{Re}\{c\ell_j(i)\}\,h(t_j).
}
\]

换回方向：

\[
\boxed{
 cT_p(x;a+ib)+\bar cT_p(x;a-ib)
=\sum_{j=0}^{p}w_jT_p(x;a+t_jb),
}
\]

其中

\[
w_j=2\operatorname{Re}\{c\ell_j(i)\}\in\mathbb R,
\qquad
a+t_jb\in\mathbb R^d.
\]

这就是通用 realification 公式。

---

## 4. 选点问题

### 4.1 任意互异实点都精确，但数值稳定性不同

理论上，只要 \(t_j\) 互异，上式对所有 degree \(\le p\) 多项式精确。

但是数值误差会被放大，主要受两个因素影响：

1. 插值权重放大：

\[
\Lambda(i):=\sum_{j=0}^{p}|\ell_j(i)|.
\]

2. 方向范数放大：

\[
\|a+t_jb\|^p.
\]

因为 \(T_p\) 是 p 次齐次，方向范数越大，数值规模越大。

所以选点应优化类似目标：

\[
\boxed{
\mathcal A(t_0,\ldots,t_p)
=\sum_{j=0}^{p}|2\operatorname{Re}(c\ell_j(i))|\,\|a+t_jb\|^p
}
\]

或不含 \(c\) 的 backend-level 目标：

\[
\boxed{
\mathcal A_0(t_0,\ldots,t_p)
=\sum_{j=0}^{p}|\ell_j(i)|\,\|a+t_jb\|^p.
}
\]

### 4.2 对称节点是必须优先考虑的结构

目标点是 \(i")，同时 conjugate pair 涉及 \(i,-i\)。因此实节点应取对称集合：

\[
\{t_j\}= -\{t_j\}.
\]

例如：

- p 偶：\(p+1\) 为奇数，节点包含 0；
- p 奇：\(p+1\) 为偶数，节点成 \(\pm\) 对。

好处：

1. 权重自动具有共轭/奇偶结构；
2. real weights 更稳定；
3. 便于 direction merge；
4. 对 pure imaginary/real special case 有更多 cancellation。

---

## 5. 三类可用选点方案

### 5.1 方案 A：Chebyshev zeros，默认稳定方案

取

\[
t_j=\rho\cos\left(\frac{2j+1}{2(p+1)}\pi\right),
\qquad j=0,\ldots,p.
\]

其中 \(\rho>0\) 是 scale。

优点：

- 对称；
- 避免等距节点 Runge 型病态；
- barycentric interpolation 权重可稳定计算；
- 实现简单；
- 对 p\(\le\)6/8 已足够。

缺点：

- Chebyshev 是区间内插值最优，不是外推到 \(i\) 的严格最优；
- \(\rho\) 需要调。

推荐作为第一版通用 realification fallback。

### 5.2 方案 B：直接优化 weighted amplification

对每个 conjugate pair \(v=a+ib\)，求解一维优化：

\[
\min_{t_0<\cdots<t_p,\; t_j=-t_{p-j}}
\sum_j |\ell_j(i)|\,\|a+t_jb\|^p.
\]

为了降低复杂度，可只优化 scale \(\rho\)：

\[
t_j=\rho\tau_j,
\]

其中 \(\tau_j\) 是 Chebyshev zeros，然后做

\[
\boxed{
\rho^*=\arg\min_{\rho>0}
\sum_j |\ell_j(i/\rho)|\,\|a+\rho\tau_j b\|^p.
}
\]

解释：令

\[
h_\rho(s)=T_p(x;a+\rho sb),
\]

则

\[
h_\rho(i/\rho)=T_p(x;a+ib).
\]

当 \(\rho\) 大，target \(i/\rho\) 离实区间更近，插值外推更稳；但实方向 \(a+\rho\tau_jb\) 范数变大。这个 tradeoff 正是 \(\rho\) 优化要解决的。

优点：

- 更贴合我们的数值目标；
- 只是一维搜索，易实现；
- 可离线对 pattern 预计算。

缺点：

- 不是 rank-optimal，只是稳定 realification；
- 每个 conjugate pair 可能有不同 \(a,b\)。

### 5.3 方案 C：Leja / max-product 节点

Leja points 通过递推最大化节点之间的乘积距离，常用于稳定插值和嵌套节点。

可考虑在实区间 \([-\rho,\rho]\) 上选 Leja 点，或针对目标点 \(i/\rho\) 设计 weighted Leja：

\[
t_{k+1}=\arg\max_{t\in[-\rho,\rho]}
|t-i/\rho|\prod_{m=0}^{k}|t-t_m|.
\]

优点：

- 节点可嵌套；
- 适合逐阶扩展 p；
- 可数值优化。

缺点：

- 理论和实现比 Chebyshev 复杂；
- 对 p\(\le\)6 暂时没必要优先。

---

## 6. 推荐的通用 realification 策略

### 6.1 第一版策略

对每个 conjugate pair \((c,v=a+ib)\)：

1. 取 Chebyshev zeros \(\tau_j\in[-1,1]\)，\(j=0,\ldots,p\)。
2. 一维搜索 scale \(\rho\)：

\[
\rho^*=\arg\min_{\rho\in[\rho_{min},\rho_{max}]}
\sum_j |\ell_j(i/\rho)|\,\|a+\rho\tau_jb\|^p.
\]

3. 实方向：

\[
\tilde v_j=a+\rho^*\tau_jb.
\]

4. 实权重：

\[
\tilde c_j=2\operatorname{Re}\{c\ell_j(i/\rho^*)\}.
\]

5. merge 相同/近似相同方向。

### 6.2 备选默认 scale

如果暂时不做优化，可取

\[
\rho=1.
\]

但更好的 heuristic 是

\[
\rho_0=\max\left(1,\frac{\|a\|}{\|b\|+10^{-12}}\right).
\]

或者直接在 log grid 上搜索：

\[
\rho\in\{2^{-4},2^{-3.5},\ldots,2^4\}.
\]

### 6.3 为什么不推荐等距节点

等距节点在插值和外推中通常具有更大的 Lebesgue 放大，尤其 p 稍大时会产生明显权重爆炸。

我们的 p 目前多为 3--6，等距未必立刻崩，但没有理由优先用它。Chebyshev + scale search 是更稳的默认选择。

---

## 7. 方向数分析

对一个 conjugate pair，通用插值 realification 使用最多 \(p+1\) 个实方向。

如果 complex Waring 有 \(R_\mathbb C\) 个方向，其中非实方向成 conjugate pair，粗略上界是

\[
M\le R_{real}+\frac{p+1}{2}R_{nonreal}.
\]

这可能比 polarization 还多，所以它不应替代已知 real-rank 公式。

推荐层级：

1. known optimal real formula；
2. Han--Moon / apolar constructive real formula；
3. complex-to-real interpolation fallback；
4. polarization fallback。

其中 complex-to-real interpolation 的角色是：

> 当我们有很好的 complex formula，但还没有对应 real formula 时，提供一个全实、精确、可实现的 fallback。

---

## 8. 对我们项目的结论

### 8.1 可以规避 complex 计算

通用公式已经说明：任意 conjugate pair 都可以通过实节点插值精确转成实方向。核心不是“能不能”，而是“选点如何稳定”。

### 8.2 推荐默认选点

第一版实现建议：

\[
\boxed{
\text{symmetric Chebyshev zeros} + \text{scale }\rho\text{ log-grid search}
}
\]

优化目标：

\[
\boxed{
\sum_j |\ell_j(i/\rho)|\,\|a+\rho\tau_jb\|^p.
}
\]

这是理论上精确、工程上可控、实现简单的方案。

### 8.3 但它不是最终最优 real Waring

该方案可能增加方向数，不能替代 real Waring rank 构造。最终 backend 应是 layered：

```text
pure power
 -> known real rank formula, e.g. min exponent = 1 / binary monomial
 -> Han--Moon apolar real construction
 -> complex-to-real Chebyshev realification
 -> polarization fallback
```

---

## 9. 下一步实现建议

1. 实现 `realify_conjugate_pair(c, a, b, p, node_rule="chebyshev", scale="grid")`。
2. 实现 barycentric Lagrange weights at complex target \(i/\rho\)。
3. 对 p=3,4,6 做 numerical unit tests：随机实 symmetric tensor / MLP TaylorJet，对比 complex formula。
4. 将 `waring_complex` 输出按 conjugate pair 分组并 realify。
5. benchmark 对比：
   - `complex_waring_jet`
   - `realified_complex_cheb_jet`
   - `polarization_jet`
   - `direct_autodiff`

重点看：direction count、coefficient norm、value error、time、memory。
