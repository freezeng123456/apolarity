# 单项 operator 的精确高阶导数计算：方法综述、现有结果与下一步机会

> 日期：2026-05-13  
> 范围：只讨论 **单项 differential operator**，即单个 mixed partial
>
> \[
> \partial^\alpha u(x),\qquad |\alpha|=p.
> \]
>
> 明确不讨论 \(\Delta^k\)、trace contraction、operator sum compression、Hermite/Wick contract、STDE-style collected estimator 等“可做 contract 的算子求和”。如果出现多个单项求和，只作为现有脚本 sanity case，不作为本文方向。

---

## 0. 目标重新定义

当前目标不是估计

\[
\mathbb E_Z[\cdots]
\]

也不是利用算子结构把很多项 contract 掉，而是针对一个给定单项

\[
\partial^\alpha u(x)=\partial_{i_1}\partial_{i_2}\cdots\partial_{i_p}u(x)
\]

做 deterministic、no-\(\sigma\)、no-MC 的精确计算。

这里“精确”指：

- 不用随机估计；
- 不用有限差分截断近似作为主方法；
- 对当前浮点神经网络程序给出 automatic differentiation / Taylor algebra 意义下的精确导数；
- 数值误差只来自浮点 round-off、复数求和 conditioning、框架实现等。

核心问题是：

> 对单个 \(\partial^\alpha\)，有哪些精确计算路径？它们的方向数、时间、显存、稳定性如何？我们能否自动生成一个比直接 autodiff 更好的通用 backend？

---

## 1. 单项 operator 的数学表示

令

\[
p=|\alpha|,
\qquad
A=D^p u(x)
\]

为对称 \(p\)-线性型。对任意方向 \(v\)，定义方向 Taylor 系数

\[
T_p(x;v)=\frac{1}{p!}A[v,\ldots,v]
=\frac{1}{p!}\frac{d^p}{dt^p}u(x+tv)\bigg|_{t=0}.
\]

如果我们找到方向和权重

\[
\partial^\alpha u(x)=\sum_{r=1}^{R}c_rT_p(x;v_r),
\]

那么单项 operator 的计算就分成两层：

1. **方向分解层**：如何用尽量少的方向 \(v_r\) 表示 \(\partial^\alpha\)；
2. **方向导数 backend 层**：如何高效、精确地计算每个 \(T_p(x;v_r)\)。

这和 contract/operator sum 完全不同。本文只关心单个 \(\alpha\) 的方向分解与计算。

---

## 2. 现有方法一览

| 方法 | 是否精确 | 是否当前已实现 | 适用对象 | 核心瓶颈 |
|---|---:|---:|---|---|
| Direct nested coordinate autodiff | 是 | 是 | 任意单项 \(\partial^\alpha\) | p 层 autograd graph，显存/时间大 |
| Scalar-path reverse AD | 是 | 是 | 已知方向 \(T_p(x;v)\) | 每方向 p 次 reverse grad |
| Scalar-path JVP | 是 | 是 | 已知方向 \(T_p(x;v)\) | PyTorch 下当前慢 |
| Taylor jet | 是 | 是 | 已知方向 \(T_p(x;v)\)，Linear/Tanh MLP | 需要 primitive jet rules |
| Polarization + Taylor jet | 是 | 是 | 任意单项 | 方向数最多约 \(2^{p-1}\) |
| Complex monomial Waring + Taylor jet | 是 | 是 | 任意单项，复方向 | complex arithmetic 常数与 conditioning |
| Real Waring / realification + Taylor jet | 是 | 未完全实现 | 任意单项，实方向 | 需要构造实 rank/near-rank 方向表 |
| Finite difference direction formula | 否 | 是 | 近似 \(T_p(x;v)\) | 截断误差与 roundoff |
| Symbolic codegen | 是 | 对特殊 exact solution 已有 | 固定解析结构 | 不适合一般 MLP |
| Hyper-dual / multi-dual | 是 | 未单独实现 | 小维/低阶单项 | coefficient 数组合爆炸 |

---

## 3. 方法细节、优势与劣势

### 3.1 Direct nested coordinate autodiff

#### 原理

直接按 multi-index 逐次求导：

\[
\partial_{i_1}\cdots\partial_{i_p}u(x)
\]

实现上就是：

1. 前向得到 \(u(x)\)；
2. 对输入 \(x\) 做 `torch.autograd.grad`；
3. 取出目标坐标；
4. 重复 \(p\) 次。

当前代码：

- `scripts/probe_polarized_jet_vs_autodiff.py::autodiff_operator`
- `kdv_hd/estimators.py::L_high_autodiff` 中也有类似逐坐标嵌套逻辑。

#### 优势

- 最直接、最可信；
- 对任意 PyTorch 可微模型都能跑；
- 不需要推导方向公式；
- 适合作 reference / correctness oracle。

#### 劣势

- 每一阶都要保留高阶 autograd graph；
- p 高时显存和时间常数很大；
- 如果要对很多点、很多单项重复算，graph 构建成本重；
- 对 repeated index 没有利用单项结构，例如 \(u_{111111}\) 仍是 6 层 nested grad。

#### 当前实测

在 `d=6, B=2, hidden=16, depth=2, fp64, T4` 下：

| 单项 | direct autodiff ms | peak MB |
|---|---:|---:|
| \(u_{111}\) | 2.52 | 17.27 |
| \(u_{112}\) | 2.39 | 17.27 |
| \(u_{123}\) | 2.31 | 17.27 |

单项三阶小网络上 direct autodiff 并不差，但这是小规模。p、B、hidden、depth 增大后通常会恶化。

---

### 3.2 Scalar-path reverse AD for \(T_p(x;v)\)

#### 原理

先把多维方向固定为一维路径：

\[
g(t)=u(x+tv).
\]

然后对 scalar \(t\) 做 p 次 reverse autodiff：

\[
T_p(x;v)=\frac{g^{(p)}(0)}{p!}.
\]

当前代码：

- `kdv_hd/estimators.py::_tp_via_reverse`

#### 优势

- 对方向导数语义清晰；
- 成本不直接依赖 ambient dimension \(d\)，只依赖 batch × direction count；
- 可作为 `TaylorJet` 的 reference。

#### 劣势

- 每个方向仍需要 p 层 reverse graph；
- 如果单项分解需要很多方向，总成本是 \(R\) 个方向的 scalar-path reverse；
- 通常不如 Taylor jet。

#### 当前实测：directional \(T_p\)

在 `B=2, K=4, hidden=16, depth=2, fp64, T4` 下，以 reverse 为 reference：

| d | p | reverse ms | jet ms | jet speedup |
|---:|---:|---:|---:|---:|
| 8 | 2 | 1.56 | 1.34 | 1.17× |
| 8 | 3 | 2.39 | 1.80 | 1.33× |
| 8 | 4 | 4.03 | 2.38 | 1.70× |
| 8 | 6 | 18.39 | 3.81 | 4.83× |
| 32 | 2 | 1.54 | 1.32 | 1.17× |
| 32 | 3 | 2.36 | 1.79 | 1.32× |
| 32 | 4 | 3.99 | 2.38 | 1.67× |
| 32 | 6 | 18.24 | 3.79 | 4.81× |

结论：scalar-path reverse 是好 reference，但高阶时 Taylor jet 更有优势。

---

### 3.3 Scalar-path JVP

#### 原理

同样定义

\[
g(t)=u(x+tv),
\]

但用 `torch.func.jvp` 递推 p 次得到 \(g^{(p)}(0)\)。

当前代码：

- `kdv_hd/estimators.py::_tp_via_jvp`

#### 优势

- 概念上更接近 forward-mode；
- 理论上适合 input dimension 大、output dimension 小的方向导数；
- 可作为另一条精确 reference。

#### 劣势

- 当前 PyTorch 环境下很慢；
- 嵌套 JVP 仍有较大框架开销；
- 不如手写 Taylor jet 可控。

#### 当前实测

同样配置下，JVP 相对 reverse 的耗时倍数：

| p | jvp / reverse time |
|---:|---:|
| 2 | 4.1× |
| 3 | 6.6× |
| 4 | 9.9–10.0× |
| 6 | 15.2× |

结论：目前不适合作主力，只保留为语义 reference。

---

### 3.4 Taylor jet for \(T_p(x;v)\)

#### 原理

每个中间变量不只保存数值，而是保存截断 Taylor 系数：

\[
y(t)=y_0+y_1t+\cdots+y_pt^p.
\]

对网络 primitive 写 jet rule：

- Linear：每阶线性传播；
- Tanh：用 recurrence 更新所有阶。

最终直接读取第 p 阶：

\[
T_p(x;v)=y_p.
\]

当前代码：

- `pinns/taylor_jet.py`
- `tp_directional_via_jet`
- `tp_directional_all_via_jet`
- `tests/test_taylor_jet.py`

#### 优势

- 对方向导数精确；
- 一次 pass 携带 0 到 p 阶信息；
- p 越高相对 reverse 优势越明显；
- 可以 batch 多个方向，是单项 direction-decomposition backend 的核心。

#### 劣势

- 当前只支持 `Linear + Tanh`；
- 需要为新 activation / primitive 写 rule；
- 本身只解决“给定方向”的计算，不解决单项 \(\partial^\alpha\) 需要多少方向。

#### 当前实测

- 和 reverse/JVP 的相对误差在 fp64 下约 `1e-16 ~ 1e-15`；
- p=6 时约 `4.8×` faster than scalar reverse；
- 小规模测试下显存差异不大，但 jet 约 `16.3–16.4 MB`，reverse 约 `17.3 MB`；更大图和 backward 场景会放大差异。

结论：**Taylor jet 是当前最适合承接单项方向分解的精确 backend。**

---

### 3.5 Polarization + Taylor jet

#### 原理

用 polarization identity 把 mixed partial 写成方向导数的 sign-sum。

对 expanded multi-index \(\alpha=(i_1,\ldots,i_p)\)：

\[
D^p u[e_{i_1},\ldots,e_{i_p}]
=2^{-p}\sum_{\epsilon\in\{\pm1\}^p}\left(\prod_r\epsilon_r\right)
D^p u\left[\sum_r\epsilon_r e_{i_r},\ldots,\sum_r\epsilon_r e_{i_r}\right].
\]

换成 \(T_p\) 后就是若干实方向的 Taylor jet。

当前代码：

- `scripts/probe_polarized_jet_vs_autodiff.py::polarization_direction_table`
- `polarized_jet_operator(..., strategy='polarization')`

#### 优势

- 完全实数；
- 对任意单项都可用；
- 和 Taylor jet 直接兼容；
- 不涉及 complex PyTorch；
- 当前实测稳定、精度好。

#### 劣势

- raw 方向数是 \(2^p\)；
- antipodal merge 后通常约 \(2^{p-1}\)，但 repeated index 可进一步减少；
- 对高阶 square-free mixed partial，方向数指数增长是硬伤。

#### 当前 probe 结果

`d=6, B=2, hidden=16, depth=2, fp64, T4`：

| 单项 | raw -> merged dirs | rel err | autodiff ms | polarization+jet ms |
|---|---:|---:|---:|---:|
| \(u_{111}\) | 8 -> 2 | 5.00e-16 | 51.95* | 14.58* |
| \(u_{112}\) | 8 -> 3 | 8.35e-16 | 2.37 | 2.22 |
| \(u_{123}\) | 8 -> 4 | 3.07e-15 | 2.48 | 2.17 |

\* 第一行受 warmup/首次 CUDA 开销影响偏大；正式 CSV 中 `u_111` 为 direct 2.52 ms、polarization 2.06 ms。

正式 CSV 中单项结果：

| 单项 | direct autodiff ms | polarization+jet ms | rel err |
|---|---:|---:|---:|
| \(u_{111}\) | 2.52 | 2.06 | 1.73e-16 |
| \(u_{112}\) | 2.39 | 2.04 | 4.93e-16 |
| \(u_{123}\) | 2.31 | 2.04 | 4.80e-16 |

结论：三阶单项上 polarization+jet 已略快且精度好；但真正价值要看 p=4/5/6 及 repeated-index pattern。

---

### 3.6 Complex monomial Waring + Taylor jet

#### 原理

把单项导数的方向数问题转成 monomial Waring rank。

设 active exponents 排序为

\[
1\le a_0\le a_1\le\cdots\le a_{s-1},
\qquad \sum_i a_i=p.
\]

复数域 monomial Waring rank 有 closed form：

\[
R_\mathbb C(\alpha)=\frac{\prod_i(a_i+1)}{a_0+1}=\prod_{i=1}^{s-1}(a_i+1).
\]

roots-of-unity 构造给出方向

\[
v_{\boldsymbol\zeta}=e_0+\sum_{i=1}^{s-1}\zeta_i e_i,
\]

和系数

\[
c_{\boldsymbol\zeta}=\frac{\alpha!}{\prod_{i=1}^{s-1}(a_i+1)}\prod_{i=1}^{s-1}\zeta_i.
\]

于是

\[
\partial^\alpha u(x)=\sum_{\boldsymbol\zeta}c_{\boldsymbol\zeta}T_p(x;v_{\boldsymbol\zeta}).
\]

当前代码：

- `pinns/monomial_waring.py::monomial_waring_directions`
- `scripts/probe_polarized_jet_vs_autodiff.py --direction_strategy waring_complex`

#### 优势

- 复数域方向数 rank-optimal；
- 对 repeated indices 优势明显；
- 给出了自动生成任意单项 \(\partial^\alpha\) 的 deterministic 方向公式；
- 与 Taylor jet 可以组合成通用 backend。

#### 劣势

- 需要 complex MLP / complex Taylor jet；
- PyTorch complex module 仍有 warning，工程成熟度不如 real；
- complex arithmetic 常数大；
- roots-of-unity 求和可能有 conditioning 问题；
- 对 square-free pattern，如 \((1,1,1,1,1,1)\)，rank 仍是 \(2^{p-1}\)，没有突破指数下界。

#### 当前 probe 结果

`d=6, B=2, hidden=16, depth=2, fp64, T4`：

| 单项 | raw -> merged dirs | rel err | autodiff ms | complex Waring+jet ms |
|---|---:|---:|---:|---:|
| \(u_{111}\) | 1 -> 1 | 5.00e-16 | 46.89* | 19.13* |
| \(u_{112}\) | 3 -> 3 | 2.90e-13 | 2.36 | 2.56 |
| \(u_{123}\) | 4 -> 4 | 1.02e-15 | 2.35 | 2.41 |

正式 CSV 中单项结果：

| 单项 | direct autodiff ms | complex Waring+jet ms | rel err |
|---|---:|---:|---:|
| \(u_{111}\) | 2.52 | 2.18 | 3.45e-16 |
| \(u_{112}\) | 2.39 | 2.32 | 5.23e-13 |
| \(u_{123}\) | 2.31 | 2.38 | 1.76e-15 |

结论：当前 complex Waring 的理论方向数好，但工程常数还没赢过 real polarization；它更像 oracle/reference，下一步应 realify。

---

### 3.7 Real Waring / realification + Taylor jet

#### 原理

目标是把 complex formula

\[
\partial^\alpha u=\sum_r c_rT_p(v_r),\qquad c_r,v_r\in\mathbb C
\]

转成纯实方向公式

\[
\partial^\alpha u=\sum_r \tilde c_rT_p(\tilde v_r),\qquad \tilde c_r,\tilde v_r\in\mathbb R.
\]

这不是把方向限制为 \(\pm1\)。实 Waring decomposition 允许任意实线性形式。

已知理论边界：

- complex rank：

\[
R_\mathbb C(\alpha)=\frac{\prod_i(a_i+1)}{a_0+1}.
\]

- Carlini--Kummer--Oneto--Ventura：

\[
R_\mathbb R(\alpha)=R_\mathbb C(\alpha)\quad\Longleftrightarrow\quad a_0=1.
\]

- binary monomial：

\[
R_\mathbb R(x^ay^b)=a+b.
\]

- Han--Moon 构造性实上界：

\[
R_\mathbb R(\alpha)\le
R_{HM}(\alpha)=\frac12\left(\prod_i(a_i+1)-\prod_i(a_i-1)\right).
\]

#### 优势

- 保留 Waring 的低方向数思想；
- 避免 complex PyTorch；
- 对训练和 backward 更稳；
- 有机会成为真正可用的单项 operator backend。

#### 劣势

- 目前还没实现完整生成器；
- 一般 monomial 的 real rank 比 complex rank 更复杂；
- 需要离线 pattern table、系数范数控制、conditioning 筛选；
- 对 all exponents > 1 且 support > 2 的 pattern，只能先做 constructive upper bound 或搜索。

#### 关键 pattern 表

| active exponent pattern | 示例 | \(R_\mathbb C\) | 已知/预期 \(R_\mathbb R\) | 备注 |
|---|---|---:|---:|---|
| \((p)\) | \(u_{111111}\) | 1 | 1 | pure directional derivative |
| \((2,1)\) | \(u_{112}\) | 3 | 3 | min exponent = 1 |
| \((1,1,1)\) | \(u_{123}\) | 4 | 4 | square-free p=3 |
| \((3,1)\) | \(u_{1112}\) | 4 | 4 | binary, min=1 |
| \((2,2)\) | \(u_{1122}\) | 3 | 4 | binary real rank = 4 |
| \((2,1,1)\) | \(u_{1123}\) | 6 | 6 | min exponent = 1 |
| \((1,1,1,1)\) | \(u_{1234}\) | 8 | 8 | square-free p=4 |
| \((2,2,2)\) | \(u_{112233}\) | 9 | 9–13 | HM upper bound 13 |
| \((1,1,1,1,1,1)\) | \(u_{123456}\) | 32 | 32 | square-free p=6 |

结论：real Waring/realification 是当前最有机会的方向，尤其是 repeated-index 单项。

---

### 3.8 Finite difference direction formula

#### 原理

对方向函数

\[
g(t)=u(x+tv)
\]

用中心差分 stencil 近似

\[
T_p(x;v)\approx \frac{1}{p!h^p}\sum_j w_jg(jh).
\]

当前代码：

- `kdv_hd/estimators.py::_tp_directional_via_findiff`

#### 优势

- 只需要 forward pass；
- 很快；
- 可作为 cheap baseline。

#### 劣势

- 不是精确导数；
- 有 \(O(h^q)\) 截断误差；
- 高阶时 \(h^{-p}\) 放大 roundoff；
- 不适合作本文主方向。

#### 当前实测

`fd_h0.05_q4` 相对 reverse：

| d | p | FD rel err |
|---:|---:|---:|
| 8 | 2 | 1.44e-04 |
| 8 | 3 | 4.20e-04 |
| 8 | 4 | 4.63e-03 |
| 8 | 6 | 1.48e-02 |
| 32 | 2 | 4.64e-04 |
| 32 | 3 | 1.58e-02 |
| 32 | 4 | 5.86e-03 |
| 32 | 6 | 1.43e-02 |

结论：速度可以作为参考，但不属于“精确计算单项 operator”的主线。

---

### 3.9 Symbolic codegen

#### 原理

对固定解析表达式直接符号求导、CSE、生成 kernel。

当前项目例子：

- `leibniz_kernels.py`
- `leibniz_kernels_threebody.py`
- `derive/derive_threebody_kernels.py`

#### 优势

- 对固定 exact solution 运行时最快；
- 可做到 \(O(d)\) 或局部闭式；
- 可作为 correctness oracle。

#### 劣势

- 不适合一般 trainable MLP；
- 每换结构都要重新推导；
- expression swell 和维护成本高。

定位：不作为 generic monomial operator backend，只作为 manufactured solution 的辅助验证。

---

### 3.10 Hyper-dual / multi-dual

#### 原理

用扩展数系携带导数 coefficient。二阶 hyper-dual 经典，更高阶可看作 truncated polynomial algebra。

#### 优势

- 数学干净；
- 对低阶、小维 mixed partial 有精确性；
- 可解释 Taylor-mode 的 algebraic foundation。

#### 劣势

- 多维高阶 coefficient 数组合爆炸；
- 对我们当前 `Linear/Tanh` MLP，手写 Taylor jet 已经覆盖了主要能力；
- 单独实现收益不大。

定位：理论参考，不作为当前工程主线。

---

## 4. 当前脚本与结果如何解读

当前和单项 operator 最相关的脚本：

| 脚本 | 作用 |
|---|---|
| `scripts/probe_polarized_jet_vs_autodiff.py` | 单项/小 operator 的 direct autodiff vs polarization/Waring + jet sanity/profiling |
| `scripts/benchmark_single_monomial_derivatives.py` | **只跑单项 operator** 的系统 benchmark，输出 direction count、精度、耗时、显存 |
| `scripts/run_single_monomial_benchmark_0513.sh` | 单项 benchmark 一键运行脚本 |
| `pinns/monomial_waring.py` | complex monomial Waring direction generator |
| `pinns/monomial_real_waring.py` | real direction generator；当前 pure power rank-1，其他 pattern 回退 polarization |
| `pinns/taylor_jet.py` | Linear/Tanh MLP 的 Taylor jet backend |

结果文件：

```text
results/0513/single_monomial_benchmark/single_monomial_benchmark.csv
results/0513/single_monomial_benchmark/single_monomial_benchmark.json
```

实验配置：`d=6, B=2, hidden=16, depth=2, fp64, T4, measure=value`。

### 4.1 单项 benchmark 结果

| 单项 | pattern | direct ms | polarization dirs/ms | complex Waring dirs/ms | real Waring dirs/ms | real rel err |
|---|---|---:|---:|---:|---:|---:|
| \(u_{111}\) | (3) | 2.49 | 2 / 2.09 | 1 / 2.21 | 1 / 1.82 | 6.25e-16 |
| \(u_{112}\) | (2,1) | 2.30 | 3 / 2.08 | 3 / 2.38 | 3 / 1.84 | 8.35e-16 |
| \(u_{123}\) | (1,1,1) | 2.29 | 4 / 2.07 | 4 / 2.42 | 4 / 1.86 | 3.07e-15 |
| \(u_{1111}\) | (4) | 3.90 | 2 / 2.69 | 1 / 2.77 | 1 / 2.39 | 4.08e-16 |
| \(u_{1112}\) | (3,1) | 3.91 | 4 / 2.67 | 4 / 3.00 | 4 / 2.40 | 5.07e-16 |
| \(u_{1122}\) | (2,2) | 3.87 | 4 / 2.63 | 3 / 2.90 | 4 / 2.42 | 9.99e-16 |
| \(u_{1123}\) | (2,1,1) | 4.05 | 6 / 2.68 | 6 / 3.11 | 6 / 2.42 | 6.69e-16 |
| \(u_{1234}\) | (1,1,1,1) | 3.90 | 8 / 2.69 | 8 / 3.21 | 8 / 2.43 | 5.93e-16 |
| \(u_{111111}\) | (6) | 17.78 | 3 / 4.32 | 1 / 4.26 | 1 / 3.85 | 4.37e-16 |
| \(u_{111122}\) | (4,2) | 18.09 | 7 / 4.27 | 5 / 4.53 | 7 / 3.92 | 3.21e-15 |
| \(u_{112233}\) | (2,2,2) | 17.98 | 13 / 4.38 | 9 / 4.76 | 13 / 3.93 | 3.06e-15 |
| \(u_{123456}\) | (1,1,1,1,1,1) | 17.82 | 32 / 4.38 | 32 / 6.14 | 32 / 3.93 | 1.59e-15 |

### 4.2 目前能确认什么

1. `TaylorJet` 本身是正确的：所有 real-direction 方法相对 direct autodiff 的误差都在 fp64 舍入量级，约 `1e-16` 到 `1e-15`。
2. 单项高阶时，directional jet 路线明显优于 direct coordinate autodiff：p=6 的单项从约 `18 ms` 降到约 `4 ms`。
3. `complex Waring` 方向数理论上更少，例如 \((2,2,2)\) 是 9 个方向而 polarization 是 13 个方向，但 complex arithmetic 常数偏大，当前未转化为速度优势。
4. 新增 `waring_real_jet` 目前只在 pure power 上真正使用 rank-1，其它 pattern 回退 polarization；即便如此已经是当前最快或接近最快的 real backend。
5. 现在最缺的是真正的 real Waring / realification generator：对于 \((3,1),(2,2),(2,1,1),(4,2),(2,2,2)\) 等 pattern，应该用实 rank/near-rank 方向替代 polarization fallback。

---

## 5. 我们当前最有机会做的方法

### 5.1 方法定位

我们不做 contract operator sum，也不做 stochastic estimator。最有机会的是：

\[
\boxed{
\text{automatic deterministic exact backend for single monomial partials}
}
\]

具体形式：

\[
\partial^\alpha u(x)
=\sum_{r=1}^{R_\mathbb R(\alpha)\text{ or upper bound}}
\tilde c_r T_p(x;\tilde v_r),
\qquad \tilde c_r,\tilde v_r\in\mathbb R,
\]

其中每个 \(T_p\) 用 Taylor jet 精确计算。

这条路线的卖点：

- no MC；
- no \(\sigma\)；
- no finite difference bias；
- 不需要完整 \(p\)-阶张量；
- 自动支持 arbitrary single multi-index \(\alpha\)；
- repeated-index 单项可显著减少方向数；
- 训练时可以对 \(\theta\) 反传。

### 5.2 为什么不是 complex Waring 直接作为主方法

Complex Waring 理论很漂亮，但工程上当前不是最佳主线：

1. PyTorch complex module 有 warning，生态不如 real；
2. complex matmul/tanh 常数偏大；
3. 复数求和有 conditioning 风险；
4. backward 到实参数时实现和调试更麻烦。

所以 complex Waring 应作为：

- 理论下界；
- direction count oracle；
- correctness reference；
- realification 的起点。

主方法应该是 real Waring / realified Waring + Taylor jet。

### 5.3 我们能做的具体贡献

#### 贡献 A：单项 operator pattern table

离线枚举 \(p\le6\) 或 \(p\le8\) 的 active exponent pattern，记录：

- pattern，例如 \((4,2), (3,1,1), (2,2,1)\)；
- complex rank；
- known real rank；
- Han--Moon upper bound；
- polarization direction count；
- best real formula；
- coefficient norm / direction norm；
- conditioning。

这是单项 operator 的核心基础设施。

#### 贡献 B：known-case real generator

先实现理论明确的两类：

1. **pure power** \((p)\)：

\[
\partial_i^p u=p!T_p(e_i)
\]

实际由于 \(T_p\) 已含 \(1/p!\)，代码里的系数按当前 convention 处理。

2. **min exponent = 1**：

\[
R_\mathbb R=R_\mathbb C.
\]

例如：

- \((2,1)\)：\(u_{112}\)，rank 3；
- \((1,1,1)\)：\(u_{123}\)，rank 4；
- \((2,1,1)\)：\(u_{1123}\)，rank 6。

3. **binary monomial** \((a,b)\)：

\[
R_\mathbb R=a+b.
\]

例如：

- \((2,2)\)：\(u_{1122}\)，real rank 4；
- \((3,2)\)：real rank 5；
- \((4,2)\)：real rank 6。

#### 贡献 C：Han--Moon constructive upper-bound generator

对 all exponents > 1 且 support > 2 的 pattern，先实现可用实上界：

\[
R_{HM}=\frac12\left(\prod_i(a_i+1)-\prod_i(a_i-1)\right).
\]

它未必 rank-optimal，但比 polarization 可能更好，且全部实数。

#### 贡献 D：real dictionary solver / formula search

对小阶 pattern，可用连续实方向候选解线性系统。

令所有 degree-p monomial basis 为 \(\beta\)，方向为 \(v_j\)，构造

\[
M_{\beta j}=v_j^\beta.
\]

目标是找到 sparse coefficients \(a_j\)，使

\[
Ma=b_\alpha,
\qquad
(b_\alpha)_\beta=\alpha!\delta_{\alpha\beta}.
\]

可优化目标：

- 最小方向数；
- 最小 coefficient norm；
- 最小 direction norm；
- 最好 conditioning；
- 可 batch 的方向 pattern。

这会比简单套理论公式更工程化。

#### 贡献 E：single monomial benchmark suite

当前 benchmark 还混入了 operator sum。下一版应只跑单项：

```text
p=1:  u1
p=2:  u11, u12
p=3:  u111, u112, u123
p=4:  u1111, u1112, u1122, u1123, u1234
p=5:  u11111, u11112, u11122, u11223, u12345
p=6:  u111111, u111122, u112233, u111123, u123456
```

比较：

- direct coordinate autodiff；
- polarization+jet；
- complex Waring+jet；
- real Waring+jet；
- optional FD only as non-exact baseline。

指标：

- direction count；
- value relative error；
- gradient relative error；
- wall-clock；
- peak allocated/reserved memory；
- coefficient norm；
- condition proxy。

---

## 6. 当前最推荐的开发路线

### Phase 1：重构 benchmark，只保留单项 operator

新增或改造脚本：

```text
scripts/benchmark_single_monomial_derivatives.py
```

要求：

- 输入 expanded multi-index 或 pattern；
- 不跑 operator sum；
- 输出每个单项的 direction count、error、time、memory；
- 支持 backward gradient check。

### Phase 2：实现 `waring_real`

新增模块：

```text
pinns/monomial_real_waring.py
```

API：

```python
V, coeff, info = monomial_real_waring_directions(alpha, d, strategy="auto")
```

策略：

1. pure power；
2. min exponent = 1；
3. binary monomial；
4. Han--Moon upper-bound；
5. fallback polarization。

### Phase 3：和 Taylor jet 合并成单项 operator API

新增统一入口：

```python
single_monomial_partial(model, x, alpha, backend="real_waring_jet")
```

可选 backend：

- `direct_autodiff`；
- `polarization_jet`；
- `complex_waring_jet`；
- `real_waring_jet`；
- `fd_polarization` only baseline。

### Phase 4：pattern table + paper claim

最终 claim 不应该是“对所有高阶导数都最优”，而应非常精确：

> For a single monomial partial derivative \(\partial^\alpha u\), we provide an automatic deterministic Taylor-jet backend using real Waring decompositions of monomials. It avoids full high-order derivative tensors and stochastic estimators, achieves known real/complex rank in major exponent patterns, and falls back to constructive real upper bounds otherwise.

中文定位：

> 一个面向单项 mixed partial 的自动化、确定性、精确高阶导数 backend；核心优势出现在 repeated-index、中高阶、需要训练反传的场景。

---

## 7. 结论

目前最清晰的方向是：

\[
\boxed{
\partial^\alpha u
\xrightarrow{\text{real Waring / realification}}
\sum_r c_rT_p(v_r)
\xrightarrow{\text{Taylor jet}}
\text{exact value and exact parameter gradient}
}
\]

短期不要再扩展到 contract/operator sum。当前应集中火力做：

1. 单项 pattern table；
2. real Waring generator；
3. 单项 benchmark suite；
4. 和 direct autodiff、polarization、complex Waring 的系统对比。

这条线既避开 stochastic estimator，也避开 full tensor materialization；在 repeated-index 高阶单项上最可能形成有说服力的速度/显存优势。
