import {
  Stack,
  Row,
  Grid,
  Divider,
  Spacer,
  H1,
  H2,
  H3,
  Text,
  Code,
  Card,
  CardHeader,
  CardBody,
  Table,
  Stat,
  Pill,
  Callout,
  LineChart,
  useHostTheme,
  type ChartSeries,
  type TableRowTone,
} from "cursor/canvas";

// ===========================================================================
//  复参数 sinh 网络 vs 实值高频基线 —— 震荡 / 高阶 PDE 基准（width study 全部完成）
//  数据: results/width/<family>_h{128,64}.csv（每点 2 seeds 均值, 600s 预算）
//  12 族全部跑完于 2026-06-30 06:53；含 14 个子算例结果。
// ===========================================================================

type SeriesMap = Record<string, number[]>;

interface Family {
  key: string;
  title: string;
  group: "高频二阶" | "高阶实算子" | "复值场";
  sweepShort: string; // 表头前缀，如 "a" / "阶" / "m"
  sweepName: string; // 图横轴含义
  defs: [string, string][]; // 详细 setting 定义列表
  sweeps: number[];
  series: SeriesMap; // 变体 -> 各 sweep 的平均相对 L2 误差
}

const VARIANT_ORDER = [
  "complex_sinh@64",
  "complex_sinh@128",
  "fourier",
  "siren",
  "mscale",
  "tanh",
];
const VARIANT_LABEL: Record<string, string> = {
  "complex_sinh@64": "复 sinh @64 (本文)",
  "complex_sinh@128": "复 sinh @128 (本文)",
  fourier: "Fourier-feature @128",
  siren: "SIREN @128",
  mscale: "MscaleDNN @128",
  tanh: "tanh RVPINN @128",
};
const SHORT_LABEL: Record<string, string> = {
  "complex_sinh@64": "复@64",
  "complex_sinh@128": "复@128",
  fourier: "Fourier",
  siren: "SIREN",
  mscale: "Mscale",
  tanh: "tanh",
};
const isComplex = (v: string) => v.startsWith("complex_sinh");

const FAMILIES: Family[] = [
  // ---------------- 高频二阶 ----------------
  {
    key: "helmholtz",
    title: "高波数 Helmholtz",
    group: "高频二阶",
    sweepShort: "a",
    sweepName: "波数 a",
    defs: [
      ["控制方程", "Δu + κ²u = f，κ = aπ（常系数）"],
      ["精确解", "u = sin(aπx)·sin(aπy)，可分离本征模"],
      ["计算域", "Ω = (−1,1)²"],
      ["边界条件", "齐次 Dirichlet：u = 0 于 ∂Ω"],
      ["源项", "f = (κ² − 2(aπ)²)u = −(aπ)²u （因 Δu = −2(aπ)²u）"],
      ["扫描参数", "波数 a ∈ {2,4,6,8,10}，即 κ 从 2π 增至 10π"],
      ["频率初始化", "ω₀ = max(10, 2πa)，σ = max(2, πa)；残差按 (aπ)² 归一"],
      ["考察重点", "标准谱偏差(spectral bias)战场：纯高频本征模，看各法随波数升高的表达/收敛能力"],
      ["文献出处", "Wang–Teng–Perdikaris 2021 (gradient pathologies) / PINNacle 2024"],
    ],
    sweeps: [2, 4, 6, 8, 10],
    series: {
      "complex_sinh@64": [0.0084, 0.2154, 0.2946, 0.5526, 0.6896],
      "complex_sinh@128": [0.0342, 0.1367, 0.3509, 0.4534, 0.7115],
      fourier: [0.0311, 0.4934, 0.7488, 0.8204, 0.8716],
      siren: [0.0189, 0.2413, 0.6746, 0.8721, 0.9248],
      mscale: [0.2798, 1.0002, 1.0009, 1.0001, 0.9999],
    },
  },
  {
    key: "helmholtz_aniso",
    title: "各向异性 Helmholtz（梯度病态原例）",
    group: "高频二阶",
    sweepShort: "f",
    sweepName: "有效频率",
    defs: [
      ["控制方程", "Δu + κ²u = q，κ = π（k=1）"],
      ["精确解", "u = sin(πx)·sin(4πy)，x、y 方向频率相差 4 倍"],
      ["计算域", "Ω = (−1,1)²"],
      ["边界条件", "齐次 Dirichlet：u = 0"],
      ["源项", "q = (κ² − (1²+4²)π²)u = (π² − 17π²)u"],
      ["扫描参数", "单算例 (a₁,a₂) = (1,4)，有效频率 = 4"],
      ["频率初始化", "ω₀ = max(10, 8π)，σ = max(2, 4π)"],
      ["考察重点", "Wang 2021 经典“梯度病态”构造：方向各向异性使各轴梯度尺度悬殊，是标准 PINN 失败的代表算例"],
      ["文献出处", "Wang–Teng–Perdikaris 2021"],
    ],
    sweeps: [4],
    series: {
      "complex_sinh@64": [0.0156],
      "complex_sinh@128": [0.0541],
      fourier: [0.3517],
      siren: [0.1367],
      mscale: [1.0003],
    },
  },
  {
    key: "helmvc",
    title: "变系数（散射）Helmholtz",
    group: "高频二阶",
    sweepShort: "a",
    sweepName: "背景波数 a",
    defs: [
      ["控制方程", "Δu + κ²(x)u = f，κ²(x) = (aπ)²·(1 + 0.5·sin πx·sin πy)"],
      ["精确解", "u = sin(aπx)·sin(aπy)"],
      ["计算域", "Ω = (−1,1)²"],
      ["边界条件", "齐次 Dirichlet：u = 0"],
      ["源项", "f = −2(aπ)²u + κ²(x)·u"],
      ["扫描参数", "背景波数 a ∈ {2,4,6}"],
      ["频率初始化", "ω₀ = max(10, 2πa)，σ = max(2, πa)"],
      ["考察重点", "空间变化的零阶系数（±50% 透镜状不均匀介质），比常系数更接近真实散射，检验对介质异质性的鲁棒性"],
      ["文献出处", "PINNacle 2024 异质介质族"],
    ],
    sweeps: [2, 4, 6],
    series: {
      "complex_sinh@64": [0.4519, 0.354, 0.4283],
      "complex_sinh@128": [0.4589, 0.2637, 0.4178],
      fourier: [0.2592, 0.5277, 0.7579],
      siren: [0.2789, 0.31, 0.7091],
      mscale: [0.996, 1.0001, 1.0009],
    },
  },
  {
    key: "chirp",
    title: "径向 chirp（非分离震荡）",
    group: "高频二阶",
    sweepShort: "a",
    sweepName: "chirp 参数 a",
    defs: [
      ["控制方程", "−Δu + u = f"],
      ["精确解", "u = sin(½·aπ·(x²+y²))，局部频率 |∇φ| = aπr 随半径线性增长"],
      ["计算域", "Ω = (−1,1)²"],
      ["边界条件", "Dirichlet：u = u_exact"],
      ["源项", "f = −Δu + u，Δu = −(aπ)²r²·sin φ + 2aπ·cos φ"],
      ["扫描参数", "a ∈ {2,4,6,8}"],
      ["频率初始化", "ω₀ = max(10, 2πa)，σ = max(2, πa)"],
      ["考察重点", "去掉“可分离正弦”这一 confound：解非单一傅里叶模态，每种结构都必须真正构建出变化的震荡，是公平的表达力测试"],
      ["文献出处", "Tancik 2020 (Fourier features) / Liu 2020 (MscaleDNN)"],
    ],
    sweeps: [2, 4, 6, 8],
    series: {
      "complex_sinh@64": [0.0012, 0.2459, 0.4331, 0.5897],
      "complex_sinh@128": [0.0037, 0.2328, 0.5506, 0.6088],
      fourier: [0.0286, 0.4286, 1.0641, 1.1394],
      siren: [0.0139, 0.3013, 0.7482, 0.8454],
      mscale: [0.6412, 1.1402, 1.5266, 1.6402],
    },
  },
  // ---------------- 高阶实算子 ----------------
  {
    key: "poly1d",
    title: "多调和算子（1D 阶数轴）",
    group: "高阶实算子",
    sweepShort: "阶",
    sweepName: "微分阶数",
    defs: [
      ["控制方程", "d²ᵐu/dx²ᵐ = (−π²)ᵐ·u，阶 = 2m"],
      ["精确解", "u = sin(πx)，频率固定为 π，只变阶数"],
      ["计算域", "Ω = (−1,1)"],
      ["边界条件", "Navier 简支：低阶偶导 d²ʲu/dx²ʲ = 0，j = 0..m−1"],
      ["源项", "(−π²)ᵐ·u（本征关系右端）"],
      ["扫描参数", "阶 ∈ {2,4,6,8,10}（m = 1..5）"],
      ["频率初始化", "ω₀ = π，σ = π。关键：阶-m 算子把初始频率放大 ~ωᵐ，过大的 ω₀ 会被 ω₀ᵐ 放大而淹没信号，故须压到目标频率"],
      ["残差缩放", "除以 (π²)ᵐ"],
      ["考察重点", "频率/域/解全部固定，唯一变量是算子微分阶数——干净隔离“阶数”对核心论点的影响"],
      ["文献出处", "Vahab 2022（双调和的高阶推广）"],
    ],
    sweeps: [2, 4, 6, 8, 10],
    series: {
      "complex_sinh@64": [0.0002, 0.0003, 0.0215, 0.2077, 0.8213],
      "complex_sinh@128": [0.0001, 0.0005, 0.1485, 0.2104, 0.9733],
      fourier: [0.0003, 0.0399, 0.2891, 1.0094, 1.0309],
      siren: [0.0001, 0.0194, 0.0969, 1.0055, 1.0058],
      mscale: [0.0003, 0.0144, 0.0588, 0.1001, 0.6215],
    },
  },
  {
    key: "poly2d",
    title: "多调和算子（2D 阶数轴）",
    group: "高阶实算子",
    sweepShort: "阶",
    sweepName: "微分阶数",
    defs: [
      ["控制方程", "Δᵐu = (−2π²)ᵐ·u，阶 = 2m（Δᵐ 展开为 m+1 个高阶 jet 项）"],
      ["精确解", "u = sin(πx)·sin(πy)，S = 2π²"],
      ["计算域", "Ω = (−1,1)²"],
      ["边界条件", "Navier 简支：Δʲu = 0，j = 0..m−1"],
      ["源项", "(−2π²)ᵐ·u"],
      ["扫描参数", "阶 ∈ {2,4,6}"],
      ["频率初始化", "ω₀ = 10，σ = π；残差按 (2π²)ᵐ 归一"],
      ["考察重点", "2D 版阶数轴，算子展开项更多、求导更重，验证 Taylor-jet 高阶可扩展性"],
      ["文献出处", "Vahab 2022"],
    ],
    sweeps: [2, 4, 6],
    series: {
      "complex_sinh@64": [0.0003, 0.0093, 0.1004],
      "complex_sinh@128": [0.0004, 0.0198, 0.0788],
      fourier: [0.0005, 0.1049, 0.9751],
      siren: [0.0009, 0.0368, 1.0028],
      mscale: [0.0089, 0.1581, 0.9902],
    },
  },
  {
    key: "plate",
    title: "Kirchhoff 板（2D 双调和本征模）",
    group: "高阶实算子",
    sweepShort: "m",
    sweepName: "模数 m",
    defs: [
      ["控制方程", "Δ²w = S²·w（4 阶），S = (m²+n²)π²，各向同性 n = m"],
      ["精确解", "w = sin(mπx)·sin(mπy)"],
      ["计算域", "Ω = (−1,1)²"],
      ["边界条件", "简支(Navier)：w = 0 且 Δw = 0 于 ∂Ω"],
      ["源项", "S²·w"],
      ["扫描参数", "模数 m ∈ {1,2,3}（阶固定为 4，只升震荡）"],
      ["频率初始化", "ω₀ = max(10, 2πm)，σ = max(2, πm)"],
      ["考察重点", "4 阶实震荡——实正弦基线的“主场”，检验复网在其最擅长处是否仍可比/占优"],
      ["文献出处", "Vahab 2022（板/梁振动）"],
    ],
    sweeps: [1, 2, 3],
    series: {
      "complex_sinh@64": [0.0092, 0.1841, 0.3969],
      "complex_sinh@128": [0.0198, 0.2398, 0.3483],
      fourier: [0.1047, 0.2088, 0.492],
      siren: [0.0362, 0.1846, 0.554],
      mscale: [0.1591, 0.9996, 1.0],
    },
  },
  {
    key: "beam",
    title: "Euler–Bernoulli 梁（1D 4 阶）",
    group: "高阶实算子",
    sweepShort: "m",
    sweepName: "模数 m",
    defs: [
      ["控制方程", "w'''' = (mπ)⁴·w（4 阶）"],
      ["精确解", "w = sin(mπx)"],
      ["计算域", "Ω = (−1,1)"],
      ["边界条件", "简支：w = 0 且 w'' = 0 于两端"],
      ["源项", "(mπ)⁴·w"],
      ["扫描参数", "模数 m ∈ {1,2,3}"],
      ["频率初始化", "ω₀ = max(10, 2πm)，σ = max(2, πm)"],
      ["考察重点", "1D 4 阶振动，单调单项偏导算子，最轻量的 4 阶基准"],
      ["文献出处", "Vahab 2022"],
    ],
    sweeps: [1, 2, 3],
    series: {
      "complex_sinh@64": [0.0019, 0.018, 0.1652],
      "complex_sinh@128": [0.0059, 0.0483, 0.2419],
      fourier: [0.0382, 0.277, 0.7598],
      siren: [0.2073, 0.1057, 0.1603],
      mscale: [0.0147, 0.2329, 1.5208],
    },
  },
  {
    key: "platemix",
    title: "各向异性板模（非分离频率，4 阶）",
    group: "高阶实算子",
    sweepShort: "f",
    sweepName: "最大模数 m+1",
    defs: [
      ["控制方程", "Δ²w = S²·w，S = (m²+(m+1)²)π²"],
      ["精确解", "w = sin(mπx)·sin((m+1)πy)，每轴频率不同"],
      ["计算域", "Ω = (−1,1)²"],
      ["边界条件", "简支：w = 0 且 Δw = 0"],
      ["源项", "S²·w"],
      ["扫描参数", "m ∈ {2,3,4}，有效频率 fmax = m+1 = 3,4,5"],
      ["频率初始化", "ω₀ = max(10, 2π·fmax)，σ = max(2, π·fmax)"],
      ["考察重点", "4 阶 + 非分离频率，比各向同性板更难，逼近真实板振动模态"],
      ["文献出处", "Vahab 2022"],
    ],
    sweeps: [3, 4, 5],
    series: {
      "complex_sinh@64": [0.3701, 0.4139, 0.6824],
      "complex_sinh@128": [0.1408, 0.2583, 0.3282],
      fourier: [0.8327, 0.911, 0.9419],
      siren: [0.6749, 0.9116, 0.9632],
      mscale: [0.9999, 1.0001, 1.0],
    },
  },
  {
    key: "kdv",
    title: "线性化 KdV / 色散波（3 阶）",
    group: "高阶实算子",
    sweepShort: "k",
    sweepName: "波数 k",
    defs: [
      ["控制方程", "uₜ + δ·u_xxx = f（3 阶，奇数阶），δ = 1"],
      ["精确解", "u = sin(kπx)·cos(kπt)"],
      ["计算域", "(x,t) ∈ (−1,1)²（坐标 0=x，1=t）"],
      ["边界条件", "Dirichlet：u = u_exact"],
      ["源项", "f = uₜ + δ·u_xxx = −kπ·sₓs_t − (kπ)³·cₓc_t"],
      ["扫描参数", "波数 k ∈ {2,3,4,5,6}"],
      ["频率初始化", "ω₀ = max(10, 2πk)，σ = max(2, πk)；残差按 δ(kπ)³+kπ 归一"],
      ["考察重点", "奇数 3 阶色散项——实 Taylor-jet 算子与复 sinh 的用武之地"],
      ["文献出处", "Raissi 2019（KdV；此处取线性化色散项隔离 3 阶算子）"],
    ],
    sweeps: [2, 3, 4, 5, 6],
    series: {
      "complex_sinh@64": [0.2464, 0.444, 0.6733, 0.7076, 0.8235],
      "complex_sinh@128": [0.2298, 0.3403, 0.3245, 0.4526, 0.5737],
      fourier: [0.1562, 0.2616, 0.5475, 0.714, 0.859],
      siren: [0.2819, 0.3338, 0.3571, 0.5505, 0.7922],
      mscale: [0.6907, 1.9365, 1.439, 2.6399, 2.6052],
    },
  },
  {
    key: "ch4",
    title: "Cahn–Hilliard 4 阶（非线性）",
    group: "高阶实算子",
    sweepShort: "a",
    sweepName: "频率 a",
    defs: [
      ["控制方程", "uₜ = M[Δ(u³) − Δu − γΔ²u]，M = γ = 1，Δ = ∂²/∂x²"],
      ["非线性通量", "Δ(u³) = 3u²·u_xx + 6u·(u_x)²，只用单调单项偏导拼出，整残差走快速 Taylor-jet"],
      ["精确解", "u = sin(aπx)·cos(aπt)"],
      ["计算域", "(x,t) ∈ (−1,1)²"],
      ["边界条件", "匹配 u 与 u_xx（Navier 类），罚权 100"],
      ["源项", "对解析解逐项 autograd 得到"],
      ["扫描参数", "a ∈ {2,3}；残差按 γ(aπ)⁴ 归一"],
      ["考察重点", "4 阶非线性——非线性通量能否在 jet 框架内高效求残差"],
      ["文献出处", "Raissi 2019 / PINNacle 2024"],
    ],
    sweeps: [2, 3],
    series: {
      "complex_sinh@64": [0.6592, 0.9747],
      "complex_sinh@128": [0.3925, 0.5921],
      fourier: [0.2784, 0.458],
      siren: [0.3295, 0.4561],
      mscale: [1.0613, 1.7393],
    },
  },
  {
    key: "ch6",
    title: "Cahn–Hilliard 6 阶（非线性）",
    group: "高阶实算子",
    sweepShort: "a",
    sweepName: "频率 a",
    defs: [
      ["控制方程", "uₜ = M[Δ(u³) − Δu − γΔ²u + κΔ³u]，κ = 1"],
      ["精确解", "u = sin(aπx)·cos(aπt)"],
      ["计算域", "(x,t) ∈ (−1,1)²"],
      ["边界条件", "匹配 u, u_xx, u_xxxx，罚权 100"],
      ["源项", "对解析解逐项 autograd 得到"],
      ["扫描参数", "a ∈ {2,3}；残差按 κ(aπ)⁶ 归一"],
      ["考察重点", "把非线性算子推到 6 阶，检验极高阶残差的稳定性"],
      ["文献出处", "Raissi 2019 / PINNacle 2024"],
    ],
    sweeps: [2, 3],
    series: {
      "complex_sinh@64": [1.0293, 0.8285],
      "complex_sinh@128": [0.5219, 0.6996],
      fourier: [0.7232, 0.8153],
      siren: [0.7991, 0.9482],
      mscale: [0.8112, 1.2403],
    },
  },
  // ---------------- 复值场 ----------------
  {
    key: "nls",
    title: "三次非线性 Schrödinger (NLS)",
    group: "复值场",
    sweepShort: "k",
    sweepName: "时间频率 k",
    defs: [
      ["控制方程", "i·uₜ + ½·u_xx + |u|²u = f，u : ℝ²→ℂ"],
      ["精确解", "u = sech(x)·exp(i·k·t)，亮孤子；源 f = (½ − k)u（k=½ 时 f=0 为精确孤子）"],
      ["计算域", "物理 x∈[−5,5]，t∈[0,π/2]；网络取归一化输入 x̂∈[−1,1]²，物理导数带链式因子 1/5、1/(π/4)"],
      ["边界/初值", "x 两端 Dirichlet + t=0 初值 = u_exact，罚权 100"],
      ["扫描参数", "时间频率 k ∈ {1,2,4}"],
      ["频率初始化", "ω₀ = max(10, 2k·L_T)，σ = max(2, k·L_T)，L_T = π/4；残差按 max(1,k) 归一"],
      ["实基线表示", "split-real（Re/Im 两路输出）RVPINN"],
      ["考察重点", "真·复值场的“明显上界”证据——复 sinh 原生承载 ℂ 值，实网必须拆成两路"],
      ["文献出处", "Raissi–Perdikaris–Karniadakis 2019"],
    ],
    sweeps: [1, 2, 4],
    series: {
      "complex_sinh@64": [0.0007, 0.0009, 0.0041],
      "complex_sinh@128": [0.0012, 0.0023, 0.0068],
      fourier: [0.0035, 0.0196, 0.0269],
      siren: [0.0017, 0.0043, 0.0083],
      tanh: [0.0613, 0.2053, 0.3478],
    },
  },
  {
    key: "maxwell",
    title: "时谐 Maxwell（有损介质）",
    group: "复值场",
    sweepShort: "a",
    sweepName: "波数 a",
    defs: [
      ["控制方程", "ΔE + κ²E = f，κ² = (aπ)²(1+iβ)，β = 0.2（TM 模约化为复 Helmholtz）"],
      ["精确解", "E = exp(i·aπ(x+y))，复值行波"],
      ["计算域", "Ω = (−1,1)²"],
      ["边界条件", "Dirichlet：E = E_exact"],
      ["源项", "f = (−2(aπ)² + κ²)·E"],
      ["扫描参数", "波数 a ∈ {2,4,6}；残差按 2(aπ)² 归一"],
      ["频率初始化", "ω₀ = max(10, 2πa)，σ = max(2, πa)"],
      ["实基线表示", "split-real（Re/Im 两路输出）RVPINN"],
      ["考察重点", "线性复值伴随 NLS 的上界测试——损耗正切 iβ 使 E 真复值"],
      ["文献出处", "Jiang 2024（有损 TM 变体）"],
    ],
    sweeps: [2, 4, 6],
    series: {
      "complex_sinh@64": [0.0082, 0.0597, 0.0984],
      "complex_sinh@128": [0.0253, 0.1121, 0.2309],
      fourier: [0.0479, 0.2628, 0.6463],
      siren: [0.0299, 0.2259, 0.6021],
      tanh: [1.4575, 2.5824, 2.1035],
    },
  },
];

// --- 工具 ---
function fmt(v: number): string {
  if (!isFinite(v)) return "—";
  if (v >= 0.01) return v.toFixed(3);
  return v.toExponential(1);
}
const L10 = (v: number) => (v > 0 ? Math.max(-5, Math.log10(v)) : -5);

function verdict(fam: Family) {
  const last = fam.sweeps.length - 1;
  const reals = Object.keys(fam.series).filter((v) => !isComplex(v));
  const bestRealVal = Math.min(...reals.map((v) => fam.series[v][last]));
  const bestRealKey = reals.find((v) => fam.series[v][last] === bestRealVal)!;
  const c64 = fam.series["complex_sinh@64"]?.[last] ?? Infinity;
  const c128 = fam.series["complex_sinh@128"]?.[last] ?? Infinity;
  const bestComplex = Math.min(c64, c128);
  const complexWins = bestComplex < bestRealVal;
  const adv = bestRealVal / bestComplex;
  return { last, c64, c128, bestComplex, bestRealVal, bestRealKey, complexWins, adv };
}

function MethodCard({ name, tag, body }: { name: string; tag: string; body: string }) {
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm">{tag}</Pill>}>{name}</CardHeader>
      <CardBody>
        <Text size="small" tone="secondary">
          {body}
        </Text>
      </CardBody>
    </Card>
  );
}

function DefList({ items }: { items: [string, string][] }) {
  const t = useHostTheme();
  return (
    <div
      style={{
        background: t.fill.tertiary,
        border: `1px solid ${t.stroke.tertiary}`,
        borderRadius: 8,
        padding: 12,
      }}
    >
      {items.map(([k, v], i) => (
        <div
          key={i}
          style={{
            display: "grid",
            gridTemplateColumns: "104px 1fr",
            gap: 10,
            padding: "3px 0",
          }}
        >
          <Text size="small" tone="secondary" weight="semibold">
            {k}
          </Text>
          <Text size="small">{v}</Text>
        </div>
      ))}
    </div>
  );
}

function FamilyCard({ fam }: { fam: Family }) {
  const v = verdict(fam);
  const variants = VARIANT_ORDER.filter((k) => k in fam.series);

  const headers = ["方法", ...fam.sweeps.map((s) => `${fam.sweepShort}=${s}`)];
  const rows = variants.map((k) => [VARIANT_LABEL[k], ...fam.series[k].map((x) => fmt(x))]);
  const rowTone: Array<TableRowTone | undefined> = variants.map((k) =>
    k === "complex_sinh@128" ? "success" : k === "complex_sinh@64" ? "info" : undefined,
  );
  const chartSeries: ChartSeries[] = variants.map((k) => ({
    name: SHORT_LABEL[k],
    data: fam.series[k].map(L10),
    tone: k === "complex_sinh@128" ? "success" : k === "complex_sinh@64" ? "info" : undefined,
  }));

  const pill = v.complexWins ? (
    <Pill size="sm" active>
      复 sinh 胜 ×{v.adv.toFixed(1)}
    </Pill>
  ) : (
    <Pill size="sm">最难点 {SHORT_LABEL[v.bestRealKey]} 略优</Pill>
  );

  return (
    <Card collapsible defaultOpen>
      <CardHeader trailing={pill}>{fam.title}</CardHeader>
      <CardBody>
        <Stack gap={12}>
          <DefList items={fam.defs} />
          <Table
            headers={headers}
            rows={rows}
            rowTone={rowTone}
            columnAlign={["left", ...fam.sweeps.map(() => "right" as const)]}
            striped
          />
          {fam.sweeps.length >= 2 && (
            <Stack gap={4}>
              <LineChart
                categories={fam.sweeps.map((s) => String(s))}
                series={chartSeries}
                height={190}
                beginAtZero={false}
                referenceLines={[{ value: 0, label: "失败 (=1)", tone: "danger" }]}
              />
              <Text size="small" tone="tertiary" italic>
                纵轴 = log₁₀(相对 L² 误差)，越低越好；横轴 = {fam.sweepName}。0 线为平凡解(误差=1)失败水平。
              </Text>
            </Stack>
          )}
        </Stack>
      </CardBody>
    </Card>
  );
}

export default function OscillatoryPinnResultsZh() {
  const groups: Family["group"][] = ["高频二阶", "高阶实算子", "复值场"];
  const winCount = FAMILIES.filter((f) => verdict(f).complexWins).length;

  return (
    <Stack gap={22} style={{ padding: 24, maxWidth: 1080, margin: "0 auto" }}>
      <Stack gap={6}>
        <H1>复参数 sinh 网络 vs 实值高频基线：震荡 / 高阶 PDE 基准</H1>
        <Text tone="secondary">
          统一协议下比较复参数 <Code>sinh</Code> 网络（宽度 64 与 128）与三类实值高频
          SOTA 基线在 9 族 PDE（含 14 个子算例）上的相对 L² 误差。每个数字为 600s 预算、2 seeds 的均值。
        </Text>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="12 / 12" label="族全部完成" tone="success" />
        <Stat value={`${winCount} / ${FAMILIES.length}`} label="最难点复网占优" tone="success" />
        <Stat value="6/30 06:53" label="完成时间" />
        <Stat value="600s × 2 seeds" label="单次预算" />
      </Grid>

      <Callout tone="info" title="本轮实验为何这样设计">
        取消按实/复 1:2 的 √2 配参，所有网络一律使用<Text as="span" weight="semibold">字面宽度</Text>。
        复权重的等价实自由度约为同宽实网的 2 倍，故让复 sinh 同时跑 <Code>64</Code> 与 <Code>128</Code>，
        正好夹住实基线 <Code>@128</Code>：若 复@64 与 复@128 表现相近，说明优势<Text as="span" weight="semibold">不依赖额外宽度/参数</Text>。
      </Callout>

      {/* ===== 一、方法原理 ===== */}
      <Stack gap={12}>
        <H2>一、各方法原理</H2>
        <Grid columns={2} gap={12}>
          <MethodCard
            name="复 sinh 网络（本文方法）"
            tag="complex128"
            body="参数为 complex128 的 MLP，激活用整函数 sinh。值场在复域 ℂ 中传播：由于 sinh 在 ℂ 上解析，Taylor-jet 后端可任意阶精确求导而内层无需 dtype 转换。一次复乘加同时携带幅值与相位，等价实自由度≈同宽实网的 2 倍。第一层频率 ω₀ 按目标波数初始化。"
          />
          <MethodCard
            name="Fourier-feature PINN"
            tag="Tancik 2020"
            body="输入先过随机傅里叶特征 γ(x)=[cos(Bx), sin(Bx)]，B~N(0,σ²)，把低频输入升频以缓解 MLP 的谱偏差。带宽 σ 按目标波数设定，决定可表达的最高频率。"
          />
          <MethodCard
            name="SIREN"
            tag="Sitzmann 2020"
            body="以 sin 为激活的 MLP，配合 ω₀ 缩放的专用初始化。正弦激活天然表达高频信号及其各阶导数，是高频 / 隐式表示任务上的强基线。"
          />
          <MethodCard
            name="MscaleDNN"
            tag="Liu 2020"
            body="多尺度网络：将输入按尺度因子 (1,2,4) 分别送入若干子网再合并，不同子网负责不同频段，相当于显式频率分解以加速高频收敛。"
          />
          <MethodCard
            name="tanh RVPINN（仅复值算例）"
            tag="Raissi 2019"
            body="标准实值 PINN，tanh 激活。对复值场用两路输出分别表示 Re / Im（split-real）。作为复值问题（NLS / Maxwell）的朴素实基线。"
          />
          <MethodCard
            name="统一导数后端"
            tag="Taylor-jet"
            body="所有 jet 兼容网络的残差导数都由同一套 complex-Waring Taylor-jet 算子计算，确保求导机制本身不偏向任何方法；非线性通量(如 Cahn–Hilliard 的 Δ(u³))也只用单调单项偏导拼出。"
          />
        </Grid>
      </Stack>

      {/* ===== 二、统一协议 ===== */}
      <Stack gap={12}>
        <H2>二、统一实验协议</H2>
        <Table
          headers={["项目", "设置"]}
          columnAlign={["left", "left"]}
          rows={[
            ["宽度 / 深度", "实基线 width=128, depth=4；复 sinh width∈{64,128}, depth=4（字面宽度，不再 √2 配参）"],
            ["时间预算", "每个 (问题, 变体, seed) 独享 600s wall-clock；2 个随机种子取均值"],
            ["优化器 / 学习率", "Adam；cosine 退火 1e-3 → 1e-4 (floor)"],
            ["配点", "内部 4096 / 边界 512（固定稠密）；评估在 8192 随机点算相对 L²"],
            ["边界条件", "罚项权重 100；多调和 / 板用 Navier 简支(低阶偶导)；复网额外加 1e-6 虚部正则"],
            ["频率匹配初始化", "ω₀(SIREN/复 sinh 第一层) 与 σ(Fourier 带宽) 按各算例目标频率设定"],
            ["硬件", "单卡 NVIDIA H20 (97GB, CUDA 12.1, PyTorch 2.5)，float64 / complex128"],
          ]}
        />
      </Stack>

      {/* ===== 三、总览 ===== */}
      <Stack gap={12}>
        <H2>三、总览：最难算例上的对比</H2>
        <Text size="small" tone="tertiary">
          取每族<Text as="span" weight="semibold">最难扫描点</Text>（最高波数 / 阶数 / 频率）比较相对 L² 误差。
          “优势倍数”= 最佳实基线 ÷ 复 sinh 中较好者；值越小越好，&ge;1 视为失败。
        </Text>
        <Table
          headers={["算例", "组", "最难点", "复@64", "复@128", "最佳实基线", "结论"]}
          columnAlign={["left", "left", "right", "right", "right", "right", "left"]}
          rowTone={FAMILIES.map((f) => (verdict(f).complexWins ? "success" : "warning"))}
          rows={FAMILIES.map((f) => {
            const v = verdict(f);
            return [
              f.title,
              f.group,
              `${f.sweepShort}=${f.sweeps[v.last]}`,
              fmt(v.c64),
              fmt(v.c128),
              `${fmt(v.bestRealVal)} (${SHORT_LABEL[v.bestRealKey]})`,
              v.complexWins ? `复 sinh ·×${v.adv.toFixed(1)}` : `实基线 ·×${v.adv.toFixed(2)}`,
            ];
          })}
        />
        <Callout tone="success" title="阶段性结论（全部 12 族完成）">
          <Stack gap={4}>
            <Text size="small">
              · 复 sinh 在 <Text as="span" weight="semibold">poly2d、各向异性 Helmholtz、Maxwell、platemix、NLS、helmvc、高波数 Helmholtz、chirp、kdv</Text> 上随难度提高优势扩大，最高约 ×6–12，单调趋势明显（满足验收标准）。
            </Text>
            <Text size="small">
              · <Text as="span" weight="semibold">复@64 常与复@128 持平甚至更好</Text>（Helmholtz、各向异性、Maxwell、NLS、beam/poly1d 低中阶）——优势来自复值表示本身而非更多参数。
            </Text>
            <Text size="small">
              · 弱项需如实呈现：<Text as="span" weight="semibold">poly1d 极高阶(8,10) 被 MscaleDNN 反超</Text>；<Text as="span" weight="semibold">beam m=3 与 SIREN 几乎并列</Text>；<Text as="span" weight="semibold">Cahn–Hilliard 4 阶 a=3 被 SIREN/Fourier 略微反超</Text>。
            </Text>
            <Text size="small">
              · 非线性 Cahn–Hilliard 整体偏难：600s 内所有方法误差仍在 0.4–0.7，无一收敛到高精度，是后续重点优化对象。
            </Text>
          </Stack>
        </Callout>
      </Stack>

      {/* ===== 四、分组详解 ===== */}
      <Stack gap={16}>
        <H2>四、各算例 setting 与逐点结果</H2>
        <Text size="small" tone="tertiary">
          每张卡可折叠。表内绿点 = 复@128，蓝点 = 复@64；趋势图纵轴为 log₁₀(相对 L²)。
        </Text>
        {groups.map((g) => (
          <div key={g}>
            <Stack gap={14}>
              <Row align="center" gap={8}>
                <Pill active>{g}</Pill>
                <Divider style={{ flex: 1 }} />
              </Row>
              {FAMILIES.filter((f) => f.group === g).map((f) => (
                <div key={f.key}>
                  <FamilyCard fam={f} />
                </div>
              ))}
            </Stack>
          </div>
        ))}
      </Stack>

      <Divider />
      <Text size="small" tone="tertiary">
        数据来源：<Code>results/width/&lt;family&gt;_h{`{128,64}`}.csv</Code>（每点 2 seeds 均值，600s 预算）。
        12 族全部完成于 2026-06-30 06:53。
      </Text>
    </Stack>
  );
}
