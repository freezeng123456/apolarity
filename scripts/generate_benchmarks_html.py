#!/usr/bin/env python3
"""Generate docs/apolarity_benchmarks.html — full benchmark reference."""
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "apolarity_benchmarks.html"

VARIANTS = ["complex_sinh", "fourier", "siren", "mscale"]

FAMILIES = [
    {
        "id": "helmholtz",
        "title": "Helmholtz 高波数（实值）",
        "script": "experiments/helmholtz/exp_helmholtz_highk.py",
        "data": "experiments/helmholtz/data/helmholtz_h128.csv",
        "purpose": "谱偏置主战场：分离正弦本征模 u=sin(aπx)sin(aπy)，随波数 a 增大考察 complex sinh 相对实基线的优势。",
        "pde": r"Δu + κ²u = f，κ = aπ，f = −(aπ)²u",
        "domain": "(-1,1)²",
        "bc": "Dirichlet：u = 0（制造解在边界自动为零）",
        "exact": r"u = sin(aπx) sin(aπy)",
        "order": 2,
        "problems": [
            ("helmholtz_a2", "a=2"), ("helmholtz_a4", "a=4"), ("helmholtz_a6", "a=6"),
            ("helmholtz_a8", "a=8"), ("helmholtz_a10", "a=10"),
        ],
        "hist_file": "helmholtz_h128_history.json",
    },
    {
        "id": "helm_aniso",
        "title": "Helmholtz 各向异性（Wang 2021）",
        "script": "experiments/helmholtz/exp_helmholtz_highk.py --aniso",
        "data": "experiments/helmholtz/data/helmholtz_aniso_h128.csv",
        "purpose": "方向各向异性 (a₁,a₂)=(1,4)，经典梯度病理算例。",
        "pde": r"Δu + k²u = q，k=1，q = (k² − ((a₁π)²+(a₂π)²))u",
        "domain": "(-1,1)²",
        "bc": "Dirichlet u = 0",
        "exact": r"u = sin(a₁πx) sin(a₂πy)",
        "order": 2,
        "problems": [("helm_aniso_1_4", "a₁=1, a₂=4")],
        "hist_file": "helmholtz_aniso_h128_history.json",
        "family_key": "helmholtz",
    },
    {
        "id": "helmholtz_vc",
        "title": "变系数 Helmholtz（散射介质）",
        "script": "experiments/helmholtz_vc/exp_helmholtz_vc.py",
        "data": "experiments/helmholtz_vc/data/helmvc_h128.csv",
        "purpose": "空间变化 zeroth 阶系数 κ²(x)，考察非均匀介质鲁棒性。",
        "pde": r"Δu + κ²(x)u = f，κ²(x)=(aπ)²(1+0.5·sin(πx)sin(πy))",
        "domain": "(-1,1)²",
        "bc": "Dirichlet u = 0",
        "exact": r"u = sin(aπx) sin(aπy)",
        "order": 2,
        "problems": [("helmvc_a2", "a=2"), ("helmvc_a4", "a=4"), ("helmvc_a6", "a=6")],
        "hist_file": "helmvc_h128_history.json",
    },
    {
        "id": "chirp",
        "title": "Chirp 非分离振荡（表达力测试）",
        "script": "experiments/chirp/exp_chirp.py",
        "data": "experiments/chirp/data/chirp_h128.csv",
        "purpose": "局部频率随半径增长的 chirp，打破纯 Fourier 模可表示性。",
        "pde": r"−Δu + u = f",
        "domain": "(-1,1)²",
        "bc": "Dirichlet u = u_exact",
        "exact": r"u = sin(aπ(x²+y²)/2)",
        "order": 2,
        "problems": [("chirp_a2", "a=2"), ("chirp_a4", "a=4"), ("chirp_a6", "a=6"), ("chirp_a8", "a=8")],
        "hist_file": "chirp_h128_history.json",
    },
    {
        "id": "polyharmonic2d",
        "title": "Polyharmonic 2D（固定频率、变阶数）",
        "script": "experiments/polyharmonic/exp_polyharmonic.py",
        "data": "experiments/polyharmonic/data/poly2d_h128.csv",
        "purpose": "固定 u=sin(πx)sin(πy)，只改变算子阶数 m，隔离高阶导数代价。",
        "pde": r"Δ^m u = (−S)^m u，S = 2π²",
        "domain": "(-1,1)²",
        "bc": "Navier：Δ^j u = 0，j = 0,…,m−1",
        "exact": r"u = sin(πx) sin(πy)",
        "order": "2,4,6",
        "problems": [("polyharm2d_o2", "m=1, order=2"), ("polyharm2d_o4", "m=2, order=4"), ("polyharm2d_o6", "m=3, order=6")],
        "hist_file": "poly2d_h128_history.json",
        "family_key": "polyharmonic",
    },
    {
        "id": "polyharmonic1d",
        "title": "Polyharmonic 1D（高阶轴推到 10）",
        "script": "experiments/polyharmonic/exp_polyharmonic.py --dim 1",
        "data": "experiments/polyharmonic/data/poly1d_h128.csv",
        "purpose": "1D 单项偏导算子便宜，可扫到 10 阶。",
        "pde": r"d^(2m)/dx^(2m) u = (−π²)^m u",
        "domain": "(-1,1)",
        "bc": "简支：低阶偶导数在边界为零",
        "exact": r"u = sin(πx)",
        "order": "2–10",
        "problems": [
            ("polyharm1d_o2", "order=2"), ("polyharm1d_o4", "order=4"), ("polyharm1d_o6", "order=6"),
            ("polyharm1d_o8", "order=8"), ("polyharm1d_o10", "order=10"),
        ],
        "hist_file": "poly1d_h128_history.json",
        "family_key": "polyharmonic",
    },
    {
        "id": "plate_beam",
        "title": "板 / 梁 四阶本征模",
        "script": "experiments/plate_beam/exp_plate_beam.py",
        "data": "experiments/plate_beam/data/plate_beam_h128.csv",
        "purpose": "Kirchhoff 板与 Euler-Bernoulli 梁，固定四阶、扫模态数 m。",
        "pde": r"板：Δ²w = S²w；梁：w'''' = (mπ)⁴w",
        "domain": "板 (-1,1)²；梁 (-1,1)",
        "bc": "简支 Navier：w=0，Δw=0（或 w''=0）",
        "exact": r"板 sin(mπx)sin(nπy)；梁 sin(mπx)",
        "order": 4,
        "problems": [
            ("plate_m1", "板 m=1"), ("plate_m2", "板 m=2"), ("plate_m3", "板 m=3"),
            ("beam_m1", "梁 m=1"), ("beam_m2", "梁 m=2"), ("beam_m3", "梁 m=3"),
        ],
        "hist_file": "plate_beam_h128_history.json",
    },
    {
        "id": "platemix",
        "title": "板 各向异性模态",
        "script": "experiments/plate_beam/exp_plate_beam.py --kind mix",
        "data": "experiments/plate_beam/data/platemix_h128.csv",
        "purpose": "非对称 (m, m+1) 板模态，更难于分离频率表示。",
        "pde": r"Δ²w = S²w，S = (m²+n²)π²，n=m+1",
        "domain": "(-1,1)²",
        "bc": "Navier 简支",
        "exact": r"w = sin(mπx) sin((m+1)πy)",
        "order": 4,
        "problems": [("platemix_m2", "m=2"), ("platemix_m3", "m=3"), ("platemix_m4", "m=4")],
        "hist_file": "platemix_h128_history.json",
        "family_key": "plate_beam",
    },
    {
        "id": "kdv",
        "title": "KdV 线性色散波（三阶）",
        "script": "experiments/kdv/exp_kdv_dispersive.py",
        "data": "experiments/kdv/data/kdv_h128.csv",
        "purpose": "奇数阶色散项 u_t + δu_xxx = f，坐标 0=x, 1=t。",
        "pde": r"u_t + δ u_xxx = f，δ=1",
        "domain": "(-1,1)² (x,t)",
        "bc": "Dirichlet u = u_exact",
        "exact": r"u = sin(kπx) cos(kπt)",
        "order": 3,
        "problems": [
            ("kdv_k2", "k=2"), ("kdv_k3", "k=3"), ("kdv_k4", "k=4"),
            ("kdv_k5", "k=5"), ("kdv_k6", "k=6"),
        ],
        "hist_file": "kdv_h128_history.json",
    },
    {
        "id": "cahn_hilliard",
        "title": "Cahn–Hilliard（非线性，4/6 阶）",
        "script": "experiments/cahn_hilliard/exp_cahn_hilliard.py",
        "data": "experiments/cahn_hilliard/data/cahn_hilliard_h128.csv",
        "purpose": "非线性通量 Δ(u³) 用网络单项偏导构造，全残差走 Taylor jet。",
        "pde": r"u_t = M[Δ(u³) − Δu − γΔ²u (+ κΔ³u 若 6 阶)]",
        "domain": "(-1,1)²，0=x, 1=t",
        "bc": "u, u_xx,（6阶时）u_xxxx  Dirichlet",
        "exact": r"u = sin(aπx) cos(aπt)",
        "order": "4 / 6",
        "problems": [
            ("ch4_a2", "4阶 a=2"), ("ch4_a3", "4阶 a=3"),
            ("ch6_a2", "6阶 a=2"), ("ch6_a3", "6阶 a=3"),
        ],
        "hist_file": "cahn_hilliard_h128_history.json",
        "loss_note": "自定义 loss：L = L_int + 100·L_bc；高阶 BC 按 a^|α| 缩放",
    },
    {
        "id": "nls",
        "title": "三次 NLS / Schrödinger（复值）",
        "script": "experiments/nls/exp_nls_schrodinger.py",
        "data": "experiments/nls/data/nls_h128.csv",
        "purpose": "复值上界证据：complex sinh 原生 C，实基线 split-real (RVPINN)。",
        "pde": r"i u_t + ½ u_xx + |u|²u = f",
        "domain": "物理域 x∈[-5,5], t∈[0,π/2]；网络输入归一化到 (-1,1)²",
        "bc": "u = u_exact 于边界/初值面",
        "exact": r"u = sech(x) exp(i k t)，f = (½−k)u",
        "order": 2,
        "problems": [("nls_k1", "k=1"), ("nls_k2", "k=2"), ("nls_k4", "k=4")],
        "hist_file": "nls_h128_history.json",
        "loss_note": "复值残差 |r|²；链式法则 u_t/ LT, u_xx/ LX²",
    },
    {
        "id": "maxwell",
        "title": "时谐 Maxwell TM 有损介质（复值线性）",
        "script": "experiments/maxwell/exp_maxwell.py",
        "data": "experiments/maxwell/data/maxwell_h128.csv",
        "purpose": "复介电常数使场 genuinely complex；线性复值 companion。",
        "pde": r"ΔE + κ²E = f，κ² = (aπ)²(1 + iβ)，β=0.2",
        "domain": "(-1,1)²",
        "bc": "Dirichlet E = E_exact",
        "exact": r"E = exp(i aπ(x+y))",
        "order": 2,
        "problems": [("maxwell_a2", "a=2"), ("maxwell_a4", "a=4"), ("maxwell_a6", "a=6")],
        "hist_file": "maxwell_h128_history.json",
    },
]


def load_agg_fixed():
    out = {}
    for csv_path in sorted((ROOT / "experiments").rglob("*_h128.csv")):
        for r in csv.DictReader(csv_path.open()):
            p, v = r["problem"], r["variant"]
            out.setdefault(p, {}).setdefault(v, {"L2": [], "L_int": [], "steps": []})
            out[p][v]["L2"].append(float(r["L2_err"]))
            out[p][v]["L_int"].append(float(r["L_int_last"]))
            out[p][v]["steps"].append(int(r["steps"]))
    for p in out:
        for v in out[p]:
            d = out[p][v]
            d["L2"] = sum(d["L2"]) / len(d["L2"])
            d["L_int"] = sum(d["L_int"]) / len(d["L_int"])
            d["steps"] = int(sum(d["steps"]) / len(d["steps"]))
    return out


def load_history(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for rec in data:
        out[(rec["problem"], rec["variant"], rec["seed"])] = rec["history"]
    return out


def fmt_sci(x):
    if x < 1e-3 or x >= 100:
        return f"{x:.3e}"
    return f"{x:.4f}"


def best_variant(problem, agg):
    if problem not in agg:
        return "—"
    return min(VARIANTS, key=lambda v: agg[problem].get(v, {"L2": 1e9})["L2"])


def results_table(problem, agg):
    if problem not in agg:
        return "<p><em>无 h128 结果数据</em></p>"
    rows = []
    for v in VARIANTS:
        if v not in agg[problem]:
            continue
        d = agg[problem][v]
        rows.append(
            f"<tr><td>{v}</td><td>{fmt_sci(d['L2'])}</td>"
            f"<td>{fmt_sci(d['L_int'])}</td><td>{d['steps']}</td></tr>"
        )
    best = min(agg[problem].items(), key=lambda x: x[1]["L2"])[0]
    return f"""<table class="results">
<caption>600s 墙钟、2 seeds 平均（<code>hidden=128</code>，字面宽度）· 最优：<code>{best}</code></caption>
<thead><tr><th>架构</th><th>rel L²</th><th>L_int 末值</th><th>步数</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>"""


def history_table(problem, hist, variant="complex_sinh"):
    samples = [0, 60, 120, 300, 600]
    by_t = {t: [] for t in samples}
    for seed in (0, 1):
        h = hist.get((problem, variant, seed))
        if not h:
            continue
        for t in samples:
            pt = min(h, key=lambda x: abs(x[0] - t))
            by_t[t].append(pt)
    if not any(by_t.values()):
        return ""
    rows = []
    for t in samples:
        pts = by_t[t]
        if not pts:
            continue
        l2 = sum(p[1] for p in pts) / len(pts)
        lint = sum(p[2] for p in pts) / len(pts)
        rows.append(f"<tr><td>{t}</td><td>{fmt_sci(l2)}</td><td>{fmt_sci(lint)}</td></tr>")
    return f"""<table class="history">
<caption><code>{variant}</code> 训练曲线采样（两 seed 平均）</caption>
<thead><tr><th>时间 (s)</th><th>rel L²</th><th>L_int</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>"""


def esc(s):
    return html.escape(str(s), quote=False)


def main():
    agg = load_agg_fixed()
    parts = []

    parts.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>apolarity 算例手册</title>
<style>
:root {
  --bg: #fafbfc; --fg: #1a1a2e; --muted: #5c6370; --accent: #2563eb;
  --border: #e2e8f0; --card: #fff; --code-bg: #f1f5f9;
  --best: #059669; --warn: #d97706;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1419; --fg: #e7ecf3; --muted: #9aa4b2; --accent: #60a5fa;
    --border: #2d3748; --card: #1a2332; --code-bg: #243044;
  }
}
* { box-sizing: border-box; }
body { font-family: "Segoe UI", system-ui, sans-serif; line-height: 1.6;
  max-width: 1100px; margin: 0 auto; padding: 1.5rem 2rem 4rem;
  background: var(--bg); color: var(--fg); }
h1 { font-size: 1.75rem; border-bottom: 2px solid var(--accent); padding-bottom: .5rem; }
h2 { margin-top: 2.5rem; font-size: 1.35rem; color: var(--accent); }
h3 { margin-top: 1.75rem; font-size: 1.1rem; }
a { color: var(--accent); }
nav.toc { background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 1rem 1.5rem; margin: 1.5rem 0; }
nav.toc ul { columns: 2; margin: .5rem 0 0; padding-left: 1.2rem; }
nav.toc li { break-inside: avoid; margin: .25rem 0; }
.card { background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
.meta { display: grid; grid-template-columns: 7rem 1fr; gap: .35rem .75rem;
  font-size: .92rem; margin: .75rem 0; }
.meta dt { color: var(--muted); font-weight: 600; }
.meta dd { margin: 0; }
code, .path { font-family: ui-monospace, monospace; font-size: .88em;
  background: var(--code-bg); padding: .1em .35em; border-radius: 4px; }
table { width: 100%; border-collapse: collapse; font-size: .88rem; margin: .75rem 0; }
th, td { border: 1px solid var(--border); padding: .4rem .6rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead { background: var(--code-bg); }
caption { caption-side: top; text-align: left; color: var(--muted); font-size: .85rem; padding: .25rem 0; }
.problem-id { font-family: ui-monospace, monospace; color: var(--accent); }
.tag { display: inline-block; background: var(--code-bg); padding: .15em .5em;
  border-radius: 4px; font-size: .8rem; margin-right: .35rem; }
.summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px,1fr)); gap: .75rem; }
.stat { text-align: center; padding: .75rem; background: var(--card); border-radius: 8px; border: 1px solid var(--border); }
.stat b { display: block; font-size: 1.5rem; color: var(--accent); }
@media print { nav.toc { break-after: page; } .card { break-inside: avoid; } }
</style>
</head>
<body>
<h1>apolarity 振荡高阶 PINN 算例手册</h1>
<p>本文档汇总 <code>experiments/</code> 下 <strong>45 个物理算例</strong> 的问题设定、损失函数、精确解与
<code>data/*_h128.csv</code> 中的代表性结果。生成命令：
<code>python scripts/generate_benchmarks_html.py</code></p>

<div class="summary-grid">
<div class="stat"><b>45</b>物理算例</div>
<div class="stat"><b>9</b>PDE 族</div>
<div class="stat"><b>4</b>对比架构</div>
<div class="stat"><b>600s</b>墙钟预算</div>
</div>
""")

    parts.append("""<nav class="toc" id="toc">
<h2 style="margin-top:0">目录</h2>
<ul>
<li><a href="#protocol">统一实验协议</a></li>
<li><a href="#arch">网络架构对比</a></li>
<li><a href="#loss">损失函数与历史格式</a></li>
<li><a href="#core">核心方法实验</a></li>
""")
    for fam in FAMILIES:
        parts.append(f'<li><a href="#{fam["id"]}">{esc(fam["title"])}</a></li>\n')
    parts.append("</ul></nav>\n")

    parts.append("""<h2 id="protocol">统一实验协议</h2>
<div class="card">
<dl class="meta">
<dt>计算域</dt><dd>默认 <code>(-1,1)^d</code>；NLS 为物理域 x∈[-5,5], t∈[0,π/2] 再归一化输入</dd>
<dt>配点</dt><dd>内点 <code>n_int=4096</code>；边界 <code>n_bc=512</code>（均匀随机重采样，每步固定一批）</dd>
<dt>优化</dt><dd>Adam <code>lr=1e-3</code>，余弦退火至 <code>lr/10</code>（<code>lr_schedule=cosine</code>）</dd>
<dt>预算</dt><dd>墙钟 <strong>600 s</strong> / run（脚本默认 80–120s，data 为 600s 正式跑）</dd>
<dt>重复</dt><dd><code>seeds=2</code>（seed 0, 1）</dd>
<dt>深度</dt><dd><code>depth=4</code></dd>
<dt>宽度</dt><dd>所有架构使用<strong>字面宽度</strong> <code>--hidden</code>（不做 √2 缩放）。正式 width study：实基线 <code>hidden=128</code>；<code>complex_sinh</code> 同时跑 <code>h=64</code> 与 <code>h=128</code>（两档复网夹住实基线参数量）</dd>
<dt>架构</dt><dd><code>complex_sinh, fourier, siren, mscale</code>；<code>omega0</code> / <code>fourier_sigma</code> 按算例频率匹配</dd>
<dt>参数量</dt><dd><code>n_params</code> 中每个复权重计 2 个实 DOF；同宽度下 complex@128 的计数约为 siren@128 的约 2 倍。对照数据见各族 <code>*_h64.*</code> 与 <code>*_h128.*</code></dd>
<dt>评估</dt><dd>8192 个均匀随机点的相对 L²：<code>‖u−u*‖₂ / ‖u*‖₂</code></dd>
<dt>导数后端</dt><dd>jet 架构用 complex-Waring + Taylor jet；Cauchy 等 fallback 标为 autograd</dd>
</dl>
</div>

<h2 id="arch">网络架构与宽度对照</h2>
<div class="card">
<p>对比原则（见 <code>experiments/README.md</code>、<code>variant_width</code>）：<strong>一律使用字面宽度</strong>，不按参数量把实基线缩放到 <code>√2·H</code>。复权在 <code>n_params</code> 里按「实部+虚部」计 2 个自由度，故在相同 <code>hidden</code> 下复网络可调参数更多。width study 让 <code>complex_sinh@64</code> 与 <code>complex_sinh@128</code> 分别位于实基线 <code>@128</code> 的两侧；若 <code>complex@64 ≈ complex@128</code>，说明方法对宽度不敏感。</p>
<table>
<thead><tr><th>variant</th><th>说明</th><th>n_params（d=2, depth=4, hidden=128）</th></tr></thead>
<tbody>
<tr><td><code>complex_sinh</code></td><td>复值 sinh + holomorphic 频率初始化（主方法）；另跑 hidden=64 一档</td><td>≈100k</td></tr>
<tr><td><code>fourier</code></td><td>Fourier features + tanh MLP，σ 随目标频率匹配</td><td>≈66k</td></tr>
<tr><td><code>siren</code></td><td>SIREN (sin 激活)，ω₀ 匹配</td><td>≈50k</td></tr>
<tr><td><code>mscale</code></td><td>多尺度 DNN（多子网，参数量最大）</td><td>≈150k</td></tr>
</tbody></table>
<p>复值算例（NLS / Maxwell）中，实基线为 <strong>split-real RVPINN</strong>：两个独立的实网络分别拟合 Re u 与 Im u，各用同一字面宽度 <code>H</code>（不是单个 √2 放大的网络）。</p>
<p>本文结果表默认来自 <code>*_h128.csv</code>；<code>*_h64.csv</code> 为 <code>complex_sinh</code> 的窄网对照，见各 <code>experiments/&lt;族&gt;/data/</code>。</p>
</div>

<h2 id="loss">损失函数与历史格式</h2>
<div class="card">
<h3>线性算例（<code>run_linear_suite</code>）</h3>
<pre>L = L_int + 100 · L_bc  (+ 1e-6 · Im(W)² 若复参数)

L_int = mean( ((Lu − f) / res_scale)² )
L_bc  = mean( (u − g)² )  + Navier 项（高阶 Δ^j u）
</pre>
<h3>非线性 / 复值算例</h3>
<ul>
<li><strong>Cahn–Hilliard</strong>：非线性项 <code>Δ(u³)=3u²u_xx+6u(u_x)²</code> 由单项偏导组装；BC 含 u, u_xx, (6阶) u_xxxx</li>
<li><strong>NLS / Maxwell</strong>：复值残差模方；split-real 基线输出 (Re, Im) 两通道</li>
</ul>
<h3>历史 JSON</h3>
<p>路径：<code>experiments/&lt;族&gt;/data/&lt;stem&gt;_h128_history.json</code></p>
<p>每条记录：<code>[elapsed_s, rel_L2_err, L_int]</code>，约 40 个时间点均匀覆盖 600s。</p>
</div>

<h2 id="core">核心方法实验（非振荡套件）</h2>
<div class="card">
<p>位于 <code>experiments/core_method/</code>，验证 Taylor-jet 后端本身，而非 PDE 扫参。</p>
<table>
<thead><tr><th>脚本</th><th>作用</th></tr></thead>
<tbody>
<tr><td><code>benchmark_single_monomial.py</code></td><td>单项高阶导数微基准：autodiff vs jet vs complex-Waring</td></tr>
<tr><td><code>profile_complex_waring_steps.py</code></td><td>complex-Waring 分步剖析</td></tr>
<tr><td><code>train_pinn_monomial.py</code></td><td>制造解 ∂^α u = f_α 的小 PINN 案例</td></tr>
<tr><td><code>train_pinn_ch_sixth_order.py</code></td><td>4D Cahn–Hilliard 六阶 PINN（论文 §5.4）</td></tr>
<tr><td><code>generate_paper_tables.py</code></td><td>CSV → LaTeX 表格行</td></tr>
</tbody></table>
</div>
""")

    total_problems = 0
    for fam in FAMILIES:
        fam_dir = fam.get("family_key", fam["id"].replace("helm_aniso", "helmholtz").split("_")[0])
        if fam["id"] == "helm_aniso":
            fam_dir = "helmholtz"
        elif fam["id"].startswith("polyharmonic"):
            fam_dir = "polyharmonic"
        elif fam["id"] == "platemix":
            fam_dir = "plate_beam"
        hist_path = ROOT / "experiments" / fam_dir / "data" / fam["hist_file"]
        hist = load_history(hist_path)

        parts.append(f'<h2 id="{fam["id"]}">{esc(fam["title"])}</h2>\n')
        parts.append('<div class="card">\n')
        parts.append(f'<p>{esc(fam["purpose"])}</p>\n')
        parts.append('<dl class="meta">\n')
        for label, key in [("PDE", "pde"), ("域", "domain"), ("边界", "bc"), ("精确解", "exact"), ("阶数", "order")]:
            parts.append(f'<dt>{label}</dt><dd>{esc(fam[key])}</dd>\n')
        parts.append(f'<dt>脚本</dt><dd><code class="path">{esc(fam["script"])}</code></dd>\n')
        parts.append(f'<dt>数据</dt><dd><code class="path">{esc(fam["data"])}</code></dd>\n')
        if fam.get("loss_note"):
            parts.append(f'<dt>Loss</dt><dd>{esc(fam["loss_note"])}</dd>\n')
        parts.append("</dl></div>\n")

        for pname, plabel in fam["problems"]:
            total_problems += 1
            parts.append(f'<h3 id="{pname}"><span class="problem-id">{esc(pname)}</span> <span class="tag">{esc(plabel)}</span></h3>\n')
            parts.append(results_table(pname, agg))
            ht = history_table(pname, hist)
            if ht:
                parts.append(ht)
            parts.append("<hr/>\n")

    parts.append(f"""
<footer style="margin-top:3rem;color:var(--muted);font-size:.85rem">
<p>共收录 <strong>{total_problems}</strong> 个算例 · 项目根目录 <code>{esc(str(ROOT))}</code></p>
<p>复现示例：<code>python experiments/helmholtz/exp_helmholtz_highk.py --seconds 600 --history --hidden 128</code></p>
</footer>
</body></html>
""")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {OUT} ({len(parts[0])} chars, {total_problems} problems)")


if __name__ == "__main__":
    main()
