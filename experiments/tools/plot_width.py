#!/usr/bin/env python3
"""Per-family figures for the 600s WIDTH study (self-contained, JCP style).

For each family it reads the family's own data folder
``experiments/<dir>/data/<stem>_h128.csv`` (complex sinh + the real baselines,
all at width 128) and ``<stem>_h64.csv`` (complex sinh at width 64), plus the
``*_history.json`` convergence sidecars, relabels the complex series by width
(``complex_sinh_128`` / ``complex_sinh_64``) and emits the 3-panel figure
``docs/paper/figures/fig_<key>.pdf``:

  (a) relative L2 vs the physics sweep (order / wavenumber / mode)
  (b) relative L2 vs training time at a representative setting
  (c) interior residual loss vs training time at a representative setting

The complex curves at 64 and 128 sitting on top of each other is the message:
the method is insensitive to width even though a complex weight carries ~2x the
real DOF of the width-128 real baselines.

Run:  python experiments/tools/plot_width.py          (FIG_PNG=1 also writes PNGs)
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
OUT = ROOT / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.4,
    "axes.grid": True, "grid.alpha": 0.25,
})

# variant -> (label, color, linestyle, marker)
STYLE = {
    "complex_sinh_128": (r"complex sinh ($W{=}128$)", "#1f77b4", "-", "o"),
    "complex_sinh_64":  (r"complex sinh ($W{=}64$)",  "#17becf", (0, (4, 2)), "o"),
    "fourier":      ("Fourier",      "#2ca02c", "--", "s"),
    "siren":        ("SIREN",        "#ff7f0e", "-.", "^"),
    "mscale":       ("MscaleDNN",    "#9467bd", ":", "D"),
    "tanh":         ("split tanh",   "#d62728", ":", "x"),
}
EPS_L2, EPS_LOSS = 1e-7, 1e-13


def csv_means(path, prob_filter=None):
    """variant -> sorted [(sweep, mean rel-L2)] from a results CSV path."""
    if path is None or not Path(path).exists():
        return {}
    by = defaultdict(lambda: defaultdict(list))
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            if prob_filter and not prob_filter(r["problem"]):
                continue
            try:
                l2 = float(r["L2_err"])
            except (KeyError, ValueError):
                continue
            if not np.isfinite(l2):
                l2 = 1.5
            by[r["variant"]][float(r["sweep"])].append(l2)
    return {v: sorted((s, float(np.mean(vals))) for s, vals in d.items())
            for v, d in by.items()}


def hist_traces(path, prob: str):
    """variant -> list of seed traces (np arrays [t, L2, loss]) from a sidecar."""
    if path is None or not Path(path).exists():
        return {}
    by = defaultdict(list)
    for rec in json.load(Path(path).open()):
        if rec.get("problem") != prob:
            continue
        arr = np.asarray(rec["history"], dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 2:
            by[rec["variant"]].append(arr)
    return by


def mean_on_grid(traces, col, eps):
    """Geometric-mean curve (+ min/max band) over seeds on a common time grid."""
    tmax = min(tr[-1, 0] for tr in traces)
    if tmax <= 0:
        tmax = max(tr[-1, 0] for tr in traces)
    grid = np.linspace(0.0, tmax, 60)
    vals = []
    for tr in traces:
        v = np.maximum(tr[:, col], eps)
        vals.append(np.interp(grid, tr[:, 0], v))
    vals = np.log(np.array(vals))
    return grid, np.exp(vals.mean(0)), np.exp(vals.min(0)), np.exp(vals.max(0))


def _order(variants):
    pref = ["complex_sinh_128", "complex_sinh_64", "fourier", "siren", "mscale", "tanh"]
    return [v for v in pref if v in variants]


def panel_sweep(ax, means, xlabel, logx=False):
    for v in _order(means):
        if not means[v]:
            continue
        lab, c, ls, mk = STYLE[v]
        xs, ys = zip(*means[v])
        ax.plot(xs, np.maximum(ys, EPS_L2), color=c, linestyle=ls, marker=mk,
                ms=4, label=lab)
    ax.axhline(1.0, color="0.5", lw=0.8, ls=(0, (1, 2)))
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("(a) accuracy vs sweep")


def panel_time(ax, traces, col, eps, ylabel, title):
    for v in _order(traces):
        trs = traces[v]
        if not trs:
            continue
        lab, c, ls, mk = STYLE[v]
        g, m, lo, hi = mean_on_grid(trs, col, eps)
        ax.plot(g, m, color=c, linestyle=ls, label=lab)
        ax.fill_between(g, lo, hi, color=c, alpha=0.12, linewidth=0)
    if col == 1:
        ax.axhline(1.0, color="0.5", lw=0.8, ls=(0, (1, 2)))
    ax.set_yscale("log")
    ax.set_xlabel("training time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


FAMILIES = [
    dict(key="poly1d", dir="polyharmonic", stem="poly1d", xlabel=r"operator order $m$",
         prob="polyharm1d_o6", rep=r"$m=6$"),
    dict(key="poly2d", dir="polyharmonic", stem="poly2d", xlabel=r"operator order $m$",
         prob="polyharm2d_o6", rep=r"$m=6$"),
    dict(key="helmholtz", dir="helmholtz", stem="helmholtz", xlabel=r"wavenumber $a$",
         prob="helmholtz_a6", rep=r"$a=6$"),
    dict(key="helmvc", dir="helmholtz_vc", stem="helmvc", xlabel=r"wavenumber $a$",
         prob="helmvc_a6", rep=r"$a=6$"),
    dict(key="chirp", dir="chirp", stem="chirp", xlabel=r"chirp rate $a$",
         prob="chirp_a4", rep=r"$a=4$"),
    dict(key="plate", dir="plate_beam", stem="plate_beam", xlabel=r"plate mode $m$",
         pf=lambda p: p.startswith("plate_m"), prob="plate_m2", rep=r"$m=2$"),
    dict(key="beam", dir="plate_beam", stem="plate_beam", xlabel=r"beam mode $m$",
         pf=lambda p: p.startswith("beam_m"), prob="beam_m2", rep=r"$m=2$"),
    dict(key="platemix", dir="plate_beam", stem="platemix", xlabel=r"mixed plate mode $m$",
         prob="platemix_m3", rep=r"modes $(3,4)$"),
    dict(key="kdv", dir="kdv", stem="kdv", xlabel=r"wavenumber $k$",
         prob="kdv_k4", rep=r"$k=4$"),
    dict(key="cahn", dir="cahn_hilliard", stem="cahn_hilliard", xlabel=r"amplitude $a$",
         pf=lambda p: p.startswith("ch6"), prob="ch6_a2", rep=r"6th order, $a=2$"),
    dict(key="nls", dir="nls", stem="nls", xlabel=r"wavenumber $k$",
         prob="nls_k2", rep=r"$k=2$"),
    dict(key="maxwell", dir="maxwell", stem="maxwell", xlabel=r"loss/freq scale $a$",
         prob="maxwell_a4", rep=r"$a=4$"),
]


def _data(fam):
    return EXP / fam["dir"] / "data"


def _exists(p):
    return p if p.exists() else None


def load(fam):
    pf = fam.get("pf")
    d = _data(fam)
    c128 = _exists(d / f"{fam['stem']}_h128.csv")
    c64 = _exists(d / f"{fam['stem']}_h64.csv")
    h128 = _exists(d / f"{fam['stem']}_h128_history.json")
    h64 = _exists(d / f"{fam['stem']}_h64_history.json")

    means = csv_means(c128, pf)
    m64 = csv_means(c64, pf)
    if "complex_sinh" in means:
        means["complex_sinh_128"] = means.pop("complex_sinh")
    if "complex_sinh" in m64:
        means["complex_sinh_64"] = m64["complex_sinh"]

    traces = hist_traces(h128, fam["prob"])
    t64 = hist_traces(h64, fam["prob"])
    if "complex_sinh" in traces:
        traces["complex_sinh_128"] = traces.pop("complex_sinh")
    if "complex_sinh" in t64:
        traces["complex_sinh_64"] = t64["complex_sinh"]
    return means, traces


def main():
    made = 0
    for fam in FAMILIES:
        means, traces = load(fam)
        if not means and not traces:
            print(f"[skip] {fam['key']}: no data in {_data(fam)}")
            continue
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
        panel_sweep(axes[0], means, fam["xlabel"])
        panel_time(axes[1], traces, 1, EPS_L2, r"relative $L^2$ error",
                   f"(b) accuracy vs time ({fam['rep']})")
        panel_time(axes[2], traces, 2, EPS_LOSS, "interior residual loss",
                   f"(c) loss vs time ({fam['rep']})")
        h, l = axes[1].get_legend_handles_labels()
        if not h:
            h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False,
                   bbox_to_anchor=(0.5, -0.06))
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        path = OUT / f"fig_{fam['key']}.pdf"
        fig.savefig(path)
        if os.environ.get("FIG_PNG"):
            fig.savefig(path.with_suffix(".png"), dpi=130)
        plt.close(fig)
        made += 1
        print(f"[ok] {path}  ({len(traces)} series)")
    print(f"\n[done] {made} figures -> {OUT}")


if __name__ == "__main__":
    main()
