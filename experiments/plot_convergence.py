#!/usr/bin/env python3
"""Per-family figures for the oscillatory PINN paper (JCP style).

Each PDE family gets one 1x3 figure:
  (a) relative L2 vs the swept parameter (mean over seeds), from the full
      v3/v4 aggregate CSVs in results/;
  (b) relative L2 vs training time (the real-time accuracy trace);
  (c) interior residual loss vs training time;
with (b),(c) read from the convergence-trace sidecars results/hist/*_history.json
written by scripts/run_history.sh (train_eval --history).

Output: docs/paper/figures/fig_<key>.pdf (vector, for \\includegraphics).
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

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
RES = ROOT / "results"            # legacy flat layout
HIST = RES / "hist"               # legacy flat layout
OUT = ROOT / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def find_csv(fam: dict) -> Path | None:
    """Per-family <fam>/data/<csv> (new layout) with legacy results/<csv> fallback."""
    cands = [EXP / fam["dir"] / "data" / fam["csv"], RES / fam["csv"]]
    return next((p for p in cands if p.exists()), None)


def find_hist(fam: dict) -> Path | None:
    name = f"{fam['hist']}_history.json"
    cands = [EXP / fam["dir"] / "data" / name, HIST / name]
    return next((p for p in cands if p.exists()), None)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.4,
    "axes.grid": True, "grid.alpha": 0.25,
})

# variant -> (label, color, linestyle, marker)
STYLE = {
    "complex_sinh": ("complex sinh", "#1f77b4", "-", "o"),
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
    out = {}
    for v, d in by.items():
        out[v] = sorted((s, float(np.mean(vals))) for s, vals in d.items())
    return out


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
    pref = ["complex_sinh_128", "complex_sinh_64", "complex_sinh",
            "fourier", "siren", "mscale", "tanh"]
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
    dict(key="poly2d", dir="polyharmonic", csv="poly2d_v3.csv",
         xlabel=r"operator order $m$",
         hist="poly2d", prob="polyharm2d_o6", rep=r"$m=6$"),
    dict(key="poly1d", dir="polyharmonic", csv="poly1d_v3.csv",
         xlabel=r"operator order $m$",
         hist="poly1d", prob="polyharm1d_o4", rep=r"$m=4$"),
    dict(key="helmholtz", dir="helmholtz", csv="helmholtz_v3.csv",
         xlabel=r"wavenumber $a$",
         hist="helmholtz", prob="helmholtz_a6", rep=r"$a=6$"),
    dict(key="helmvc", dir="helmholtz_vc", csv="helmvc_v3.csv",
         xlabel=r"wavenumber $a$",
         hist="helmvc", prob="helmvc_a6", rep=r"$a=6$"),
    dict(key="chirp", dir="chirp", csv="chirp_v3.csv", xlabel=r"chirp rate $a$",
         hist="chirp", prob="chirp_a4", rep=r"$a=4$"),
    dict(key="plate", dir="plate_beam", csv="plate_beam_v3.csv",
         xlabel=r"plate mode $m$", pf=lambda p: p.startswith("plate_m"),
         hist="plate_beam", prob="plate_m2", rep=r"$m=2$"),
    dict(key="beam", dir="plate_beam", csv="plate_beam_v3.csv",
         xlabel=r"beam mode $m$", pf=lambda p: p.startswith("beam_m"),
         hist="plate_beam", prob="beam_m2", rep=r"$m=2$"),
    dict(key="platemix", dir="plate_beam", csv="platemix_v3.csv",
         xlabel=r"mixed plate mode $m$",
         hist="platemix", prob="platemix_m3", rep=r"modes $(3,4)$"),
    dict(key="kdv", dir="kdv", csv="kdv_v3.csv", xlabel=r"wavenumber $k$",
         hist="kdv", prob="kdv_k4", rep=r"$k=4$"),
    dict(key="cahn", dir="cahn_hilliard", csv="cahn_hilliard_v3.csv",
         xlabel=r"amplitude $a$", pf=lambda p: p.startswith("ch6"),
         hist="cahn_hilliard", prob="ch6_a2", rep=r"6th order, $a=2$"),
    dict(key="nls", dir="nls", csv="nls_v3.csv", xlabel=r"wavenumber $k$",
         hist="nls", prob="nls_k2", rep=r"$k=2$"),
    dict(key="maxwell", dir="maxwell", csv="maxwell_v3.csv",
         xlabel=r"loss/freq scale $a$",
         hist="maxwell", prob="maxwell_a4", rep=r"$a=4$"),
]


def main():
    made = []
    for fam in FAMILIES:
        means = csv_means(find_csv(fam), fam.get("pf"))
        traces = hist_traces(find_hist(fam), fam["prob"])
        if not means and not traces:
            print(f"[skip] {fam['key']}: no data")
            continue
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
        panel_sweep(axes[0], means, fam["xlabel"])
        panel_time(axes[1], traces, 1, EPS_L2, r"relative $L^2$ error",
                   f"(b) accuracy vs time ({fam['rep']})")
        panel_time(axes[2], traces, 2, EPS_LOSS, "interior residual loss",
                   f"(c) loss vs time ({fam['rep']})")
        # one shared legend across the row
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
        made.append((fam["key"], len(traces)))
        print(f"[ok] {path}  ({len(traces)} variant traces)")
    print(f"\n[done] {len(made)} figures -> {OUT}")


if __name__ == "__main__":
    main()
