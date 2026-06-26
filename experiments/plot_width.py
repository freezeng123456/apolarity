#!/usr/bin/env python3
"""Per-family figures for the 600s WIDTH study.

For each family it reads results/width/<stem>_h128.csv (complex_sinh + the real
baselines, all at width 128) and <stem>_h64.csv (complex_sinh at width 64), plus
their *_history.json sidecars, relabels the complex series by width
(complex_sinh_128 / complex_sinh_64) and emits the 3-panel figure
docs/paper/figures/fig_<key>.pdf:

  (a) relative L2 vs the physics sweep (order / wavenumber / mode)
  (b) relative L2 vs training time at the representative setting
  (c) interior residual loss vs training time at the representative setting

The complex curves at 64 and 128 sitting on top of each other is the message:
the method is insensitive to width even though a complex weight carries ~2x the
real DOF of the width-128 real baselines.
"""
from __future__ import annotations

import os
from pathlib import Path

import plot_convergence as pc  # shared helpers, STYLE, panels
import matplotlib.pyplot as plt

DATA = Path(os.environ.get("WIDTH_DIR", pc.RES / "width"))
OUT = pc.OUT

FAMILIES = [
    dict(key="poly1d", stem="poly1d", xlabel=r"operator order $m$",
         prob="polyharm1d_o6", rep=r"$m=6$"),
    dict(key="poly2d", stem="poly2d", xlabel=r"operator order $m$",
         prob="polyharm2d_o6", rep=r"$m=6$"),
    dict(key="helmholtz", stem="helmholtz", xlabel=r"wavenumber $a$",
         prob="helmholtz_a6", rep=r"$a=6$"),
    dict(key="helmvc", stem="helmvc", xlabel=r"wavenumber $a$",
         prob="helmvc_a6", rep=r"$a=6$"),
    dict(key="chirp", stem="chirp", xlabel=r"chirp rate $a$",
         prob="chirp_a4", rep=r"$a=4$"),
    dict(key="plate", stem="plate_beam", xlabel=r"plate mode $m$",
         pf=lambda p: p.startswith("plate_m"), prob="plate_m2", rep=r"$m=2$"),
    dict(key="beam", stem="plate_beam", xlabel=r"beam mode $m$",
         pf=lambda p: p.startswith("beam_m"), prob="beam_m2", rep=r"$m=2$"),
    dict(key="platemix", stem="platemix", xlabel=r"mixed plate mode $m$",
         prob="platemix_m3", rep=r"modes $(3,4)$"),
    dict(key="kdv", stem="kdv", xlabel=r"wavenumber $k$",
         prob="kdv_k4", rep=r"$k=4$"),
    dict(key="cahn", stem="cahn_hilliard", xlabel=r"amplitude $a$",
         pf=lambda p: p.startswith("ch6"), prob="ch6_a2", rep=r"6th order, $a=2$"),
    dict(key="nls", stem="nls", xlabel=r"wavenumber $k$",
         prob="nls_k2", rep=r"$k=2$"),
    dict(key="maxwell", stem="maxwell", xlabel=r"loss/freq scale $a$",
         prob="maxwell_a4", rep=r"$a=4$"),
]


def _exists(p):
    return p if p.exists() else None


def load(fam):
    pf = fam.get("pf")
    c128 = _exists(DATA / f"{fam['stem']}_h128.csv")
    c64 = _exists(DATA / f"{fam['stem']}_h64.csv")
    h128 = _exists(DATA / f"{fam['stem']}_h128_history.json")
    h64 = _exists(DATA / f"{fam['stem']}_h64_history.json")

    means = pc.csv_means(c128, pf)
    m64 = pc.csv_means(c64, pf)
    if "complex_sinh" in means:
        means["complex_sinh_128"] = means.pop("complex_sinh")
    if "complex_sinh" in m64:
        means["complex_sinh_64"] = m64["complex_sinh"]

    traces = pc.hist_traces(h128, fam["prob"])
    t64 = pc.hist_traces(h64, fam["prob"])
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
            print(f"[skip] {fam['key']}: no data in {DATA}")
            continue
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
        pc.panel_sweep(axes[0], means, fam["xlabel"])
        pc.panel_time(axes[1], traces, 1, pc.EPS_L2, r"relative $L^2$ error",
                      f"(b) accuracy vs time ({fam['rep']})")
        pc.panel_time(axes[2], traces, 2, pc.EPS_LOSS, "interior residual loss",
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
