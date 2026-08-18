#!/usr/bin/env python3
"""Build the derivative-backend figures from validated jsc_v3 bundles.

Two figures are produced.  ``fig_v3_cost`` shows how the cost of one optimizer
step and its peak memory grow with the derivative order for the two backends,
which is the quantity the rank-optimal schedule is meant to control.
``fig_v3_history`` shows the accuracy trajectories that the saved time buys.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from paper_style import (  # noqa: E402
    TEXT_WIDTH_IN,
    check_width,
    save_figure,
    set_plot_style,
    style_axis,
    thin_log_ticks,
)

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "experiments" / "results" / "jsc_v3"
OUT = ROOT / "docs" / "paper" / "figures"
PROTOCOL_ID = "jsc_v3"
JET = "complex_sinh"
AUTODIFF = "complex_sinh_autodiff"
LABEL = {JET: "rank-optimal jet", AUTODIFF: "nested autodiff"}
STYLE = {
    JET: {"color": "#1450a0", "linestyle": "-", "marker": "o"},
    AUTODIFF: {"color": "#c0392b", "linestyle": "--", "marker": "s"},
}
#: The polyharmonic family is the one that sweeps the derivative order.
ORDER_TASKS = (("poly_d2_o2", 2), ("poly_d2_o4", 4), ("poly_d2_o6", 6))
HISTORY_TASKS = (
    ("poly_d2_o4", r"polyharmonic, $2m=4$"),
    ("chirp_a2", r"radial chirp, $a=2$"),
    ("maxwell_a4", r"Maxwell, $a=4$"),
)
FIG_HEIGHT_IN = 2.42


def load(task_id: str) -> dict[str, list[dict]]:
    task_dir = RESULT_ROOT / task_id
    if not (task_dir / "VALIDATED").exists():
        raise ValueError(f"{task_id} carries no VALIDATED marker")
    rows = json.loads((task_dir / f"{task_id}.json").read_text())
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["protocol_id"] != PROTOCOL_ID:
            raise ValueError(f"{task_id} contains a non-{PROTOCOL_ID} row")
        grouped.setdefault(str(row["variant"]), []).append(row)
    if set(grouped) != {JET, AUTODIFF}:
        raise ValueError(f"{task_id} does not contain exactly the two formal lines")
    return grouped


def load_histories(task_id: str) -> dict[str, list[np.ndarray]]:
    records = json.loads((RESULT_ROOT / task_id / f"{task_id}_history.json").read_text())
    grouped: dict[str, list[np.ndarray]] = {}
    for record in records:
        array = np.asarray(record["history"], dtype=float)
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError(f"{task_id}: expected [time, rel_error, loss, L_int] rows")
        grouped.setdefault(str(record["variant"]), []).append(array)
    return grouped


def median_trace(traces: list[np.ndarray], column: int, floor: float):
    tmax = min(trace[-1, 0] for trace in traces)
    grid = np.linspace(0.0, tmax, 200)
    values = np.asarray([
        np.interp(grid, trace[:, 0], np.maximum(trace[:, column], floor))
        for trace in traces
    ])
    logs = np.log(values)
    return grid, np.exp(np.median(logs, axis=0)), np.exp(logs.min(0)), np.exp(logs.max(0))


def marker_line(variant: str) -> Line2D:
    style = STYLE[variant]
    return Line2D([0], [0], color=style["color"], linestyle=style["linestyle"],
                  marker=style["marker"], markersize=4.0, markerfacecolor="white",
                  markeredgewidth=1.0, linewidth=1.6, label=LABEL[variant])


def figure_cost() -> list[Path]:
    orders = [order for _, order in ORDER_TASKS]
    stats = {variant: {"ms": [], "mb": []} for variant in (JET, AUTODIFF)}
    for task_id, _ in ORDER_TASKS:
        grouped = load(task_id)
        for variant in (JET, AUTODIFF):
            stats[variant]["ms"].append(fmean(float(r["ms_per_step"]) for r in grouped[variant]))
            stats[variant]["mb"].append(fmean(float(r["peak_mb"]) for r in grouped[variant]))

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, FIG_HEIGHT_IN))
    for axis, key, ylabel, title in (
        (axes[0], "ms", "ms per optimizer step", "(a)  step time"),
        (axes[1], "mb", "peak memory (MiB)", "(b)  peak memory"),
    ):
        for variant in (JET, AUTODIFF):
            style = STYLE[variant]
            axis.plot(orders, stats[variant][key], color=style["color"],
                      linestyle=style["linestyle"], marker=style["marker"],
                      markersize=4.0, markerfacecolor="white", markeredgewidth=1.0,
                      linewidth=1.6, zorder=3)
        axis.set_yscale("log")
        axis.set_xticks(orders)
        axis.set_xlabel(r"derivative order $p$")
        axis.set_ylabel(ylabel)
        axis.set_title(title)

    ratio_ms = [a / j for j, a in zip(stats[JET]["ms"], stats[AUTODIFF]["ms"])]
    ratio_mb = [a / j for j, a in zip(stats[JET]["mb"], stats[AUTODIFF]["mb"])]
    axes[2].plot(orders, ratio_ms, color="#1450a0", linestyle="-", marker="o",
                 markersize=4.0, markerfacecolor="white", markeredgewidth=1.0,
                 linewidth=1.6, label="time", zorder=3)
    axes[2].plot(orders, ratio_mb, color="#7b3fa0", linestyle=":", marker="D",
                 markersize=4.0, markerfacecolor="white", markeredgewidth=1.0,
                 linewidth=1.6, label="memory", zorder=3)
    axes[2].axhline(1.0, color="0.45", linewidth=0.7, zorder=2)
    axes[2].set_xticks(orders)
    axes[2].set_xlabel(r"derivative order $p$")
    axes[2].set_ylabel("nested / jet")
    axes[2].set_title("(c)  saving")
    axes[2].legend(loc="upper left", frameon=True, framealpha=0.93, edgecolor="0.75")

    for axis in axes:
        style_axis(axis)
    for axis in axes[:2]:
        thin_log_ticks(axis)

    fig.legend(handles=[marker_line(JET), marker_line(AUTODIFF)], loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False, handlelength=2.1,
               columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(left=0.098, right=0.976, bottom=0.205, top=0.775, wspace=0.50)
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = save_figure(fig, OUT / "fig_v3_cost")
    plt.close(fig)
    return outputs


def figure_history() -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, FIG_HEIGHT_IN))
    for axis, (task_id, label), tag in zip(axes, HISTORY_TASKS, "abc"):
        histories = load_histories(task_id)
        extent = []
        for variant in (JET, AUTODIFF):
            style = STYLE[variant]
            grid, mid, low, high = median_trace(histories[variant], 1, 1e-12)
            axis.plot(grid, mid, color=style["color"], linestyle=style["linestyle"],
                      linewidth=1.6, zorder=3)
            axis.fill_between(grid, low, high, color=style["color"], alpha=0.13,
                              linewidth=0, zorder=2)
            axis.plot(grid[-1], mid[-1], color=style["color"], marker=style["marker"],
                      markersize=4.0, markerfacecolor="white", markeredgewidth=1.0,
                      zorder=4)
            extent.append((float(mid.min()), float(mid.max())))
        axis.set_yscale("log")
        axis.set_ylim(min(e[0] for e in extent) / 3.0, max(e[1] for e in extent) * 3.0)
        axis.set_xlim(0.0, 1000.0)
        axis.set_xticks([0, 500, 1000])
        axis.set_xlabel("training time (s)")
        if tag == "a":
            axis.set_ylabel(r"relative $L^2$ error")
        axis.set_title(f"({tag})  {label}")
        style_axis(axis)
        thin_log_ticks(axis)

    fig.legend(handles=[marker_line(JET), marker_line(AUTODIFF)], loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False, handlelength=2.1,
               columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(left=0.098, right=0.972, bottom=0.205, top=0.775, wspace=0.30)
    outputs = save_figure(fig, OUT / "fig_v3_history")
    plt.close(fig)
    return outputs


def main() -> None:
    if not RESULT_ROOT.exists():
        print(f"[skip] no {PROTOCOL_ID} results in {RESULT_ROOT}")
        return
    set_plot_style()
    for builder in (figure_cost, figure_history):
        for path in builder():
            if path.suffix == ".pdf":
                check_width(path)
            print(f"[ok] {path}")


if __name__ == "__main__":
    main()
