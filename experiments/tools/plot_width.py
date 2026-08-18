#!/usr/bin/env python3
"""Generate paper figures exclusively from validated jsc_v2 bundles.

Layout, styling, and output conventions follow the group's reference plotting
implementation: a grid of panels with lettered titles, error bars on the
profile panel, a star on the best entry, per-panel legends, one figure legend
plus a suptitle, explicit figure margins, and PDF/PNG/SVG output.  The one
deliberate departure is scale: the canvas is sized so that the saved figure is
as wide as the manuscript text block, so ``width=\\linewidth`` neither enlarges
nor shrinks it and the type sizes below are the sizes that reach the page.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

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
RESULT_ROOT = ROOT / "experiments" / "results" / "jsc_v2"
OUT = ROOT / "docs" / "paper" / "figures"
PROTOCOL_ID = "jsc_v2"
METHODS = ("complex_sinh", "siren", "fourier", "mscale")
METHOD_LABEL = {
    "complex_sinh": r"Complex $\sinh$",
    "siren": "SIREN",
    "fourier": "mFF-PINN",
    "mscale": "MscaleDNN-2-sin",
}
STYLE = {
    "complex_sinh": {"color": "#1450a0", "linestyle": "-", "marker": "o"},
    "siren": {"color": "#c0392b", "linestyle": "--", "marker": "s"},
    "fourier": {"color": "#1b7f4d", "linestyle": "-.", "marker": "^"},
    "mscale": {"color": "#7b3fa0", "linestyle": ":", "marker": "D"},
}
FAMILY_TITLE = {
    "poly_d2": r"Polyharmonic benchmark, $d=2$",
    "poly_d3": r"Polyharmonic benchmark, $d=3$",
    "chirp": "Radial-chirp benchmark",
    "maxwell": "Time-harmonic Maxwell benchmark",
}
SWEEP_LABEL = {
    "poly_d2": r"operator order $2m$",
    "poly_d3": r"operator order $2m$",
    "chirp": r"chirp rate $a$",
    "maxwell": r"wavenumber parameter $a$",
}
#: The canvas is the text block itself.  A tight bounding box is deliberately
#: not used: its size depends on the tick labels, which would give the four
#: benchmark figures four different widths, four different scale factors on the
#: page, and four different effective type sizes.
FIG_WIDTH_IN = TEXT_WIDTH_IN
FIG_HEIGHT_IN = 2.42


def validated_bundles() -> list[tuple[Path, list[dict], list[dict]]]:
    bundles = []
    if not RESULT_ROOT.exists():
        return bundles
    for marker in sorted(RESULT_ROOT.glob("*/VALIDATED")):
        task_dir = marker.parent
        task_id = task_dir.name
        row_path = task_dir / f"{task_id}.json"
        history_path = task_dir / f"{task_id}_history.json"
        if not row_path.exists() or not history_path.exists():
            raise ValueError(f"validated task {task_id} lacks canonical files")
        rows = json.loads(row_path.read_text())
        histories = json.loads(history_path.read_text())
        if not rows or any(row.get("protocol_id") != PROTOCOL_ID for row in rows):
            raise ValueError(f"{task_id} contains non-jsc_v2 rows")
        bundles.append((task_dir, rows, histories))
    return bundles


def grouped_bundles():
    grouped = defaultdict(list)
    for task_dir, rows, histories in validated_bundles():
        first = rows[0]
        key = (
            f"poly_d{int(first['dimension'])}"
            if first["family"] == "poly"
            else first["family"]
        )
        grouped[key].append((task_dir, rows, histories))
    for items in grouped.values():
        items.sort(key=lambda item: float(item[1][0]["sweep"]))
    return grouped


def sweep_statistics(items, field: str) -> dict[str, list[tuple[float, float, float]]]:
    """Per method, the sweep value with the seed mean and sample deviation."""
    values: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for _, rows, _ in items:
        for row in rows:
            values[row["variant"]][float(row["sweep"])].append(float(row[field]))
    return {
        method: [
            (sweep, float(np.mean(samples)), float(np.std(samples, ddof=1)))
            for sweep, samples in sorted(by_sweep.items())
        ]
        for method, by_sweep in values.items()
    }


def histories_for(item) -> dict[str, list[np.ndarray]]:
    _, _, histories = item
    grouped = defaultdict(list)
    for record in histories:
        array = np.asarray(record["history"], dtype=float)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("validated history has an invalid shape")
        grouped[record["variant"]].append(array)
    return grouped


def mean_trace(traces: list[np.ndarray], column: int, floor: float):
    tmax = min(trace[-1, 0] for trace in traces)
    grid = np.linspace(0.0, tmax, 160)
    values = np.asarray([
        np.interp(grid, trace[:, 0], np.maximum(trace[:, column], floor))
        for trace in traces
    ])
    logs = np.log(values)
    return grid, np.exp(logs.mean(0)), np.exp(logs.min(0)), np.exp(logs.max(0))


def plot_profile(axis: plt.Axes, stats, tag: str, ylabel: str, xlabel: str,
                 title: str, mark_best: bool) -> None:
    sweeps = [point[0] for point in next(iter(stats.values()))]
    for method in METHODS:
        points = stats.get(method)
        if not points:
            continue
        style = STYLE[method]
        x, mean, deviation = (np.asarray(column) for column in zip(*points))
        axis.errorbar(
            x, mean, yerr=deviation,
            color=style["color"], linestyle=style["linestyle"], marker=style["marker"],
            markersize=4.2, markerfacecolor="white", markeredgewidth=1.1,
            linewidth=1.6, elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=3,
        )
    if mark_best:
        for index, sweep in enumerate(sweeps):
            best = min(
                (stats[method][index][1], method)
                for method in METHODS if stats.get(method)
            )
            axis.scatter(
                [sweep], [best[0]], marker="*", s=70,
                color=STYLE[best[1]]["color"], edgecolor="black", linewidth=0.4,
                zorder=5,
            )
    axis.set_xticks(sweeps)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(f"({tag})  {title}")


def make_figure(key: str, items) -> list[Path]:
    error_stats = sweep_statistics(items, "L2_err")
    representative = items[len(items) // 2]
    representative_sweep = float(representative[1][0]["sweep"])
    traces = histories_for(representative)
    setting = (
        rf"$2m={representative_sweep:g}$" if key.startswith("poly_")
        else rf"$a={representative_sweep:g}$"
    )

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    plot_profile(
        axes[0], error_stats, "a",
        r"relative $L^2$ error", SWEEP_LABEL[key],
        "final accuracy", mark_best=True,
    )
    axes[0].set_yscale("log")

    for axis, column, floor, tag, ylabel, name in (
        (axes[1], 1, 1e-12, "b", r"relative $L^2$ error", "error"),
        (axes[2], 2, 1e-16, "c", "normalized residual", "residual"),
    ):
        extent = []
        for method in METHODS:
            method_traces = traces.get(method)
            if not method_traces:
                continue
            style = STYLE[method]
            grid, mean, low, high = mean_trace(method_traces, column, floor)
            axis.plot(grid, mean, color=style["color"], linestyle=style["linestyle"],
                      linewidth=1.6, zorder=3)
            axis.fill_between(grid, low, high, color=style["color"], alpha=0.13,
                              linewidth=0, zorder=2)
            axis.plot(grid[-1], mean[-1], color=style["color"], marker=style["marker"],
                      markersize=4.2, markerfacecolor="white", markeredgewidth=1.1,
                      zorder=4)
            extent.append((float(mean.min()), float(mean.max())))
        axis.set_yscale("log")
        if extent:
            axis.set_ylim(min(item[0] for item in extent) / 3.0,
                          max(item[1] for item in extent) * 3.0)
        axis.set_xlim(0.0, 1200.0)
        axis.set_xticks([0, 600, 1200])
        axis.set_xlabel("training time (s)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"({tag})  {name}, {setting}")

    handles = [
        Line2D([0], [0], color=STYLE[method]["color"], linestyle=STYLE[method]["linestyle"],
               marker=STYLE[method]["marker"], markersize=4.2, markerfacecolor="white",
               markeredgewidth=1.1, linewidth=1.6, label=METHOD_LABEL[method])
        for method in METHODS
    ]
    for axis in axes.flat:
        style_axis(axis)
        thin_log_ticks(axis, max_labels=4)

    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=4, frameon=False, handlelength=1.9, columnspacing=0.9,
               handletextpad=0.35)
    fig.subplots_adjust(left=0.088, right=0.976, bottom=0.205, top=0.775,
                        wspace=0.46)

    OUT.mkdir(parents=True, exist_ok=True)
    outputs = save_figure(fig, OUT / f"fig_{key}")
    plt.close(fig)
    return outputs


def main() -> None:
    grouped = grouped_bundles()
    if not grouped:
        print(f"[skip] no validated {PROTOCOL_ID} bundles in {RESULT_ROOT}")
        return
    set_plot_style()
    for key, items in sorted(grouped.items()):
        for path in make_figure(key, items):
            if path.suffix == ".pdf":
                check_width(path)
            print(f"[ok] {path}")


if __name__ == "__main__":
    main()
