#!/usr/bin/env python3
"""Generate paper figures exclusively from validated jsc_v2 bundles.

Figures are drawn at the SIAM text width of the manuscript (370.4 pt) so that
``\\includegraphics[width=\\linewidth]`` scales them by one and the type sizes
below are the sizes that reach the page.
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
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "experiments" / "results" / "jsc_v2"
OUT = ROOT / "docs" / "paper" / "figures"
PROTOCOL_ID = "jsc_v2"
METHODS = ("complex_sinh", "siren", "fourier", "mscale")
STYLE = {
    "complex_sinh": (r"Complex $\sinh$", "#1450a0", "-", "o"),
    "siren": ("SIREN", "#c0392b", "--", "s"),
    "fourier": ("mFF-PINN", "#1b7f4d", "-.", "^"),
    "mscale": ("MscaleDNN-2-sin", "#7b3fa0", ":", "D"),
}
#: SIAM \textwidth of jsc_paper_main.tex, in inches (370.38374 pt / 72.27).
TEXT_WIDTH_IN = 5.125
FIG_HEIGHT_IN = 2.55
SWEEP_LABEL = {
    "poly_d2": r"operator order $2m$",
    "poly_d3": r"operator order $2m$",
    "chirp": r"chirp rate $a$",
    "maxwell": r"wavenumber parameter $a$",
}
PANEL_TITLE = {
    "poly_d2": "polyharmonic, $d=2$",
    "poly_d3": "polyharmonic, $d=3$",
    "chirp": "radial chirp",
    "maxwell": "Maxwell",
}


def set_plot_style() -> None:
    """Apply the house figure style at the manuscript's own type sizes."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    """Grid, tick, and spine treatment shared by every panel."""
    axis.grid(True, which="major", color="#D0D0D0", linewidth=0.5, zorder=0)
    axis.grid(True, which="minor", color="#ECECEC", linewidth=0.35, zorder=0)
    axis.tick_params(direction="in", top=True, right=True, width=0.6, length=2.6)
    axis.tick_params(which="minor", direction="in", top=True, right=True, length=1.5)
    axis.set_axisbelow(True)


def thin_log_ticks(axis: plt.Axes, max_decades: int = 5) -> None:
    """Keep a log axis readable inside a narrow panel."""
    low, high = axis.get_ylim()
    decades = np.log10(high) - np.log10(low)
    step = max(1, int(np.ceil(decades / max_decades)))
    axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=max_decades + 2)
                                 if step == 1 else
                                 LogLocator(base=10.0, subs=(1.0,), numticks=99))
    if step > 1:
        axis.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=max_decades + 1))
    axis.yaxis.set_minor_formatter(NullFormatter())


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


def means_for(items) -> dict[str, list[tuple[float, float, float, float]]]:
    values = defaultdict(lambda: defaultdict(list))
    for _, rows, _ in items:
        for row in rows:
            values[row["variant"]][float(row["sweep"])].append(float(row["L2_err"]))
    return {
        method: [
            (sweep, float(np.mean(v)), float(np.min(v)), float(np.max(v)))
            for sweep, v in sorted(by_sweep.items())
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


def make_figure(key: str, items) -> list[Path]:
    means = means_for(items)
    representative = items[len(items) // 2]
    representative_sweep = float(representative[1][0]["sweep"])
    traces = histories_for(representative)

    fig, axes = plt.subplots(
        1, 3, figsize=(TEXT_WIDTH_IN, FIG_HEIGHT_IN), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.0)
    # Seed bands of a diverging residual can span fifteen decades; the history
    # panels are framed on the seed-mean curves so that every method stays
    # legible, and the bands are allowed to run off the top of the panel.
    mean_extent: dict[plt.Axes, list[tuple[float, float]]] = {axes[1]: [], axes[2]: []}
    for method in METHODS:
        label, color, linestyle, marker = STYLE[method]
        points = means.get(method, [])
        if points:
            x, mean, lo, hi = map(np.asarray, zip(*points))
            axes[0].plot(
                x, mean,
                color=color, linestyle=linestyle, marker=marker,
                markersize=3.6, markerfacecolor="white", markeredgewidth=1.0,
                zorder=3,
            )
            axes[0].fill_between(x, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=2)
        method_traces = traces.get(method, [])
        if method_traces:
            for axis, column, floor in ((axes[1], 1, 1e-12), (axes[2], 2, 1e-16)):
                grid, mean, lo, hi = mean_trace(method_traces, column, floor)
                axis.plot(grid, mean, color=color, linestyle=linestyle, zorder=3)
                axis.fill_between(grid, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=2)
                axis.plot(
                    grid[-1], mean[-1],
                    color=color, marker=marker, markersize=3.6,
                    markerfacecolor="white", markeredgewidth=1.0, zorder=4,
                )
                mean_extent[axis].append((float(mean.min()), float(mean.max())))

    axes[0].set_xlabel(SWEEP_LABEL[key])
    axes[0].set_ylabel(r"relative $L^2$ error")
    axes[0].set_title("(a) final error")
    axes[0].set_xticks([point[0] for point in next(iter(means.values()))])

    sweep_tag = (
        rf"$2m={representative_sweep:g}$" if key.startswith("poly_")
        else rf"$a={representative_sweep:g}$"
    )
    axes[1].set_xlabel("training time (s)")
    axes[1].set_ylabel(r"relative $L^2$ error")
    axes[1].set_title(f"(b) error, {sweep_tag}")
    axes[2].set_xlabel("training time (s)")
    axes[2].set_ylabel("normalized residual")
    axes[2].set_title(f"(c) residual, {sweep_tag}")

    for axis in axes:
        axis.set_yscale("log")
    for axis, extents in mean_extent.items():
        if extents:
            low = min(item[0] for item in extents)
            high = max(item[1] for item in extents)
            axis.set_ylim(low / 3.0, high * 3.0)
    for axis in axes:
        style_axis(axis)
        thin_log_ticks(axis)
    for axis in axes[1:]:
        axis.set_xlim(left=0.0)
        axis.set_xticks([0, 400, 800, 1200])

    handles = [
        Line2D([0], [0], color=STYLE[method][1], linestyle=STYLE[method][2],
               marker=STYLE[method][3], markersize=3.6, markerfacecolor="white",
               markeredgewidth=1.0, linewidth=1.5, label=STYLE[method][0])
        for method in METHODS
    ]
    fig.legend(
        handles=handles, loc="outside lower center", ncol=4, frameon=False,
        handlelength=2.2, columnspacing=1.3, handletextpad=0.45, borderaxespad=0.0,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    outputs = []
    stem = OUT / f"fig_{key}"
    for suffix, options in (("pdf", {"metadata": {"CreationDate": None}}), ("png", {"dpi": 300})):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, **options)
        outputs.append(path)
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
            print(f"[ok] {path}")


if __name__ == "__main__":
    main()
