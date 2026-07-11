#!/usr/bin/env python3
"""Generate paper figures exclusively from validated jsc_v2 bundles."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "experiments" / "results" / "jsc_v2"
OUT = ROOT / "docs" / "paper" / "figures"
PROTOCOL_ID = "jsc_v2"
METHODS = ("complex_sinh", "siren", "fourier", "mscale")
STYLE = {
    "complex_sinh": ("Complex Sinh", "#1f77b4", "-", "o"),
    "siren": ("SIREN", "#ff7f0e", "-.", "^"),
    "fourier": ("mFF-PINN", "#2ca02c", "--", "s"),
    "mscale": ("MscaleDNN-2-sin", "#9467bd", ":", "D"),
}


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
    grid = np.linspace(0.0, tmax, 80)
    values = np.asarray([
        np.interp(grid, trace[:, 0], np.maximum(trace[:, column], floor))
        for trace in traces
    ])
    logs = np.log(values)
    return grid, np.exp(logs.mean(0)), np.exp(logs.min(0)), np.exp(logs.max(0))


def make_figure(key: str, items) -> Path:
    means = means_for(items)
    representative = items[len(items) // 2]
    traces = histories_for(representative)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    for method in METHODS:
        label, color, linestyle, marker = STYLE[method]
        points = means.get(method, [])
        if points:
            x, mean, lo, hi = map(np.asarray, zip(*points))
            axes[0].plot(
                x, mean, color=color, linestyle=linestyle, marker=marker, label=label
            )
            axes[0].fill_between(x, lo, hi, color=color, alpha=0.12)
        method_traces = traces.get(method, [])
        if method_traces:
            for axis, column, floor in ((axes[1], 1, 1e-12), (axes[2], 2, 1e-16)):
                grid, mean, lo, hi = mean_trace(method_traces, column, floor)
                axis.plot(grid, mean, color=color, linestyle=linestyle, label=label)
                axis.fill_between(grid, lo, hi, color=color, alpha=0.12)

    xlabel = "operator order" if key.startswith("poly_") else "sweep value a"
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(r"relative $L^2$ error")
    axes[0].set_title("(a) validated accuracy")
    axes[1].set_xlabel("training time (s)")
    axes[1].set_ylabel(r"relative $L^2$ error")
    axes[1].set_title("(b) accuracy vs time")
    axes[2].set_xlabel("training time (s)")
    axes[2].set_ylabel("interior residual loss")
    axes[2].set_title("(c) residual vs time")
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"fig_{key}.pdf"
    fig.savefig(path)
    if os.environ.get("FIG_PNG"):
        fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    grouped = grouped_bundles()
    if not grouped:
        print(f"[skip] no validated {PROTOCOL_ID} bundles in {RESULT_ROOT}")
        return
    for key, items in sorted(grouped.items()):
        path = make_figure(key, items)
        print(f"[ok] {path}")


if __name__ == "__main__":
    main()
