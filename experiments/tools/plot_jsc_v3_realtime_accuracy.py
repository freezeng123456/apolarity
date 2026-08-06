#!/usr/bin/env python3
"""Create the compact jsc_v3 real-time relative-error figure.

The figure uses the three central settings (Poly d2/o4, Chirp a2, Maxwell a4)
as paper panels.  Each curve is the seed median of the raw history traces in
log-error space; the translucent band is the seed IQR.  The complete nine-task
summary remains in ``experiments/results/jsc_v3/summary.csv``.

The renderer deliberately emits dependency-free SVG so the paper figure can be
reproduced on the server and viewed without a plotting GUI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PROTOCOL_ID = "jsc_v3"
METHODS = ("complex_sinh", "complex_sinh_autodiff")
METHOD_LABELS = {
    "complex_sinh": "Complex Sinh (jet)",
    "complex_sinh_autodiff": "Complex Sinh (direct autodiff)",
}
COLORS = {"complex_sinh": "#2166ac", "complex_sinh_autodiff": "#d6604d"}
DEFAULT_TASKS = ("poly_d2_o4", "chirp_a2", "maxwell_a4")
TASK_TITLES = {
    "poly_d2_o4": "(a) Polyharmonic d=2, order=4",
    "chirp_a2": "(b) Chirp a=2",
    "maxwell_a4": "(c) Maxwell a=4",
}
FLOOR = 1.0e-7


def _load_task(results: Path, task_id: str) -> tuple[list[dict], list[dict]]:
    task_dir = results / task_id
    rows = json.loads((task_dir / f"{task_id}.json").read_text())
    histories = json.loads((task_dir / f"{task_id}_history.json").read_text())
    if not (task_dir / "VALIDATED").exists():
        raise ValueError(f"{task_id}: missing VALIDATED marker")
    if len(rows) != 6 or len(histories) != 6:
        raise ValueError(f"{task_id}: expected 6 rows and histories")
    if any(row.get("protocol_id") != PROTOCOL_ID for row in rows):
        raise ValueError(f"{task_id}: row protocol mismatch")
    return rows, histories


def _quantile(values: np.ndarray, q: float) -> np.ndarray:
    return np.quantile(values, q, axis=0)


def _summary(results: Path, task_id: str, n_grid: int) -> dict[str, dict[str, np.ndarray]]:
    rows, histories = _load_task(results, task_id)
    output: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        traces = [item for item in histories if item["variant"] == method]
        final_rows = [row for row in rows if row["variant"] == method]
        if len(traces) != 3 or len(final_rows) != 3:
            raise ValueError(f"{task_id}/{method}: expected three seeds")
        tmax = min(float(trace["history"][-1][0]) for trace in traces)
        grid = np.linspace(0.0, tmax, n_grid)
        values = []
        for trace in traces:
            points = trace["history"]
            times = np.asarray([float(point[0]) for point in points])
            errors = np.maximum(np.asarray([float(point[1]) for point in points]), FLOOR)
            values.append(np.exp(np.interp(grid, times, np.log(errors))))
        values_array = np.asarray(values)
        output[method] = {
            "time_s": grid,
            "median": np.exp(np.median(np.log(values_array), axis=0)),
            "q25": np.exp(_quantile(np.log(values_array), 0.25)),
            "q75": np.exp(_quantile(np.log(values_array), 0.75)),
            "final": np.asarray([float(row["rel_error"]) for row in final_rows]),
        }
    return output


def _esc(value: object) -> str:
    text = str(value)
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _write_svg(summaries: dict[str, dict[str, dict[str, np.ndarray]]], path: Path) -> None:
    width, height = 1800, 760
    left, right, top, bottom, gap = 112, 38, 88, 112, 58
    panel_width = (width - left - right - 2 * gap) / 3
    panel_height = height - top - bottom
    y_min, y_max = 1.0e-5, 1.0

    def xmap(value: float, x0: float) -> float:
        return x0 + value / 1000.0 * panel_width

    def ymap(value: float, y0: float) -> float:
        log_value = math.log10(max(value, FLOOR))
        return y0 + panel_height - (log_value - math.log10(y_min)) / (
            math.log10(y_max) - math.log10(y_min)
        ) * panel_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<title>jsc_v3 real-time relative accuracy</title>',
        '<desc>Median and interquartile range over three seeds for the two derivative backends.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202020} '
        '.small{font-size:15px}.tick{font-size:14px}.title{font-size:19px;font-weight:600} '
        '.axis{stroke:#303030;stroke-width:1.5}.grid{stroke:#dfe3e8;stroke-width:1} '
        '.curve{fill:none;stroke-width:3.2;stroke-linejoin:round;stroke-linecap:round}</style>',
    ]

    for index, task_id in enumerate(DEFAULT_TASKS):
        x0 = left + index * (panel_width + gap)
        y0 = top
        x1, y1 = x0 + panel_width, y0 + panel_height
        lines.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{panel_width:.2f}" height="{panel_height:.2f}" fill="none" class="axis"/>')
        for exponent in range(-5, 1):
            value = 10.0 ** exponent
            y = ymap(value, y0)
            lines.append(f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" class="grid"/>')
            lines.append(f'<text x="{x0 - 12:.2f}" y="{y + 5:.2f}" text-anchor="end" class="tick">1e{exponent}</text>')
        for tick in (0, 500, 1000):
            x = xmap(tick, x0)
            lines.append(f'<line x1="{x:.2f}" y1="{y0:.2f}" x2="{x:.2f}" y2="{y1:.2f}" class="grid"/>')
            lines.append(f'<text x="{x:.2f}" y="{y1 + 28:.2f}" text-anchor="middle" class="tick">{tick}</text>')
        lines.append(f'<text x="{x0 + panel_width / 2:.2f}" y="{y0 - 34:.2f}" text-anchor="middle" class="title">{_esc(TASK_TITLES[task_id])}</text>')

        for method in METHODS:
            data = summaries[task_id][method]
            times = data["time_s"]
            med = data["median"]
            lo = data["q25"]
            hi = data["q75"]
            upper = [(xmap(float(t), x0), ymap(float(v), y0)) for t, v in zip(times, hi)]
            lower = [(xmap(float(t), x0), ymap(float(v), y0)) for t, v in zip(times[::-1], lo[::-1])]
            color = COLORS[method]
            lines.append(f'<polygon points="{_polyline(upper + lower)}" fill="{color}" opacity="0.16"/>')
            median_points = [(xmap(float(t), x0), ymap(float(v), y0)) for t, v in zip(times, med)]
            lines.append(f'<polyline points="{_polyline(median_points)}" stroke="{color}" class="curve"/>')

        legend_x = x0 + 14
        legend_y = y0 + 18
        for legend_index, method in enumerate(METHODS):
            yy = legend_y + legend_index * 24
            color = COLORS[method]
            lines.append(f'<line x1="{legend_x:.2f}" y1="{yy:.2f}" x2="{legend_x + 28:.2f}" y2="{yy:.2f}" stroke="{color}" class="curve"/>')
            lines.append(f'<text x="{legend_x + 38:.2f}" y="{yy + 5:.2f}" class="small">{_esc(METHOD_LABELS[method])}</text>')

    lines.extend([
        f'<text x="{width / 2:.2f}" y="{height - 34:.2f}" text-anchor="middle" class="small">wall-clock training time (s)</text>',
        f'<text transform="translate(25 {height / 2:.2f}) rotate(-90)" text-anchor="middle" class="small">relative L2 error (log scale)</text>',
        '</svg>',
    ])
    path.write_text("\n".join(lines) + "\n")


def _write_curve_csv(summaries: dict[str, dict[str, dict[str, np.ndarray]]], path: Path) -> None:
    fields = ["task_id", "variant", "time_s", "median_rel_error", "q25_rel_error", "q75_rel_error"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task_id in DEFAULT_TASKS:
            for method in METHODS:
                data = summaries[task_id][method]
                for time_s, median, q25, q75 in zip(data["time_s"], data["median"], data["q25"], data["q75"]):
                    writer.writerow({
                        "task_id": task_id,
                        "variant": method,
                        "time_s": f"{time_s:.6f}",
                        "median_rel_error": f"{median:.10e}",
                        "q25_rel_error": f"{q25:.10e}",
                        "q75_rel_error": f"{q75:.10e}",
                    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("experiments/results/jsc_v3"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/paper/figures"))
    parser.add_argument("--grid", type=int, default=101)
    args = parser.parse_args()
    summaries = {task: _summary(args.results, task, args.grid) for task in DEFAULT_TASKS}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_svg(summaries, args.out_dir / "jsc_v3_realtime_accuracy.svg")
    _write_curve_csv(summaries, args.out_dir / "jsc_v3_realtime_accuracy.csv")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "tasks": list(DEFAULT_TASKS),
        "methods": list(METHODS),
        "seeds": [0, 1, 2],
        "history_field": "rel_error",
        "aggregation": "median curve with q25-q75 seed band in log-error space",
        "grid_points": args.grid,
        "figure": "jsc_v3_realtime_accuracy.svg",
        "curve_csv": "jsc_v3_realtime_accuracy.csv",
    }
    (args.out_dir / "jsc_v3_realtime_accuracy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
