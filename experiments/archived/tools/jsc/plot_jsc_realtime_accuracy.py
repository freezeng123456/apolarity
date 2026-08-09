#!/usr/bin/env python3
"""Plot active ``jsc_v2`` real-time accuracy curves.

Only the three active main families are accepted by this figure builder.  The
default panels use the central setting in each family (Polyharmonic d=2,
order=4; Chirp a=2; Maxwell a=4).  Every curve is the five-seed median of the
raw held-out relative L2 history; the translucent band is the seed IQR and the
dotted curve is the median best-so-far value.  The script deliberately does not
load anything from ``experiments/archived`` and has a dependency-light PIL /
ReportLab renderer because the desktop runtime may not include Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PROTOCOL_ID = "jsc_v2"
FORMAL_METHODS = ("complex_sinh", "siren", "fourier", "mscale")
METHOD_LABELS = {
    "complex_sinh": "Complex Sinh",
    "siren": "SIREN",
    "fourier": "mFF-PINN",
    "mscale": "MscaleDNN-2-sin",
}
COLORS = {
    "complex_sinh": "#1f77b4",
    "siren": "#d62728",
    "fourier": "#2ca02c",
    "mscale": "#9467bd",
}
DEFAULT_TASKS = ("poly_d2_o4", "chirp_a2", "maxwell_a4")
TASK_TITLES = {
    "poly_d2_o4": "(a) Polyharmonic, d=2, order=4",
    "chirp_a2": "(b) Chirp, a=2",
    "maxwell_a4": "(c) Maxwell, a=4",
}
FLOOR = 1.0e-12


def _hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index:index + 2], 16) for index in (1, 3, 5))


def _load_task(results: Path, task_id: str) -> tuple[list[dict], list[dict]]:
    task_dir = results / task_id
    marker = task_dir / "VALIDATED"
    manifest_path = task_dir / "manifest.json"
    rows_path = task_dir / f"{task_id}.json"
    history_path = task_dir / f"{task_id}_history.json"
    for path in (marker, manifest_path, rows_path, history_path):
        if not path.exists():
            raise FileNotFoundError(f"{task_id}: missing {path}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"{task_id}: expected protocol_id={PROTOCOL_ID}")
    family = manifest.get("task", {}).get("family")
    if family not in {"poly", "chirp", "maxwell"}:
        raise ValueError(f"{task_id}: not one of the three active families")

    rows = json.loads(rows_path.read_text())
    if not isinstance(rows, list) or len(rows) != 20:
        raise ValueError(f"{task_id}: expected 20 canonical rows, found {len(rows)}")
    keys = {(str(row.get("variant")), int(row.get("seed", -1))) for row in rows}
    expected = {(method, seed) for method in FORMAL_METHODS for seed in range(5)}
    if keys != expected:
        raise ValueError(f"{task_id}: canonical method/seed keys are incomplete")
    for row in rows:
        if row.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"{task_id}: row has non-jsc_v2 protocol")
        if float(row.get("budget_seconds", -1.0)) != 1200.0:
            raise ValueError(f"{task_id}: row is not a 1200-second run")
        if int(row.get("n_int", -1)) != 4096 or int(row.get("n_bc", -1)) != 512:
            raise ValueError(f"{task_id}: collocation sizes do not match jsc_v2")
        if str(row.get("backend")) != "jet":
            raise ValueError(f"{task_id}: active formal rows must use the jet backend")
        if str(row.get("nan")).lower() == "true":
            raise ValueError(f"{task_id}: NaN row cannot be plotted")

    histories = json.loads(history_path.read_text())
    if not isinstance(histories, list) or len(histories) != 20:
        raise ValueError(f"{task_id}: expected 20 history traces, found {len(histories)}")
    hkeys = {(str(item.get("variant")), int(item.get("seed", -1))) for item in histories}
    if hkeys != expected:
        raise ValueError(f"{task_id}: history method/seed keys are incomplete")
    for item in histories:
        points = item.get("history")
        if not isinstance(points, list) or not points:
            raise ValueError(f"{task_id}: empty history for {item.get('variant')} seed {item.get('seed')}")
        last_time = -math.inf
        for point in points:
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(f"{task_id}: malformed history point")
            time_s, l2, lint = (float(value) for value in point)
            if not all(math.isfinite(value) for value in (time_s, l2, lint)) or l2 <= 0.0:
                raise ValueError(f"{task_id}: non-finite/non-positive history point")
            if time_s < last_time:
                raise ValueError(f"{task_id}: non-monotone history time")
            last_time = time_s
    return rows, histories


def _summarize(traces: list[dict], rows: list[dict], n_grid: int) -> dict:
    # Use the common prefix of the traces so no seed is extrapolated past its
    # last checkpoint.  Interpolation is performed in log-error space.
    tmax = min(float(trace["time"][-1]) for trace in traces)
    grid = np.linspace(0.0, tmax, n_grid)
    raw, best = [], []
    for trace in traces:
        x = np.asarray(trace["time"], dtype=float)
        y = np.maximum(np.asarray(trace["value"], dtype=float), FLOOR)
        raw_y = np.exp(np.interp(grid, x, np.log(y)))
        best_y = np.minimum.accumulate(y)
        best_y = np.exp(np.interp(grid, x, np.log(best_y)))
        raw.append(raw_y)
        best.append(best_y)
    raw = np.asarray(raw)
    best = np.asarray(best)
    final = np.asarray([float(row["L2_err"]) for row in rows], dtype=float)
    return {
        "time_s": grid,
        "median_l2": np.exp(np.median(np.log(raw), axis=0)),
        "q25_l2": np.exp(np.quantile(np.log(raw), 0.25, axis=0)),
        "q75_l2": np.exp(np.quantile(np.log(raw), 0.75, axis=0)),
        "median_best_l2": np.exp(np.median(np.log(best), axis=0)),
        "q25_best_l2": np.exp(np.quantile(np.log(best), 0.25, axis=0)),
        "q75_best_l2": np.exp(np.quantile(np.log(best), 0.75, axis=0)),
        "n_seeds": len(traces),
        "final_median_l2": float(np.median(final)),
        "final_q25_l2": float(np.quantile(final, 0.25)),
        "final_q75_l2": float(np.quantile(final, 0.75)),
        "history_points_min": min(len(trace["time"]) for trace in traces),
        "history_points_max": max(len(trace["time"]) for trace in traces),
        "tmax_s": tmax,
    }


def load_and_summarize(results: Path, task_ids: tuple[str, ...], n_grid: int) -> dict:
    summaries = {}
    for task_id in task_ids:
        rows, history = _load_task(results, task_id)
        by_variant_rows = {method: [row for row in rows if row["variant"] == method]
                           for method in FORMAL_METHODS}
        by_variant_history = {
            method: [
                {
                    "time": [float(point[0]) for point in item["history"]],
                    "value": [float(point[1]) for point in item["history"]],
                }
                for item in history if item["variant"] == method
            ]
            for method in FORMAL_METHODS
        }
        for method in FORMAL_METHODS:
            if len(by_variant_rows[method]) != 5 or len(by_variant_history[method]) != 5:
                raise ValueError(f"{task_id}/{method}: expected five seeds")
            summaries[(task_id, method)] = _summarize(
                by_variant_history[method], by_variant_rows[method], n_grid
            )
    return summaries


def write_summary(summaries: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id", "variant", "time_s", "median_l2", "q25_l2", "q75_l2",
        "median_best_l2", "q25_best_l2", "q75_best_l2", "n_seeds",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for (task_id, variant), summary in summaries.items():
            for index, time_s in enumerate(summary["time_s"]):
                writer.writerow({
                    "task_id": task_id,
                    "variant": variant,
                    "time_s": f"{time_s:.6f}",
                    "median_l2": f"{summary['median_l2'][index]:.10e}",
                    "q25_l2": f"{summary['q25_l2'][index]:.10e}",
                    "q75_l2": f"{summary['q75_l2'][index]:.10e}",
                    "median_best_l2": f"{summary['median_best_l2'][index]:.10e}",
                    "q25_best_l2": f"{summary['q25_best_l2'][index]:.10e}",
                    "q75_best_l2": f"{summary['q75_best_l2'][index]:.10e}",
                    "n_seeds": summary["n_seeds"],
                })


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _map_xy(x, y, x0, y0, width, height, xmax, ymin, ymax):
    px = x0 + float(x) / xmax * width
    py = y0 + height - (math.log10(max(float(y), FLOOR)) - math.log10(ymin)) \
        / (math.log10(ymax) - math.log10(ymin)) * height
    return px, py


def _axis_limits(summaries: dict, task_id: str):
    values = []
    for (task, _), summary in summaries.items():
        if task == task_id:
            values.extend(summary["q25_l2"])
            values.extend(summary["q75_l2"])
    ymin = max(1.0e-5, 10.0 ** math.floor(math.log10(max(min(values), 1e-12)) - 0.25))
    ymax = max(2.0, 10.0 ** math.ceil(math.log10(max(values) * 1.15)))
    return ymin, ymax


def _draw_png(summaries: dict, path: Path, task_ids: tuple[str, ...]) -> None:
    from PIL import Image, ImageDraw

    scale = 2
    width, height = 2160, 720
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(22, bold=True)
    tick_font = _font(16)
    label_font = _font(18)
    legend_font = _font(14)
    left, right, top, bottom, gap = 160, 46, 74, 122, 72
    panel_width = (width - left - right - 2 * gap) / 3
    panel_height = height - top - bottom

    for index, task_id in enumerate(task_ids):
        x0, y0 = left + index * (panel_width + gap), top
        x1, y1 = x0 + panel_width, y0 + panel_height
        ymin, ymax = _axis_limits(summaries, task_id)
        draw.rectangle((x0, y0, x1, y1), outline=(30, 30, 30), width=2)
        for exponent in range(-8, 3):
            value = 10.0 ** exponent
            if not ymin <= value <= ymax:
                continue
            _, py = _map_xy(0.0, value, x0, y0, panel_width, panel_height, 1200.0, ymin, ymax)
            draw.line((x0, py, x1, py), fill=(222, 226, 232), width=1)
            draw.text((x0 - 70, py - 11), f"1e{exponent}", fill=(45, 45, 45), font=tick_font)
        for xtick in (0.0, 600.0, 1200.0):
            px, _ = _map_xy(xtick, ymin, x0, y0, panel_width, panel_height, 1200.0, ymin, ymax)
            draw.line((px, y0, px, y1), fill=(235, 237, 240), width=1)
            draw.text((px - 16, y1 + 12), str(int(xtick)), fill=(45, 45, 45), font=tick_font)
        draw.text((x0 + 4, y0 - 42), TASK_TITLES.get(task_id, task_id),
                  fill=(20, 20, 20), font=title_font)

        legend_y = y0 + 12
        for method in FORMAL_METHODS:
            summary = summaries[(task_id, method)]
            color = _hex_rgb(COLORS[method])
            x = summary["time_s"]
            raw_points = [_map_xy(t, v, x0, y0, panel_width, panel_height, 1200.0, ymin, ymax)
                          for t, v in zip(x, summary["median_l2"])]
            lo_points = [_map_xy(t, v, x0, y0, panel_width, panel_height, 1200.0, ymin, ymax)
                         for t, v in zip(x, summary["q25_l2"])]
            hi_points = [_map_xy(t, v, x0, y0, panel_width, panel_height, 1200.0, ymin, ymax)
                         for t, v in zip(x, summary["q75_l2"])]
            overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
            ImageDraw.Draw(overlay).polygon(hi_points + list(reversed(lo_points)),
                                             fill=(*color, 38))
            image = Image.alpha_composite(image, overlay)
            draw = ImageDraw.Draw(image)
            draw.line(raw_points, fill=(*color, 255), width=4, joint="curve")
            best_points = [_map_xy(t, v, x0, y0, panel_width, panel_height, 1200.0, ymin, ymax)
                           for t, v in zip(x, summary["median_best_l2"])]
            for a, b in zip(best_points[::2], best_points[1::2]):
                draw.line((a, b), fill=(*color, 200), width=2)
            ex, ey = _map_xy(x[-1], summary["final_median_l2"], x0, y0,
                             panel_width, panel_height, 1200.0, ymin, ymax)
            draw.line((ex - 6, ey - 6, ex + 6, ey + 6), fill=(*color, 255), width=3)
            draw.line((ex - 6, ey + 6, ex + 6, ey - 6), fill=(*color, 255), width=3)
            ly = legend_y
            draw.line((x1 - 282, ly + 8, x1 - 247, ly + 8), fill=(*color, 255), width=3)
            draw.text((x1 - 240, ly - 1), METHOD_LABELS[method], fill=(35, 35, 35), font=legend_font)
            legend_y += 24
        draw.text((x0 + panel_width / 2 - 92, y1 + 46), "wall-clock time (s)",
                  fill=(35, 35, 35), font=label_font)
        if index == 0:
            ylabel = Image.new("RGBA", (260, 42), (255, 255, 255, 0))
            ImageDraw.Draw(ylabel).text((0, 0), "held-out relative L2 (lower is better)",
                                        fill=(35, 35, 35), font=label_font)
            ylabel = ylabel.rotate(90, expand=True)
            image.alpha_composite(ylabel, (8, int(y0 + panel_height / 2 - ylabel.height / 2)))
    draw = ImageDraw.Draw(image)
    draw.text((width / 2 - 610, height - 40),
              "solid: raw median; shading: seed IQR; dotted: best-so-far median; x: final median",
              fill=(55, 55, 55), font=legend_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, dpi=(300, 300))


def _draw_pdf(summaries: dict, path: Path, task_ids: tuple[str, ...]) -> None:
    from reportlab.lib.colors import Color, HexColor, black
    from reportlab.pdfgen import canvas

    width, height = 10.8 * 72, 3.6 * 72
    c = canvas.Canvas(str(path), pagesize=(width, height))
    left, right, top, bottom, gap = 50, 18, 34, 54, 36
    panel_width = (width - left - right - 2 * gap) / 3
    panel_height = height - top - bottom
    for index, task_id in enumerate(task_ids):
        x0, y0 = left + index * (panel_width + gap), bottom
        x1, y1 = x0 + panel_width, y0 + panel_height
        ymin, ymax = _axis_limits(summaries, task_id)
        c.setStrokeColor(black)
        c.rect(x0, y0, panel_width, panel_height)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(x0 + 2, y1 + 11, TASK_TITLES.get(task_id, task_id))
        c.setFont("Helvetica", 6.2)
        for exponent in range(-8, 3):
            value = 10.0 ** exponent
            if not ymin <= value <= ymax:
                continue
            _, py = _map_xy(0.0, value, x0, y0, panel_width, panel_height, 1200.0, ymin, ymax)
            c.setStrokeColorRGB(0.87, 0.88, 0.90)
            c.line(x0, py, x1, py)
            c.setFillColorRGB(0.18, 0.18, 0.18)
            c.drawRightString(x0 - 4, py - 2, f"1e{exponent}")
        for xtick in (0.0, 600.0, 1200.0):
            px, _ = _map_xy(xtick, ymin, x0, y0, panel_width, panel_height, 1200.0, ymin, ymax)
            c.setStrokeColorRGB(0.92, 0.93, 0.94)
            c.line(px, y0, px, y1)
            c.setFillColorRGB(0.18, 0.18, 0.18)
            c.drawCentredString(px, y0 - 13, str(int(xtick)))
        legend_y = y1 - 12
        for method in FORMAL_METHODS:
            summary = summaries[(task_id, method)]
            color = HexColor(COLORS[method])
            x = summary["time_s"]
            raw = [_map_xy(t, v, x0, y0, panel_width, panel_height, 1200.0, ymin, ymax)
                   for t, v in zip(x, summary["median_l2"])]
            lo = [_map_xy(t, v, x0, y0, panel_width, panel_height, 1200.0, ymin, ymax)
                  for t, v in zip(x, summary["q25_l2"])]
            hi = [_map_xy(t, v, x0, y0, panel_width, panel_height, 1200.0, ymin, ymax)
                  for t, v in zip(x, summary["q75_l2"])]
            c.setFillColor(Color(color.red, color.green, color.blue, alpha=0.15))
            polygon = c.beginPath()
            polygon.moveTo(*hi[0])
            for point in hi[1:] + list(reversed(lo)):
                polygon.lineTo(*point)
            polygon.close()
            c.drawPath(polygon, fill=1, stroke=0)
            c.setStrokeColor(color)
            c.setLineWidth(1.2)
            for a, b in zip(raw, raw[1:]):
                c.line(*a, *b)
            best = [_map_xy(t, v, x0, y0, panel_width, panel_height, 1200.0, ymin, ymax)
                    for t, v in zip(x, summary["median_best_l2"])]
            c.setDash(2, 2)
            c.setLineWidth(0.55)
            for a, b in zip(best, best[1:]):
                c.line(*a, *b)
            c.setDash()
            ex, ey = _map_xy(x[-1], summary["final_median_l2"], x0, y0,
                             panel_width, panel_height, 1200.0, ymin, ymax)
            c.setLineWidth(0.9)
            c.line(ex - 3, ey - 3, ex + 3, ey + 3)
            c.line(ex - 3, ey + 3, ex + 3, ey - 3)
            c.setStrokeColor(color)
            c.line(x1 - 95, legend_y, x1 - 82, legend_y)
            c.setFillColorRGB(0.15, 0.15, 0.15)
            c.setFont("Helvetica", 5.1)
            c.drawString(x1 - 79, legend_y - 2, METHOD_LABELS[method])
            legend_y -= 11
        c.setFillColorRGB(0.15, 0.15, 0.15)
        c.setFont("Helvetica", 6.2)
        c.drawCentredString(x0 + panel_width / 2, y0 - 29, "wall-clock time (s)")
        if index == 0:
            c.saveState()
            c.translate(x0 - 30, y0 + panel_height / 2)
            c.rotate(90)
            c.drawCentredString(0, 0, "held-out relative L2 (lower is better)")
            c.restoreState()
    c.setFillColorRGB(0.2, 0.2, 0.2)
    c.setFont("Helvetica", 5.8)
    c.drawCentredString(width / 2, 9,
                        "solid: raw median; shading: seed IQR; dotted: best-so-far median; x: final median")
    c.save()


def _write_manifest(summaries: dict, results: Path, task_ids: tuple[str, ...], out_dir: Path) -> None:
    groups = {}
    for (task_id, method), summary in summaries.items():
        groups[f"{task_id}:{method}"] = {
            "n_seeds": summary["n_seeds"],
            "history_points_min": summary["history_points_min"],
            "history_points_max": summary["history_points_max"],
            "tmax_s": summary["tmax_s"],
            "final_median_l2": summary["final_median_l2"],
            "final_q25_l2": summary["final_q25_l2"],
            "final_q75_l2": summary["final_q75_l2"],
        }
    payload = {
        "protocol_id": PROTOCOL_ID,
        "source": str(results),
        "active_scope": ["polyharmonic", "chirp", "maxwell"],
        "tasks": list(task_ids),
        "methods": list(FORMAL_METHODS),
        "budget_seconds": 1200.0,
        "timing_note": "1200 s is the training-time budget; evaluation time is excluded, and the final optimizer step may overshoot the timestamp slightly",
        "statistic": "median and seed IQR in log-error space; raw held-out relative L2 histories",
        "best_so_far": "median of each seed's cumulative minimum; shown dotted for context",
        "excluded": ["auto", "experiments/archived"],
        "groups": groups,
        "summary_csv": str(out_dir / "jsc_realtime_accuracy.csv"),
        "figure_pdf": str(out_dir / "jsc_realtime_accuracy.pdf"),
        "figure_png": str(out_dir / "jsc_realtime_accuracy.png"),
    }
    (out_dir / "jsc_realtime_accuracy_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS),
                        help="three representative active task ids; defaults to the central setting in each family")
    parser.add_argument("--grid-points", type=int, default=180)
    args = parser.parse_args()
    task_ids = tuple(args.tasks)
    if len(task_ids) != 3:
        raise SystemExit("the paper figure requires exactly three task ids")
    if args.grid_points < 20:
        raise SystemExit("--grid-points must be at least 20")
    summaries = load_and_summarize(args.results, task_ids, args.grid_points)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(summaries, args.out_dir / "jsc_realtime_accuracy.csv")
    _draw_png(summaries, args.out_dir / "jsc_realtime_accuracy.png", task_ids)
    _draw_pdf(summaries, args.out_dir / "jsc_realtime_accuracy.pdf", task_ids)
    _write_manifest(summaries, args.results, task_ids, args.out_dir)
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "tasks": list(task_ids),
        "figure_pdf": str(args.out_dir / "jsc_realtime_accuracy.pdf"),
        "figure_png": str(args.out_dir / "jsc_realtime_accuracy.png"),
        "summary_csv": str(args.out_dir / "jsc_realtime_accuracy.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
