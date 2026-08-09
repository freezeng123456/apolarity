#!/usr/bin/env python3
"""Plot paper-ready real-time accuracy curves for the three main experiments.

The script intentionally excludes the experimental ``auto`` selector.  It reads
the validated overnight archive, interpolates each seed's history in log-error
space on a common time grid, and writes both the plotted summary table and the
figure.  The shaded band is the seed-wise interquartile range; the solid line
is the median of the raw (not best-so-far) held-out relative L2 error.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np  # noqa: E402

try:  # The local desktop runtime may not ship matplotlib.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    HAVE_MATPLOTLIB = True
except ModuleNotFoundError:  # pragma: no cover - exercised on the desktop host
    HAVE_MATPLOTLIB = False


BACKEND_LABELS = {
    "direct_autodiff": "Direct AD",
    "polarization_jet": "Polarization jet",
    "waring_complex_jet": "Waring jet",
}
RISK_LABELS = {
    "vanilla": "Vanilla tanh (direct AD)",
    "complex_sinh": "Complex Sinh",
    "pwnn": "PWNN",
}
STYLE = {
    "direct_autodiff": ("#d62728", "-"),
    "polarization_jet": ("#1f77b4", "-"),
    "waring_complex_jet": ("#2ca02c", "-"),
    "vanilla": ("#ff7f0e", "-"),
    "complex_sinh": ("#1f77b4", "-"),
    "pwnn": ("#9467bd", "-"),
}
FLOOR = 1.0e-12


def load_backend_traces(results: Path):
    grouped = defaultdict(list)
    for path in sorted((results / "pinn_fixed_time").glob("seed*_*.json")):
        payload = json.loads(path.read_text())
        backend = payload["manifest"]["backend"]
        if backend not in BACKEND_LABELS:
            continue
        points = [
            (float(item["wall_seconds"]), float(item["probe_L2"]))
            for item in payload.get("history", [])
            if "probe_L2" in item
            and np.isfinite(float(item["wall_seconds"]))
            and np.isfinite(float(item["probe_L2"]))
            and float(item["probe_L2"]) > 0.0
        ]
        points.sort()
        if len(points) < 3:
            raise ValueError(f"insufficient P0-B history in {path}")
        grouped[("p0b", backend)].append({
            "seed": int(payload["manifest"]["seed"]),
            "time": np.asarray([p[0] for p in points]),
            "value": np.asarray([p[1] for p in points]),
            "final": float(payload["final_L2"]),
        })
    return grouped


def load_risk_traces(results: Path):
    grouped = defaultdict(list)
    for path in sorted((results / "risk_baselines").glob("*_history.json")):
        payload = json.loads(path.read_text())
        if not payload:
            continue
        record = payload[0]
        problem = record["problem"]
        variant = record["variant"]
        if problem not in {"chirp_a2", "maxwell_a4"}:
            continue
        if variant not in RISK_LABELS:
            continue
        points = [
            (float(row[0]), float(row[1]))
            for row in record.get("history", [])
            if len(row) >= 2
            and np.isfinite(float(row[0]))
            and np.isfinite(float(row[1]))
            and float(row[1]) > 0.0
        ]
        points.sort()
        if len(points) < 3:
            raise ValueError(f"insufficient P0-C history in {path}")
        row_path = path.with_name(path.name.replace("_history.json", ".csv"))
        with row_path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        grouped[(problem, variant)].append({
            "seed": int(record["seed"]),
            "time": np.asarray([p[0] for p in points]),
            "value": np.asarray([p[1] for p in points]),
            "final": float(row["L2_err"]),
        })
    return grouped


def summarize(traces, *, n_grid: int = 180):
    tmax = min(float(trace["time"][-1]) for trace in traces)
    grid = np.linspace(0.0, tmax, n_grid)
    raw = []
    best = []
    for trace in traces:
        x = trace["time"]
        y = np.maximum(trace["value"], FLOOR)
        log_y = np.interp(grid, x, np.log(y))
        interp = np.exp(log_y)
        raw.append(interp)
        best_values = np.minimum.accumulate(y)
        best.append(np.exp(np.interp(grid, x, np.log(best_values))))
    raw = np.asarray(raw)
    best = np.asarray(best)
    return {
        "time_s": grid,
        "median_l2": np.exp(np.median(np.log(raw), axis=0)),
        "q25_l2": np.exp(np.quantile(np.log(raw), 0.25, axis=0)),
        "q75_l2": np.exp(np.quantile(np.log(raw), 0.75, axis=0)),
        "median_best_l2": np.exp(np.median(np.log(best), axis=0)),
        "q25_best_l2": np.exp(np.quantile(np.log(best), 0.25, axis=0)),
        "q75_best_l2": np.exp(np.quantile(np.log(best), 0.75, axis=0)),
        "n_seeds": len(traces),
        "final_median_l2": float(np.median([trace["final"] for trace in traces])),
        "final_q25_l2": float(np.quantile([trace["final"] for trace in traces], 0.25)),
        "final_q75_l2": float(np.quantile([trace["final"] for trace in traces], 0.75)),
    }


def write_summary(summaries, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "panel", "variant", "time_s", "median_l2", "q25_l2", "q75_l2",
        "median_best_l2", "q25_best_l2", "q75_best_l2", "n_seeds",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for (panel, variant), summary in summaries.items():
            for i, time_s in enumerate(summary["time_s"]):
                writer.writerow({
                    "panel": panel,
                    "variant": variant,
                    "time_s": f"{time_s:.6f}",
                    "median_l2": f"{summary['median_l2'][i]:.10e}",
                    "q25_l2": f"{summary['q25_l2'][i]:.10e}",
                    "q75_l2": f"{summary['q75_l2'][i]:.10e}",
                    "median_best_l2": f"{summary['median_best_l2'][i]:.10e}",
                    "q25_best_l2": f"{summary['q25_best_l2'][i]:.10e}",
                    "q75_best_l2": f"{summary['q75_best_l2'][i]:.10e}",
                    "n_seeds": summary["n_seeds"],
                })


def _panels():
    return [
        ("p0b", "(a) 4D mixed derivative PINN", 1200.0),
        ("chirp_a2", "(b) Chirp, a=2", 600.0),
        ("maxwell_a4", "(c) Maxwell, a=4", 600.0),
    ]


def _plot_matplotlib(summaries, path: Path, png_path: Path | None):
    panels = _panels()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), constrained_layout=True)
    for axis, (panel, title, xmax) in zip(axes, panels):
        items = [(key[1], value) for key, value in summaries.items() if key[0] == panel]
        for variant, summary in sorted(items):
            label = BACKEND_LABELS.get(variant, RISK_LABELS.get(variant, variant))
            color, linestyle = STYLE[variant]
            x = summary["time_s"]
            axis.plot(x, summary["median_l2"], color=color, lw=1.8, label=label)
            axis.fill_between(x, summary["q25_l2"], summary["q75_l2"], color=color, alpha=0.16, lw=0)
            axis.plot(x, summary["median_best_l2"], color=color, lw=0.85,
                      linestyle=(0, (2, 2)), alpha=0.75)
            axis.scatter([x[-1]], [summary["final_median_l2"]], marker="x", s=28,
                         color=color, linewidths=1.2, zorder=4)
        axis.set_title(title, fontsize=9)
        axis.set_xlim(0.0, xmax)
        axis.set_yscale("log")
        axis.set_ylim(5.0e-5 if panel == "p0b" else 3.0e-5, 2.0)
        axis.set_xlabel("wall-clock time (s)", fontsize=8)
        axis.set_ylabel(r"held-out relative $L^2$ error", fontsize=8)
        axis.grid(True, which="both", alpha=0.22, linewidth=0.6)
        axis.tick_params(labelsize=8)
        axis.legend(frameon=False, fontsize=7.1, loc="best")
    fig.text(0.5, -0.045, "solid: median raw accuracy; shading: seed IQR; dotted: median best-so-far; ×: median final checkpoint",
             ha="center", fontsize=7.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if png_path is not None:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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


def _rgb(hex_color: str):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def _map_xy(x, y, x0, y0, width, height, xmax, ymin, ymax):
    px = x0 + (float(x) / xmax) * width
    log_min, log_max = np.log10(ymin), np.log10(ymax)
    py = y0 + height - (np.log10(max(float(y), FLOOR)) - log_min) / (log_max - log_min) * height
    return px, py


def _draw_pil(summaries, path: Path, png_path: Path | None):
    """Small dependency-free fallback: PNG with PIL and vector PDF below."""
    from PIL import Image, ImageDraw

    scale = 2
    width, height = 2160, 650
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(22, bold=True)
    tick_font = _font(16)
    label_font = _font(18)
    legend_font = _font(15)
    left, right, top, bottom, gap = 150, 55, 58, 100, 72
    panel_width = (width - left - right - 2 * gap) / 3
    panel_height = height - top - bottom
    for index, (panel, title, xmax) in enumerate(_panels()):
        x0 = left + index * (panel_width + gap)
        y0 = top
        ymin = 5e-5 if panel == "p0b" else 3e-5
        ymax = 2.0
        x1, y1 = x0 + panel_width, y0 + panel_height
        draw.rectangle((x0, y0, x1, y1), outline=(30, 30, 30), width=2)
        for exponent in range(-5, 1):
            value = 10.0 ** exponent
            if value < ymin or value > ymax:
                continue
            _, py = _map_xy(0, value, x0, y0, panel_width, panel_height, xmax, ymin, ymax)
            draw.line((x0, py, x1, py), fill=(222, 226, 232), width=1)
            draw.text((x0 - 76, py - 12), f"1e{exponent}", fill=(45, 45, 45), font=tick_font)
        for xtick in (0.0, xmax / 2.0, xmax):
            px, _ = _map_xy(xtick, ymin, x0, y0, panel_width, panel_height, xmax, ymin, ymax)
            draw.line((px, y0, px, y1), fill=(235, 237, 240), width=1)
            label = f"{int(xtick)}"
            draw.text((px - 16, y1 + 12), label, fill=(45, 45, 45), font=tick_font)
        draw.text((x0 + 5, y0 - 40), title, fill=(20, 20, 20), font=title_font)
        items = [(key[1], value) for key, value in summaries.items() if key[0] == panel]
        legend_y = y0 + 14
        for variant, summary in sorted(items):
            label = BACKEND_LABELS.get(variant, RISK_LABELS.get(variant, variant))
            color = _rgb(STYLE[variant][0])
            x = summary["time_s"]
            raw_points = [_map_xy(t, v, x0, y0, panel_width, panel_height, xmax, ymin, ymax)
                          for t, v in zip(x, summary["median_l2"])]
            lo_points = [_map_xy(t, v, x0, y0, panel_width, panel_height, xmax, ymin, ymax)
                         for t, v in zip(x, summary["q25_l2"])]
            hi_points = [_map_xy(t, v, x0, y0, panel_width, panel_height, xmax, ymin, ymax)
                         for t, v in zip(x, summary["q75_l2"])]
            overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.polygon(hi_points + list(reversed(lo_points)), fill=(*color, 38))
            image = Image.alpha_composite(image, overlay)
            draw = ImageDraw.Draw(image)
            draw.line(raw_points, fill=(*color, 255), width=4, joint="curve")
            best_points = [_map_xy(t, v, x0, y0, panel_width, panel_height, xmax, ymin, ymax)
                           for t, v in zip(x, summary["median_best_l2"])]
            for a, b in zip(best_points[::2], best_points[1::2]):
                draw.line((a, b), fill=(*color, 200), width=2)
            ex, ey = _map_xy(x[-1], summary["final_median_l2"], x0, y0, panel_width, panel_height, xmax, ymin, ymax)
            draw.line((ex - 6, ey - 6, ex + 6, ey + 6), fill=(*color, 255), width=3)
            draw.line((ex - 6, ey + 6, ex + 6, ey - 6), fill=(*color, 255), width=3)
            ly = legend_y
            draw.line((x1 - 220, ly + 8, x1 - 185, ly + 8), fill=(*color, 255), width=3)
            draw.text((x1 - 178, ly - 2), label, fill=(35, 35, 35), font=legend_font)
            legend_y += 24
        draw.text((x0 + panel_width / 2 - 70, y1 + 45), "wall-clock time (s)", fill=(35, 35, 35), font=label_font)
        if index == 0:
            ylabel = Image.new("RGBA", (260, 40), (255, 255, 255, 0))
            ylabel_draw = ImageDraw.Draw(ylabel)
            ylabel_draw.text((0, 0), "held-out relative L2", fill=(35, 35, 35), font=label_font)
            ylabel = ylabel.rotate(90, expand=True)
            image.alpha_composite(ylabel, (8, int(y0 + panel_height / 2 - ylabel.height / 2)))
            draw = ImageDraw.Draw(image)
    draw.text((width / 2 - 560, height - 38),
              "solid: median raw accuracy; shading: seed IQR; dotted: median best-so-far; x: median final",
              fill=(55, 55, 55), font=legend_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(png_path or path.with_suffix(".png"), dpi=(300, 300))
    _draw_pdf_fallback(summaries, path)


def _draw_pdf_fallback(summaries, path: Path):
    from reportlab.lib.colors import HexColor, black, Color
    from reportlab.pdfgen import canvas

    width, height = 10.8 * 72, 3.25 * 72
    c = canvas.Canvas(str(path), pagesize=(width, height))
    left, right, top, bottom, gap = 50, 18, 30, 50, 36
    panel_width = (width - left - right - 2 * gap) / 3
    panel_height = height - top - bottom
    for index, (panel, title, xmax) in enumerate(_panels()):
        x0 = left + index * (panel_width + gap)
        y0 = bottom
        ymin = 5e-5 if panel == "p0b" else 3e-5
        ymax = 2.0
        x1, y1 = x0 + panel_width, y0 + panel_height
        c.setStrokeColor(black)
        c.rect(x0, y0, panel_width, panel_height)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(x0 + 2, y1 + 10, title)
        c.setFont("Helvetica", 6.5)
        for exponent in range(-5, 1):
            value = 10.0 ** exponent
            if value < ymin or value > ymax:
                continue
            _, py = _map_xy(0, value, x0, y0, panel_width, panel_height, xmax, ymin, ymax)
            c.setStrokeColorRGB(0.87, 0.88, 0.90)
            c.line(x0, py, x1, py)
            c.setFillColorRGB(0.18, 0.18, 0.18)
            c.drawRightString(x0 - 4, py - 2, f"1e{exponent}")
        for xtick in (0.0, xmax / 2.0, xmax):
            px, _ = _map_xy(xtick, ymin, x0, y0, panel_width, panel_height, xmax, ymin, ymax)
            c.setStrokeColorRGB(0.92, 0.93, 0.94)
            c.line(px, y0, px, y1)
            c.setFillColorRGB(0.18, 0.18, 0.18)
            c.drawCentredString(px, y0 - 12, f"{int(xtick)}")
        items = [(key[1], value) for key, value in summaries.items() if key[0] == panel]
        legend_y = y1 - 12
        for variant, summary in sorted(items):
            label = BACKEND_LABELS.get(variant, RISK_LABELS.get(variant, variant))
            color = HexColor(STYLE[variant][0])
            x = summary["time_s"]
            raw = [_map_xy(t, v, x0, y0, panel_width, panel_height, xmax, ymin, ymax)
                   for t, v in zip(x, summary["median_l2"])]
            lo = [_map_xy(t, v, x0, y0, panel_width, panel_height, xmax, ymin, ymax)
                  for t, v in zip(x, summary["q25_l2"])]
            hi = [_map_xy(t, v, x0, y0, panel_width, panel_height, xmax, ymin, ymax)
                  for t, v in zip(x, summary["q75_l2"])]
            c.setFillColor(Color(color.red, color.green, color.blue, alpha=0.15))
            polygon = c.beginPath()
            polygon.moveTo(*hi[0])
            for point in hi[1:] + list(reversed(lo)):
                polygon.lineTo(*point)
            polygon.close()
            c.drawPath(polygon, fill=1, stroke=0)
            c.setStrokeColor(color)
            c.setLineWidth(1.3)
            for a, b in zip(raw, raw[1:]):
                c.line(*a, *b)
            best = [_map_xy(t, v, x0, y0, panel_width, panel_height, xmax, ymin, ymax)
                    for t, v in zip(x, summary["median_best_l2"])]
            c.setDash(2, 2)
            c.setLineWidth(0.6)
            for a, b in zip(best, best[1:]):
                c.line(*a, *b)
            c.setDash()
            ex, ey = _map_xy(x[-1], summary["final_median_l2"], x0, y0, panel_width, panel_height, xmax, ymin, ymax)
            c.setLineWidth(1.0)
            c.line(ex - 3, ey - 3, ex + 3, ey + 3)
            c.line(ex - 3, ey + 3, ex + 3, ey - 3)
            c.setStrokeColor(color)
            c.line(x1 - 95, legend_y, x1 - 82, legend_y)
            c.setFillColorRGB(0.15, 0.15, 0.15)
            c.setFont("Helvetica", 5.4)
            c.drawString(x1 - 79, legend_y - 2, label)
            legend_y -= 11
        c.setFillColorRGB(0.15, 0.15, 0.15)
        c.setFont("Helvetica", 6.2)
        c.drawCentredString(x0 + panel_width / 2, y0 - 28, "wall-clock time (s)")
        if index == 0:
            c.saveState()
            c.translate(x0 - 30, y0 + panel_height / 2)
            c.rotate(90)
            c.drawCentredString(0, 0, "held-out relative L2")
            c.restoreState()
    c.setFont("Helvetica", 5.8)
    c.drawCentredString(width / 2, 8, "solid: median raw; shading: seed IQR; dotted: median best-so-far; x: median final")
    c.save()


def plot(summaries, path: Path, png_path: Path | None):
    if HAVE_MATPLOTLIB:
        _plot_matplotlib(summaries, path, png_path)
    else:
        _draw_pil(summaries, path, png_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    grouped = load_backend_traces(args.results)
    grouped.update(load_risk_traces(args.results))
    expected = {
        ("p0b", "direct_autodiff"), ("p0b", "polarization_jet"),
        ("p0b", "waring_complex_jet"), ("chirp_a2", "vanilla"),
        ("chirp_a2", "complex_sinh"), ("maxwell_a4", "pwnn"),
        ("maxwell_a4", "complex_sinh"),
    }
    missing = expected - set(grouped)
    if missing:
        raise SystemExit(f"missing trace groups: {sorted(missing)}")
    summaries = {key: summarize(traces) for key, traces in sorted(grouped.items())}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(summaries, args.out_dir / "paper_realtime_accuracy.csv")
    plot(summaries, args.out_dir / "paper_realtime_accuracy.pdf",
         args.out_dir / "paper_realtime_accuracy.png")
    manifest = {
        "source": str(args.results),
        "excluded": ["auto_uncached", "auto_cached_selected"],
        "summary_csv": str(args.out_dir / "paper_realtime_accuracy.csv"),
        "figure_pdf": str(args.out_dir / "paper_realtime_accuracy.pdf"),
        "figure_png": str(args.out_dir / "paper_realtime_accuracy.png"),
        "statistic": "median and seed IQR in log-error space; raw histories",
        "groups": {f"{key[0]}:{key[1]}": {
            "n_seeds": value["n_seeds"],
            "final_median_l2": value["final_median_l2"],
            "final_q25_l2": value["final_q25_l2"],
            "final_q75_l2": value["final_q75_l2"],
        } for key, value in summaries.items()},
    }
    (args.out_dir / "paper_realtime_accuracy_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
