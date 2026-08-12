#!/usr/bin/env python3
"""Create auditable HO-04 formal tables and figures on the experiment host."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


TASK = "hyperviscous_ns_2d_o4"
METHODS = ("war", "real_tanh_autodiff")
COLORS = {"war": "#0072B2", "real_tanh_autodiff": "#D55E00"}
LABELS = {"war": "WAR (complex64)", "real_tanh_autodiff": "AD (float32)"}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "median": statistics.median(values),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    formal_root = args.formal_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (formal_root / "FORMAL_COMPLETE").is_file():
        raise RuntimeError("FORMAL_COMPLETE marker is missing")

    final_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    results: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for seed in range(5):
        for method in METHODS:
            result = load_json(
                formal_root / TASK / f"seed_{seed:03d}" / f"{method}.json"
            )
            if result.get("status") != "complete":
                raise RuntimeError(f"incomplete formal result seed={seed} {method}")
            results[method].append(result)
            metrics = result["metrics"]
            final_rows.append({
                "seed": seed,
                "method": method,
                "loss": result["loss"],
                "velocity_rel_error": result["rel_error"],
                "u_rel_error": metrics["u_rel_error"],
                "v_rel_error": metrics["v_rel_error"],
                "pressure_rel_error": metrics["pressure_rel_error"],
                "divergence_rms": metrics["divergence_rms"],
                "pressure_mean_max_abs": metrics["pressure_mean_max_abs"],
                "energy_relative_rmse": metrics["energy_relative_rmse"],
                "steps": result["steps"],
                "ms_per_step": result["ms_per_step"],
                "peak_mb": result["peak_mb"],
            })
            for point in result["history"]:
                curve_rows.append({
                    "seed": seed,
                    "method": method,
                    "elapsed_seconds": point["elapsed_seconds"],
                    "step": point["step"],
                    "velocity_rel_error": point["rel_error"],
                    "pressure_rel_error": point.get("pressure_rel_error"),
                    "loss": point["loss"],
                })
    write_csv(output / "final_metrics.csv", final_rows)
    write_csv(output / "realtime_accuracy.csv", curve_rows)

    common_time = np.linspace(0.0, 1200.0, 241)
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for method in METHODS:
        interpolated: list[np.ndarray] = []
        for seed, result in enumerate(results[method]):
            times = np.asarray(
                [float(point["elapsed_seconds"]) for point in result["history"]]
            )
            errors = np.asarray(
                [float(point["rel_error"]) for point in result["history"]]
            )
            order = np.argsort(times)
            times = times[order]
            errors = errors[order]
            curve = np.interp(common_time, times, errors)
            interpolated.append(curve)
            axis.plot(
                times,
                errors,
                color=COLORS[method],
                alpha=0.18,
                linewidth=0.9,
            )
        stack = np.stack(interpolated)
        median = np.median(stack, axis=0)
        q1 = np.quantile(stack, 0.25, axis=0)
        q3 = np.quantile(stack, 0.75, axis=0)
        axis.plot(
            common_time,
            median,
            color=COLORS[method],
            linewidth=2.2,
            label=LABELS[method],
        )
        axis.fill_between(
            common_time, q1, q3, color=COLORS[method], alpha=0.16
        )
    axis.set_yscale("log")
    axis.set_xlabel("Training wall time (s)")
    axis.set_ylabel("Velocity relative L2 error")
    axis.set_title("2D fourth-order hyperviscous Navier–Stokes")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(output / "realtime_velocity_rel_error.png", dpi=240)
    fig.savefig(output / "realtime_velocity_rel_error.pdf")
    plt.close(fig)

    summary: dict[str, Any] = {"task": TASK, "methods": {}}
    for method in METHODS:
        method_rows = [row for row in final_rows if row["method"] == method]
        summary["methods"][method] = {
            key: distribution([float(row[key]) for row in method_rows])
            for key in (
                "velocity_rel_error",
                "pressure_rel_error",
                "divergence_rms",
                "pressure_mean_max_abs",
                "energy_relative_rmse",
                "ms_per_step",
                "peak_mb",
            )
        }
    war_median = summary["methods"]["war"]["velocity_rel_error"]["median"]
    ad_median = summary["methods"]["real_tanh_autodiff"]["velocity_rel_error"]["median"]
    summary["ad_over_war_median_error"] = ad_median / max(war_median, 1e-30)
    summary["war_seed_wins"] = sum(
        float(next(row["velocity_rel_error"] for row in final_rows if row["seed"] == seed and row["method"] == "war"))
        < float(next(row["velocity_rel_error"] for row in final_rows if row["seed"] == seed and row["method"] == "real_tanh_autodiff"))
        for seed in range(5)
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    files = sorted(path for path in output.iterdir() if path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("\n".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in files if path.is_file()
    ) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

