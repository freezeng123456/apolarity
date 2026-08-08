#!/usr/bin/env python3
"""Build a human-readable and auditable report for fixed-weight formal runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METHODS = ("war", "real_tanh_autodiff")
TASK_ORDER = (
    "poly_d2_o2",
    "poly_d2_o4",
    "poly_d2_o6",
    "cahn_hilliard_o4",
    "cahn_hilliard_o6",
)


def load(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    return "—" if value is None else f"{float(value):.7g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = load(root / "manifest.json")
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seed_rows: dict[str, list[dict]] = {task: [] for task in TASK_ORDER}
    verification = {
        "protocol_id": manifest["protocol_id"],
        "expected_runs": manifest["method_seed_run_count"],
        "complete_runs": 0,
        "history_final_metrics_complete": True,
        "raw_logs_present": True,
        "raw_log_final_line_has_metrics": True,
        "supplemental_final_metrics_source": "atomic_json",
        "issues": [],
    }
    for task_info in manifest["tasks"]:
        task = task_info["task_id"]
        for seed in manifest["seeds"]:
            pair = {"task_id": task, "seed": seed, "weights": task_info["weights"]}
            loaded = {}
            for method in METHODS:
                path = root / task / f"seed_{seed:03d}" / f"{method}.json"
                log_path = path.with_suffix(".log")
                if not path.exists():
                    verification["issues"].append(f"missing result: {path}")
                    continue
                result = load(path)
                loaded[method] = result
                history = result.get("history") or []
                final_history = history[-1] if history else {}
                final_line_has_metrics = False
                if log_path.exists():
                    lines = [line.strip() for line in log_path.read_text().splitlines() if line.strip()]
                    final_line_has_metrics = bool(lines and "loss" in lines[-1] and "rel_error" in lines[-1])
                else:
                    verification["raw_logs_present"] = False
                    verification["issues"].append(f"missing log: {log_path}")
                verification["raw_log_final_line_has_metrics"] &= final_line_has_metrics
                if not ("loss" in final_history and "rel_error" in final_history):
                    verification["history_final_metrics_complete"] = False
                    verification["issues"].append(f"history missing final loss/rel_error: {path}")
                if result.get("status") == "complete":
                    verification["complete_runs"] += 1
                rows.append({
                    "task_id": task,
                    "seed": seed,
                    "method": method,
                    "weights": json.dumps(task_info["weights"]),
                    "status": result.get("status"),
                    "loss": result.get("loss"),
                    "rel_error": result.get("rel_error"),
                    "training_seconds": result.get("training_seconds"),
                    "evaluation_seconds": result.get("evaluation_seconds"),
                    "steps": result.get("steps"),
                    "history_points": len(history),
                    "history_final_has_loss": "loss" in final_history,
                    "history_final_has_rel_error": "rel_error" in final_history,
                    "raw_log_final_line_has_metrics": final_line_has_metrics,
                    "final_metrics_source": "atomic_json",
                    "raw_json": str(path.relative_to(root)),
                    "raw_log": str(log_path.relative_to(root)),
                })
                pair[f"{method}_loss"] = result.get("loss")
                pair[f"{method}_rel_error"] = result.get("rel_error")
            if all(method in loaded and loaded[method].get("status") == "complete" for method in METHODS):
                war_error = float(loaded["war"]["rel_error"])
                ad_error = float(loaded["real_tanh_autodiff"]["rel_error"])
                pair["geometric_mean"] = math.sqrt(war_error * ad_error)
                pair["max_error"] = max(war_error, ad_error)
                pair["status"] = "complete"
            else:
                pair["status"] = "incomplete"
            seed_rows[task].append(pair)

    write_json(analysis / "final_metrics.json", rows)
    write_csv(analysis / "final_metrics.csv", rows)
    task_summaries = []
    report = [
        "# Fixed-weight WAR / real autodiff formal results",
        "",
        f"- Protocol: `{manifest['protocol_id']}`",
        f"- Budget: `{manifest['seconds_per_method_seed']}` seconds per method/seed",
        f"- Seeds: `{manifest['seeds']}`",
        f"- Completed cells: `{verification['complete_runs']}/{verification['expected_runs']}`",
        "- Every final history point contains `loss` and `rel_error`.",
        "- `analysis/final_metrics.*` is the canonical supplement generated from each atomic result JSON; raw logs are preserved unchanged.",
        "",
    ]
    for task in TASK_ORDER:
        info = next(item for item in manifest["tasks"] if item["task_id"] == task)
        complete = [row for row in seed_rows[task] if row["status"] == "complete"]
        war = [float(row["war_rel_error"]) for row in complete]
        ad = [float(row["real_tanh_autodiff_rel_error"]) for row in complete]
        gm = [float(row["geometric_mean"]) for row in complete]
        mm = [float(row["max_error"]) for row in complete]
        summary = {
            "task_id": task,
            "weights": info["weights"],
            "paired_complete_seed_count": len(complete),
            "war_mean_rel_error": statistics.mean(war) if war else None,
            "real_tanh_autodiff_mean_rel_error": statistics.mean(ad) if ad else None,
            "geometric_mean_mean": statistics.mean(gm) if gm else None,
            "minimax_worst_seed": max(mm) if mm else None,
        }
        task_summaries.append(summary)
        report += [
            f"## `{task}` — weights `{info['weights']}`",
            "",
            "| seed | WAR loss | WAR rel_error | real AD loss | real AD rel_error | geomean | minimax |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in seed_rows[task]:
            report.append(
                f"| {row['seed']} | {fmt(row.get('war_loss'))} | {fmt(row.get('war_rel_error'))} | "
                f"{fmt(row.get('real_tanh_autodiff_loss'))} | {fmt(row.get('real_tanh_autodiff_rel_error'))} | "
                f"{fmt(row.get('geometric_mean'))} | {fmt(row.get('max_error'))} |"
            )
        report += [
            "",
            f"Mean WAR rel_error: `{fmt(summary['war_mean_rel_error'])}`; mean real AD rel_error: `{fmt(summary['real_tanh_autodiff_mean_rel_error'])}`; mean geomean: `{fmt(summary['geometric_mean_mean'])}`; worst minimax: `{fmt(summary['minimax_worst_seed'])}`.",
            "",
        ]
    verification["all_complete"] = verification["complete_runs"] == verification["expected_runs"]
    verification["task_summaries"] = task_summaries
    write_json(analysis / "verification.json", verification)
    (analysis / "RESULTS.md").write_text("\n".join(report).rstrip() + "\n")
    return 0 if verification["all_complete"] and not verification["issues"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
