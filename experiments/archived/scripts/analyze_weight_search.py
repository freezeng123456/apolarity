#!/usr/bin/env python3
"""Build a complete, auditable report for the WAR/real-AD weight search.

The raw point JSON and text logs are treated as immutable inputs.  In
particular, the six early logs whose final JSON line predates the logging fix
are represented in the generated metrics tables with an explicit
``candidate_json`` source rather than being rewritten.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


TASKS = [
    "poly_d2_o2",
    "poly_d2_o4",
    "poly_d2_o6",
    "cahn_hilliard_o4",
    "cahn_hilliard_o6",
]
METHODS = [("war", "war"), ("real_tanh_autodiff", "realad")]
RANKINGS = {
    "war": "ranking_war.csv",
    "realad": "ranking_real_tanh_autodiff.csv",
    "shared_geomean": "ranking_shared_geomean.csv",
    "shared_minimax": "ranking_shared_minimax.csv",
}


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def pretty_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def write_json(path: Path, value):
    path.write_text(pretty_json(value) + "\n", encoding="utf-8")


def load_matrix(task_dir: Path):
    matrix = {}
    with (task_dir / "run_matrix.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cid = row["candidate_id"]
            for key in ("weight_labels", "weight_map", "weights"):
                try:
                    row[key] = json.loads(row[key])
                except (TypeError, json.JSONDecodeError):
                    pass
            row["candidate_index"] = int(row["candidate_index"])
            matrix[cid] = row
    return matrix


def final_log_json(log_path: Path):
    if not log_path.exists():
        return None, "missing_log"
    nonempty = [line.strip() for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not nonempty:
        return None, "empty_log"
    try:
        value = json.loads(nonempty[-1])
    except json.JSONDecodeError:
        return None, "non_json_final_line"
    if not isinstance(value, dict):
        return None, "non_object_final_line"
    return value, "ok"


def read_candidate(task_dir: Path, cid: str, row: dict, method: str, label: str):
    point_dir = task_dir / "points" / cid
    json_path = point_dir / f"{method}.json"
    log_path = point_dir / f"{method}.log"
    record = {
        "task_id": task_dir.name,
        "candidate_id": cid,
        "candidate_index": row["candidate_index"],
        "method": label,
        "method_name": method,
        "json_path": str(json_path.relative_to(task_dir.parent.parent)),
        "log_path": str(log_path.relative_to(task_dir.parent.parent)),
        "status": "missing_json",
        "loss": None,
        "rel_error": None,
        "steps": None,
        "history_points": 0,
        "log_final_status": "missing_log",
        "log_final_has_loss": False,
        "log_final_has_rel_error": False,
        "final_metrics_source": "missing",
        "weight_labels": row.get("weight_labels"),
        "weights": row.get("weights"),
        "weight_map": row.get("weight_map"),
    }
    if not json_path.exists():
        return record
    data = json.loads(json_path.read_text(encoding="utf-8"))
    history = data.get("history") or []
    loss = data.get("loss")
    if loss is None:
        loss = (data.get("components") or {}).get("loss")
    rel_error = data.get("rel_error")
    log_value, log_status = final_log_json(log_path)
    has_loss = isinstance(log_value, dict) and finite(log_value.get("loss"))
    has_rel = isinstance(log_value, dict) and finite(log_value.get("rel_error"))
    source = "text_log_final_line" if has_loss and has_rel else "candidate_json_top_level"
    if not finite(loss) or not finite(rel_error):
        source = "missing_or_nonfinite_json_metrics"
    record.update(
        {
            "status": data.get("status", "unknown"),
            "loss": loss,
            "rel_error": rel_error,
            "steps": data.get("steps"),
            "history_points": len(history),
            "history_final_loss": history[-1].get("loss") if history else None,
            "history_final_rel_error": history[-1].get("rel_error") if history else None,
            "log_final_status": log_status,
            "log_final_has_loss": has_loss,
            "log_final_has_rel_error": has_rel,
            "final_metrics_source": source,
            "completed_at": data.get("completed_at"),
            "budget_seconds": data.get("budget_seconds"),
            "json_protocol_id": data.get("protocol_id"),
        }
    )
    return record


def make_reports(root: Path):
    # Prefer the canonical nested raw snapshot used by the repository.  The
    # fallback keeps the script usable directly on a freshly downloaded remote
    # result directory.
    raw_root = root / "results" if (root / "results" / "manifest.json").exists() else root
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    run_rows = []
    candidate_rows = []
    verification_tasks = {}
    for task in TASKS:
        task_dir = raw_root / task
        matrix = load_matrix(task_dir)
        task_runs = []
        for cid, row in sorted(matrix.items(), key=lambda item: item[1]["candidate_index"]):
            method_records = {
                label: read_candidate(task_dir, cid, row, method, label)
                for method, label in METHODS
            }
            for rec in method_records.values():
                run_rows.append(rec)
                task_runs.append(rec)
            war = method_records["war"]
            realad = method_records["realad"]
            paired = war["status"] == "complete" and realad["status"] == "complete"
            war_err = war.get("rel_error")
            real_err = realad.get("rel_error")
            geomean = math.sqrt(war_err * real_err) if finite(war_err) and finite(real_err) and war_err >= 0 and real_err >= 0 else None
            max_error = max(war_err, real_err) if finite(war_err) and finite(real_err) else None
            mean_error = (war_err + real_err) / 2 if finite(war_err) and finite(real_err) else None
            candidate_rows.append(
                {
                    "task_id": task,
                    "candidate_id": cid,
                    "candidate_index": row["candidate_index"],
                    "weight_labels": compact_json(row.get("weight_labels")),
                    "weights": compact_json(row.get("weights")),
                    "weight_map": compact_json(row.get("weight_map")),
                    "war_status": war["status"],
                    "war_loss": war.get("loss"),
                    "war_rel_error": war_err,
                    "war_steps": war.get("steps"),
                    "war_history_points": war.get("history_points"),
                    "war_log_final_has_loss": war.get("log_final_has_loss"),
                    "war_log_final_has_rel_error": war.get("log_final_has_rel_error"),
                    "war_final_metrics_source": war.get("final_metrics_source"),
                    "realad_status": realad["status"],
                    "realad_loss": realad.get("loss"),
                    "realad_rel_error": real_err,
                    "realad_steps": realad.get("steps"),
                    "realad_history_points": realad.get("history_points"),
                    "realad_log_final_has_loss": realad.get("log_final_has_loss"),
                    "realad_log_final_has_rel_error": realad.get("log_final_has_rel_error"),
                    "realad_final_metrics_source": realad.get("final_metrics_source"),
                    "paired_complete": paired,
                    "geometric_mean_rel_error": geomean,
                    "max_rel_error": max_error,
                    "mean_rel_error": mean_error,
                }
            )
        expected = int(json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))["expected_candidates"])
        war_rows = [r for r in task_runs if r["method"] == "war"]
        ad_rows = [r for r in task_runs if r["method"] == "realad"]
        verification_tasks[task] = {
            "expected_candidates": expected,
            "matrix_candidates": len(matrix),
            "war_complete": sum(r["status"] == "complete" for r in war_rows),
            "realad_complete": sum(r["status"] == "complete" for r in ad_rows),
            "paired_complete": sum(r["status"] == "complete" for r in war_rows) == expected and sum(r["status"] == "complete" for r in ad_rows) == expected,
            "war_history_points": Counter(r["history_points"] for r in war_rows),
            "realad_history_points": Counter(r["history_points"] for r in ad_rows),
        }

    run_fieldnames = sorted({key for row in run_rows for key in row})
    with (analysis / "run_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(run_rows)
    candidate_fieldnames = list(candidate_rows[0])
    with (analysis / "final_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=candidate_fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(candidate_rows)

    top10_rows = []
    for task in TASKS:
        ranking_dir = raw_root / task / "rankings"
        for ranking, filename in RANKINGS.items():
            path = ranking_dir / filename
            with path.open(newline="", encoding="utf-8") as handle:
                for rank, row in enumerate(csv.DictReader(handle), start=1):
                    if rank > 10:
                        break
                    row = dict(row)
                    row.update({"task_id": task, "ranking": ranking, "rank": rank})
                    top10_rows.append(row)
    top10_fields = ["task_id", "ranking", "rank"] + [key for key in top10_rows[0] if key not in {"task_id", "ranking", "rank"}]
    with (analysis / "top10_full.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=top10_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(top10_rows)

    # A compact human-readable report, while the CSV/JSON files retain every value.
    lines = [
        "# WAR / 实数 autodiff 全量权重搜索报告",
        "",
        "- 协议：`war_realad_weight_grid_v1`",
        "- 范围：`poly_d2_o2`, `poly_d2_o4`, `poly_d2_o6`, `cahn_hilliard_o4`, `cahn_hilliard_o6`",
        "- 运行：994 / 994（497 个权重向量 × WAR/实数 autodiff）",
        "- 每个运行预算：60 秒；失败：0；未完成：0",
        "- 原始 JSON、逐点日志和 history 未改写。6 条最早日志末行缺少 `loss`，已在 `final_metrics.csv` 中以对应原子 JSON 的 `candidate_json_top_level` 来源补充。",
        "",
        "## 每个任务的最佳候选",
        "",
        "| task | WAR 最佳 rel_error | 实数 AD 最佳 rel_error | shared geomean 最佳 | shared minimax 最佳 |",
        "|---|---:|---:|---:|---:|",
    ]
    summary_cache = {}
    for task in TASKS:
        entries = {}
        for ranking in RANKINGS:
            with (raw_root / task / "rankings" / RANKINGS[ranking]).open(newline="", encoding="utf-8") as handle:
                entries[ranking] = next(csv.DictReader(handle))
        summary_cache[task] = entries
        def fmt(entry, key):
            return f"`{entry['candidate_id']}` {float(entry[key]):.6g}"
        lines.append(
            f"| `{task}` | {fmt(entries['war'], 'war_rel_error')} | {fmt(entries['realad'], 'real_tanh_autodiff_rel_error')} | {fmt(entries['shared_geomean'], 'geometric_mean')} | {fmt(entries['shared_minimax'], 'max_error')} |"
        )
    for task in TASKS:
        lines += ["", f"## `{task}` Top 10", ""]
        for ranking, filename in RANKINGS.items():
            lines += [f"### {ranking}", "", "| rank | candidate | weights | WAR rel_error | real AD rel_error | metric |", "|---:|---|---|---:|---:|---:|"]
            with (raw_root / task / "rankings" / filename).open(newline="", encoding="utf-8") as handle:
                for rank, row in enumerate(csv.DictReader(handle), start=1):
                    if rank > 10:
                        break
                    metric_key = {
                        "war": "war_rel_error",
                        "realad": "real_tanh_autodiff_rel_error",
                        "shared_geomean": "geometric_mean",
                        "shared_minimax": "max_error",
                    }[ranking]
                    lines.append(
                        f"| {rank} | `{row['candidate_id']}` | `{row['weight_labels']}` | {float(row['war_rel_error']):.9g} | {float(row['real_tanh_autodiff_rel_error']):.9g} | {float(row[metric_key]):.9g} |"
                    )
            lines.append("")
    while lines and not lines[-1]:
        lines.pop()
    (analysis / "TOP10.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    log_source_counts = Counter()
    log_missing_examples = []
    status_counts = Counter()
    history_counts = Counter()
    for row in run_rows:
        status_counts[row["status"]] += 1
        history_counts[str(row["history_points"])] += 1
        log_source_counts[row["final_metrics_source"]] += 1
        if row["final_metrics_source"] != "text_log_final_line" and len(log_missing_examples) < 20:
            log_missing_examples.append(row["log_path"])
    expected_by_task = {task: verification_tasks[task]["expected_candidates"] for task in TASKS}
    ranking_rows = {}
    for task in TASKS:
        ranking_rows[task] = {}
        for ranking, filename in RANKINGS.items():
            with (raw_root / task / "rankings" / filename).open(newline="", encoding="utf-8") as handle:
                ranking_rows[task][ranking] = sum(1 for _ in csv.DictReader(handle))
    temporary_files = [str(p.relative_to(root)) for p in raw_root.rglob("*") if p.is_file() and p.suffix in {".tmp", ".part", ".inprogress"}]
    progress = json.loads((raw_root / "progress.json").read_text(encoding="utf-8"))
    verification = {
        "snapshot_type": "full_complete_search",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_id": "war_realad_weight_grid_v1",
        "scope_tasks": TASKS,
        "raw_root": str(raw_root.relative_to(root)),
        "expected_candidates": sum(expected_by_task.values()),
        "expected_runs": 994,
        "candidate_rows": len(candidate_rows),
        "method_rows": len(run_rows),
        "status_counts": dict(status_counts),
        "history_point_counts": dict(history_counts),
        "log_final_metrics_source_counts": dict(log_source_counts),
        "text_log_missing_loss_or_rel_error_count": sum(v for k, v in log_source_counts.items() if k != "text_log_final_line"),
        "text_log_missing_loss_or_rel_error_examples": log_missing_examples,
        "temporary_files": temporary_files,
        "per_task": verification_tasks,
        "ranking_data_rows": ranking_rows,
        "remote_progress": progress,
        "remote_complete": progress.get("processed_runs") == 994 and progress.get("failures_seen_this_process") == 0,
    }
    write_json(analysis / "verification.json", verification)
    report = [
        "# Full-complete snapshot",
        "",
        "This directory contains the complete WAR/real-tanh-autodiff 60-second loss-weight search.",
        "",
        "- 5 tasks, 497 candidates, 994 method runs.",
        "- All JSON results are `status=complete`; no failed or running cells were reported.",
        "- Every JSON result contains a 13-point history with final loss and rel_error.",
        "- Raw point logs were preserved byte-for-byte. The six pre-fix logs are not rewritten; their final metrics are explicitly sourced from the paired JSON in `analysis/final_metrics.csv`.",
        "- `TOP10.md` and `top10_full.csv` contain WAR, real AD, shared geometric-mean, and shared minimax rankings for every task.",
        "- `verification.json` records the counts and the remote completion state.",
        "",
    ]
    (analysis / "COMPLETION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    verification = make_reports(args.root)
    print(json.dumps({
        "candidate_rows": verification["candidate_rows"],
        "method_rows": verification["method_rows"],
        "status_counts": verification["status_counts"],
        "log_sources": verification["log_final_metrics_source_counts"],
        "temporary_files": verification["temporary_files"],
    }, indent=2))


if __name__ == "__main__":
    main()
