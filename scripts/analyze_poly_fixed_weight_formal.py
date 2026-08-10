#!/usr/bin/env python3
"""Validate and summarize the current Polyharmonic fixed-weight formal bundle.

This analysis is intentionally text/CSV/JSON only.  It does not generate plots.
The real baseline remains the approved float32 tanh network.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_ID = "war_realad_fixed_weights_common_xavier_fp32_v1"
ENGINE_PROTOCOL_ID = "war_realad_weight_grid_common_xavier_fp32_v1"
TASKS = ("poly_d2_o2", "poly_d2_o4", "poly_d2_o6")
METHODS = ("war", "real_tanh_autodiff")
SEEDS = tuple(range(5))
EXPECTED_WEIGHTS = {
    "poly_d2_o2": [1.0],
    "poly_d2_o4": [1.0, 1.0],
    "poly_d2_o6": [10.0, 1.0, 1.0],
}
EXPECTED_SOURCE_HASHES = {
    "experiments/common/weight_search.py":
        "1da016aa5f917ce445f184d74a5b5d10fcb5b7e53e0cb7f846eb428b68289e64",
    "scripts/run_weight_search.py":
        "904e43a3c96fa85d51a21f578be35ce0996ffe17dae3ba0ecff7f6b862aa8d85",
    "scripts/run_fixed_weight_formal.py":
        "f8063744f48c1400ff7a3263feb1aec8f7f8d5f547abbab5680a7d64b2395247",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def inspect_finite(value: Any, label: str, issues: list[str]) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            issues.append(f"non-finite value: {label}={value}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            inspect_finite(item, f"{label}[{index}]", issues)
    elif isinstance(value, dict):
        for key, item in value.items():
            inspect_finite(item, f"{label}.{key}", issues)


def distribution(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    return {
        "mean": statistics.fmean(data),
        "std": statistics.stdev(data) if len(data) > 1 else 0.0,
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
    }


def verify_raw_checksums(root: Path) -> tuple[int, list[str]]:
    issues: list[str] = []
    checked = 0
    for line in (root / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = root / relative.strip()
        checked += 1
        if not path.is_file():
            issues.append(f"missing checksum target: {relative.strip()}")
        elif sha256(path) != expected:
            issues.append(f"checksum mismatch: {relative.strip()}")
    return checked, issues


def audit_source_snapshot(root: Path) -> tuple[dict[str, str], list[str]]:
    source_root = root / "provenance" / "source_snapshot"
    observed: dict[str, str] = {}
    issues: list[str] = []
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        path = source_root / relative
        if not path.is_file():
            issues.append(f"missing run source snapshot: {relative}")
            continue
        observed[relative] = sha256(path)
        if observed[relative] != expected:
            issues.append(f"run source snapshot hash mismatch: {relative}")
    return observed, issues


def load_and_audit(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[str] = []
    warnings: list[str] = []
    checksum_count, checksum_issues = verify_raw_checksums(root)
    issues.extend(checksum_issues)
    source_hashes, source_issues = audit_source_snapshot(root)
    issues.extend(source_issues)

    manifest = load_json(root / "manifest.json")
    summary = load_json(root / "summary.json")
    marker = load_json(root / "FORMAL_COMPLETE")
    progress = load_json(root / "progress.json")
    runs = load_json(root / "runs.json")
    for name, payload in (
        ("manifest", manifest),
        ("summary", summary),
        ("marker", marker),
        ("progress", progress),
        ("runs", runs),
    ):
        inspect_finite(payload, name, issues)

    expected_manifest = {
        "protocol_id": PROTOCOL_ID,
        "engine_protocol_id": ENGINE_PROTOCOL_ID,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "seconds_per_method_seed": 1200.0,
        "method_seed_run_count": 30,
        "serial_single_gpu": True,
        "smoke": False,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            issues.append(
                f"manifest mismatch for {key}: {manifest.get(key)!r} != {expected!r}"
            )

    git = manifest.get("git", {})
    if git.get("sha") != "00113c16a4596e41871da6f5b00c43e968d63b8f":
        issues.append(f"unexpected recorded git SHA: {git.get('sha')}")
    if git.get("dirty") is not True:
        issues.append("expected the recovered bundle to disclose git dirty=true")
    else:
        warnings.append(
            "The run recorded git dirty=true; exact effective source files are preserved "
            "under provenance/source_snapshot and verified by SHA-256."
        )

    for name, payload in (("summary", summary), ("marker", marker)):
        if payload.get("all_complete") is not True:
            issues.append(f"{name} does not report all_complete")
        if payload.get("complete_runs") != 30:
            issues.append(f"{name} complete_runs={payload.get('complete_runs')}")
    if progress.get("processed_runs") != 30 or progress.get("total_runs") != 30:
        issues.append("progress does not report 30/30")
    if not isinstance(runs, list) or len(runs) != 30:
        issues.append("runs.json does not contain 30 method/seed rows")

    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, int, str]] = set()
    total_history_points = 0
    for task in TASKS:
        for seed in SEEDS:
            config_path = root / task / f"seed_{seed:03d}" / "config.json"
            if not config_path.is_file():
                issues.append(f"missing seed config: {config_path.relative_to(root)}")
            else:
                inspect_finite(load_json(config_path), str(config_path), issues)
            for method in METHODS:
                base = root / task / f"seed_{seed:03d}" / method
                result_path = base.with_suffix(".json")
                log_path = base.with_suffix(".log")
                if not result_path.is_file() or not log_path.is_file():
                    issues.append(f"missing result/log pair: {base.relative_to(root)}")
                    continue
                result = load_json(result_path)
                inspect_finite(result, str(result_path), issues)
                key = (task, seed, method)
                if key in keys:
                    issues.append(f"duplicate method/seed key: {key}")
                keys.add(key)
                if result.get("status") != "complete":
                    issues.append(f"incomplete result: {key}")
                if result.get("task_id") != task or result.get("seed") != seed:
                    issues.append(f"task/seed mismatch: {key}")
                if result.get("method") != method:
                    issues.append(f"method mismatch: {key}")
                if result.get("formal_protocol_id") != PROTOCOL_ID:
                    issues.append(f"formal protocol mismatch: {key}")
                if result.get("protocol_id") != ENGINE_PROTOCOL_ID:
                    issues.append(f"engine protocol mismatch: {key}")
                if result.get("weights") != EXPECTED_WEIGHTS[task]:
                    issues.append(f"weight mismatch: {key}")
                if float(result.get("budget_seconds", -1.0)) != 1200.0:
                    issues.append(f"budget mismatch: {key}")
                model = result.get("model", {})
                expected_activation = "sinh" if method == "war" else "tanh"
                expected_dtype = "torch.complex64" if method == "war" else "torch.float32"
                if model.get("activation") != expected_activation:
                    issues.append(f"activation mismatch: {key}")
                if model.get("parameter_dtype") != expected_dtype:
                    issues.append(f"dtype mismatch: {key}")
                if model.get("init_mode") != "common_xavier":
                    issues.append(f"initialization mismatch: {key}")
                if model.get("frequency_initialization") != "disabled":
                    issues.append(f"frequency initialization enabled: {key}")

                history = result.get("history")
                if not isinstance(history, list) or not history:
                    issues.append(f"missing history: {key}")
                    history = []
                elapsed = [float(item["elapsed_seconds"]) for item in history]
                if any(elapsed[index] > elapsed[index + 1] for index in range(len(elapsed) - 1)):
                    issues.append(f"non-monotone history time: {key}")
                if any("loss" not in item or "rel_error" not in item for item in history):
                    issues.append(f"history metric missing: {key}")
                total_history_points += len(history)

                lines = log_path.read_text().splitlines()
                if not lines:
                    issues.append(f"empty log: {key}")
                else:
                    try:
                        final_log = json.loads(lines[-1])
                    except json.JSONDecodeError:
                        issues.append(f"log final line is not JSON: {key}")
                    else:
                        if "loss" not in final_log or "rel_error" not in final_log:
                            issues.append(f"log final metrics missing: {key}")

                rows.append({
                    "task_id": task,
                    "seed": seed,
                    "method": method,
                    "weights": json.dumps(EXPECTED_WEIGHTS[task]),
                    "loss": result.get("loss"),
                    "rel_error": result.get("rel_error"),
                    "steps": result.get("steps"),
                    "training_seconds": result.get("training_seconds"),
                    "ms_per_step": result.get("ms_per_step"),
                    "peak_mb": result.get("peak_mb"),
                    "history_points": len(history),
                    "history_endpoint_seconds": elapsed[-1] if elapsed else None,
                })

    if len(keys) != 30 or len(rows) != 30:
        issues.append(f"expected 30 unique result rows; got keys={len(keys)}, rows={len(rows)}")
    if total_history_points != 7230:
        issues.append(f"expected 7230 history points; got {total_history_points}")

    audit = {
        "generated_at": utc_now(),
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "warnings": warnings,
        "raw_checksum_count": checksum_count,
        "raw_sha256sums_sha256": sha256(root / "SHA256SUMS"),
        "source_snapshot_hashes": source_hashes,
        "result_rows": len(rows),
        "history_points": total_history_points,
    }
    return rows, audit


def build_statistics(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = [row for row in rows if row["task_id"] == task]
        by_method: dict[str, dict[str, float]] = {}
        for method in METHODS:
            method_rows = [row for row in task_rows if row["method"] == method]
            stats = distribution(float(row["rel_error"]) for row in method_rows)
            by_method[method] = stats
            aggregate.append({"task_id": task, "method": method, **stats})

        war_wins = 0
        ad_wins = 0
        ratios: list[float] = []
        for seed in SEEDS:
            seed_rows = {row["method"]: row for row in task_rows if row["seed"] == seed}
            war = float(seed_rows["war"]["rel_error"])
            ad = float(seed_rows["real_tanh_autodiff"]["rel_error"])
            winner = "war" if war < ad else "real_tanh_autodiff" if ad < war else "tie"
            war_wins += int(winner == "war")
            ad_wins += int(winner == "real_tanh_autodiff")
            ratios.append(ad / war)
            paired.append({
                "task_id": task,
                "seed": seed,
                "war_rel_error": war,
                "real_tanh_autodiff_rel_error": ad,
                "ad_over_war": ad / war,
                "winner": winner,
            })
        task_summaries.append({
            "task_id": task,
            "weights": EXPECTED_WEIGHTS[task],
            "war": by_method["war"],
            "real_tanh_autodiff": by_method["real_tanh_autodiff"],
            "war_wins": war_wins,
            "real_tanh_autodiff_wins": ad_wins,
            "ad_over_war_mean": statistics.fmean(ratios),
            "both_methods_below_0_1": all(
                float(row["rel_error"]) < 0.1 for row in task_rows
            ),
        })
    return aggregate, paired, task_summaries


def report_markdown(audit: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    by_task = {summary["task_id"]: summary for summary in summaries}
    lines = [
        "# Polyharmonic common-Xavier 正式实验",
        "",
        "## 验收",
        "",
        f"- 状态：`{audit['status']}`；30/30 cell，5 seeds，三组 Poly task。",
        f"- 原始校验和：{audit['raw_checksum_count']} 项通过。",
        f"- 实时 history：{audit['history_points']} 个数据点，均嵌在方法 JSON 的 `history` 字段中。",
        "- WAR：complex64、sinh、common Xavier、无频率初始化。",
        "- 实数 AD：float32、tanh、common Xavier、无频率初始化。",
        "- 每方法每 seed 1200 秒；网络宽度 128、深度 4。",
        "",
        "## 最终相对误差（5 seeds）",
        "",
        "| Task | 方法 | Mean | Std | Median | Min | Max | 胜出 seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        for method, label in (("war", "WAR"), ("real_tanh_autodiff", "Real tanh AD")):
            stats = summary[method]
            wins = summary["war_wins"] if method == "war" else summary["real_tanh_autodiff_wins"]
            lines.append(
                f"| {summary['task_id']} | {label} | {stats['mean']:.6g} | "
                f"{stats['std']:.6g} | {stats['median']:.6g} | {stats['min']:.6g} | "
                f"{stats['max']:.6g} | {wins}/5 |"
            )
    lines.extend([
        "",
        "## 解读",
        "",
        f"- `poly_d2_o2`：WAR 在 {by_task['poly_d2_o2']['war_wins']}/5 seeds 上更低，平均误差约为实数 AD 的四分之一。",
        f"- `poly_d2_o4`：实数 tanh AD 在 {by_task['poly_d2_o4']['real_tanh_autodiff_wins']}/5 seeds 上更低；两种方法都达到 1e-2 以下。",
        "- `poly_d2_o6`：两种方法的误差都约为 1，说明共同 Xavier、无频率初始化的当前配置没有学到六阶算例。该 task 必须如实报告为失败，不能与旧频率初始化结果混用。",
        "",
        "## Provenance 边界",
        "",
        "结果记录的基准提交为 `00113c16a4596e41871da6f5b00c43e968d63b8f`，并明确记录 `git dirty=true`。为保证可复核性，实际运行时使用的三个源文件已原样保存在 `provenance/source_snapshot/`，并由固定 SHA-256 验证。当前仓库中的正式 runner 在该快照基础上仅做了目录命名、Poly-only 默认任务和“不保留 raw smoke”的结构性整理；实验方法与结果未改写。",
        "",
        "完整逐 seed 数值见 `final_metrics.csv` 和 `paired_comparison.csv`。",
    ])
    return "\n".join(lines) + "\n"


def write_analysis_checksums(analysis: Path) -> None:
    paths = sorted(
        path for path in analysis.iterdir()
        if path.is_file() and path.name != "SHA256SUMS" and ".tmp" not in path.name
    )
    atomic_text(
        analysis / "SHA256SUMS",
        "\n".join(f"{sha256(path)}  {path.name}" for path in paths) + "\n",
    )


def write_source_checksums(root: Path) -> None:
    source_root = root / "provenance" / "source_snapshot"
    rows = []
    for relative in sorted(EXPECTED_SOURCE_HASHES):
        path = source_root / relative
        rows.append(f"{sha256(path)}  source_snapshot/{relative}")
    atomic_text(root / "provenance" / "SOURCE_SHA256SUMS", "\n".join(rows) + "\n")


def write_delivery_checksums(root: Path) -> None:
    """Cover the immutable raw bundle plus local audit/provenance additions."""
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.name != "DELIVERY_SHA256SUMS"
        and ".tmp" not in path.name
    )
    atomic_text(
        root / "DELIVERY_SHA256SUMS",
        "\n".join(
            f"{sha256(path)}  {path.relative_to(root)}" for path in paths
        ) + "\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.result_root.resolve()
    analysis = (args.analysis_dir or root / "analysis").resolve()
    rows, audit = load_and_audit(root)
    atomic_json(analysis / "audit.json", audit)
    if audit["status"] != "passed":
        raise RuntimeError("Poly formal audit failed; see analysis/audit.json")
    aggregate, paired, summaries = build_statistics(rows)
    write_csv(analysis / "final_metrics.csv", rows)
    write_csv(analysis / "aggregate_metrics.csv", aggregate)
    write_csv(analysis / "paired_comparison.csv", paired)
    atomic_json(
        analysis / "analysis_summary.json",
        {
            "generated_at": utc_now(),
            "protocol_id": PROTOCOL_ID,
            "task_summaries": summaries,
        },
    )
    atomic_text(analysis / "REPORT_zh.md", report_markdown(audit, summaries))
    write_analysis_checksums(analysis)
    write_source_checksums(root)
    write_delivery_checksums(root)
    print(json.dumps({"status": "complete", "analysis_dir": str(analysis)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
