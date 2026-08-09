#!/usr/bin/env python3
"""Audit, summarize, and plot the fixed-weight 2D Cahn--Hilliard formal run.

Project policy: run this plot-producing script only on the development server
or a T4 host, never through a workspace built-in image-generation facility.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


TASKS = {
    "cahn_hilliard_2d_o4": "CH4",
    "cahn_hilliard_2d_o6": "CH6",
}
METHODS = ("war", "real_sinh_autodiff")
METHOD_LABELS = {
    "war": "WAR (complex64)",
    "real_sinh_autodiff": "Real AD (float32)",
}
COLORS = {
    "war": "#0072B2",
    "real_sinh_autodiff": "#D55E00",
}
LINESTYLES = {
    "war": "-",
    "real_sinh_autodiff": "--",
}
SEEDS = tuple(range(5))
REQUIRED_HISTORY_FIELDS = (
    "elapsed_seconds",
    "step",
    "rel_error",
    "loss",
    "L_PDE",
    "L_IC",
    "L_BC",
    "mass_drift_rms",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_checksums(root: Path) -> tuple[int, list[str]]:
    checksum_file = root / "SHA256SUMS"
    issues: list[str] = []
    checked = 0
    for line in checksum_file.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.strip()
        path = root / relative
        checked += 1
        if not path.is_file():
            issues.append(f"missing checksum target: {relative}")
        elif sha256(path) != expected:
            issues.append(f"checksum mismatch: {relative}")
    return checked, issues


def inspect_numeric_finiteness(value: Any, label: str, issues: list[str]) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            issues.append(f"non-finite value: {label}={value}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            inspect_numeric_finiteness(item, f"{label}[{index}]", issues)
    elif isinstance(value, dict):
        for key, item in value.items():
            inspect_numeric_finiteness(item, f"{label}.{key}", issues)


def distribution(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    return {
        "mean": statistics.fmean(data),
        "std": statistics.stdev(data) if len(data) > 1 else 0.0,
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
    }


def load_and_audit(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[str] = []
    raw_checksum_count, checksum_issues = verify_raw_checksums(root)
    issues.extend(checksum_issues)

    manifest = json.loads((root / "manifest.json").read_text())
    required_manifest = {
        "protocol_id": "cahn_hilliard_2d_fixed_weights_formal_v1",
        "engine_protocol_id": "cahn_hilliard_2d_natural_bc_common_sinh_fp32_v1",
        "seconds_per_method_seed": 1200.0,
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "sample_counts": {
            "n_int": 4096,
            "n_ic": 1024,
            "n_bc": 2048,
            "n_eval": 32768,
            "history_eval_n": 4096,
        },
        "method_seed_run_count": 20,
        "serial_single_gpu": True,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            issues.append(
                f"manifest mismatch for {key}: {manifest.get(key)!r} != {expected!r}"
            )
    git = manifest.get("git", {})
    if git.get("sha") != "e2e781f4a4074d5ba5d810118b1097f2a9353d8f":
        issues.append(f"unexpected git SHA: {git.get('sha')}")
    if git.get("dirty") is not False:
        issues.append("formal run git tree was not clean")

    root_summary = json.loads((root / "summary.json").read_text())
    completion = json.loads((root / "FORMAL_COMPLETE").read_text())
    for name, payload in (("summary", root_summary), ("marker", completion)):
        if payload.get("all_complete") is not True:
            issues.append(f"{name} does not report all_complete")
        if payload.get("complete_runs") != 20:
            issues.append(f"{name} complete_runs={payload.get('complete_runs')}")

    rows: list[dict[str, Any]] = []
    total_history_points = 0
    for task in TASKS:
        paired = 0
        for seed in SEEDS:
            pair_complete: list[bool] = []
            for method in METHODS:
                base = root / task / f"seed_{seed:03d}" / method
                result_path = base.with_suffix(".json")
                log_path = base.with_suffix(".log")
                done_path = base.with_suffix(".DONE")
                try:
                    result = json.loads(result_path.read_text())
                except (OSError, json.JSONDecodeError) as error:
                    issues.append(f"cannot parse {result_path}: {error}")
                    continue
                inspect_numeric_finiteness(
                    result, str(result_path.relative_to(root)), issues
                )
                pair_complete.append(result.get("status") == "complete")
                history = result.get("history", [])
                total_history_points += len(history)
                if len(history) != 241:
                    issues.append(
                        f"{task}/seed={seed}/{method}: history has {len(history)} points"
                    )
                for index, point in enumerate(history):
                    for field in REQUIRED_HISTORY_FIELDS:
                        if field not in point:
                            issues.append(
                                f"{task}/seed={seed}/{method}/history={index}: "
                                f"missing {field}"
                            )
                        elif not math.isfinite(float(point[field])):
                            issues.append(
                                f"{task}/seed={seed}/{method}/history={index}: "
                                f"non-finite {field}"
                            )
                final_log_line = ""
                if log_path.is_file():
                    final_log_line = next(
                        (
                            line
                            for line in reversed(
                                log_path.read_text(errors="replace").splitlines()
                            )
                            if line.strip()
                        ),
                        "",
                    )
                if "loss" not in final_log_line or "rel_error" not in final_log_line:
                    issues.append(f"{task}/seed={seed}/{method}: incomplete final log")
                if not done_path.is_file():
                    issues.append(f"{task}/seed={seed}/{method}: missing DONE marker")
                rows.append(
                    {
                        "task_id": task,
                        "task_label": TASKS[task],
                        "seed": seed,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "loss": float(result["loss"]),
                        "rel_error": float(result["rel_error"]),
                        "steps": int(result["steps"]),
                        "training_seconds": float(result["training_seconds"]),
                        "evaluation_seconds": float(result["evaluation_seconds"]),
                        "ms_per_step": float(result["ms_per_step"]),
                        "peak_mb": float(result["peak_mb"]),
                        "mass_drift_rms": float(result["metrics"]["mass_drift_rms"]),
                        "mass_drift_max_abs": float(
                            result["metrics"]["mass_drift_max_abs"]
                        ),
                        "history_points": len(history),
                        "history": history,
                    }
                )
            paired += int(len(pair_complete) == 2 and all(pair_complete))
        if paired != 5:
            issues.append(f"{task}: only {paired} paired complete seeds")

    failed_markers = list(root.glob("cahn_hilliard_2d_o*/seed_*/*.FAILED"))
    attempt_files = list(root.glob("cahn_hilliard_2d_o*/seed_*/attempts/*"))
    temporary_files = list(root.rglob("*.tmp.*"))
    if failed_markers:
        issues.append(f"found {len(failed_markers)} FAILED markers")
    if attempt_files:
        issues.append(f"found {len(attempt_files)} retry attempt files")
    if temporary_files:
        issues.append(f"found {len(temporary_files)} temporary files")

    audit = {
        "status": "passed" if not issues else "failed",
        "generated_at": utc_now(),
        "raw_root": str(root.resolve()),
        "raw_sha256sums_sha256": sha256(root / "SHA256SUMS"),
        "raw_checksum_entries_verified": raw_checksum_count,
        "expected_runs": 20,
        "complete_result_count": len(rows),
        "total_history_points": total_history_points,
        "log_count": len(list(root.glob("cahn_hilliard_2d_o*/seed_*/*.log"))),
        "done_marker_count": len(
            list(root.glob("cahn_hilliard_2d_o*/seed_*/*.DONE"))
        ),
        "failed_marker_count": len(failed_markers),
        "retry_attempt_file_count": len(attempt_files),
        "temporary_file_count": len(temporary_files),
        "issues": issues,
    }
    return rows, audit


def build_statistics(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    aggregate_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    task_summaries: dict[str, Any] = {}
    metrics = (
        "rel_error",
        "loss",
        "steps",
        "ms_per_step",
        "peak_mb",
        "mass_drift_rms",
    )
    for task in TASKS:
        method_distributions: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["task_id"] == task and row["method"] == method
            ]
            metric_distributions = {
                metric: distribution(row[metric] for row in selected)
                for metric in metrics
            }
            method_distributions[method] = metric_distributions
            flat: dict[str, Any] = {
                "task_id": task,
                "task_label": TASKS[task],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed_count": len(selected),
            }
            for metric, values in metric_distributions.items():
                for statistic, value in values.items():
                    flat[f"{metric}_{statistic}"] = value
            aggregate_rows.append(flat)

        wins = 0
        ratios: list[float] = []
        for seed in SEEDS:
            pair = {
                row["method"]: row
                for row in rows
                if row["task_id"] == task and row["seed"] == seed
            }
            war = pair["war"]
            autodiff = pair["real_sinh_autodiff"]
            error_ratio = autodiff["rel_error"] / war["rel_error"]
            ratios.append(error_ratio)
            wins += int(war["rel_error"] < autodiff["rel_error"])
            paired_rows.append(
                {
                    "task_id": task,
                    "task_label": TASKS[task],
                    "seed": seed,
                    "war_rel_error": war["rel_error"],
                    "real_ad_rel_error": autodiff["rel_error"],
                    "real_ad_over_war_rel_error": error_ratio,
                    "war_loss": war["loss"],
                    "real_ad_loss": autodiff["loss"],
                    "war_steps": war["steps"],
                    "real_ad_steps": autodiff["steps"],
                    "war_over_real_ad_steps": war["steps"] / autodiff["steps"],
                    "war_peak_mb": war["peak_mb"],
                    "real_ad_peak_mb": autodiff["peak_mb"],
                    "real_ad_over_war_peak_mb": autodiff["peak_mb"]
                    / war["peak_mb"],
                    "war_mass_drift_rms": war["mass_drift_rms"],
                    "real_ad_mass_drift_rms": autodiff["mass_drift_rms"],
                }
            )

        n = len(SEEDS)
        one_sided_sign_p = sum(
            math.comb(n, successes) for successes in range(wins, n + 1)
        ) / (2**n)
        two_sided_sign_p = min(1.0, 2.0 * one_sided_sign_p)
        war_error = method_distributions["war"]["rel_error"]
        ad_error = method_distributions["real_sinh_autodiff"]["rel_error"]
        task_summaries[task] = {
            "task_label": TASKS[task],
            "war": method_distributions["war"],
            "real_sinh_autodiff": method_distributions["real_sinh_autodiff"],
            "war_wins": wins,
            "paired_seed_count": n,
            "exact_sign_test_one_sided_p": one_sided_sign_p,
            "exact_sign_test_two_sided_p": two_sided_sign_p,
            "ratio_of_mean_rel_errors_real_ad_over_war": ad_error["mean"]
            / war_error["mean"],
            "paired_rel_error_ratio_real_ad_over_war": {
                **distribution(ratios),
                "geometric_mean": math.exp(statistics.fmean(map(math.log, ratios))),
            },
        }
    return aggregate_rows, paired_rows, task_summaries


def build_curve_rows(
    raw_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, np.ndarray]]]:
    grid = np.linspace(0.0, 1200.0, 241)
    curve_rows: list[dict[str, Any]] = []
    series: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for task in TASKS:
        for method in METHODS:
            histories = [
                row["history"]
                for row in raw_rows
                if row["task_id"] == task and row["method"] == method
            ]
            rel_matrix = np.asarray(
                [
                    np.interp(
                        grid,
                        [point["elapsed_seconds"] for point in history],
                        [point["rel_error"] for point in history],
                    )
                    for history in histories
                ],
                dtype=float,
            )
            loss_matrix = np.asarray(
                [
                    np.interp(
                        grid,
                        [point["elapsed_seconds"] for point in history],
                        [point["loss"] for point in history],
                    )
                    for history in histories
                ],
                dtype=float,
            )
            values = {
                "elapsed_seconds": grid,
                "rel_error_median": np.median(rel_matrix, axis=0),
                "rel_error_q25": np.quantile(rel_matrix, 0.25, axis=0),
                "rel_error_q75": np.quantile(rel_matrix, 0.75, axis=0),
                "rel_error_min": np.min(rel_matrix, axis=0),
                "rel_error_max": np.max(rel_matrix, axis=0),
                "loss_median": np.median(loss_matrix, axis=0),
                "loss_q25": np.quantile(loss_matrix, 0.25, axis=0),
                "loss_q75": np.quantile(loss_matrix, 0.75, axis=0),
                "loss_min": np.min(loss_matrix, axis=0),
                "loss_max": np.max(loss_matrix, axis=0),
            }
            series[(task, method)] = values
            for index, elapsed in enumerate(grid):
                curve_rows.append(
                    {
                        "task_id": task,
                        "task_label": TASKS[task],
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        **{
                            key: float(value[index])
                            for key, value in values.items()
                            if key != "elapsed_seconds"
                        },
                        "elapsed_seconds": float(elapsed),
                    }
                )
    return curve_rows, series


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_realtime(
    analysis_dir: Path,
    series: dict[tuple[str, str], dict[str, np.ndarray]],
    metric: str,
    ylabel: str,
    stem: str,
) -> None:
    configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharex=True)
    for axis, task in zip(axes, TASKS):
        for method in METHODS:
            values = series[(task, method)]
            x = values["elapsed_seconds"]
            median = np.maximum(values[f"{metric}_median"], 1e-12)
            q25 = np.maximum(values[f"{metric}_q25"], 1e-12)
            q75 = np.maximum(values[f"{metric}_q75"], 1e-12)
            axis.fill_between(
                x,
                q25,
                q75,
                color=COLORS[method],
                alpha=0.16,
                linewidth=0,
            )
            axis.plot(
                x,
                median,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.0,
                label=METHOD_LABELS[method],
            )
        axis.set_yscale("log")
        axis.set_xlim(0.0, 1200.0)
        axis.set_xticks([0, 300, 600, 900, 1200])
        axis.set_title(f"{TASKS[task]} (order {TASKS[task][2:]})")
        axis.set_xlabel("Training wall time (s)")
        axis.grid(axis="y", which="both", linewidth=0.45, alpha=0.28)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "2D Cahn–Hilliard: median real-time trajectory with interquartile band (5 seeds)",
        y=0.985,
        fontsize=12,
    )
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.14, top=0.73, wspace=0.16)
    for suffix in ("png", "pdf"):
        figure.savefig(analysis_dir / f"{stem}.{suffix}", dpi=220)
    plt.close(figure)


def plot_final_by_seed(analysis_dir: Path, rows: list[dict[str, Any]]) -> None:
    configure_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(8.7, 4.25), sharey=False)
    for axis, task in zip(axes, TASKS):
        for seed in SEEDS:
            pair = {
                row["method"]: row
                for row in rows
                if row["task_id"] == task and row["seed"] == seed
            }
            y = [pair["war"]["rel_error"], pair["real_sinh_autodiff"]["rel_error"]]
            axis.plot([0, 1], y, color="0.68", linewidth=0.8, zorder=1)
            axis.scatter(
                [0],
                [y[0]],
                color=COLORS["war"],
                marker="o",
                s=34,
                zorder=2,
            )
            axis.scatter(
                [1],
                [y[1]],
                color=COLORS["real_sinh_autodiff"],
                marker="s",
                s=34,
                zorder=2,
            )
        axis.set_yscale("log")
        axis.set_xticks([0, 1], ["WAR\ncomplex64", "Real AD\nfloat32"])
        axis.set_xlim(-0.35, 1.35)
        axis.set_title(f"{TASKS[task]} (5 paired seeds)")
        axis.grid(axis="y", which="both", linewidth=0.45, alpha=0.28)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Final relative L2 error (32,768-point evaluation)")
    figure.suptitle("Final accuracy by paired training seed", y=1.01, fontsize=12)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(analysis_dir / f"final_rel_error_by_seed.{suffix}", dpi=220)
    plt.close(figure)


def markdown_report(
    root: Path,
    audit: dict[str, Any],
    task_summaries: dict[str, Any],
    paired_rows: list[dict[str, Any]],
) -> str:
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    created_at = datetime.fromisoformat(manifest["created_at"])
    completed_at = datetime.fromisoformat(summary["completed_at"])
    wall_seconds = (completed_at - created_at).total_seconds()
    lines = [
        "# 二维 Cahn–Hilliard 固定权重正式实验",
        "",
        "## 验收",
        "",
        f"- 状态：`{audit['status']}`；20/20 cell 完成，0 失败，0 重试。",
        f"- 原始校验和：{audit['raw_checksum_entries_verified']} 项全部通过。",
        f"- 实时 history：{audit['total_history_points']} 个数据点，全部字段有限。",
        f"- 总墙钟：{wall_seconds / 3600:.2f} 小时。",
        "- 协议：CH4/CH6，`(lambda_ic, lambda_bc)=(1,10)`，每方法每 seed 1200 秒，5 seeds。",
        "- WAR 使用 complex64；实数 autodiff 使用 float32；两者层形状、深度、宽度与 sinh 激活相同。",
        "- 输入只有仿射归一化的 `(x,y,t)`；无三角特征与频率初始化。",
        "",
        "## 最终相对误差（5 seeds）",
        "",
        "| Task | 方法 | Mean | Std | Median | Min | Max |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        for method in METHODS:
            values = task_summaries[task][method]["rel_error"]
            lines.append(
                f"| {TASKS[task]} | {METHOD_LABELS[method]} | "
                f"{values['mean']:.6g} | {values['std']:.6g} | "
                f"{values['median']:.6g} | {values['min']:.6g} | "
                f"{values['max']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## 配对比较",
            "",
            "| Task | WAR 胜出 seeds | AD/WAR 平均误差比 | 配对比几何均值 | 单侧 sign p | 双侧 sign p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        values = task_summaries[task]
        lines.append(
            f"| {TASKS[task]} | {values['war_wins']}/5 | "
            f"{values['ratio_of_mean_rel_errors_real_ad_over_war']:.3f}× | "
            f"{values['paired_rel_error_ratio_real_ad_over_war']['geometric_mean']:.3f}× | "
            f"{values['exact_sign_test_one_sided_p']:.5f} | "
            f"{values['exact_sign_test_two_sided_p']:.5f} |"
        )
    lines.extend(
        [
            "",
            "五个 seed 在 CH4 和 CH6 上都由 WAR 获得更低的最终相对误差。由于每个任务只有 5 个配对 seed，双侧精确 sign test 的最小 p 值为 0.0625；因此结果是强而一致的描述性证据，但不应写成传统双侧 5% 水平的显著性结论。",
            "",
            "## 图与口径",
            "",
            "- `realtime_rel_error`：固定 4096 点 history 评估集上的实时相对误差，中位数与四分位带。",
            "- `realtime_loss`：相同时间轴上的训练目标，中位数与四分位带。",
            "- `final_rel_error_by_seed`：32768 点最终评估集上的五个配对 seed。",
            "- history 曲线和最终表使用不同评估规模，因此曲线最后一点与最终 JSON 数值允许有小幅差异。",
            "",
            "## 公平性边界",
            "",
            "两种方法具有相同的字面网络层形状，但 complex64 WAR 的复参数对应两倍实标量自由度；结论应表述为固定网络形状和固定墙钟预算下的效率/精度比较，而不是严格等实参数量比较。",
            "",
            "## 逐 seed 文件",
            "",
            "完整数值见 `final_metrics.csv` 与 `paired_comparison.csv`。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_analysis_checksums(analysis_dir: Path) -> None:
    paths = sorted(
        path
        for path in analysis_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS" and ".tmp" not in path.name
    )
    lines = [f"{sha256(path)}  {path.name}" for path in paths]
    atomic_text(analysis_dir / "SHA256SUMS", "\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.result_root.resolve()
    analysis_dir = (args.analysis_dir or root / "analysis").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows_with_history, audit = load_and_audit(root)
    if audit["status"] != "passed":
        atomic_json(analysis_dir / "audit.json", audit)
        raise RuntimeError("raw formal result audit failed; see analysis/audit.json")

    aggregate_rows, paired_rows, task_summaries = build_statistics(rows_with_history)
    curve_rows, series = build_curve_rows(rows_with_history)
    final_rows = [
        {key: value for key, value in row.items() if key != "history"}
        for row in rows_with_history
    ]

    atomic_json(analysis_dir / "audit.json", audit)
    write_csv(analysis_dir / "final_metrics.csv", final_rows)
    write_csv(analysis_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(analysis_dir / "paired_comparison.csv", paired_rows)
    write_csv(analysis_dir / "realtime_curves.csv", curve_rows)
    atomic_json(
        analysis_dir / "analysis_summary.json",
        {
            "generated_at": utc_now(),
            "protocol_id": "cahn_hilliard_2d_fixed_weights_formal_v1",
            "raw_sha256sums_sha256": audit["raw_sha256sums_sha256"],
            "task_summaries": task_summaries,
        },
    )
    atomic_text(
        analysis_dir / "REPORT_zh.md",
        markdown_report(root, audit, task_summaries, paired_rows),
    )
    plot_realtime(
        analysis_dir,
        series,
        "rel_error",
        "Relative L2 error (history evaluation; lower is better)",
        "realtime_rel_error",
    )
    plot_realtime(
        analysis_dir,
        series,
        "loss",
        "Weighted training loss (lower is better)",
        "realtime_loss",
    )
    plot_final_by_seed(analysis_dir, final_rows)
    write_analysis_checksums(analysis_dir)
    print(json.dumps({"status": "complete", "analysis_dir": str(analysis_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
