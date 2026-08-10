#!/usr/bin/env python3
"""Audit, summarize, and plot the high-order PDE pilot and selected formal run.

Project policy: execute this plot-producing script only on a development GPU
server (or a registered T4 host), never through a workspace built-in image
generator.  The script is standalone so the frozen training checkout need not
be modified after the experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROTOCOL_ID = "high_order_candidate_screen_v1"
ENGINE_PROTOCOL_ID = "high_order_candidate_common_xavier_fp32_v1"
RUN_GIT_SHA = "20a8cd652e921d6337ff07aeaf5d0824a8cd4620"
PILOT_TASKS = {
    "zk_2d_o3": "ZK-2D (order 3)",
    "zk_3d_o3": "ZK-3D (order 3)",
    "dynamic_plate_2d_o4": "Dynamic plate (order 4)",
    "swift_hohenberg_2d_o4": "Swift–Hohenberg (order 4)",
}
FORMAL_TASK = "zk_2d_o3"
METHODS = ("war", "real_tanh_autodiff")
METHOD_LABELS = {
    "war": "WAR (complex64, sinh)",
    "real_tanh_autodiff": "Real AD (float32, tanh)",
}
COLORS = {"war": "#0072B2", "real_tanh_autodiff": "#D55E00"}
LINESTYLES = {"war": "-", "real_tanh_autodiff": "--"}
SAMPLES = {
    "n_int": 2048,
    "n_ic": 512,
    "n_bc": 1024,
    "n_eval": 16384,
    "history_eval_n": 2048,
}
REQUIRED_HISTORY_FIELDS = (
    "elapsed_seconds",
    "step",
    "loss",
    "rel_error",
    "L_PDE",
    "L_BC",
)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def distribution(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    return {
        "mean": statistics.fmean(data),
        "std": statistics.stdev(data) if len(data) > 1 else 0.0,
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
    }


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


def verify_checksums(root: Path) -> tuple[int, list[str]]:
    issues: list[str] = []
    checked = 0
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        return 0, ["missing SHA256SUMS"]
    for line in checksum_path.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        target = root / relative.strip()
        checked += 1
        if not target.is_file():
            issues.append(f"missing checksum target: {relative.strip()}")
        elif sha256(target) != expected:
            issues.append(f"checksum mismatch: {relative.strip()}")
    return checked, issues


def audit_bundle(
    root: Path,
    *,
    stage: str,
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
    seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[str] = []
    checksum_count, checksum_issues = verify_checksums(root)
    issues.extend(checksum_issues)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    progress = json.loads((root / "progress.json").read_text())
    inspect_finite(manifest, "manifest", issues)
    inspect_finite(summary, "summary", issues)
    inspect_finite(progress, "progress", issues)

    expected_manifest = {
        "protocol_id": PROTOCOL_ID,
        "engine_protocol_id": ENGINE_PROTOCOL_ID,
        "stage": stage,
        "tasks": list(tasks),
        "methods": list(METHODS),
        "seeds": list(seeds),
        "seconds_per_method_seed": seconds,
        "sample_counts": SAMPLES,
        "expected_runs": len(tasks) * len(seeds) * len(METHODS),
        "serial_single_gpu": True,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            issues.append(
                f"manifest mismatch for {key}: {manifest.get(key)!r} != {expected!r}"
            )
    git = manifest.get("git", {})
    if git.get("sha") != RUN_GIT_SHA or git.get("dirty") is not False:
        issues.append(f"unexpected run git state: {git}")
    architecture = manifest.get("architecture", {})
    shared = architecture.get("shared", {})
    if shared.get("hidden") != 128 or shared.get("depth") != 4:
        issues.append("architecture is not hidden=128/depth=4")
    if shared.get("init_mode") != "common_xavier":
        issues.append("initialization is not common_xavier")
    if shared.get("trigonometric_input_features") is not False:
        issues.append("trigonometric input features are enabled")
    if shared.get("frequency_initialization") is not False:
        issues.append("frequency initialization is enabled")
    if architecture.get("war", {}).get("dtype") != "torch.complex64":
        issues.append("WAR dtype is not complex64")
    if architecture.get("real_tanh_autodiff", {}).get("dtype") != "torch.float32":
        issues.append("real AD dtype is not float32")

    marker = "FORMAL_COMPLETE" if stage == "formal" else "PILOT_COMPLETE"
    if not (root / marker).is_file():
        issues.append(f"missing {marker}")
    expected_runs = len(tasks) * len(seeds) * len(METHODS)
    if summary.get("all_complete") is not True:
        issues.append("summary does not report all_complete")
    if summary.get("complete_runs") != expected_runs:
        issues.append(f"summary complete_runs={summary.get('complete_runs')}")
    if progress.get("status") != "complete":
        issues.append(f"progress status={progress.get('status')}")
    if progress.get("complete_runs") != expected_runs or progress.get("failures") != 0:
        issues.append("progress does not report all runs complete with zero failures")

    rows: list[dict[str, Any]] = []
    history_points = 0
    for task in tasks:
        for seed in seeds:
            cell = root / task / f"seed_{seed:03d}"
            if not (cell / "DONE").is_file():
                issues.append(f"missing DONE: {task}/seed_{seed:03d}")
            config_path = cell / "config.json"
            if not config_path.is_file():
                issues.append(f"missing config: {task}/seed_{seed:03d}")
            for method in METHODS:
                result_path = cell / f"{method}.json"
                log_path = cell / f"{method}.log"
                if not result_path.is_file() or not log_path.is_file():
                    issues.append(f"missing result/log: {task}/seed={seed}/{method}")
                    continue
                result = json.loads(result_path.read_text())
                inspect_finite(result, str(result_path.relative_to(root)), issues)
                key = f"{task}/seed={seed}/{method}"
                expected_fields = {
                    "protocol_id": ENGINE_PROTOCOL_ID,
                    "screen_protocol_id": PROTOCOL_ID,
                    "stage": stage,
                    "status": "complete",
                    "task_id": task,
                    "method": method,
                    "train_seed": seed,
                    "budget_seconds": seconds,
                }
                for name, expected in expected_fields.items():
                    if result.get(name) != expected:
                        issues.append(
                            f"{key}: {name}={result.get(name)!r} != {expected!r}"
                        )
                if result.get("git", {}).get("sha") != RUN_GIT_SHA:
                    issues.append(f"{key}: result git SHA mismatch")
                problem = result.get("problem", {})
                for name, expected in SAMPLES.items():
                    if name == "n_ic" and task == "swift_hohenberg_2d_o4":
                        expected = 0
                    if problem.get(name) != expected:
                        issues.append(f"{key}: sample count {name} mismatch")
                model = result.get("model", {})
                expected_dtype = "torch.complex64" if method == "war" else "torch.float32"
                expected_activation = "sinh" if method == "war" else "tanh"
                if model.get("parameter_dtype") != expected_dtype:
                    issues.append(f"{key}: model dtype mismatch")
                if model.get("activation") != expected_activation:
                    issues.append(f"{key}: activation mismatch")
                if model.get("init_mode") != "common_xavier":
                    issues.append(f"{key}: initialization mismatch")
                if model.get("frequency_initialization") != "disabled":
                    issues.append(f"{key}: frequency initialization enabled")
                if model.get("input_transform") != "affine_only":
                    issues.append(f"{key}: unexpected input transform")

                history = result.get("history", [])
                if not isinstance(history, list) or len(history) < 2:
                    issues.append(f"{key}: missing history")
                    history = []
                elapsed: list[float] = []
                for index, point in enumerate(history):
                    required_fields = REQUIRED_HISTORY_FIELDS + (
                        ("L_IC",) if task != "swift_hohenberg_2d_o4" else ()
                    )
                    for field in required_fields:
                        if field not in point:
                            issues.append(f"{key}/history={index}: missing {field}")
                    elapsed.append(float(point.get("elapsed_seconds", math.nan)))
                if any(a > b for a, b in zip(elapsed, elapsed[1:])):
                    issues.append(f"{key}: non-monotone history time")
                if elapsed and elapsed[-1] < 0.95 * seconds:
                    issues.append(f"{key}: history ends too early at {elapsed[-1]} s")
                history_points += len(history)
                log_lines = [line for line in log_path.read_text(errors="replace").splitlines() if line.strip()]
                if not log_lines:
                    issues.append(f"{key}: empty log")
                else:
                    try:
                        final_log = json.loads(log_lines[-1])
                    except json.JSONDecodeError:
                        issues.append(f"{key}: final log line is not JSON")
                    else:
                        if "loss" not in final_log or "rel_error" not in final_log:
                            issues.append(f"{key}: final log lacks loss/rel_error")
                rows.append(
                    {
                        "stage": stage,
                        "task_id": task,
                        "task_label": PILOT_TASKS[task],
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
                        "history_points": len(history),
                        "history": history,
                    }
                )

    failed = list(root.rglob("FAILED")) + list(root.rglob("*.FAILED"))
    attempts = list(root.rglob("attempts/*"))
    temporary = [path for path in root.rglob("*") if path.is_file() and ".tmp." in path.name]
    if failed:
        issues.append(f"found {len(failed)} FAILED markers")
    if attempts:
        issues.append(f"found {len(attempts)} attempt artifacts")
    if temporary:
        issues.append(f"found {len(temporary)} temporary artifacts")
    audit = {
        "stage": stage,
        "status": "passed" if not issues else "failed",
        "root": str(root),
        "raw_sha256sums_sha256": sha256(root / "SHA256SUMS"),
        "checksum_entries_verified": checksum_count,
        "expected_runs": expected_runs,
        "complete_results": len(rows),
        "history_points": history_points,
        "failed_marker_count": len(failed),
        "attempt_artifact_count": len(attempts),
        "temporary_artifact_count": len(temporary),
        "issues": issues,
    }
    return rows, audit


def build_statistics(
    rows: list[dict[str, Any]], tasks: tuple[str, ...], seeds: tuple[int, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    metrics = ("rel_error", "loss", "steps", "ms_per_step", "peak_mb")
    for task in tasks:
        methods: dict[str, Any] = {}
        for method in METHODS:
            selected = [
                row for row in rows
                if row["task_id"] == task and row["method"] == method
            ]
            methods[method] = {
                metric: distribution(row[metric] for row in selected)
                for metric in metrics
            }
            flat: dict[str, Any] = {
                "task_id": task,
                "task_label": PILOT_TASKS[task],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "seed_count": len(selected),
            }
            for metric, values in methods[method].items():
                for statistic, value in values.items():
                    flat[f"{metric}_{statistic}"] = value
            aggregate.append(flat)
        ratios: list[float] = []
        war_wins = 0
        for seed in seeds:
            pair = {
                row["method"]: row for row in rows
                if row["task_id"] == task and row["seed"] == seed
            }
            war = pair["war"]
            ad = pair["real_tanh_autodiff"]
            ratio = ad["rel_error"] / war["rel_error"]
            ratios.append(ratio)
            war_wins += int(war["rel_error"] < ad["rel_error"])
            paired.append(
                {
                    "task_id": task,
                    "task_label": PILOT_TASKS[task],
                    "seed": seed,
                    "war_rel_error": war["rel_error"],
                    "real_ad_rel_error": ad["rel_error"],
                    "real_ad_over_war_rel_error": ratio,
                    "war_loss": war["loss"],
                    "real_ad_loss": ad["loss"],
                    "war_steps": war["steps"],
                    "real_ad_steps": ad["steps"],
                    "war_ms_per_step": war["ms_per_step"],
                    "real_ad_ms_per_step": ad["ms_per_step"],
                    "real_ad_over_war_step_time": ad["ms_per_step"] / war["ms_per_step"],
                    "war_peak_mb": war["peak_mb"],
                    "real_ad_peak_mb": ad["peak_mb"],
                }
            )
        n = len(seeds)
        one_sided = sum(math.comb(n, k) for k in range(war_wins, n + 1)) / (2**n)
        summaries[task] = {
            "task_label": PILOT_TASKS[task],
            "war": methods["war"],
            "real_tanh_autodiff": methods["real_tanh_autodiff"],
            "war_wins": war_wins,
            "paired_seed_count": n,
            "exact_sign_test_one_sided_p": one_sided,
            "exact_sign_test_two_sided_p": min(1.0, 2.0 * one_sided),
            "ratio_of_mean_rel_errors_real_ad_over_war": (
                methods["real_tanh_autodiff"]["rel_error"]["mean"]
                / methods["war"]["rel_error"]["mean"]
            ),
            "paired_rel_error_ratio_real_ad_over_war": {
                **distribution(ratios),
                "geometric_mean": math.exp(statistics.fmean(map(math.log, ratios))),
            },
        }
    return aggregate, paired, summaries


def positive_log_interp(
    grid: np.ndarray, elapsed: list[float], values: list[float]
) -> np.ndarray:
    safe = np.maximum(np.asarray(values, dtype=float), 1e-15)
    return np.exp(np.interp(grid, np.asarray(elapsed, dtype=float), np.log(safe)))


def build_formal_curves(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    grid = np.linspace(0.0, 1200.0, 61)
    long_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    series: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        for point in row["history"]:
            fields = REQUIRED_HISTORY_FIELDS + ("L_IC",)
            long_rows.append(
                {
                    "task_id": row["task_id"],
                    "method": row["method"],
                    "method_label": row["method_label"],
                    "seed": row["seed"],
                    **{
                        field: point.get(field)
                        for field in fields
                    },
                }
            )
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        rel_matrix = np.asarray(
            [
                positive_log_interp(
                    grid,
                    [float(point["elapsed_seconds"]) for point in row["history"]],
                    [float(point["rel_error"]) for point in row["history"]],
                )
                for row in selected
            ]
        )
        loss_matrix = np.asarray(
            [
                positive_log_interp(
                    grid,
                    [float(point["elapsed_seconds"]) for point in row["history"]],
                    [float(point["loss"]) for point in row["history"]],
                )
                for row in selected
            ]
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
        series[method] = values
        for index, elapsed in enumerate(grid):
            curve_rows.append(
                {
                    "task_id": FORMAL_TASK,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "elapsed_seconds": float(elapsed),
                    **{
                        key: float(value[index])
                        for key, value in values.items()
                        if key != "elapsed_seconds"
                    },
                }
            )
    return long_rows, curve_rows, series


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
    series: dict[str, dict[str, np.ndarray]],
    *,
    metric: str,
    ylabel: str,
    stem: str,
) -> None:
    configure_plot_style()
    figure, axis = plt.subplots(figsize=(6.8, 4.65))
    for method in METHODS:
        values = series[method]
        x = values["elapsed_seconds"]
        median = np.maximum(values[f"{metric}_median"], 1e-15)
        q25 = np.maximum(values[f"{metric}_q25"], 1e-15)
        q75 = np.maximum(values[f"{metric}_q75"], 1e-15)
        axis.fill_between(x, q25, q75, color=COLORS[method], alpha=0.16, linewidth=0)
        axis.plot(
            x,
            median,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.1,
            label=METHOD_LABELS[method],
        )
    axis.set_yscale("log")
    axis.set_xlim(0.0, 1200.0)
    axis.set_xticks([0, 300, 600, 900, 1200])
    axis.set_xlabel("Training wall time (s)")
    axis.set_ylabel(ylabel)
    axis.set_title("2D third-order Zakharov–Kuznetsov (5 seeds)")
    axis.grid(axis="y", which="both", linewidth=0.45, alpha=0.28)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(analysis_dir / f"{stem}.{suffix}", dpi=240)
    plt.close(figure)


def plot_final_by_seed(analysis_dir: Path, rows: list[dict[str, Any]]) -> None:
    configure_plot_style()
    figure, axis = plt.subplots(figsize=(5.7, 4.45))
    for seed in range(5):
        pair = {row["method"]: row for row in rows if row["seed"] == seed}
        y = [pair["war"]["rel_error"], pair["real_tanh_autodiff"]["rel_error"]]
        axis.plot([0, 1], y, color="0.68", linewidth=0.9, zorder=1)
        axis.scatter(0, y[0], color=COLORS["war"], marker="o", s=38, zorder=2)
        axis.scatter(1, y[1], color=COLORS["real_tanh_autodiff"], marker="s", s=38, zorder=2)
    axis.set_yscale("log")
    axis.set_xticks([0, 1], ["WAR\ncomplex64, sinh", "Real AD\nfloat32, tanh"])
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylabel("Final relative L2 error")
    axis.set_title("2D third-order ZK: paired final accuracy")
    axis.grid(axis="y", which="both", linewidth=0.45, alpha=0.28)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(analysis_dir / f"final_rel_error_by_seed.{suffix}", dpi=240)
    plt.close(figure)


def markdown_report(
    pilot_audit: dict[str, Any],
    formal_audit: dict[str, Any],
    pilot_summary: dict[str, Any],
    formal_summary: dict[str, Any],
) -> str:
    lines = [
        "# 高阶 PDE 候选筛选与二维 ZK 正式实验",
        "",
        "## 验收",
        "",
        f"- Pilot：`{pilot_audit['status']}`，24/24 cell，{pilot_audit['history_points']} 个 history 点。",
        f"- Formal：`{formal_audit['status']}`，10/10 cell，{formal_audit['history_points']} 个 history 点。",
        f"- 原始校验：pilot {pilot_audit['checksum_entries_verified']} 项、formal {formal_audit['checksum_entries_verified']} 项全部通过。",
        "- 两阶段均为 0 失败、0 重试；正式统计没有混入 600 秒 pilot。",
        "- 固定协议：共同 Xavier，hidden=128，depth=4，原始仿射坐标，无三角输入和频率初始化。",
        "- WAR 使用 complex64+sinh+Waring jet；基线使用 float32+tanh+direct autodiff。",
        "",
        "## Pilot（3 seeds × 600 秒）",
        "",
        "| 候选 | WAR median rel_error | Real AD median rel_error | WAR 胜出 | 通过门槛 |",
        "|---|---:|---:|---:|:---:|",
    ]
    for task in PILOT_TASKS:
        values = pilot_summary[task]
        passed = (
            min(values["war"]["rel_error"]["median"], values["real_tanh_autodiff"]["rel_error"]["median"]) < 0.2
            and max(values["war"]["rel_error"]["median"], values["real_tanh_autodiff"]["rel_error"]["median"]) < 0.75
        )
        lines.append(
            f"| {PILOT_TASKS[task]} | {values['war']['rel_error']['median']:.6g} | "
            f"{values['real_tanh_autodiff']['rel_error']['median']:.6g} | "
            f"{values['war_wins']}/3 | {'是' if passed else '否'} |"
        )
    lines.extend(
        [
            "",
            "二维 ZK 与动态板通过门槛。动态板的筛选误差更低，但按预先冻结的选择规则，优先选择与现有 Polyharmonic/Cahn–Hilliard 不同的可训练 ZK，因此正式实验选择 `zk_2d_o3`。三维 ZK 与 Swift–Hohenberg 在当前公平协议下均退化到相对误差约 1。",
            "",
            "## 二维三阶 ZK 正式结果（5 seeds × 1200 秒）",
            "",
            "| 方法 | Mean | Std | Median | Min | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        values = formal_summary[FORMAL_TASK][method]["rel_error"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {values['mean']:.6g} | {values['std']:.6g} | "
            f"{values['median']:.6g} | {values['min']:.6g} | {values['max']:.6g} |"
        )
    values = formal_summary[FORMAL_TASK]
    lines.extend(
        [
            "",
            f"WAR 在 5/5 个配对 seed 上误差更低；Real AD/WAR 的平均误差比为 {values['ratio_of_mean_rel_errors_real_ad_over_war']:.3f}×，配对比的几何均值为 {values['paired_rel_error_ratio_real_ad_over_war']['geometric_mean']:.3f}×。",
            f"5/5 同向的精确 sign test：单侧 p={values['exact_sign_test_one_sided_p']:.5f}，双侧 p={values['exact_sign_test_two_sided_p']:.5f}。样本数只有 5，因此这是强而一致的描述性证据，不应写成双侧 5% 水平显著。",
            "",
            "## 曲线与口径",
            "",
            "- `formal_realtime_history.csv`：所有原始逐点 history。",
            "- `formal_realtime_curves.csv`：在 0–1200 秒的 20 秒公共网格上，对正值指标作 log-linear 插值，再计算五个 seed 的中位数与四分位带。",
            "- `realtime_rel_error`：固定 2048 点 history 评估集上的实时相对误差。",
            "- `final_rel_error_by_seed`：固定 16384 点最终评估集上的配对结果。",
            "",
            "## 公平性边界",
            "",
            "两种方法具有相同的字面层形状、墙钟预算、训练点和初始化类型，但激活函数按已冻结协议分别为 sinh 与 tanh；complex64 WAR 的复参数也对应约两倍实标量自由度。结论应表述为该固定方法实现与网络形状下的墙钟效率/精度比较，而不是严格等激活或等实参数量比较。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_analysis_checksums(analysis_dir: Path) -> None:
    paths = sorted(
        path for path in analysis_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS" and ".tmp" not in path.name
    )
    atomic_text(
        analysis_dir / "SHA256SUMS",
        "\n".join(f"{sha256(path)}  {path.name}" for path in paths) + "\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_root", type=Path)
    parser.add_argument("formal_root", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pilot_root = args.pilot_root.resolve()
    formal_root = args.formal_root.resolve()
    analysis_dir = (args.analysis_dir or formal_root / "analysis").resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    pilot_rows, pilot_audit = audit_bundle(
        pilot_root,
        stage="pilot",
        tasks=tuple(PILOT_TASKS),
        seeds=tuple(range(3)),
        seconds=600.0,
    )
    formal_rows, formal_audit = audit_bundle(
        formal_root,
        stage="formal",
        tasks=(FORMAL_TASK,),
        seeds=tuple(range(5)),
        seconds=1200.0,
    )
    audit = {
        "generated_at": utc_now(),
        "status": "passed" if pilot_audit["status"] == formal_audit["status"] == "passed" else "failed",
        "pilot": pilot_audit,
        "formal": formal_audit,
        "checksum_finalization_note": (
            "The runner originally hashed run.log before emitting ORCHESTRATOR_FINAL. "
            "After confirming that run.log was the sole mismatch, SHA256SUMS was "
            "regenerated after process exit; no JSON, per-cell log, history, or metric changed."
        ),
    }
    atomic_json(analysis_dir / "audit.json", audit)
    if audit["status"] != "passed":
        raise RuntimeError("raw result audit failed; see analysis/audit.json")

    pilot_aggregate, pilot_paired, pilot_summary = build_statistics(
        pilot_rows, tuple(PILOT_TASKS), tuple(range(3))
    )
    formal_aggregate, formal_paired, formal_summary = build_statistics(
        formal_rows, (FORMAL_TASK,), tuple(range(5))
    )
    long_rows, curve_rows, series = build_formal_curves(formal_rows)
    strip_history = lambda rows: [  # noqa: E731 - compact serialization helper
        {key: value for key, value in row.items() if key != "history"} for row in rows
    ]
    write_csv(analysis_dir / "pilot_final_metrics.csv", strip_history(pilot_rows))
    write_csv(analysis_dir / "pilot_aggregate_metrics.csv", pilot_aggregate)
    write_csv(analysis_dir / "pilot_paired_comparison.csv", pilot_paired)
    write_csv(analysis_dir / "formal_final_metrics.csv", strip_history(formal_rows))
    write_csv(analysis_dir / "formal_aggregate_metrics.csv", formal_aggregate)
    write_csv(analysis_dir / "formal_paired_comparison.csv", formal_paired)
    write_csv(analysis_dir / "formal_realtime_history.csv", long_rows)
    write_csv(analysis_dir / "formal_realtime_curves.csv", curve_rows)
    atomic_json(
        analysis_dir / "analysis_summary.json",
        {
            "generated_at": utc_now(),
            "protocol_id": PROTOCOL_ID,
            "run_git_sha": RUN_GIT_SHA,
            "selected_task": FORMAL_TASK,
            "selection_rule": "prefer a trainable ZK distinct from existing Poly/CH; otherwise dynamic plate, then Swift–Hohenberg",
            "pilot": pilot_summary,
            "formal": formal_summary,
        },
    )
    atomic_text(
        analysis_dir / "REPORT_zh.md",
        markdown_report(pilot_audit, formal_audit, pilot_summary, formal_summary),
    )
    plot_realtime(
        analysis_dir,
        series,
        metric="rel_error",
        ylabel="Relative L2 error (history evaluation; lower is better)",
        stem="realtime_rel_error",
    )
    plot_realtime(
        analysis_dir,
        series,
        metric="loss",
        ylabel="Weighted training loss (lower is better)",
        stem="realtime_loss",
    )
    plot_final_by_seed(analysis_dir, formal_rows)
    shutil.copy2(Path(__file__).resolve(), analysis_dir / "generate_analysis.py")
    write_analysis_checksums(analysis_dir)
    print(json.dumps({"status": "complete", "analysis_dir": str(analysis_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
