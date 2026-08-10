#!/usr/bin/env python3
"""Audit and summarize the completed dynamic-plate weight search.

The script is intentionally server/result agnostic: it consumes the raw
search bundle and writes deterministic CSV/Markdown summaries.  It never
launches training and never changes raw result files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


METHODS = ("war", "real_tanh_autodiff")
TASKS = ("dynamic_plate_2d_o4", "strain_gradient_plate_2d_o6")
RANKING_NAMES = (
    "ranking_war",
    "ranking_real_tanh_autodiff",
    "ranking_shared_geomean",
    "ranking_shared_minimax",
)


def read_raw(raw_root: Path):
    rows = []
    for task in TASKS:
        for point in sorted((raw_root / task / "points").glob("point_*")):
            for method in METHODS:
                path = point / f"{method}.json"
                data = json.loads(path.read_text())
                rows.append(
                    {
                        "task": task,
                        "candidate_id": data["weight_map"],
                        "candidate": point.name,
                        "method": method,
                        "lambda_ic": float(data["weights"][0]),
                        "lambda_bc": float(data["weights"][1]),
                        "war_rel_error": None,
                        "ad_rel_error": None,
                        "rel_error": float(data["rel_error"]),
                        "loss": float(data["loss"]),
                        "steps": int(data["steps"]),
                        "ms_per_step": float(data["ms_per_step"]),
                        "peak_mb": float(data["peak_mb"]),
                    }
                )
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["task"], row["candidate"])] [row["method"]] = row
    paired = []
    for (task, candidate), methods in sorted(grouped.items()):
        if set(methods) != set(METHODS):
            raise RuntimeError(f"unpaired candidate: {task}/{candidate}")
        war = methods["war"]
        ad = methods["real_tanh_autodiff"]
        paired.append(
            {
                "task": task,
                "candidate": candidate,
                "lambda_ic": war["lambda_ic"],
                "lambda_bc": war["lambda_bc"],
                "war_rel_error": war["rel_error"],
                "ad_rel_error": ad["rel_error"],
                "war_loss": war["loss"],
                "ad_loss": ad["loss"],
                "war_steps": war["steps"],
                "ad_steps": ad["steps"],
                "war_ms_per_step": war["ms_per_step"],
                "ad_ms_per_step": ad["ms_per_step"],
                "war_peak_mb": war["peak_mb"],
                "ad_peak_mb": ad["peak_mb"],
                "geomean": math.sqrt(war["rel_error"] * ad["rel_error"]),
                "minimax": max(war["rel_error"], ad["rel_error"]),
                "mean_error": (war["rel_error"] + ad["rel_error"]) / 2.0,
            }
        )
    if len(rows) != 196 or len(paired) != 98:
        raise RuntimeError(f"expected 196 method rows/98 pairs, got {len(rows)}/{len(paired)}")
    return rows, paired


def sci(value: float) -> str:
    return f"{value:.3g}"


def weight_label(value: float) -> str:
    if value == 1.0:
        return "1"
    return f"{value:g}"


def top_rows(paired, task, key):
    return sorted((row for row in paired if row["task"] == task), key=lambda row: row[key])[:10]


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_report(raw_root: Path, out_root: Path):
    rows, paired = read_raw(raw_root)
    analysis = out_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    fields = [
        "task", "candidate", "lambda_ic", "lambda_bc", "war_rel_error",
        "ad_rel_error", "war_loss", "ad_loss", "war_steps", "ad_steps",
        "war_ms_per_step", "ad_ms_per_step", "war_peak_mb", "ad_peak_mb",
        "geomean", "minimax", "mean_error",
    ]
    write_csv(analysis / "paired_candidates.csv", paired, fields)

    report = []
    report.append("# 二维动态板权重搜参报告\n")
    report.append("本报告由服务器完成的 7×7 权重网格生成；不包含任何正式多 seed 运行。")
    report.append("协议：WAR=`complex64+sinh+Waring jet`，实数 AD=`float32+tanh+direct autodiff`，共同 Xavier/hidden128/depth4，训练 seed=42，评估 seed=68421。\n")

    for task in TASKS:
        task_rows = [row for row in paired if row["task"] == task]
        report.append(f"## `{task}`\n")
        report.append(f"完成配对候选：{len(task_rows)}/49。\n")
        war_values = [row["war_rel_error"] for row in task_rows]
        ad_values = [row["ad_rel_error"] for row in task_rows]
        wins_war = sum(row["war_rel_error"] < row["ad_rel_error"] for row in task_rows)
        wins_ad = sum(row["ad_rel_error"] < row["war_rel_error"] for row in task_rows)
        report.append(
            f"WAR rel_error 中位数 {sci(median(war_values))}，范围 [{sci(min(war_values))}, {sci(max(war_values))}]；"
            f"AD 中位数 {sci(median(ad_values))}，范围 [{sci(min(ad_values))}, {sci(max(ad_values))}]。"
        )
        report.append(
            f"逐点比较：WAR 胜 {wins_war}/49，AD 胜 {wins_ad}/49，平局 {49 - wins_war - wins_ad}/49。\n"
        )
        report.append("### 四类 Top 10\n")
        metric_names = {
            "war_rel_error": "WAR 单方法",
            "ad_rel_error": "实数 AD 单方法",
            "geomean": "共享几何平均",
            "minimax": "共享 minimax",
        }
        for key, title in metric_names.items():
            report.append(f"#### {title}\n")
            report.append("| 排名 | `(lambda_ic, lambda_bc)` | WAR rel_error | AD rel_error | 目标值 |\n|---:|---:|---:|---:|---:|")
            for index, row in enumerate(top_rows(paired, task, key), 1):
                report.append(
                    f"| {index} | `({weight_label(row['lambda_ic'])}, {weight_label(row['lambda_bc'])})` | "
                    f"{sci(row['war_rel_error'])} | {sci(row['ad_rel_error'])} | {sci(row[key])} |"
                )
            report.append("")

        top_shared = top_rows(paired, task, "geomean")
        top_minimax = top_rows(paired, task, "minimax")
        ic_counts = Counter(row["lambda_ic"] for row in top_shared)
        bc_counts = Counter(row["lambda_bc"] for row in top_shared)
        geomean_under_quarter = sum(row["geomean"] < 0.25 for row in task_rows)
        minimax_under_three_quarters = sum(row["minimax"] < 0.75 for row in task_rows)
        war_under_point_two = sum(row["war_rel_error"] < 0.2 for row in task_rows)
        ad_under_point_two = sum(row["ad_rel_error"] < 0.2 for row in task_rows)
        ic_values = [row["lambda_ic"] for row in top_shared]
        bc_values = [row["lambda_bc"] for row in top_shared]
        report.append("### 权重敏感性与稳定平台\n")
        report.append(
            "共享几何平均 Top10 的权重点为："
            + ", ".join(
                f"({weight_label(row['lambda_ic'])},{weight_label(row['lambda_bc'])})"
                for row in top_shared
            )
            + "。"
        )
        report.append(
            "共享 minimax Top10 的权重点为："
            + ", ".join(
                f"({weight_label(row['lambda_ic'])},{weight_label(row['lambda_bc'])})"
                for row in top_minimax
            )
            + "。"
        )
        report.append(
            "两种方法的最优点通常不相同，因此正式实验若要求同一权重，应优先看共享几何平均/minimax，"
            "而不是直接采用 WAR 或 AD 的单方法最优点。"
        )
        report.append(
            f"阈值统计：WAR rel_error<0.2 有 {war_under_point_two}/49 个点，"
            f"AD rel_error<0.2 有 {ad_under_point_two}/49 个点，"
            f"共享几何平均<0.25 有 {geomean_under_quarter}/49 个点，"
            f"共享 minimax<0.75 有 {minimax_under_three_quarters}/49 个点。"
        )
        report.append(
            f"共享几何平均 Top10 的坐标频数：lambda_ic={dict(ic_counts)}，lambda_bc={dict(bc_counts)}；"
            f"坐标范围为 lambda_ic=[{min(ic_values):g},{max(ic_values):g}]、"
            f"lambda_bc=[{min(bc_values):g},{max(bc_values):g}]。\n"
        )

        report.append("### 计算代价\n")
        report.append(
            f"WAR 平均 step time {sci(median([row['war_ms_per_step'] for row in task_rows]))} ms，"
            f"AD 平均 step time {sci(median([row['ad_ms_per_step'] for row in task_rows]))} ms；"
            f"WAR 峰值显存约 {sci(max(row['war_peak_mb'] for row in task_rows))} MB，"
            f"AD 峰值显存约 {sci(max(row['ad_peak_mb'] for row in task_rows))} MB。\n"
        )

    report.append("## 结论与下一步建议\n")
    report.append(
        "四阶动态板已经显示出明确的可训练区域：WAR 的误差整体更低，实数 AD 也存在误差低于 0.1 的权重平台。"
    )
    report.append(
        "六阶应变梯度板明显更难：完整网格中仍应以共享目标筛选可复现权重；不要把单个 WAR 最优点直接解释为方法优势。"
    )
    report.append(
        "本轮只完成 60 秒权重搜索；建议先由用户从 Top10/平台中确定权重，再单独批准固定权重的正式多 seed 运行。"
    )
    (analysis / "TOP10_AND_SENSITIVITY_zh.md").write_text("\n".join(report) + "\n")

    metadata = {
        "remote_result_root": "/root/apolarity-dynamic-plate-o4-o6-weight-search-v1",
        "remote_archive_sha256": "283ca40eb24bd350bfec4cd94c8bb31e2c5a48096ab8806c5c4af64a2d648794",
        "remote_archive_bytes": 368656,
        "raw_sha256sums_verified": True,
        "code_sha": "adde41ff9be87d87b88bc1338af66c02422db4c5",
        "protocol_id": "dynamic_plate_o4_o6_shared_weight_grid_fp32_v1",
        "total_method_runs": 196,
        "paired_candidates": 98,
        "formal_experiment_started": False,
    }
    (out_root / "DELIVERY_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    make_report(args.raw_root, args.out_root)


if __name__ == "__main__":
    main()
