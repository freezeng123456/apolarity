"""Plot the validated paired summary; no timing values are hand-entered."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    checks = json.loads((args.report / "validation.json").read_text())
    if checks["missing_cells"] or not checks["all_outputs_consistent"]:
        raise RuntimeError("do not plot incomplete/inconsistent results as a complete comparison")
    with (args.report / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    order = ["111111", "111222", "112233", "123456", "11223344", "111222333"]
    main_rows = sorted([r for r in rows if int(r["batch"]) == 100], key=lambda r: order.index(r["target"]))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    x = np.arange(len(main_rows))
    for offset, name, label, color in [(-0.18, "nested", "Nested JVP", "#596b80"),
                                       (0.18, "waring", "Waring + batched Taylor jet", "#16847e")]:
        axes[0].bar(x + offset, [float(r[f"{name}_mean_ms"]) for r in main_rows], .36,
                    yerr=[float(r[f"{name}_std_ms"]) for r in main_rows], label=label,
                    color=color, capsize=3)
    labels = [r["target"] + "\nR=" + r["rank"] for r in main_rows]
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Derivative-value time (ms, log scale)")
    axes[0].set_title("Six targets | batch 100")
    axes[0].legend(fontsize=9)
    ratios = [float(r["paired_speedup_mean"]) for r in main_rows]
    axes[1].bar(x, ratios, color=["#16847e" if v >= 1 else "#c58b42" for v in ratios])
    axes[1].axhline(1, color="black", linewidth=1, linestyle="--")
    axes[1].set_xticks(x, [r["target"] for r in main_rows], rotation=30, ha="right")
    axes[1].set_ylabel("Nested JVP / Waring time")
    axes[1].set_title("Paired speedup | above 1 favors Waring")
    for i, v in enumerate(ratios):
        axes[1].annotate(f"{v:.2f}x", (i, v), xytext=(0, 4), textcoords="offset points", ha="center")
    axes[1].set_ylim(0, max(1.2, max(ratios) * 1.18))
    fig.suptitle("H20 · JAX 0.4.38 · float32 / complex64 · JIT on both methods\nThree seeds; 30 synchronized calls per seed; no parameter backward", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(args.report / f"first_group_main.{ext}")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, target in zip(axes, ("123456", "111222333")):
        selected = sorted([r for r in rows if r["target"] == target], key=lambda r: int(r["batch"]))
        for name, label, color in [("nested", "Nested JVP", "#596b80"), ("waring", "Waring + jet", "#16847e")]:
            ax.errorbar([int(r["batch"]) for r in selected], [float(r[f"{name}_mean_ms"]) for r in selected],
                        yerr=[float(r[f"{name}_std_ms"]) for r in selected], marker="o", label=label, color=color, capsize=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([100, 400, 1600], ["100", "400", "1600"])
        ax.set_xlabel("Input batch size")
        ax.set_ylabel("Derivative-value time (ms)")
        ax.set_title(target)
        ax.legend()
    fig.suptitle("H20 · batch scaling · same network and derivative targets")
    for ext in ("png", "pdf"):
        fig.savefig(args.report / f"first_group_scaling.{ext}")


if __name__ == "__main__":
    main()
