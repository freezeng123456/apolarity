#!/usr/bin/env python3
"""Aggregate exp_oscillatory_suite.py CSV into per-(problem,variant) summaries.

Usage:  python experiments/aggregate_oscillatory.py results/oscillatory_suite.csv
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from statistics import mean, pstdev


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main(path: str) -> None:
    rows = list(csv.DictReader(open(path)))
    problems, variants = [], []
    for r in rows:
        if r["problem"] not in problems:
            problems.append(r["problem"])
        if r["variant"] not in variants:
            variants.append(r["variant"])

    agg: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"L2": [], "ms": [], "steps": [], "mb": [], "params": []})
    for r in rows:
        a = agg[(r["problem"], r["variant"])]
        a["L2"].append(_f(r["L2_err"]))
        a["ms"].append(_f(r["ms_per_step"]))
        a["steps"].append(_f(r["steps"]))
        a["mb"].append(_f(r["peak_mb"]))
        a["params"].append(_f(r["params"]))

    for prob in problems:
        order = next((r["order"] for r in rows if r["problem"] == prob), "?")
        print(f"\n=== {prob}  (order {order}) ===")
        print(f"{'variant':<20} {'params':>7} {'L2 mean':>10} {'L2 std':>9} "
              f"{'L2 best':>10} {'ms/step':>8} {'steps':>7}")
        ranked = []
        for v in variants:
            a = agg.get((prob, v))
            if not a or not a["L2"]:
                continue
            finite = [x for x in a["L2"] if math.isfinite(x)]
            l2m = mean(finite) if finite else float("inf")
            ranked.append((l2m, v, a, finite))
        ranked.sort(key=lambda t: t[0])
        for l2m, v, a, finite in ranked:
            l2s = pstdev(finite) if len(finite) > 1 else 0.0
            best = min(finite) if finite else float("inf")
            print(f"{v:<20} {int(mean(a['params'])):>7} {l2m:>10.3e} {l2s:>9.2e} "
                  f"{best:>10.3e} {mean(a['ms']):>8.1f} {int(mean(a['steps'])):>7}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/oscillatory_suite.csv")
