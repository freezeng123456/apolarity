#!/usr/bin/env python3
"""Exhaustive Cartesian search for shared Poly boundary-loss weights."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "experiments" / "archived" / "scripts" / "run_poly_shared_weights.py"
METHODS = {"vanilla_tanh_direct_ad", "complex_sinh"}


def parse_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("grid must contain at least one value")
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("grid values must be finite and positive")
    if len(set(values)) != len(values):
        raise ValueError("grid values must be unique")
    return values


def cartesian_weights(order: int, grid: tuple[float, ...]):
    return tuple(itertools.product(grid, repeat=order // 2))


def load_complete(path: Path, expected_weights: tuple[float, ...]):
    if not path.exists():
        return None
    try:
        rows = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if len(rows) != 2 or {row.get("variant") for row in rows} != METHODS:
        return None
    if any(tuple(row.get("bc_weights", ())) != expected_weights for row in rows):
        return None
    if any(not math.isfinite(float(row.get("L2_err", math.nan))) for row in rows):
        return None
    return rows


def score_rows(rows: list[dict]) -> dict:
    errors = {row["variant"]: float(row["L2_err"]) for row in rows}
    vanilla = errors["vanilla_tanh_direct_ad"]
    sinh = errors["complex_sinh"]
    weights = tuple(float(value) for value in rows[0]["bc_weights"])
    return {
        "bc_weights": list(weights),
        "vanilla_L2_err": vanilla,
        "sinh_L2_err": sinh,
        "geometric_mean": math.sqrt(vanilla * sinh),
        "max_error": max(vanilla, sinh),
        "weight_sum": sum(weights),
    }


def run_point(
    args, weights: tuple[float, ...], stem: Path, seconds: float
) -> list[dict]:
    json_path = stem.with_suffix(".json")
    rows = load_complete(json_path, weights) if args.resume else None
    if rows is not None:
        print(f"[resume] {stem.name} weights={weights}", flush=True)
        return rows
    command = [
        sys.executable,
        str(RUNNER),
        "--order",
        str(args.order),
        "--dim",
        str(args.dim),
        "--method",
        "both",
        "--bc-weights",
        ",".join(f"{value:g}" for value in weights),
        "--seconds",
        str(seconds),
        "--seed",
        str(args.seed),
        "--eval-seed",
        str(args.eval_seed),
        "--out",
        str(stem),
    ]
    print(f"[start] {stem.name} weights={weights} seconds={seconds:g}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    rows = load_complete(json_path, weights)
    if rows is None:
        raise RuntimeError(f"incomplete result: {json_path}")
    return rows


def write_summary(path: Path, records: list[dict], metadata: dict) -> None:
    ranked = sorted(
        records,
        key=lambda row: (
            row["geometric_mean"],
            row["max_error"],
            row["weight_sum"],
        ),
    )
    path.mkdir(parents=True, exist_ok=True)
    (path / "ranking.json").write_text(json.dumps(ranked, indent=2) + "\n")
    (path / "manifest.json").write_text(
        json.dumps({**metadata, "best": ranked[0]}, indent=2) + "\n"
    )
    with (path / "ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ranked[0].keys())
        writer.writeheader()
        writer.writerows(ranked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, choices=(2, 4, 6), required=True)
    parser.add_argument("--dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--grid", default="0.01,0.03,0.1,0.3,1,3")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--confirm-top-k", type=int, default=0)
    parser.add_argument("--confirm-seconds", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=54321)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    grid = parse_grid(args.grid)
    combinations = cartesian_weights(args.order, grid)
    grid_dir = args.out_dir / "grid30"
    grid_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, weights in enumerate(combinations):
        rows = run_point(args, weights, grid_dir / f"point_{index:03d}", args.seconds)
        records.append(score_rows(rows))
    metadata = {
        "order": args.order,
        "dimension": args.dim,
        "grid": list(grid),
        "seconds_per_method": args.seconds,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "combinations": len(combinations),
        "selection_metric": "geometric_mean",
        "tie_breakers": ["max_error", "weight_sum"],
    }
    write_summary(args.out_dir / "grid30_summary", records, metadata)

    if args.confirm_top_k:
        ranked = sorted(
            records,
            key=lambda row: (
                row["geometric_mean"], row["max_error"], row["weight_sum"]
            ),
        )
        confirm_dir = args.out_dir / "confirm90"
        confirm_dir.mkdir(parents=True, exist_ok=True)
        confirmations = []
        for rank, record in enumerate(ranked[: args.confirm_top_k], start=1):
            weights = tuple(record["bc_weights"])
            rows = run_point(
                args,
                weights,
                confirm_dir / f"rank_{rank:02d}",
                args.confirm_seconds,
            )
            confirmations.append(score_rows(rows))
        write_summary(
            args.out_dir / "confirm90_summary",
            confirmations,
            {
                **metadata,
                "source": "top candidates from grid30_summary/ranking.json",
                "confirm_top_k": args.confirm_top_k,
                "seconds_per_method": args.confirm_seconds,
            },
        )


if __name__ == "__main__":
    main()
