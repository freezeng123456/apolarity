#!/usr/bin/env python3
"""Run a power-grid Poly sweep and formal comparison for one dimension."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_SCRIPTS = ROOT / "experiments" / "archived" / "scripts"
GRID_RUNNER = ARCHIVED_SCRIPTS / "run_poly_weight_grid.py"
FORMAL_RUNNER = ARCHIVED_SCRIPTS / "run_poly_shared_weights.py"
ORDERS = (2, 4, 6)


def load_best_weights(ranking_path: Path) -> tuple[float, ...]:
    ranking = json.loads(ranking_path.read_text())
    if not ranking:
        raise ValueError(f"empty ranking: {ranking_path}")
    return tuple(float(value) for value in ranking[0]["bc_weights"])


def run(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=(2, 3), required=True)
    parser.add_argument("--grid", default="0.001,0.01,0.1,1,10,100,1000")
    parser.add_argument("--grid-seconds", type=float, default=30.0)
    parser.add_argument("--formal-seconds", type=float, default=1200.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-eval-seed", type=int, default=54321)
    parser.add_argument("--formal-eval-seed", type=int, default=12345)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.grid_root.mkdir(parents=True, exist_ok=True)
    args.formal_root.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[float]] = {}
    for order in ORDERS:
        grid_dir = args.grid_root / f"o{order}"
        run(
            [
                sys.executable,
                str(GRID_RUNNER),
                "--dim",
                str(args.dim),
                "--order",
                str(order),
                "--seconds",
                str(args.grid_seconds),
                "--grid",
                args.grid,
                "--seed",
                str(args.seed),
                "--eval-seed",
                str(args.grid_eval_seed),
                "--out-dir",
                str(grid_dir),
                *( ["--resume"] if args.resume else [] ),
            ]
        )
        weights = load_best_weights(grid_dir / "grid30_summary" / "ranking.json")
        selected[f"o{order}"] = list(weights)
        run(
            [
                sys.executable,
                str(FORMAL_RUNNER),
                "--dim",
                str(args.dim),
                "--order",
                str(order),
                "--method",
                "both",
                "--bc-weights",
                ",".join(f"{weight:g}" for weight in weights),
                "--seconds",
                str(args.formal_seconds),
                "--seed",
                str(args.seed),
                "--eval-seed",
                str(args.formal_eval_seed),
                "--out",
                str(args.formal_root / f"o{order}"),
            ]
        )
    (args.formal_root / "selected_weights.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
