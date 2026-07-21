#!/usr/bin/env python3
"""Complete Maxwell seed 1--4 runs and aggregate them with seed 0."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from run_osc_shared_weight_suite import SETTINGS, run_one, write


def finite_rows(rows):
    for row in rows:
        for value in row.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite value in {row['problem']} seed {row['seed']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formal-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=1200.0)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seed-stop", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=12345)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n-int", type=int, default=4096)
    ap.add_argument("--n-bc", type=int, default=512)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    selected = json.loads((args.formal_root / "selected_weights.json").read_text())
    args.out_root.mkdir(parents=True, exist_ok=True)
    for a in SETTINGS["maxwell"]:
        key = f"maxwell_a{a}"
        weight = float(selected[key])
        rows = []
        original = args.formal_root / f"{key}.json"
        rows.extend(json.loads(original.read_text()))
        for seed in range(args.seed_start, args.seed_stop + 1):
            path = args.out_root / f"{key}_seed{seed}.json"
            if args.resume and path.exists():
                new_rows = json.loads(path.read_text())
            else:
                new_rows = [
                    run_one("maxwell", a, method, weight, args.seconds, seed,
                            args.eval_seed, args.hidden, args.depth, args.n_int,
                            args.n_bc)
                    for method in ("vanilla", "sinh")
                ]
                write(path, new_rows)
            rows.extend(new_rows)
            print(f"[done] {key} seed={seed}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(rows) != 10 or sorted({row["seed"] for row in rows}) != [0, 1, 2, 3, 4]:
            raise ValueError(f"incomplete five-seed bundle for {key}")
        finite_rows(rows)
        summary = {"problem": key, "bc_weight": weight, "seeds": [0, 1, 2, 3, 4], "rows": rows}
        for method in ("vanilla", "sinh"):
            values = [row["L2_err"] for row in rows if row["variant"] == method]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            summary[method] = {
                "L2_mean": mean,
                "L2_std": math.sqrt(variance),
                "L2_min": min(values),
                "L2_max": max(values),
            }
        write(args.out_root / f"{key}_5seed.json", summary)


if __name__ == "__main__":
    main()
