#!/usr/bin/env python3
"""Run the five-scale MscaleDNN sensitivity study on the nine 2D paper tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "jsc_v2_mscale5_sensitivity_v1"
SCALES = (1.0, 2.0, 4.0, 8.0, 16.0)
TASKS = (
    ("poly_d2_o2", "polyharmonic/exp_polyharmonic.py", ("--dim", "2", "--orders", "2")),
    ("poly_d2_o4", "polyharmonic/exp_polyharmonic.py", ("--dim", "2", "--orders", "4")),
    ("poly_d2_o6", "polyharmonic/exp_polyharmonic.py", ("--dim", "2", "--orders", "6")),
    ("chirp_a1", "chirp/exp_chirp.py", ("--sweeps", "1")),
    ("chirp_a2", "chirp/exp_chirp.py", ("--sweeps", "2")),
    ("chirp_a3", "chirp/exp_chirp.py", ("--sweeps", "3")),
    ("maxwell_a2", "maxwell/exp_maxwell.py", ("--sweeps", "2")),
    ("maxwell_a4", "maxwell/exp_maxwell.py", ("--sweeps", "4")),
    ("maxwell_a6", "maxwell/exp_maxwell.py", ("--sweeps", "6")),
)


def git_state() -> tuple[str, bool]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return sha, dirty


def write_rows(path: Path, rows: list[dict]) -> None:
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = sorted({key for row in rows for key in row})
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_and_enrich(
    task_id: str,
    output: Path,
    *,
    sha: str,
    smoke: bool,
) -> None:
    rows = json.loads(output.with_suffix(".json").read_text())
    expected = 1 if smoke else 5
    if len(rows) != expected:
        raise ValueError(f"{task_id}: wrote {len(rows)} rows, expected {expected}")
    expected_seeds = list(range(expected))
    if sorted(int(row["seed"]) for row in rows) != expected_seeds:
        raise ValueError(f"{task_id}: seed set does not match {expected_seeds}")
    for row in rows:
        if row.get("variant") != "mscale5":
            raise ValueError(f"{task_id}: unexpected variant {row.get('variant')!r}")
        if row.get("nan"):
            raise ValueError(f"{task_id}: seed {row['seed']} reported NaN")
        if not math.isfinite(float(row["L2_err"])):
            raise ValueError(f"{task_id}: seed {row['seed']} has invalid L2_err")
        row.update(
            {
                "protocol_id": PROTOCOL_ID,
                "git_sha": sha,
                "git_dirty": False,
                "task_id": task_id,
                "actual_width": 128,
                "mscale_scales": json.dumps(SCALES),
                "representation": (
                    "split_real" if task_id.startswith("maxwell") else "real"
                ),
                "hardware": (
                    f"{torch.cuda.get_device_name(0)}; torch={torch.__version__}; "
                    f"cuda={torch.version.cuda}"
                    if torch.cuda.is_available()
                    else f"CPU; torch={torch.__version__}"
                ),
            }
        )
    write_rows(output, rows)
    task_dir = output.parent
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "task_id": task_id,
        "method": "mscale5",
        "scales": SCALES,
        "width": 128,
        "depth": 4,
        "seconds_per_seed": 1.0 if smoke else 1200.0,
        "seeds": expected_seeds,
        "git_sha": sha,
        "smoke": smoke,
    }
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (task_dir / "VALIDATED").write_text("ok\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    sha, dirty = git_state()
    if dirty:
        raise RuntimeError("mscale5 runs require a clean Git worktree")
    result_root = ROOT / "experiments" / "results"
    root = result_root / (
        "_smoke/mscale5_main9" if args.smoke else "mscale5_main9"
    )
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "git_sha": sha,
                "method": "mscale5",
                "scales": SCALES,
                "tasks": [task[0] for task in TASKS],
                "smoke": args.smoke,
            },
            indent=2,
        )
        + "\n"
    )

    seconds = 1.0 if args.smoke else 1200.0
    seeds = 1 if args.smoke else 5
    n_int = 32 if args.smoke else 4096
    n_bc = 16 if args.smoke else 512
    history_n = 64 if args.smoke else 4096
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    for task_id, relative_script, setting in TASKS:
        task_dir = root / task_id
        task_dir.mkdir()
        output = task_dir / "mscale5_part.csv"
        command = [
            sys.executable,
            str(ROOT / "experiments" / relative_script),
            *setting,
            "--variants",
            "mscale5",
            "--hidden",
            "128",
            "--depth",
            "4",
            "--seconds",
            str(seconds),
            "--seeds",
            str(seeds),
            "--seed-start",
            "0",
            "--n-int",
            str(n_int),
            "--n-bc",
            str(n_bc),
            "--lr",
            "0.001",
            "--lr-schedule",
            "cosine",
            "--lr-final",
            "0.0001",
            "--history",
            "--history-every-steps",
            "20",
            "--history-eval-n",
            str(history_n),
            "--out",
            str(output),
        ]
        print("[run]", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        validate_and_enrich(task_id, output, sha=sha, smoke=args.smoke)
        print(f"[validated] {task_id}", flush=True)

    (root / "VALIDATED").write_text("ok\n")
    print(f"[complete] {root}", flush=True)


if __name__ == "__main__":
    main()
