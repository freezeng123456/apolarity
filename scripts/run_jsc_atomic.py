#!/usr/bin/env python3
"""Pure Python executor for one preregistered jsc_v2 atomic task."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(ROOT / "scripts"))

from protocol import (  # noqa: E402
    BUDGET_SECONDS,
    COLLOCATION_PROTOCOL,
    DEPTH,
    EVALUATION_PROTOCOL,
    FORMAL_METHODS,
    HISTORY_EVAL_N,
    HISTORY_EVERY_STEPS,
    LEARNING_RATE,
    LEARNING_RATE_FINAL,
    LEARNING_RATE_SCHEDULE,
    N_BOUNDARY,
    N_INTERIOR,
    PROTOCOL_ID,
    RESULT_ROOT,
    SEEDS,
    get_task,
    validate_budget_table,
)
from validate_jsc_results import validate_task_directory  # noqa: E402


def _git_state() -> tuple[str, bool]:
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


def _hardware() -> str:
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        return (
            f"{torch.cuda.get_device_name(index)}; "
            f"torch={torch.__version__}; cuda={torch.version.cuda}"
        )
    return f"CPU; torch={torch.__version__}"


def _experiment_command(
    task,
    method: str,
    width: int,
    output: Path,
    *,
    smoke: bool,
) -> list[str]:
    if task.family == "poly":
        script = ROOT / "experiments" / "polyharmonic" / "exp_polyharmonic.py"
        setting = ["--dim", str(task.dimension), "--orders", str(task.order)]
    elif task.family == "chirp":
        script = ROOT / "experiments" / "chirp" / "exp_chirp.py"
        setting = ["--sweeps", str(task.sweep)]
    else:
        script = ROOT / "experiments" / "maxwell" / "exp_maxwell.py"
        setting = ["--sweeps", str(task.sweep)]

    seconds = 1.0 if smoke else BUDGET_SECONDS
    seeds = 1 if smoke else len(SEEDS)
    n_int = 32 if smoke else N_INTERIOR
    n_bc = 16 if smoke else N_BOUNDARY
    return [
        sys.executable,
        str(script),
        *setting,
        "--variants",
        method,
        "--hidden",
        str(width),
        "--depth",
        str(DEPTH),
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
        str(LEARNING_RATE),
        "--lr-schedule",
        LEARNING_RATE_SCHEDULE,
        "--lr-final",
        str(LEARNING_RATE_FINAL),
        "--history",
        "--history-every-steps",
        str(HISTORY_EVERY_STEPS),
        "--history-eval-n",
        str(64 if smoke else HISTORY_EVAL_N),
        "--out",
        str(output),
    ]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = sorted({key for row in rows for key in row})
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _enrich_part(
    output: Path,
    *,
    task,
    method: str,
    budget,
    sha: str,
    dirty: bool,
    hardware: str,
    smoke: bool,
) -> None:
    json_path = output.with_suffix(".json")
    history_path = output.with_name(output.stem + "_history").with_suffix(".json")
    rows = json.loads(json_path.read_text())
    expected_rows = 1 if smoke else len(SEEDS)
    if len(rows) != expected_rows:
        raise ValueError(f"{method} wrote {len(rows)} rows, expected {expected_rows}")
    expected_representation = budget.representation
    frequency = {
        "complex_sinh_omega0": task.omega0,
        "siren_first_omega0": 30.0,
        "siren_hidden_omega0": 30.0,
        "fourier_branch_sigmas": [1.0, task.fourier_sigma],
        "fourier_input_mean": [0.0] * task.dimension,
        "fourier_input_std": [3.0 ** -0.5] * task.dimension,
        "mscale_scales": [1.0, 2.0, 4.0],
    }
    for row in rows:
        if row["variant"] != method:
            raise ValueError(f"{output} contains unexpected method {row['variant']}")
        if int(row["params"]) != budget.real_dof:
            raise ValueError(
                f"{method} emitted {row['params']} DOF, expected {budget.real_dof}"
            )
        row.update({
            "protocol_id": PROTOCOL_ID,
            "git_sha": sha,
            "git_dirty": dirty,
            "task_id": task.task_id,
            "family": task.family,
            "dimension": task.dimension,
            "actual_width": budget.width,
            "real_dof": budget.real_dof,
            "target_real_dof": budget.target_real_dof,
            "parameter_relative_error": budget.relative_error,
            "representation": expected_representation,
            "collocation": COLLOCATION_PROTOCOL,
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "frequency_initialization": json.dumps(frequency, sort_keys=True),
            "complex_sinh_omega0": task.omega0,
            "siren_first_omega0": 30.0,
            "siren_hidden_omega0": 30.0,
            "fourier_branch_sigmas": json.dumps([1.0, task.fourier_sigma]),
            "fourier_input_mean": json.dumps([0.0] * task.dimension),
            "fourier_input_std": json.dumps([3.0 ** -0.5] * task.dimension),
            "mscale_scales": json.dumps([1.0, 2.0, 4.0]),
            "hardware": hardware,
        })
    _write_rows(output, rows)

    histories = json.loads(history_path.read_text())
    for history in histories:
        history.update({
            "protocol_id": PROTOCOL_ID,
            "task_id": task.task_id,
            "family": task.family,
            "dimension": task.dimension,
            "variant": method,
            "representation": expected_representation,
            "actual_width": budget.width,
            "real_dof": budget.real_dof,
        })
    history_path.write_text(json.dumps(histories, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=["poly", "chirp", "maxwell"])
    parser.add_argument("--dim", type=int)
    parser.add_argument("--order", type=int)
    parser.add_argument("--sweep", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task = get_task(
        args.family,
        dimension=args.dim,
        order=args.order,
        sweep=args.sweep,
    )
    budgets = validate_budget_table(task)
    sha, dirty = _git_state()
    if dirty and not (args.smoke or args.dry_run):
        raise RuntimeError(
            "formal jsc_v2 runs require a clean Git worktree so git_sha fully "
            "identifies the executed code"
        )
    root = args.output_root or (
        ROOT / "experiments" / "results" / "_smoke" if args.smoke else RESULT_ROOT
    )
    task_dir = root / task.task_id
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "task": asdict(task),
        "methods": {method: asdict(budget) for method, budget in budgets.items()},
        "git_sha": sha,
        "git_dirty": dirty,
        "smoke": args.smoke,
        "estimated_runs": len(FORMAL_METHODS) * (1 if args.smoke else len(SEEDS)),
        "estimated_training_seconds": (
            len(FORMAL_METHODS)
            * (1 if args.smoke else len(SEEDS))
            * (1.0 if args.smoke else BUDGET_SECONDS)
        ),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if not args.smoke:
        smoke_dir = ROOT / "experiments" / "results" / "_smoke" / task.task_id
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir)
    if task_dir.exists() and any(task_dir.iterdir()):
        raise FileExistsError(
            f"{task_dir} is not empty; move or remove the incomplete task first"
        )
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    hardware = _hardware()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    for method in FORMAL_METHODS:
        budget = budgets[method]
        output = task_dir / f"{method}_part.csv"
        command = _experiment_command(
            task,
            method,
            budget.width,
            output,
            smoke=args.smoke,
        )
        print("[run]", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        _enrich_part(
            output,
            task=task,
            method=method,
            budget=budget,
            sha=sha,
            dirty=dirty,
            hardware=hardware,
            smoke=args.smoke,
        )

    canonical = validate_task_directory(task_dir, smoke=args.smoke)
    print(f"[complete] validated canonical bundle: {canonical}", flush=True)


if __name__ == "__main__":
    main()
