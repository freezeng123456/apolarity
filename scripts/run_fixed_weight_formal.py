#!/usr/bin/env python3
"""Run the fixed-weight 2D Cahn--Hilliard formal comparison.

The protocol contains only the two active two-spatial-dimensional benchmarks:
``cahn_hilliard_2d_o4`` and ``cahn_hilliard_2d_o6``.  Both use the shared
``(lambda_ic, lambda_bc) = (1, 10)`` objective and compare complex64 WAR with
the width/depth/activation-matched float32 real-sinh autodiff baseline.

The formal run is five training seeds, 1200 seconds per method/seed, strictly
serial on one GPU.  Results, histories, logs, progress, completion markers and
checksums are written atomically and the orchestrator is safely resumable.

Examples
--------
Run an ephemeral full-scale CUDA smoke and retain only its conclusion::

    python scripts/run_fixed_weight_formal.py smoke --seconds 3

Run or resume the approved formal experiment::

    python scripts/run_fixed_weight_formal.py orchestrate \
        --seconds 1200 --seeds 5 --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from run_cahn2d_weight_search import (
    COMPLEX_DTYPE,
    DEPTH,
    EVAL_SEED,
    HIDDEN,
    HISTORY_INTERVAL_SECONDS,
    LEARNING_RATE,
    LEARNING_RATE_FINAL,
    METHODS,
    PROTOCOL_ID as ENGINE_PROTOCOL_ID,
    REAL_DTYPE,
    TASKS,
    TASK_ORDER as ENGINE_TASK_ORDER,
    atomic_write_json,
    atomic_write_text,
    git_state,
    hardware_metadata,
    load_result,
    train_one,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "cahn_hilliard_2d_fixed_weights_formal_v1"
DEFAULT_ROOT = ROOT / "outputs" / PROTOCOL_ID
TASK_ORDER = ("cahn_hilliard_2d_o4", "cahn_hilliard_2d_o6")
FIXED_WEIGHTS: dict[str, tuple[float, float]] = {
    "cahn_hilliard_2d_o4": (1.0, 10.0),
    "cahn_hilliard_2d_o6": (1.0, 10.0),
}
FORMAL_SAMPLE_COUNTS = {
    "n_int": 4096,
    "n_ic": 1024,
    "n_bc": 2048,
    "n_eval": 32768,
    "history_eval_n": 4096,
}

if TASK_ORDER != ENGINE_TASK_ORDER:
    raise RuntimeError(
        f"formal task order {TASK_ORDER} differs from 2D engine {ENGINE_TASK_ORDER}"
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def write_checksums(root: Path) -> None:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and ".tmp." not in path.name
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
        for path in paths
    ]
    atomic_write_text(root / "SHA256SUMS", "\n".join(lines) + "\n")


def selected_tasks(value: str) -> tuple[str, ...]:
    names = TASK_ORDER if value == "all" else tuple(
        part.strip() for part in value.split(",") if part.strip()
    )
    unknown = [name for name in names if name not in FIXED_WEIGHTS]
    if unknown:
        raise ValueError(f"unknown 2D formal tasks: {unknown}")
    return names


def seed_list(count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("--seeds must be positive")
    return tuple(range(count))


def sample_counts(args: argparse.Namespace) -> dict[str, int]:
    values = {
        "n_int": int(args.n_int),
        "n_ic": int(args.n_ic),
        "n_bc": int(args.n_bc),
        "n_eval": int(args.n_eval),
        "history_eval_n": int(args.history_eval_n),
    }
    if min(values.values()) <= 0:
        raise ValueError("all sample counts must be positive")
    if values["history_eval_n"] > values["n_eval"]:
        raise ValueError("history_eval_n cannot exceed n_eval")
    return values


def complete_formal_result(
    path: Path,
    *,
    task: str,
    method: str,
    seed: int,
    seconds: float,
    samples: dict[str, int],
) -> bool:
    result = load_result(path)
    if not result:
        return False
    problem = result.get("problem", {})
    return bool(
        result.get("protocol_id") == ENGINE_PROTOCOL_ID
        and result.get("formal_protocol_id") == PROTOCOL_ID
        and result.get("status") == "complete"
        and result.get("task_id") == task
        and result.get("method") == method
        and tuple(float(value) for value in result.get("weights", ()))
        == FIXED_WEIGHTS[task]
        and int(result.get("train_seed", -1)) == seed
        and math.isclose(float(result.get("budget_seconds", -1.0)), seconds)
        and math.isfinite(float(result.get("loss", math.inf)))
        and math.isfinite(float(result.get("rel_error", math.inf)))
        and int(problem.get("n_int", -1)) == samples["n_int"]
        and int(problem.get("n_ic", -1)) == samples["n_ic"]
        and int(problem.get("n_bc_total", -1)) == samples["n_bc"]
        and int(problem.get("n_eval", -1)) == samples["n_eval"]
        and int(problem.get("history_eval_n", -1)) == samples["history_eval_n"]
    )


def manifest(
    tasks: Iterable[str],
    seconds: float,
    seeds: tuple[int, ...],
    samples: dict[str, int],
    *,
    smoke: bool,
) -> dict[str, Any]:
    task_list = list(tasks)
    parameter_elements = (
        3 * HIDDEN
        + HIDDEN
        + (DEPTH - 1) * (HIDDEN * HIDDEN + HIDDEN)
        + HIDDEN
        + 1
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "engine_protocol_id": ENGINE_PROTOCOL_ID,
        "created_at": utc_now(),
        "smoke": smoke,
        "tasks": [
            {
                "task_id": task,
                "family": "cahn_hilliard_2d",
                "order": TASKS[task].order,
                "q": TASKS[task].q,
                "eta_q": TASKS[task].eta,
                "weight_names": list(TASKS[task].weight_names),
                "weights": list(FIXED_WEIGHTS[task]),
                "weight_labels": ["1e+0", "1e+1"],
            }
            for task in task_list
        ],
        "methods": list(METHODS),
        "architecture": {
            "shared": {
                "physical_input": "affine-normalized raw (x,y,t)",
                "trigonometric_input_features": False,
                "activation": "sinh",
                "hidden": HIDDEN,
                "depth": DEPTH,
                "init_mode": "common_xavier",
                "frequency_initialization": "disabled",
            },
            "war": {
                "representation": "native_complex",
                "backend": "waring_complex_jet",
                "parameter_dtype": str(COMPLEX_DTYPE),
                "parameter_elements": parameter_elements,
                "real_dof": 2 * parameter_elements,
            },
            "real_sinh_autodiff": {
                "representation": "real",
                "backend": "direct_autodiff",
                "parameter_dtype": str(REAL_DTYPE),
                "parameter_elements": parameter_elements,
                "real_dof": parameter_elements,
            },
            "capacity_note": (
                "literal layer shapes match; native complex parameters contain "
                "two real scalar degrees of freedom"
            ),
        },
        "sample_counts": dict(samples),
        "seconds_per_method_seed": seconds,
        "seeds": list(seeds),
        "eval_seed": EVAL_SEED,
        "task_count": len(task_list),
        "method_seed_run_count": len(task_list) * len(METHODS) * len(seeds),
        "nominal_training_seconds": (
            len(task_list) * len(METHODS) * len(seeds) * seconds
        ),
        "learning_rate": LEARNING_RATE,
        "learning_rate_final": LEARNING_RATE_FINAL,
        "history_interval_seconds": HISTORY_INTERVAL_SECONDS,
        "serial_single_gpu": True,
        "history_required_fields": [
            "elapsed_seconds",
            "step",
            "rel_error",
            "loss",
            "L_PDE",
            "L_IC",
            "L_BC",
            "mass_drift_rms",
        ],
        "git": git_state(),
        "hardware": hardware_metadata(),
    }


def worker_command(
    task: str,
    method: str,
    seed: int,
    eval_seed: int,
    seconds: float,
    output: Path,
    samples: dict[str, int],
    *,
    smoke: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--task",
        task,
        "--method",
        method,
        "--weights",
        ",".join(f"{value:.16g}" for value in FIXED_WEIGHTS[task]),
        "--seed",
        str(seed),
        "--eval-seed",
        str(eval_seed),
        "--seconds",
        str(seconds),
        "--output",
        str(output),
    ]
    if smoke:
        command.append("--smoke")
    for flag, key in (
        ("--n-int", "n_int"),
        ("--n-ic", "n_ic"),
        ("--n-bc", "n_bc"),
        ("--n-eval", "n_eval"),
        ("--history-eval-n", "history_eval_n"),
    ):
        command.extend([flag, str(samples[key])])
    return command


def run_worker(args: argparse.Namespace) -> int:
    task = TASKS[args.task]
    expected_weights = FIXED_WEIGHTS[args.task]
    parsed_weights = tuple(
        float(value) for value in args.weights.split(",") if value
    )
    if parsed_weights != expected_weights:
        raise ValueError(
            f"weights for {args.task} are fixed at {expected_weights}; "
            f"received {parsed_weights}"
        )
    samples = sample_counts(args)
    output = args.output.resolve()
    base: dict[str, Any] = {
        "protocol_id": ENGINE_PROTOCOL_ID,
        "formal_protocol_id": PROTOCOL_ID,
        "status": "running",
        "task_id": args.task,
        "method": args.method,
        "weights": list(expected_weights),
        "seed": args.seed,
        "train_seed": args.seed,
        "eval_seed": args.eval_seed,
        "budget_seconds": args.seconds,
        "sample_counts": samples,
        "smoke": args.smoke,
        "started_at": utc_now(),
    }
    try:
        result = train_one(
            task,
            args.method,
            expected_weights,
            seconds=args.seconds,
            smoke=args.smoke,
            n_int=samples["n_int"],
            n_ic=samples["n_ic"],
            n_bc=samples["n_bc"],
            n_eval=samples["n_eval"],
            history_eval_n=samples["history_eval_n"],
            train_seed=args.seed,
            eval_seed=args.eval_seed,
        )
        result.update(
            {
                "formal_protocol_id": PROTOCOL_ID,
                "seed": args.seed,
                "fixed_weight_policy": {
                    "lambda_ic": expected_weights[0],
                    "lambda_bc": expected_weights[1],
                },
                "git": git_state(),
                "hardware": hardware_metadata(),
            }
        )
        atomic_write_json(output, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "task_id": args.task,
                    "method": args.method,
                    "seed": args.seed,
                    "steps": result.get("steps"),
                    "loss": result.get("loss"),
                    "rel_error": result.get("rel_error"),
                    "mass_drift_rms": result.get("metrics", {}).get(
                        "mass_drift_rms"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if result["status"] == "complete" else 2
    except BaseException as error:  # noqa: BLE001 - persist every failure
        failure = {
            **base,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
            "traceback": traceback.format_exc(limit=40),
            "completed_at": utc_now(),
        }
        try:
            failure.update({"git": git_state(), "hardware": hardware_metadata()})
        except Exception as metadata_error:  # noqa: BLE001
            failure["metadata_error"] = repr(metadata_error)[:1000]
        atomic_write_json(output, failure)
        print(
            json.dumps(
                {
                    "status": "failed",
                    "task_id": args.task,
                    "method": args.method,
                    "seed": args.seed,
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1


def archive_incomplete(output: Path, point_dir: Path, method: str) -> None:
    if not output.exists():
        return
    attempts = point_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    shutil.copy2(output, attempts / f"{method}.{stamp}.json")


def flatten_result(
    task: str,
    method: str,
    seed: int,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": task,
        "method": method,
        "seed": seed,
        "weights": json.dumps(FIXED_WEIGHTS[task]),
        "status": "missing" if result is None else result.get("status", "unknown"),
    }
    if result:
        for key in (
            "loss",
            "rel_error",
            "steps",
            "training_seconds",
            "evaluation_seconds",
            "ms_per_step",
            "peak_mb",
            "started_at",
            "completed_at",
        ):
            if key in result:
                row[key] = result[key]
        metrics = result.get("metrics", {})
        row["mass_drift_rms"] = metrics.get("mass_drift_rms")
        row["mass_drift_max_abs"] = metrics.get("mass_drift_max_abs")
        row["history_points"] = len(result.get("history", []))
        row["final_history_has_loss"] = bool(
            result.get("history") and "loss" in result["history"][-1]
        )
        row["final_history_has_rel_error"] = bool(
            result.get("history") and "rel_error" in result["history"][-1]
        )
    return row


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def build_summary(
    root: Path, tasks: tuple[str, ...], seeds: tuple[int, ...]
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for task in tasks:
        seed_rows: list[dict[str, Any]] = []
        for seed in seeds:
            pair: dict[str, Any] = {
                "task_id": task,
                "seed": seed,
                "weights": list(FIXED_WEIGHTS[task]),
            }
            results: dict[str, dict[str, Any] | None] = {}
            for method in METHODS:
                output = root / task / f"seed_{seed:03d}" / f"{method}.json"
                result = load_result(output)
                results[method] = result
                all_rows.append(flatten_result(task, method, seed, result))
                if result is not None and result.get("status") == "complete":
                    pair[f"{method}_loss"] = float(result["loss"])
                    pair[f"{method}_rel_error"] = float(result["rel_error"])
            if all(
                results.get(method)
                and results[method].get("status") == "complete"
                and math.isfinite(float(results[method].get("rel_error", math.inf)))
                for method in METHODS
            ):
                war_error = float(results["war"]["rel_error"])
                ad_error = float(results["real_sinh_autodiff"]["rel_error"])
                pair.update(
                    {
                        "geometric_mean": math.sqrt(war_error * ad_error),
                        "max_error": max(war_error, ad_error),
                        "mean_error": 0.5 * (war_error + ad_error),
                        "status": "complete",
                    }
                )
            else:
                pair["status"] = "incomplete"
            seed_rows.append(pair)

        complete_pairs = [row for row in seed_rows if row["status"] == "complete"]
        ranking_dir = root / task / "rankings"
        ranking_dir.mkdir(parents=True, exist_ok=True)
        ranking_specs = {
            "war": lambda row: row["war_rel_error"],
            "real_sinh_autodiff": lambda row: row["real_sinh_autodiff_rel_error"],
            "geometric_mean": lambda row: row["geometric_mean"],
            "minimax": lambda row: row["max_error"],
        }
        for name, key in ranking_specs.items():
            ranked = sorted(complete_pairs, key=lambda row: (key(row), row["seed"]))
            atomic_write_json(ranking_dir / f"{name}.json", ranked)
            write_csv(ranking_dir / f"{name}.csv", ranked)

        war_values = [row["war_rel_error"] for row in complete_pairs]
        ad_values = [row["real_sinh_autodiff_rel_error"] for row in complete_pairs]
        gm_values = [row["geometric_mean"] for row in complete_pairs]
        mm_values = [row["max_error"] for row in complete_pairs]
        summary = {
            "task_id": task,
            "weights": list(FIXED_WEIGHTS[task]),
            "expected_seed_count": len(seeds),
            "paired_complete_seed_count": len(complete_pairs),
            "war_complete": len(war_values),
            "real_sinh_autodiff_complete": len(ad_values),
            "war_rel_error": _distribution(war_values),
            "real_sinh_autodiff_rel_error": _distribution(ad_values),
            "shared_geometric_mean": _distribution(gm_values),
            "shared_minimax": _distribution(mm_values),
            "seed_metrics": seed_rows,
        }
        atomic_write_json(root / task / "summary.json", summary)
        task_summaries.append(summary)

    write_csv(root / "runs.csv", all_rows)
    atomic_write_json(root / "runs.json", all_rows)
    expected = len(tasks) * len(seeds) * len(METHODS)
    complete = sum(1 for row in all_rows if row["status"] == "complete")
    final = {
        "protocol_id": PROTOCOL_ID,
        "engine_protocol_id": ENGINE_PROTOCOL_ID,
        "updated_at": utc_now(),
        "tasks": list(tasks),
        "seeds": list(seeds),
        "expected_runs": expected,
        "complete_runs": complete,
        "all_complete": complete == expected,
        "task_summaries": task_summaries,
    }
    atomic_write_json(root / "summary.json", final)
    return final


def run_smoke(args: argparse.Namespace) -> int:
    tasks = selected_tasks(args.tasks)
    samples = sample_counts(args)
    conclusion = (
        args.conclusion or DEFAULT_ROOT / "SMOKE_CONCLUSION.json"
    ).resolve()
    cells: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="apolarity-cahn2d-formal-smoke-") as raw:
        raw_root = Path(raw)
        for task in tasks:
            for method in METHODS:
                cell_dir = raw_root / task
                output = cell_dir / f"{method}.json"
                log = cell_dir / f"{method}.log"
                output.parent.mkdir(parents=True, exist_ok=True)
                command = worker_command(
                    task,
                    method,
                    0,
                    args.eval_seed,
                    args.seconds,
                    output,
                    samples,
                    smoke=True,
                )
                started = time.perf_counter()
                with log.open("w") as handle:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        timeout=max(600.0, args.seconds * 30.0),
                        check=False,
                    )
                elapsed = time.perf_counter() - started
                result = load_result(output) or {}
                passed = completed.returncode == 0 and complete_formal_result(
                    output,
                    task=task,
                    method=method,
                    seed=0,
                    seconds=args.seconds,
                    samples=samples,
                )
                cell = {
                    "task_id": task,
                    "method": method,
                    "status": result.get("status", "missing"),
                    "passed": passed,
                    "returncode": completed.returncode,
                    "wall_seconds": elapsed,
                    "steps": result.get("steps"),
                    "loss": result.get("loss"),
                    "rel_error": result.get("rel_error"),
                    "peak_mb": result.get("peak_mb"),
                    "mass_drift_rms": result.get("metrics", {}).get(
                        "mass_drift_rms"
                    ),
                }
                if not passed:
                    cell["error_type"] = result.get("error_type")
                    cell["error"] = result.get("error")
                    try:
                        cell["log_tail"] = "\n".join(
                            log.read_text(errors="replace").splitlines()[-30:]
                        )[-6000:]
                    except OSError:
                        pass
                cells.append(cell)
                print(json.dumps(cell, sort_keys=True), flush=True)

    passed = all(bool(cell["passed"]) for cell in cells)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "engine_protocol_id": ENGINE_PROTOCOL_ID,
        "created_at": utc_now(),
        "passed": passed,
        "failure_count": sum(not bool(cell["passed"]) for cell in cells),
        "cell_count": len(cells),
        "seconds_per_cell": args.seconds,
        "sample_counts": samples,
        "fixed_weights": {
            task: list(FIXED_WEIGHTS[task]) for task in tasks
        },
        "methods": list(METHODS),
        "raw_artifacts_retained": False,
        "cells": cells,
        "git": git_state(),
        "hardware": hardware_metadata(),
        "conclusion": (
            "full-scale CUDA startup/finite-gradient/data-pipeline gate only; "
            "not a formal accuracy result"
        ),
    }
    atomic_write_json(conclusion, payload)
    return 0 if passed else 1


def run_orchestrator(args: argparse.Namespace) -> int:
    tasks = selected_tasks(args.tasks)
    seeds = seed_list(args.seeds)
    samples = sample_counts(args)
    root = (args.output_root or DEFAULT_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    expected_manifest = manifest(
        tasks, args.seconds, seeds, samples, smoke=False
    )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in (
            "protocol_id",
            "engine_protocol_id",
            "seconds_per_method_seed",
            "seeds",
            "tasks",
            "methods",
            "sample_counts",
        ):
            if existing.get(key) != expected_manifest.get(key):
                raise ValueError(
                    f"incompatible manifest field {key!r} at {manifest_path}"
                )
    else:
        atomic_write_json(manifest_path, expected_manifest)

    print(
        json.dumps(
            {
                "event": "FORMAL_START_OR_RESUME",
                "protocol_id": PROTOCOL_ID,
                "engine_protocol_id": ENGINE_PROTOCOL_ID,
                "python_executable": sys.executable,
                "tasks": list(tasks),
                "methods": list(METHODS),
                "weights": {
                    task: list(FIXED_WEIGHTS[task]) for task in tasks
                },
                "seeds": list(seeds),
                "seconds_per_method_seed": args.seconds,
                "sample_counts": samples,
                "git": expected_manifest["git"],
                "hardware": expected_manifest["hardware"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    total = len(tasks) * len(seeds) * len(METHODS)
    processed = 0
    attempted = 0
    failures = 0
    started = time.time()
    for task in tasks:
        task_dir = root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            task_dir / "manifest.json",
            {
                "protocol_id": PROTOCOL_ID,
                "engine_protocol_id": ENGINE_PROTOCOL_ID,
                "task_id": task,
                "weights": list(FIXED_WEIGHTS[task]),
                "weight_names": list(TASKS[task].weight_names),
                "methods": list(METHODS),
                "seeds": list(seeds),
                "seconds_per_method_seed": args.seconds,
                "sample_counts": samples,
            },
        )
        for seed in seeds:
            point_dir = task_dir / f"seed_{seed:03d}"
            point_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                point_dir / "config.json",
                {
                    "protocol_id": PROTOCOL_ID,
                    "engine_protocol_id": ENGINE_PROTOCOL_ID,
                    "task_id": task,
                    "weights": list(FIXED_WEIGHTS[task]),
                    "seed": seed,
                    "eval_seed": args.eval_seed,
                    "seconds": args.seconds,
                    "methods": list(METHODS),
                    "sample_counts": samples,
                    "serial_single_gpu": True,
                },
            )
            for method in METHODS:
                output = point_dir / f"{method}.json"
                log = point_dir / f"{method}.log"
                done_marker = point_dir / f"{method}.DONE"
                failed_marker = point_dir / f"{method}.FAILED"
                if args.resume and complete_formal_result(
                    output,
                    task=task,
                    method=method,
                    seed=seed,
                    seconds=args.seconds,
                    samples=samples,
                ):
                    processed += 1
                    atomic_write_text(done_marker, "resume-validated\n")
                    continue
                if output.exists():
                    archive_incomplete(output, point_dir, method)

                success = False
                last_returncode: int | None = None
                for attempt in range(1, args.retries + 2):
                    attempted += 1
                    command = worker_command(
                        task,
                        method,
                        seed,
                        args.eval_seed,
                        args.seconds,
                        output,
                        samples,
                        smoke=False,
                    )
                    with log.open("a") as handle:
                        handle.write(
                            f"\n# attempt={attempt} started_at={utc_now()}\n"
                        )
                        handle.flush()
                        completed = subprocess.run(
                            command,
                            cwd=ROOT,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            timeout=max(1800.0, args.seconds * 2.0 + 600.0),
                            check=False,
                        )
                    last_returncode = completed.returncode
                    success = completed.returncode == 0 and complete_formal_result(
                        output,
                        task=task,
                        method=method,
                        seed=seed,
                        seconds=args.seconds,
                        samples=samples,
                    )
                    if success:
                        atomic_write_text(
                            done_marker,
                            f"completed_at={utc_now()} attempt={attempt}\n",
                        )
                        break
                    failures += 1
                    if attempt <= args.retries:
                        archive_incomplete(output, point_dir, method)
                        print(
                            f"[retry] {task} seed={seed} {method} "
                            f"attempt={attempt}",
                            flush=True,
                        )
                if not success:
                    atomic_write_text(
                        failed_marker,
                        f"failed_at={utc_now()} returncode={last_returncode}\n",
                    )

                processed += 1
                elapsed = time.time() - started
                eta = elapsed / max(1, processed) * max(0, total - processed)
                result = load_result(output) or {}
                atomic_write_json(
                    root / "progress.json",
                    {
                        "protocol_id": PROTOCOL_ID,
                        "updated_at": utc_now(),
                        "total_runs": total,
                        "processed_runs": processed,
                        "attempted_runs": attempted,
                        "failures_seen_this_process": failures,
                        "current_task": task,
                        "current_seed": seed,
                        "current_method": method,
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": eta,
                        "last_status": result.get("status"),
                        "last_loss": result.get("loss"),
                        "last_rel_error": result.get("rel_error"),
                    },
                )
                print(
                    f"[run {processed}/{total}] {task} seed={seed} {method} "
                    f"status={result.get('status')} loss={result.get('loss')} "
                    f"rel_error={result.get('rel_error')}",
                    flush=True,
                )

    final = build_summary(root, tasks, seeds)
    final.update(
        {
            "attempted_runs": attempted,
            "failures_seen_this_process": failures,
            "completed_at": utc_now(),
        }
    )
    atomic_write_json(root / "summary.json", final)
    marker = (
        "FORMAL_COMPLETE" if final["all_complete"] else "FORMAL_INCOMPLETE"
    )
    atomic_write_text(root / marker, json.dumps(final, indent=2) + "\n")
    write_checksums(root)
    return 0 if final["all_complete"] else 1


def add_sample_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-int", type=int, default=FORMAL_SAMPLE_COUNTS["n_int"])
    parser.add_argument("--n-ic", type=int, default=FORMAL_SAMPLE_COUNTS["n_ic"])
    parser.add_argument("--n-bc", type=int, default=FORMAL_SAMPLE_COUNTS["n_bc"])
    parser.add_argument("--n-eval", type=int, default=FORMAL_SAMPLE_COUNTS["n_eval"])
    parser.add_argument(
        "--history-eval-n",
        type=int,
        default=FORMAL_SAMPLE_COUNTS["history_eval_n"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--task", choices=TASK_ORDER, required=True)
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    worker.add_argument("--weights", required=True)
    worker.add_argument("--seconds", type=float, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--smoke", action="store_true")
    add_sample_arguments(worker)
    worker.set_defaults(func=run_worker)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--tasks", default="all")
    smoke.add_argument("--seconds", type=float, default=3.0)
    smoke.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    smoke.add_argument("--conclusion", type=Path)
    add_sample_arguments(smoke)
    smoke.set_defaults(func=run_smoke)

    orchestrate = subparsers.add_parser("orchestrate")
    orchestrate.add_argument("--tasks", default="all")
    orchestrate.add_argument("--seconds", type=float, default=1200.0)
    orchestrate.add_argument("--seeds", type=int, default=5)
    orchestrate.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    orchestrate.add_argument("--output-root", type=Path)
    orchestrate.add_argument("--resume", action="store_true")
    orchestrate.add_argument("--retries", type=int, default=1)
    add_sample_arguments(orchestrate)
    orchestrate.set_defaults(func=run_orchestrator)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
