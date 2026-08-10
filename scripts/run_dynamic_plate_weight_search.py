#!/usr/bin/env python3
"""Search shared IC/BC weights for the 2D fourth-/sixth-order plates.

The exhaustive grid is

    (lambda_ic, lambda_bc) in {1e-3, ..., 1e3}^2,

with one fixed search seed and equal 60-second wall-clock budgets for WAR and
the real-tanh direct-autodiff baseline.  This wrapper reuses the audited
ranking/serialization helpers while retaining the gradient-clipped training
loop used by the original high-order candidate screen.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_high_order_candidate_screen as screen  # noqa: E402
import run_weight_search as runner  # noqa: E402

from experiments.dynamic_plate_weight_search import problem  # noqa: E402


DEFAULT_ROOT = (
    ROOT / "outputs" / "search" / "dynamic-plate-o4-o6-weight-search-v1"
)
MAX_RETRIES = 1


for module in (screen, runner):
    module.COMPLEX_DTYPE = problem.COMPLEX_DTYPE
    module.DEPTH = problem.DEPTH
    module.EVAL_SEED = problem.EVAL_SEED
    module.HIDDEN = problem.HIDDEN
    module.HISTORY_INTERVAL_SECONDS = problem.HISTORY_INTERVAL_SECONDS
    module.LEARNING_RATE = problem.LEARNING_RATE
    module.LEARNING_RATE_FINAL = problem.LEARNING_RATE_FINAL
    module.METHODS = problem.METHODS
    module.PROTOCOL_ID = problem.PROTOCOL_ID
    module.REAL_DTYPE = problem.REAL_DTYPE
    module.TASKS = problem.TASKS
    module.TASK_ORDER = problem.TASK_ORDER
    module.build_model = problem.build_model
    module.model_metadata = problem.model_metadata
    module.tensor_components_to_float = problem.tensor_components_to_float

screen.ENGINE_PROTOCOL_ID = problem.PROTOCOL_ID
screen.GRAD_CLIP = problem.GRAD_CLIP
screen.make_loss_bundle = problem.make_loss_bundle

runner.GRID_VALUES = problem.GRID_VALUES
runner.INIT_MODE = problem.INIT_MODE
runner.TRAIN_SEED = problem.TRAIN_SEED
runner.DEFAULT_RESULT_ROOT = DEFAULT_ROOT
runner.DEFAULT_ACTIVE_TASKS = ",".join(problem.TASK_ORDER)
runner.SearchTask = problem.PlateSearchTask
runner.build_search_model = problem.build_model


def _plate_train_one(
    task: problem.PlateSearchTask,
    method: str,
    weights: tuple[float, ...],
    *,
    seconds: float,
    smoke: bool,
    train_seed: int = problem.TRAIN_SEED,
    eval_seed: int = problem.EVAL_SEED,
) -> dict[str, Any]:
    weighted_task = problem.with_weights(task, weights)
    result = screen.train_one(
        weighted_task,
        method,
        stage="weight_search_smoke" if smoke else "weight_search",
        seconds=seconds,
        samples=dict(problem.SAMPLE_COUNTS),
        train_seed=train_seed,
        eval_seed=eval_seed,
        smoke=smoke,
    )
    result.update({
        "protocol_id": problem.PROTOCOL_ID,
        "search_protocol_id": problem.PROTOCOL_ID,
        "underlying_problem_protocol_id": problem.problem_protocol_id(
            weighted_task
        ),
        "search_sample_counts": dict(problem.SAMPLE_COUNTS),
    })
    return result


runner.train_one = _plate_train_one


_base_worker_command = runner.worker_command


def _worker_command(*args: Any, **kwargs: Any) -> list[str]:
    command = _base_worker_command(*args, **kwargs)
    command[1] = str(Path(__file__).resolve())
    return command


runner.worker_command = _worker_command


def _parameter_elements(input_dim: int) -> int:
    return (
        input_dim * problem.HIDDEN
        + problem.HIDDEN
        + (problem.DEPTH - 1)
        * (problem.HIDDEN * problem.HIDDEN + problem.HIDDEN)
        + problem.HIDDEN
        + 1
    )


def _root_manifest(
    tasks: Any, seconds: float, smoke: bool
) -> dict[str, Any]:
    task_list = list(tasks)
    candidate_count = sum(
        len(runner.candidate_vectors(task)) for task in task_list
    )
    return {
        "protocol_id": problem.PROTOCOL_ID,
        "created_at": runner.utc_now(),
        "smoke": smoke,
        "tasks": [
            {
                "task_id": task.task_id,
                "family": task.family,
                "spatial_dim": task.spatial_dim,
                "order": task.order,
                "weight_names": list(task.weight_names),
                "center_weights": list(task.center_weights),
                "candidate_count": len(runner.candidate_vectors(task)),
                "residual_scale": task.residual_scale,
                "underlying_problem_protocol_id": problem.problem_protocol_id(
                    task
                ),
                "uniqueness_basis": task.uniqueness,
            }
            for task in task_list
        ],
        "grid_values": list(problem.GRID_VALUES),
        "grid_labels": [
            runner.weight_label(value) for value in problem.GRID_VALUES
        ],
        "grid_type": "complete_ordered_cartesian_product",
        "candidate_count": candidate_count,
        "methods": list(problem.METHODS),
        "method_run_count": candidate_count * len(problem.METHODS),
        "seconds_per_task_weight_method": seconds,
        "nominal_training_seconds": (
            candidate_count * len(problem.METHODS) * seconds
        ),
        "train_seed": problem.TRAIN_SEED,
        "eval_seed": problem.EVAL_SEED,
        "sample_counts": dict(problem.SAMPLE_COUNTS),
        "architecture": {
            "shared": {
                "physical_input_dim": 3,
                "input": "affine-normalized raw (x,y,t)",
                "trigonometric_input_features": False,
                "frequency_initialization": False,
                "hidden": problem.HIDDEN,
                "depth": problem.DEPTH,
                "init_mode": problem.INIT_MODE,
                "literal_layer_shape_matched": True,
            },
            "war": {
                "representation": "native_complex",
                "activation": "sinh",
                "backend": "waring_complex_jet",
                "parameter_dtype": str(problem.COMPLEX_DTYPE),
                "parameter_elements": _parameter_elements(3),
                "real_dof": 2 * _parameter_elements(3),
            },
            "real_tanh_autodiff": {
                "representation": "real",
                "activation": "tanh",
                "backend": "direct_autodiff",
                "parameter_dtype": str(problem.REAL_DTYPE),
                "parameter_elements": _parameter_elements(3),
                "real_dof": _parameter_elements(3),
            },
        },
        "learning_rate": problem.LEARNING_RATE,
        "learning_rate_final": problem.LEARNING_RATE_FINAL,
        "lr_schedule": "wall_clock_cosine",
        "history_interval_seconds": problem.HISTORY_INTERVAL_SECONDS,
        "gradient_clip": problem.GRAD_CLIP,
        "method_order_policy": (
            "WAR first on even candidate indices; real AD first on odd indices"
        ),
        "retry_policy": {
            "max_retries_after_initial_attempt": MAX_RETRIES,
            "same_protocol_only": True,
            "preserve_attempt_json_and_log": True,
        },
        "selection_outputs": [
            "shared_minimax",
            "shared_geometric_mean",
            "war_specific",
            "real_tanh_autodiff_specific",
        ],
        "serial_single_gpu": True,
        "git": runner.git_state(),
        "hardware": runner.hardware_metadata(),
    }


runner.root_manifest = _root_manifest


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS", "run.pid"}
        and ".tmp." not in path.name
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(root)}"
        for path in paths
    ]
    runner.atomic_write_text(root / "SHA256SUMS", "\n".join(lines) + "\n")


runner.write_checksums = _write_checksums


def _same_weights(left: Any, right: tuple[float, ...]) -> bool:
    try:
        values = tuple(float(value) for value in left)
    except (TypeError, ValueError):
        return False
    return len(values) == len(right) and all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-14)
        for a, b in zip(values, right)
    )


def _cell_complete(
    output: Path,
    *,
    task: problem.PlateSearchTask,
    method: str,
    weights: tuple[float, ...],
    seconds: float,
) -> bool:
    result = runner.load_result(output)
    metadata = result.get("problem", {}) if result else {}
    return bool(
        result
        and result.get("protocol_id") == problem.PROTOCOL_ID
        and result.get("stage") == "weight_search"
        and result.get("status") == "complete"
        and result.get("task_id") == task.task_id
        and result.get("method") == method
        and _same_weights(result.get("weights"), weights)
        and math.isclose(
            float(result.get("budget_seconds", -1.0)),
            seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and int(result.get("train_seed", -1)) == problem.TRAIN_SEED
        and int(result.get("eval_seed", -1)) == problem.EVAL_SEED
        and math.isfinite(float(result.get("loss", math.inf)))
        and math.isfinite(float(result.get("rel_error", math.inf)))
        and all(
            int(metadata.get(name, -1)) == expected
            for name, expected in problem.SAMPLE_COUNTS.items()
        )
    )


def _archive_attempt(
    output: Path,
    log: Path,
    point_dir: Path,
    method: str,
    reason: str,
) -> None:
    if not output.exists() and not log.exists():
        return
    attempts = point_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if output.exists():
        shutil.copy2(output, attempts / f"{method}.{stamp}.json")
    if log.exists():
        shutil.copy2(log, attempts / f"{method}.{stamp}.log")
    runner.atomic_write_json(
        attempts / f"{method}.{stamp}.attempt.json",
        {"archived_at": runner.utc_now(), "method": method, "reason": reason},
    )


def _stable_manifest_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "protocol_id",
            "tasks",
            "grid_values",
            "methods",
            "seconds_per_task_weight_method",
            "train_seed",
            "eval_seed",
            "sample_counts",
            "architecture",
            "gradient_clip",
            "method_order_policy",
            "retry_policy",
        )
    } | {"git_sha": value.get("git", {}).get("sha")}


def _run_orchestrator(args: Any) -> int:
    tasks = runner.selected_tasks(args.tasks)
    root = (args.output_root or DEFAULT_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    expected_manifest = _root_manifest(tasks, args.seconds, False)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if _stable_manifest_fields(existing) != _stable_manifest_fields(
            expected_manifest
        ):
            raise ValueError(f"incompatible manifest at {manifest_path}")
    else:
        runner.atomic_write_json(manifest_path, expected_manifest)

    total = int(expected_manifest["method_run_count"])
    processed = 0
    completed = 0
    failures = 0
    attempted_subprocesses = 0
    timed_cells = 0
    started = time.time()

    for task in tasks:
        vectors = runner.candidate_vectors(task)
        task_dir = root / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        runner.atomic_write_json(task_dir / "manifest.json", {
            "protocol_id": problem.PROTOCOL_ID,
            "task_id": task.task_id,
            "underlying_problem_protocol_id": problem.problem_protocol_id(task),
            "weight_names": list(task.weight_names),
            "grid_values": list(problem.GRID_VALUES),
            "candidate_count": len(vectors),
            "methods": list(problem.METHODS),
            "expected_runs": len(vectors) * len(problem.METHODS),
            "seconds_per_run": args.seconds,
            "train_seed": problem.TRAIN_SEED,
            "eval_seed": problem.EVAL_SEED,
            "sample_counts": dict(problem.SAMPLE_COUNTS),
        })

        for index, weights in enumerate(vectors):
            candidate = runner.candidate_record(task, index, weights)
            method_order = (
                problem.METHODS
                if index % 2 == 0
                else tuple(reversed(problem.METHODS))
            )
            candidate["method_execution_order"] = list(method_order)
            point_dir = task_dir / "points" / candidate["candidate_id"]
            point_dir.mkdir(parents=True, exist_ok=True)
            runner.atomic_write_json(point_dir / "candidate.json", candidate)

            for method in method_order:
                output = point_dir / f"{method}.json"
                log = point_dir / f"{method}.log"
                if args.resume and _cell_complete(
                    output,
                    task=task,
                    method=method,
                    weights=weights,
                    seconds=args.seconds,
                ):
                    processed += 1
                    completed += 1
                    continue

                if output.exists() or log.exists():
                    _archive_attempt(
                        output, log, point_dir, method, "preexisting_incomplete"
                    )

                success = False
                for attempt in range(1, MAX_RETRIES + 2):
                    attempted_subprocesses += 1
                    elapsed = time.time() - started
                    remaining = total - processed
                    eta = (
                        elapsed / timed_cells * remaining
                        if timed_cells > 0
                        else total * (args.seconds + 15.0)
                    )
                    progress = {
                        "protocol_id": problem.PROTOCOL_ID,
                        "status": "running",
                        "updated_at": runner.utc_now(),
                        "processed_runs": processed,
                        "complete_runs": completed,
                        "total_runs": total,
                        "failures": failures,
                        "attempted_subprocesses": attempted_subprocesses,
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": eta,
                        "current": {
                            "task": task.task_id,
                            "candidate_id": candidate["candidate_id"],
                            "weights": list(weights),
                            "method": method,
                            "attempt": attempt,
                        },
                    }
                    runner.atomic_write_json(root / "progress.json", progress)
                    print(json.dumps({"event": "LAUNCH_CELL", **progress}, sort_keys=True), flush=True)

                    command = _worker_command(
                        task,
                        method,
                        weights,
                        output,
                        args.seconds,
                        smoke=False,
                    )
                    with log.open("w") as handle:
                        handle.write(
                            f"# attempt={attempt} started_at={runner.utc_now()}\n"
                        )
                        handle.flush()
                        completed_process = subprocess.run(
                            command,
                            cwd=ROOT,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            timeout=max(900.0, args.seconds * 15.0),
                            check=False,
                        )
                    success = completed_process.returncode == 0 and _cell_complete(
                        output,
                        task=task,
                        method=method,
                        weights=weights,
                        seconds=args.seconds,
                    )
                    if success:
                        break
                    _archive_attempt(
                        output,
                        log,
                        point_dir,
                        method,
                        f"failed_attempt_{attempt}",
                    )

                processed += 1
                timed_cells += 1
                if success:
                    completed += 1
                else:
                    failures += 1
                task_summary = runner.summarize_task(task, task_dir)
                elapsed = time.time() - started
                eta = (
                    elapsed / timed_cells * (total - processed)
                    if timed_cells > 0
                    else 0.0
                )
                result = runner.load_result(output) or {}
                progress = {
                    "protocol_id": problem.PROTOCOL_ID,
                    "status": "running" if processed < total else "summarizing",
                    "updated_at": runner.utc_now(),
                    "processed_runs": processed,
                    "complete_runs": completed,
                    "total_runs": total,
                    "failures": failures,
                    "attempted_subprocesses": attempted_subprocesses,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": eta,
                    "last_completed": {
                        "task": task.task_id,
                        "candidate_id": candidate["candidate_id"],
                        "weights": list(weights),
                        "method": method,
                        "valid": success,
                        "loss": result.get("loss"),
                        "rel_error": result.get("rel_error"),
                        "steps": result.get("steps"),
                        "ms_per_step": result.get("ms_per_step"),
                        "peak_mb": result.get("peak_mb"),
                    },
                    "task_summary": task_summary,
                }
                runner.atomic_write_json(root / "progress.json", progress)
                print(json.dumps({"event": "CELL_RETURN", **progress}, sort_keys=True), flush=True)

        runner.summarize_task(task, task_dir)

    summaries = [
        runner.summarize_task(task, root / task.task_id) for task in tasks
    ]
    complete_runs = sum(int(value["complete_runs"]) for value in summaries)
    paired = sum(int(value["paired_complete_candidates"]) for value in summaries)
    all_complete = (
        complete_runs == total
        and failures == 0
        and paired == sum(len(runner.candidate_vectors(task)) for task in tasks)
    )
    final = {
        "protocol_id": problem.PROTOCOL_ID,
        "completed_at": runner.utc_now(),
        "expected_runs": total,
        "complete_runs": complete_runs,
        "failed_runs": failures,
        "paired_complete_candidates": paired,
        "all_complete": all_complete,
        "task_summaries": summaries,
    }
    runner.atomic_write_json(root / "summary.json", final)
    runner.atomic_write_json(root / "progress.json", {
        "protocol_id": problem.PROTOCOL_ID,
        "status": "complete" if all_complete else "incomplete",
        "updated_at": runner.utc_now(),
        "processed_runs": processed,
        "complete_runs": complete_runs,
        "total_runs": total,
        "failures": failures,
        "attempted_subprocesses": attempted_subprocesses,
        "elapsed_seconds": time.time() - started,
        "estimated_remaining_seconds": 0.0,
    })
    marker = "SEARCH_COMPLETE" if all_complete else "SEARCH_INCOMPLETE"
    runner.atomic_write_text(root / marker, json.dumps(final, indent=2) + "\n")
    print(json.dumps({"event": "ORCHESTRATOR_FINAL", **final}, sort_keys=True), flush=True)
    _write_checksums(root)
    return 0 if all_complete else 1


runner.run_orchestrator = _run_orchestrator


_base_run_smoke = runner.run_smoke


def _run_smoke(args: Any) -> int:
    result = _base_run_smoke(args)
    conclusion = (args.conclusion or DEFAULT_ROOT / "SMOKE_CONCLUSION.json").resolve()
    payload = json.loads(conclusion.read_text())
    payload.update({
        "git": runner.git_state(),
        "hardware": runner.hardware_metadata(),
        "sample_counts": dict(problem.SAMPLE_COUNTS),
        "train_seed": problem.TRAIN_SEED,
        "eval_seed": problem.EVAL_SEED,
        "interpretation": (
            "full search-scale startup/finite-gradient/data-pipeline gate only; "
            "raw smoke artifacts were removed and no smoke metric enters rankings"
        ),
    })
    runner.atomic_write_json(conclusion, payload)
    return result


runner.run_smoke = _run_smoke


if __name__ == "__main__":
    raise SystemExit(runner.main())
