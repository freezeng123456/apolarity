#!/usr/bin/env python3
"""Run and resume the exhaustive WAR/real-autodiff loss-weight search.

Examples
--------
Smoke every task/method without entering the ranked grid::

    python scripts/run_weight_search.py smoke --seconds 5

Run the complete 497-vector grid (994 method runs) and resume safely::

    python scripts/run_weight_search.py orchestrate --seconds 60 --resume

Rebuild rankings without training::

    python scripts/run_weight_search.py summarize
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
SRC = ROOT / "src"
for path in (str(COMMON), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from weight_search import (
    COMPLEX_DTYPE,
    DEPTH,
    EVAL_SEED,
    GRID_VALUES,
    HIDDEN,
    HISTORY_INTERVAL_SECONDS,
    INIT_MODE,
    LEARNING_RATE,
    LEARNING_RATE_FINAL,
    METHODS,
    PROTOCOL_ID,
    REAL_DTYPE,
    TASKS,
    TRAIN_SEED,
    SearchTask,
    build_search_model,
    make_loss_bundle,
    model_metadata,
    tensor_components_to_float,
)

DEFAULT_RESULT_ROOT = ROOT / "experiments" / "results" / PROTOCOL_ID
TASK_ORDER = (
    "poly_d2_o2",
    "poly_d2_o4",
    "cahn_hilliard_o4",
    "cahn_hilliard_o6",
    "poly_d2_o6",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


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


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "sha": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def hardware_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        metadata.update({
            "cuda_device_index": index,
            "cuda_device_name": props.name,
            "cuda_total_memory_bytes": props.total_memory,
        })
    return metadata


def candidate_vectors(task: SearchTask) -> tuple[tuple[float, ...], ...]:
    return tuple(itertools.product(GRID_VALUES, repeat=task.weight_count))


def weight_label(value: float) -> str:
    exponent = round(math.log10(value))
    return f"1e{exponent:+d}"


def candidate_record(task: SearchTask, index: int, weights: tuple[float, ...]) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "candidate_id": f"point_{index:03d}",
        "weights": list(weights),
        "weight_labels": [weight_label(value) for value in weights],
        "weight_map": dict(zip(task.weight_names, weights)),
    }


def finite_mapping(values: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in values.values())


def train_one(
    task: SearchTask,
    method: str,
    weights: tuple[float, ...],
    *,
    seconds: float,
    smoke: bool,
    train_seed: int = TRAIN_SEED,
    eval_seed: int = EVAL_SEED,
) -> dict[str, Any]:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("weight-search training requires CUDA")
    run_started_at = utc_now()
    device = torch.device("cuda")
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    model, dtype, backend = build_search_model(task, method, device)
    bundle = make_loss_bundle(
        task,
        model,
        dtype,
        backend,
        weights,
        device,
        smoke=smoke,
        train_seed=train_seed,
        eval_seed=eval_seed,
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=LEARNING_RATE,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    initial_loss, initial_components = bundle.loss_fn()
    initial_component_values = tensor_components_to_float(initial_components)
    initial_error = float(bundle.history_eval_fn())
    if not finite_mapping(initial_component_values) or not math.isfinite(initial_error):
        raise FloatingPointError("non-finite initial loss or relative error")
    history: list[dict[str, Any]] = [{
        "elapsed_seconds": 0.0,
        "step": 0,
        "learning_rate": LEARNING_RATE,
        "rel_error": initial_error,
        **initial_component_values,
    }]
    del initial_loss, initial_components

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    eval_seconds = 0.0
    next_history = HISTORY_INTERVAL_SECONDS
    steps = 0
    latest_component_values = initial_component_values
    nan_hit = False

    while True:
        elapsed_before = time.perf_counter() - started - eval_seconds
        if elapsed_before >= seconds:
            break
        fraction = min(1.0, elapsed_before / seconds)
        current_lr = LEARNING_RATE_FINAL + 0.5 * (
            LEARNING_RATE - LEARNING_RATE_FINAL
        ) * (1.0 + math.cos(math.pi * fraction))
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        loss, components = bundle.loss_fn()
        if not bool(torch.isfinite(loss).item()):
            nan_hit = True
            latest_component_values = tensor_components_to_float(components)
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        steps += 1
        torch.cuda.synchronize(device)
        elapsed_after = time.perf_counter() - started - eval_seconds

        if elapsed_after >= next_history:
            latest_component_values = tensor_components_to_float(components)
            evaluation_started = time.perf_counter()
            error = float(bundle.history_eval_fn())
            torch.cuda.synchronize(device)
            evaluation_duration = time.perf_counter() - evaluation_started
            eval_seconds += evaluation_duration
            elapsed_after = time.perf_counter() - started - eval_seconds
            history.append({
                "elapsed_seconds": round(elapsed_after, 6),
                "step": steps,
                "learning_rate": current_lr,
                "rel_error": error,
                **latest_component_values,
            })
            while next_history <= elapsed_after:
                next_history += HISTORY_INTERVAL_SECONDS
        del loss, components

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started - eval_seconds
    final_loss, final_components = bundle.loss_fn()
    final_component_values = tensor_components_to_float(final_components)
    final_error = float(bundle.eval_fn())
    final_lr = float(optimizer.param_groups[0]["lr"])
    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20
    if history[-1]["step"] != steps:
        history.append({
            "elapsed_seconds": round(training_seconds, 6),
            "step": steps,
            "learning_rate": final_lr,
            "rel_error": final_error,
            **final_component_values,
        })

    status = "complete"
    if nan_hit or not math.isfinite(final_error) or not finite_mapping(final_component_values):
        status = "failed_nonfinite"
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "task_id": task.task_id,
        "family": task.family,
        "order": task.order,
        "method": method,
        "weights": list(weights),
        "weight_map": dict(zip(task.weight_names, weights)),
        "budget_seconds": seconds,
        "smoke": smoke,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "started_at": run_started_at,
        "steps": steps,
        "training_seconds": training_seconds,
        "evaluation_seconds": eval_seconds,
        "ms_per_step": 1000.0 * training_seconds / max(1, steps),
        "peak_mb": peak_mb,
        "learning_rate_initial": LEARNING_RATE,
        "learning_rate_final_target": LEARNING_RATE_FINAL,
        "learning_rate_last": final_lr,
        "rel_error": final_error,
        "L2_err": final_error,
        "loss": float(final_loss.detach().item()),
        "components": final_component_values,
        "initial_rel_error": initial_error,
        "initial_components": initial_component_values,
        "history": history,
        "history_schema": [
            "elapsed_seconds",
            "step",
            "learning_rate",
            "rel_error",
            "loss",
            "loss_components",
        ],
        "model": model_metadata(model, method),
        "problem": bundle.metadata,
        "completed_at": utc_now(),
    }
    del final_loss, final_components, optimizer, model, bundle
    torch.cuda.empty_cache()
    return result


def result_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("protocol_id") == PROTOCOL_ID
        and result.get("status") == "complete"
        and math.isfinite(float(result.get("rel_error", math.inf)))
    )


def run_worker(args: argparse.Namespace) -> int:
    task = TASKS[args.task]
    weights = tuple(float(value) for value in args.weights.split(",") if value)
    output = args.output.resolve()
    base: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "status": "running",
        "task_id": task.task_id,
        "method": args.method,
        "weights": list(weights),
        "budget_seconds": args.seconds,
        "smoke": args.smoke,
        "started_at": utc_now(),
    }
    try:
        result = train_one(
            task,
            args.method,
            weights,
            seconds=args.seconds,
            smoke=args.smoke,
        )
        result.update({
            "git": git_state(),
            "hardware": hardware_metadata(),
        })
        atomic_write_json(output, result)
        print(json.dumps({
            "status": result["status"],
            "task_id": task.task_id,
            "method": args.method,
            "weights": list(weights),
            "steps": result["steps"],
            "loss": result["loss"],
            "rel_error": result["rel_error"],
        }, sort_keys=True), flush=True)
        return 0 if result["status"] == "complete" else 2
    except BaseException as error:  # noqa: BLE001 - persist all worker failures
        failure = {
            **base,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
            "traceback": traceback.format_exc(limit=30),
            "completed_at": utc_now(),
        }
        try:
            failure.update({"git": git_state(), "hardware": hardware_metadata()})
        except Exception as metadata_error:  # noqa: BLE001
            # Preserve the primary training failure even if the optional
            # provenance probe also fails (for example outside a Git clone).
            failure["metadata_error"] = repr(metadata_error)[:1000]
        atomic_write_json(output, failure)
        print(json.dumps({
            "status": "failed",
            "task_id": task.task_id,
            "method": args.method,
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        }, sort_keys=True), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 1


def load_result(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in result.items()
        if key not in {"history", "components", "initial_components", "model", "problem", "hardware", "git"}
    }
    row["weights"] = json.dumps(result.get("weights", []))
    row["weight_map"] = json.dumps(result.get("weight_map", {}), sort_keys=True)
    for prefix in ("components", "initial_components", "model", "problem"):
        for key, value in result.get(prefix, {}).items():
            row[f"{prefix}.{key}"] = json.dumps(value) if isinstance(value, (list, dict)) else value
    return row


def summarize_task(task: SearchTask, task_dir: Path) -> dict[str, Any]:
    vectors = candidate_vectors(task)
    matrix: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    complete_runs = 0
    failed_runs = 0
    for index, weights in enumerate(vectors):
        candidate = candidate_record(task, index, weights)
        point_dir = task_dir / "points" / candidate["candidate_id"]
        results = {method: load_result(point_dir / f"{method}.json") for method in METHODS}
        row: dict[str, Any] = {**candidate}
        row["weights"] = json.dumps(row["weights"])
        row["weight_labels"] = json.dumps(row["weight_labels"])
        row["weight_map"] = json.dumps(row["weight_map"], sort_keys=True)
        for method, result in results.items():
            status = "missing" if result is None else str(result.get("status", "unknown"))
            row[f"{method}_status"] = status
            if status == "complete":
                error = float(result["rel_error"])
                row[f"{method}_rel_error"] = error
                row[f"{method}_steps"] = int(result["steps"])
                row[f"{method}_loss"] = float(result["loss"])
                complete_runs += 1
            elif status not in {"missing", "running"}:
                failed_runs += 1
        matrix.append(row)
        if all(
            results[method] is not None
            and results[method].get("status") == "complete"
            and math.isfinite(float(results[method].get("rel_error", math.inf)))
            for method in METHODS
        ):
            war = float(results["war"]["rel_error"])
            autodiff = float(results["real_tanh_autodiff"]["rel_error"])
            paired.append({
                **candidate,
                "weights": list(weights),
                "war_rel_error": war,
                "real_tanh_autodiff_rel_error": autodiff,
                "geometric_mean": math.sqrt(war * autodiff),
                "max_error": max(war, autodiff),
                "mean_error": 0.5 * (war + autodiff),
                "weight_sum": sum(weights),
            })

    ranking_specs = {
        "ranking_shared_minimax": lambda row: (
            row["max_error"], row["geometric_mean"], row["weight_sum"]
        ),
        "ranking_shared_geomean": lambda row: (
            row["geometric_mean"], row["max_error"], row["weight_sum"]
        ),
        "ranking_war": lambda row: (
            row["war_rel_error"], row["real_tanh_autodiff_rel_error"], row["weight_sum"]
        ),
        "ranking_real_tanh_autodiff": lambda row: (
            row["real_tanh_autodiff_rel_error"], row["war_rel_error"], row["weight_sum"]
        ),
    }
    ranking_dir = task_dir / "rankings"
    ranking_dir.mkdir(parents=True, exist_ok=True)
    for name, key in ranking_specs.items():
        ranked = sorted(paired, key=key)
        atomic_write_json(ranking_dir / f"{name}.json", ranked)
        write_csv(
            ranking_dir / f"{name}.csv",
            [{**row, "weights": json.dumps(row["weights"]), "weight_labels": json.dumps(row["weight_labels"]), "weight_map": json.dumps(row["weight_map"], sort_keys=True)} for row in ranked],
        )
    write_csv(task_dir / "run_matrix.csv", matrix)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "task_id": task.task_id,
        "expected_candidates": len(vectors),
        "expected_runs": len(vectors) * len(METHODS),
        "complete_runs": complete_runs,
        "failed_runs": failed_runs,
        "paired_complete_candidates": len(paired),
        "missing_or_running_runs": len(vectors) * len(METHODS) - complete_runs - failed_runs,
        "best_shared_minimax": min(paired, key=ranking_specs["ranking_shared_minimax"]) if paired else None,
        "best_shared_geomean": min(paired, key=ranking_specs["ranking_shared_geomean"]) if paired else None,
        "best_war": min(paired, key=ranking_specs["ranking_war"]) if paired else None,
        "best_real_tanh_autodiff": min(paired, key=ranking_specs["ranking_real_tanh_autodiff"]) if paired else None,
        "updated_at": utc_now(),
    }
    atomic_write_json(task_dir / "summary.json", summary)
    return summary


def write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and ".tmp." not in path.name
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root)}")
    atomic_write_text(root / "SHA256SUMS", "\n".join(lines) + "\n")


def root_manifest(tasks: Iterable[SearchTask], seconds: float, smoke: bool) -> dict[str, Any]:
    task_list = list(tasks)
    candidate_count = sum(len(candidate_vectors(task)) for task in task_list)
    return {
        "protocol_id": PROTOCOL_ID,
        "created_at": utc_now(),
        "smoke": smoke,
        "grid_values": list(GRID_VALUES),
        "grid_labels": [weight_label(value) for value in GRID_VALUES],
        "grid_type": "complete_ordered_cartesian_product",
        "methods": list(METHODS),
        "architecture": {
        "war": {
                "representation": "native_complex",
                "activation": "sinh",
                "backend": "waring_complex_jet",
                "hidden": HIDDEN,
                "depth": DEPTH,
                "init_mode": INIT_MODE,
            "frequency_initialization": "disabled",
            "parameter_dtype": str(COMPLEX_DTYPE),
        },
            "real_tanh_autodiff": {
                "representation": "real",
                "activation": "tanh",
                "backend": "direct_autodiff",
                "hidden": HIDDEN,
                "depth": DEPTH,
                "init_mode": INIT_MODE,
            "frequency_initialization": "disabled",
            "parameter_dtype": str(REAL_DTYPE),
        },
        },
        "tasks": [{
            "task_id": task.task_id,
            "family": task.family,
            "order": task.order,
            "weight_names": list(task.weight_names),
            "candidate_count": len(candidate_vectors(task)),
        } for task in task_list],
        "seconds_per_task_weight_method": seconds,
        "candidate_count": candidate_count,
        "method_run_count": candidate_count * len(METHODS),
        "nominal_training_seconds": candidate_count * len(METHODS) * seconds,
        "train_seed": TRAIN_SEED,
        "eval_seed": EVAL_SEED,
        "input_features": {
            "poly": "raw_coordinates",
            "cahn_hilliard": "periodic_embedding_shared_by_methods",
        },
        "learning_rate": LEARNING_RATE,
        "learning_rate_final": LEARNING_RATE_FINAL,
        "lr_schedule": "wall_clock_cosine",
        "history_interval_seconds": HISTORY_INTERVAL_SECONDS,
        "selection_outputs": [
            "shared_minimax",
            "shared_geometric_mean",
            "war_specific",
            "real_tanh_autodiff_specific",
        ],
        "git": git_state(),
        "hardware": hardware_metadata(),
    }


def selected_tasks(text: str) -> tuple[SearchTask, ...]:
    names = TASK_ORDER if text == "all" else tuple(part for part in text.split(",") if part)
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    return tuple(TASKS[name] for name in names)


def worker_command(
    task: SearchTask,
    method: str,
    weights: tuple[float, ...],
    output: Path,
    seconds: float,
    *,
    smoke: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--task",
        task.task_id,
        "--method",
        method,
        "--weights",
        ",".join(f"{value:.16g}" for value in weights),
        "--seconds",
        str(seconds),
        "--output",
        str(output),
    ]
    if smoke:
        command.append("--smoke")
    return command


def run_smoke(args: argparse.Namespace) -> int:
    tasks = selected_tasks(args.tasks)
    root = (args.output_root or DEFAULT_RESULT_ROOT / "_smoke").resolve()
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "manifest.json", root_manifest(tasks, args.seconds, True))
    failures = 0
    for task in tasks:
        for method in METHODS:
            output = root / task.task_id / f"{method}.json"
            log = root / task.task_id / f"{method}.log"
            output.parent.mkdir(parents=True, exist_ok=True)
            command = worker_command(
                task, method, task.center_weights, output, args.seconds, smoke=True
            )
            print(f"[smoke] {task.task_id} {method} weights={task.center_weights}", flush=True)
            with log.open("w") as handle:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=max(300.0, args.seconds * 20.0),
                    check=False,
                )
            if completed.returncode != 0 or not result_complete(output):
                failures += 1
                print(f"[smoke-failed] see {log}", flush=True)
            else:
                result = load_result(output) or {}
                print(
                    f"[smoke-ok] steps={result.get('steps')} "
                    f"rel_error={result.get('rel_error'):.6g}",
                    flush=True,
                )
    write_checksums(root)
    marker = "SMOKE_COMPLETE" if failures == 0 else "SMOKE_COMPLETE_WITH_FAILURES"
    atomic_write_text(root / marker, f"failures={failures}\n")
    return 0 if failures == 0 else 1


def run_orchestrator(args: argparse.Namespace) -> int:
    tasks = selected_tasks(args.tasks)
    root = (args.output_root or DEFAULT_RESULT_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"incompatible manifest at {manifest_path}")
        if float(existing.get("seconds_per_task_weight_method")) != args.seconds:
            raise ValueError("cannot resume with a different seconds-per-run budget")
    else:
        atomic_write_json(manifest_path, root_manifest(tasks, args.seconds, False))

    total_runs = sum(len(candidate_vectors(task)) for task in tasks) * len(METHODS)
    started = time.time()
    attempted = 0
    skipped = 0
    failures = 0
    for task in tasks:
        vectors = candidate_vectors(task)
        task_dir = root / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        task_manifest = {
            "protocol_id": PROTOCOL_ID,
            "task_id": task.task_id,
            "family": task.family,
            "order": task.order,
            "weight_names": list(task.weight_names),
            "grid_values": list(GRID_VALUES),
            "candidate_count": len(vectors),
            "method_count": len(METHODS),
            "expected_runs": len(vectors) * len(METHODS),
            "seconds_per_run": args.seconds,
        }
        atomic_write_json(task_dir / "manifest.json", task_manifest)
        for index, weights in enumerate(vectors):
            candidate = candidate_record(task, index, weights)
            point_dir = task_dir / "points" / candidate["candidate_id"]
            point_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(point_dir / "candidate.json", candidate)
            for method in METHODS:
                output = point_dir / f"{method}.json"
                log = point_dir / f"{method}.log"
                if args.resume and result_complete(output):
                    skipped += 1
                    continue
                attempted += 1
                command = worker_command(
                    task, method, weights, output, args.seconds, smoke=False
                )
                print(
                    f"[run {attempted + skipped}/{total_runs}] {task.task_id} "
                    f"{candidate['candidate_id']} {method} weights={weights}",
                    flush=True,
                )
                try:
                    with log.open("a" if log.exists() else "w") as handle:
                        completed = subprocess.run(
                            command,
                            cwd=ROOT,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            timeout=max(900.0, args.seconds * 10.0),
                            check=False,
                        )
                    if completed.returncode != 0 or not result_complete(output):
                        failures += 1
                        print(f"[failed] see {log}", flush=True)
                except subprocess.TimeoutExpired as error:
                    failures += 1
                    atomic_write_json(output, {
                        "protocol_id": PROTOCOL_ID,
                        "status": "failed_timeout",
                        "task_id": task.task_id,
                        "method": method,
                        "weights": list(weights),
                        "timeout_seconds": error.timeout,
                        "completed_at": utc_now(),
                    })
                    print(f"[timeout] see {log}", flush=True)

                summary = summarize_task(task, task_dir)
                elapsed = time.time() - started
                processed = attempted + skipped
                eta = elapsed / max(1, attempted) * max(0, total_runs - processed)
                atomic_write_json(root / "progress.json", {
                    "protocol_id": PROTOCOL_ID,
                    "updated_at": utc_now(),
                    "total_runs": total_runs,
                    "processed_runs": processed,
                    "attempted_runs": attempted,
                    "resumed_complete_runs": skipped,
                    "failures_seen_this_process": failures,
                    "current_task": task.task_id,
                    "current_candidate_id": candidate["candidate_id"],
                    "current_method": method,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": eta,
                    "task_summary": summary,
                })
        summarize_task(task, task_dir)

    summaries = [summarize_task(task, root / task.task_id) for task in tasks]
    complete = sum(int(summary["complete_runs"]) for summary in summaries)
    recorded_failed = sum(int(summary["failed_runs"]) for summary in summaries)
    final = {
        "protocol_id": PROTOCOL_ID,
        "completed_at": utc_now(),
        "expected_runs": total_runs,
        "complete_runs": complete,
        "failed_runs": recorded_failed,
        "all_complete": complete == total_runs,
        "task_summaries": summaries,
    }
    atomic_write_json(root / "summary.json", final)
    write_checksums(root)
    marker = "SEARCH_COMPLETE" if complete == total_runs else "SEARCH_COMPLETE_WITH_FAILURES"
    atomic_write_text(root / marker, json.dumps(final, indent=2) + "\n")
    return 0 if complete == total_runs else 1


def run_summarize(args: argparse.Namespace) -> int:
    root = (args.output_root or DEFAULT_RESULT_ROOT).resolve()
    tasks = selected_tasks(args.tasks)
    summaries = [summarize_task(task, root / task.task_id) for task in tasks]
    atomic_write_json(root / "summary.json", {
        "protocol_id": PROTOCOL_ID,
        "updated_at": utc_now(),
        "task_summaries": summaries,
    })
    write_checksums(root)
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker", help="run one task/weight/method cell")
    worker.add_argument("--task", choices=TASK_ORDER, required=True)
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--weights", required=True)
    worker.add_argument("--seconds", type=float, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--smoke", action="store_true")
    worker.set_defaults(func=run_worker)

    smoke = subparsers.add_parser("smoke", help="smoke all selected tasks and methods")
    smoke.add_argument("--tasks", default="all")
    smoke.add_argument("--seconds", type=float, default=5.0)
    smoke.add_argument("--output-root", type=Path)
    smoke.set_defaults(func=run_smoke)

    orchestrate = subparsers.add_parser("orchestrate", help="run the exhaustive grid")
    orchestrate.add_argument("--tasks", default="all")
    orchestrate.add_argument("--seconds", type=float, default=60.0)
    orchestrate.add_argument("--output-root", type=Path)
    orchestrate.add_argument("--resume", action="store_true")
    orchestrate.set_defaults(func=run_orchestrator)

    summarize = subparsers.add_parser("summarize", help="rebuild rankings from raw results")
    summarize.add_argument("--tasks", default="all")
    summarize.add_argument("--output-root", type=Path)
    summarize.set_defaults(func=run_summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
