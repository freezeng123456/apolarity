#!/usr/bin/env python3
"""Smoke, run, resume, and summarize a two-weight PDE grid.

The default problem is the 2D natural-boundary Cahn--Hilliard benchmark.  A
thin wrapper may set ``APOLARITY_PROBLEM_FAMILY`` to select another problem
module implementing the same audited runner contract.  Every shared
``(lambda_ic, lambda_bc)`` vector is trained once with WAR and once with the
problem's architecture-matched real-autodiff baseline.
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
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
PROBLEM_FAMILY = os.environ.get(
    "APOLARITY_PROBLEM_FAMILY", "cahn_hilliard_2d"
)
PROBLEM_DIR = ROOT / "experiments" / PROBLEM_FAMILY
SRC = ROOT / "src"
for path in (str(COMMON), str(PROBLEM_DIR), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import problem as _problem  # noqa: E402
from problem import (  # noqa: E402
    COMPLEX_DTYPE,
    DEPTH,
    EVAL_SEED,
    GRID_VALUES,
    HIDDEN,
    HISTORY_INTERVAL_SECONDS,
    LEARNING_RATE,
    LEARNING_RATE_FINAL,
    METHODS,
    PROTOCOL_ID,
    REAL_DTYPE,
    TASKS,
    TRAIN_SEED,
    Cahn2DTask,
    build_model,
    make_loss_bundle,
    model_metadata,
    tensor_components_to_float,
)


DEFAULT_ROOT = ROOT / "outputs" / PROTOCOL_ID
TASK_ORDER = tuple(getattr(_problem, "TASK_ORDER", tuple(TASKS)))
RUNNER_FAMILY_NAME = str(
    getattr(_problem, "RUNNER_FAMILY_NAME", PROBLEM_FAMILY)
)
BASELINE_METHOD = METHODS[1]
GRAD_CLIP = 10.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
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
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
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
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        metadata.update({
            "cuda_device_index": device_index,
            "cuda_device_name": properties.name,
            "cuda_total_memory_bytes": properties.total_memory,
        })
    return metadata


def candidate_vectors(task: Cahn2DTask) -> tuple[tuple[float, ...], ...]:
    return tuple(itertools.product(GRID_VALUES, repeat=task.weight_count))


def weight_label(value: float) -> str:
    return f"1e{round(math.log10(value)):+d}"


def selected_tasks(value: str) -> tuple[str, ...]:
    names = TASK_ORDER if value == "all" else tuple(
        part.strip() for part in value.split(",") if part.strip()
    )
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    return names


def finite_mapping(values: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in values.values())


def _metrics_finite(metrics: dict[str, object]) -> bool:
    for value in metrics.values():
        if isinstance(value, dict):
            if not _metrics_finite(value):
                return False
        elif isinstance(value, (float, int)) and not math.isfinite(float(value)):
            return False
    return True


def train_one(
    task: Cahn2DTask,
    method: str,
    weights: tuple[float, ...],
    *,
    seconds: float,
    smoke: bool,
    n_int: int | None,
    n_ic: int | None,
    n_bc: int | None,
    n_eval: int | None,
    history_eval_n: int | None,
    train_seed: int,
    eval_seed: int,
) -> dict[str, Any]:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{RUNNER_FAMILY_NAME} training requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    run_started_at = utc_now()
    model, dtype, backend = build_model(task, method, device)
    bundle = make_loss_bundle(
        task,
        model,
        dtype,
        backend,
        weights,
        device,
        smoke=smoke,
        n_int=n_int,
        n_ic=n_ic,
        n_bc=n_bc,
        n_eval=n_eval,
        history_eval_n=history_eval_n,
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
    initial_metrics = bundle.history_metrics_fn()
    if (
        not bool(torch.isfinite(initial_loss).item())
        or not finite_mapping(initial_component_values)
        or not _metrics_finite(initial_metrics)
    ):
        raise FloatingPointError("non-finite initial loss, component, or metric")
    history: list[dict[str, Any]] = [{
        "elapsed_seconds": 0.0,
        "step": 0,
        "learning_rate": LEARNING_RATE,
        "grad_norm": None,
        **initial_component_values,
        **initial_metrics,
    }]
    del initial_loss, initial_components

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    evaluation_seconds = 0.0
    next_history = HISTORY_INTERVAL_SECONDS
    steps = 0
    latest_components = initial_component_values
    latest_grad_norm = 0.0
    nan_hit = False

    while True:
        elapsed_before = time.perf_counter() - started - evaluation_seconds
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
            latest_components = tensor_components_to_float(components)
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm).item()):
            nan_hit = True
            latest_components = tensor_components_to_float(components)
            break
        latest_grad_norm = float(grad_norm.detach().item())
        optimizer.step()
        steps += 1
        torch.cuda.synchronize(device)
        elapsed_after = time.perf_counter() - started - evaluation_seconds

        if elapsed_after >= next_history:
            latest_components = tensor_components_to_float(components)
            evaluation_started = time.perf_counter()
            metrics = bundle.history_metrics_fn()
            torch.cuda.synchronize(device)
            evaluation_seconds += time.perf_counter() - evaluation_started
            elapsed_after = time.perf_counter() - started - evaluation_seconds
            history.append({
                "elapsed_seconds": round(elapsed_after, 6),
                "step": steps,
                "learning_rate": current_lr,
                "grad_norm": latest_grad_norm,
                **latest_components,
                **metrics,
            })
            while next_history <= elapsed_after:
                next_history += HISTORY_INTERVAL_SECONDS
        del loss, components

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started - evaluation_seconds
    final_loss, final_components = bundle.loss_fn()
    final_component_values = tensor_components_to_float(final_components)
    final_metrics = bundle.eval_metrics_fn()
    final_lr = float(optimizer.param_groups[0]["lr"])
    peak_mb = torch.cuda.max_memory_allocated(device) / (2**20)
    if history[-1]["step"] != steps:
        history.append({
            "elapsed_seconds": round(training_seconds, 6),
            "step": steps,
            "learning_rate": final_lr,
            "grad_norm": latest_grad_norm,
            **final_component_values,
            **final_metrics,
        })

    status = "complete"
    if (
        nan_hit
        or not bool(torch.isfinite(final_loss).item())
        or not finite_mapping(final_component_values)
        or not _metrics_finite(final_metrics)
    ):
        status = "failed_nonfinite"
    result = {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "task_id": task.task_id,
        "family": str(getattr(task, "family", RUNNER_FAMILY_NAME)),
        "order": task.order,
        "q": int(getattr(task, "q", task.order // 2)),
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
        "evaluation_seconds": evaluation_seconds,
        "ms_per_step": 1000.0 * training_seconds / max(1, steps),
        "peak_mb": peak_mb,
        "learning_rate_initial": LEARNING_RATE,
        "learning_rate_final_target": LEARNING_RATE_FINAL,
        "learning_rate_last": final_lr,
        "gradient_clip": GRAD_CLIP,
        "last_grad_norm_before_clip": latest_grad_norm,
        "rel_error": float(final_metrics["rel_error"]),
        "loss": float(final_loss.detach().item()),
        "components": final_component_values,
        "metrics": final_metrics,
        "initial_components": initial_component_values,
        "initial_metrics": initial_metrics,
        "history": history,
        "history_schema": {
            "x_axis": "training wall time excluding evaluation",
            "required": [
                "elapsed_seconds",
                "step",
                "learning_rate",
                "grad_norm",
                "loss",
                "rel_error",
                "L_PDE",
                "L_IC",
                "L_BC",
                "mass_drift_rms",
            ],
        },
        "model": model_metadata(model, method),
        "problem": bundle.metadata,
        "completed_at": utc_now(),
    }
    del final_loss, final_components, optimizer, model, bundle
    torch.cuda.empty_cache()
    return result


def load_result(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def result_complete(
    path: Path,
    *,
    task_id: str | None = None,
    method: str | None = None,
    weights: tuple[float, ...] | None = None,
    seconds: float | None = None,
    smoke: bool | None = None,
    train_seed: int | None = None,
    eval_seed: int | None = None,
    reference_sha256: str | None = None,
) -> bool:
    result = load_result(path)
    if not bool(
        result
        and result.get("protocol_id") == PROTOCOL_ID
        and result.get("status") == "complete"
        and math.isfinite(float(result.get("loss", math.inf)))
        and math.isfinite(float(result.get("rel_error", math.inf)))
    ):
        return False
    assert result is not None
    checks = (
        task_id is None or result.get("task_id") == task_id,
        method is None or result.get("method") == method,
        seconds is None
        or math.isclose(float(result.get("budget_seconds", -1.0)), seconds),
        smoke is None or bool(result.get("smoke")) is smoke,
        train_seed is None or int(result.get("train_seed", -1)) == train_seed,
        eval_seed is None or int(result.get("eval_seed", -1)) == eval_seed,
    )
    if not all(checks):
        return False
    if weights is not None:
        found = tuple(float(value) for value in result.get("weights", ()))
        if len(found) != len(weights) or any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-14)
            for left, right in zip(found, weights, strict=True)
        ):
            return False
    if reference_sha256 is not None:
        found_reference = (
            result.get("problem", {}).get("reference", {})
            if isinstance(result.get("problem"), dict)
            else {}
        )
        if found_reference.get("reference_sha256") != reference_sha256:
            return False
    return True


def worker_command(
    *,
    task: str,
    method: str,
    weights: tuple[float, ...],
    seconds: float,
    output: Path,
    smoke: bool,
    train_seed: int,
    eval_seed: int,
    n_int: int | None,
    n_ic: int | None,
    n_bc: int | None,
    n_eval: int | None,
    history_eval_n: int | None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--task", task,
        "--method", method,
        "--weights", ",".join(f"{value:.16g}" for value in weights),
        "--seconds", str(seconds),
        "--output", str(output),
        "--train-seed", str(train_seed),
        "--eval-seed", str(eval_seed),
    ]
    if smoke:
        command.append("--smoke")
    for name, value in (
        ("--n-int", n_int),
        ("--n-ic", n_ic),
        ("--n-bc", n_bc),
        ("--n-eval", n_eval),
        ("--history-eval-n", history_eval_n),
    ):
        if value is not None:
            command.extend([name, str(value)])
    return command


def run_worker(args: argparse.Namespace) -> int:
    task = TASKS[args.task]
    weights = tuple(float(value) for value in args.weights.split(",") if value)
    output = args.output.resolve()
    base = {
        "protocol_id": PROTOCOL_ID,
        "status": "running",
        "task_id": args.task,
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
            n_int=args.n_int,
            n_ic=args.n_ic,
            n_bc=args.n_bc,
            n_eval=args.n_eval,
            history_eval_n=args.history_eval_n,
            train_seed=args.train_seed,
            eval_seed=args.eval_seed,
        )
        result.update({"git": git_state(), "hardware": hardware_metadata()})
        atomic_write_json(output, result)
        print(json.dumps({
            "status": result["status"],
            "task_id": result["task_id"],
            "method": result["method"],
            "steps": result["steps"],
            "loss": result["loss"],
            "rel_error": result["rel_error"],
            "mass_drift_rms": result.get("metrics", {}).get("mass_drift_rms"),
        }, sort_keys=True), flush=True)
        return 0 if result["status"] == "complete" else 2
    except BaseException as error:  # noqa: BLE001 - preserve every failure
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
        print(json.dumps({
            "status": "failed",
            "task_id": args.task,
            "method": args.method,
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        }, sort_keys=True), flush=True)
        return 1


def manifest(
    tasks: Iterable[str],
    *,
    seconds: float,
    smoke: bool,
    n_int: int | None,
    n_ic: int | None,
    n_bc: int | None,
    n_eval: int | None,
    history_eval_n: int | None,
) -> dict[str, Any]:
    task_list = list(tasks)
    candidate_count = sum(len(candidate_vectors(TASKS[name])) for name in task_list)
    metadata_factory = getattr(_problem, "runner_manifest_metadata", None)
    problem_metadata = (
        metadata_factory(smoke=smoke)
        if callable(metadata_factory)
        else dict(getattr(_problem, "RUNNER_MANIFEST_METADATA", {}))
    )
    input_dim = int(getattr(_problem, "INPUT_DIM", 3))
    parameter_elements = (
        input_dim * HIDDEN + HIDDEN
        + (DEPTH - 1) * (HIDDEN * HIDDEN + HIDDEN)
        + HIDDEN + 1
    )
    baseline_activation = str(
        getattr(_problem, "BASELINE_ACTIVATION", "sinh")
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "created_at": utc_now(),
        "smoke": smoke,
        "tasks": [
            {
                "task_id": name,
                "order": TASKS[name].order,
                "family": str(
                    getattr(TASKS[name], "family", RUNNER_FAMILY_NAME)
                ),
                "q": int(getattr(TASKS[name], "q", TASKS[name].order // 2)),
                "eta_q": float(getattr(TASKS[name], "eta", math.nan)),
                "weight_names": list(TASKS[name].weight_names),
                "candidate_count": len(candidate_vectors(TASKS[name])),
            }
            for name in task_list
        ],
        "methods": list(METHODS),
        "architecture": {
            "shared": {
                "input": "affine-normalized raw (x,y,t)",
                "trigonometric_input_features": False,
                "hidden": HIDDEN,
                "depth": DEPTH,
                "init": "common_xavier",
            },
            "war": {
                "parameter_dtype": str(COMPLEX_DTYPE),
                "activation": "sinh",
                "backend": "waring_complex_jet",
                "parameter_elements": parameter_elements,
                "real_dof": 2 * parameter_elements,
            },
            BASELINE_METHOD: {
                "parameter_dtype": str(REAL_DTYPE),
                "activation": baseline_activation,
                "backend": "direct_autodiff",
                "parameter_elements": parameter_elements,
                "real_dof": parameter_elements,
            },
            "capacity_note": (
                "literal layer shapes are matched; native complex parameters "
                "contain two real scalar degrees of freedom"
            ),
        },
        "grid_values": list(GRID_VALUES),
        "grid_type": "complete_ordered_cartesian_product",
        "candidate_count": candidate_count,
        "method_run_count": candidate_count * len(METHODS),
        "seconds_per_method_run": seconds,
        "nominal_training_seconds": candidate_count * len(METHODS) * seconds,
        "sample_overrides": {
            "n_int": n_int,
            "n_ic": n_ic,
            "n_bc": n_bc,
            "n_eval": n_eval,
            "history_eval_n": history_eval_n,
        },
        "learning_rate": LEARNING_RATE,
        "learning_rate_final": LEARNING_RATE_FINAL,
        "history_interval_seconds": HISTORY_INTERVAL_SECONDS,
        "gradient_clip": GRAD_CLIP,
        "serial_single_gpu": True,
        "method_order_policy": (
            "WAR-first for even candidates; AD-first for odd candidates"
            if bool(getattr(_problem, "ALTERNATE_METHOD_ORDER", False))
            else "fixed declared method order"
        ),
        "problem_metadata": problem_metadata,
        "git": git_state(),
        "hardware": hardware_metadata(),
    }


def candidate_record(
    task: Cahn2DTask, index: int, weights: tuple[float, ...]
) -> dict[str, Any]:
    return {
        "candidate_index": index,
        "candidate_id": f"point_{index:03d}",
        "weights": list(weights),
        "weight_labels": [weight_label(value) for value in weights],
        "weight_map": dict(zip(task.weight_names, weights)),
    }


def summarize_task(task: Cahn2DTask, task_dir: Path) -> dict[str, Any]:
    matrix: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    complete_runs = 0
    failed_runs = 0
    for index, weights in enumerate(candidate_vectors(task)):
        candidate = candidate_record(task, index, weights)
        point_dir = task_dir / "points" / candidate["candidate_id"]
        results = {
            method: load_result(point_dir / f"{method}.json") for method in METHODS
        }
        row: dict[str, Any] = {
            **candidate,
            "weights": json.dumps(candidate["weights"]),
            "weight_labels": json.dumps(candidate["weight_labels"]),
            "weight_map": json.dumps(candidate["weight_map"], sort_keys=True),
        }
        for method, result in results.items():
            status = "missing" if result is None else str(result.get("status", "unknown"))
            row[f"{method}_status"] = status
            if status == "complete":
                row[f"{method}_rel_error"] = float(result["rel_error"])
                row[f"{method}_loss"] = float(result["loss"])
                row[f"{method}_steps"] = int(result["steps"])
                row[f"{method}_mass_drift_rms"] = float(
                    result.get("metrics", {}).get("mass_drift_rms", math.nan)
                )
                complete_runs += 1
            elif status not in {"missing", "running"}:
                failed_runs += 1
        matrix.append(row)
        if all(
            results[method]
            and results[method].get("status") == "complete"
            and math.isfinite(float(results[method].get("rel_error", math.inf)))
            for method in METHODS
        ):
            war_error = float(results["war"]["rel_error"])
            ad_error = float(results[BASELINE_METHOD]["rel_error"])
            paired.append({
                **candidate,
                "war_rel_error": war_error,
                f"{BASELINE_METHOD}_rel_error": ad_error,
                "geometric_mean": math.sqrt(war_error * ad_error),
                "max_error": max(war_error, ad_error),
                "mean_error": 0.5 * (war_error + ad_error),
                "weight_sum": sum(weights),
            })

    ranking_specs = {
        "shared_minimax": lambda row: (
            row["max_error"], row["geometric_mean"], row["weight_sum"]
        ),
        "shared_geomean": lambda row: (
            row["geometric_mean"], row["max_error"], row["weight_sum"]
        ),
        "war": lambda row: (
            row["war_rel_error"], row[f"{BASELINE_METHOD}_rel_error"], row["weight_sum"]
        ),
        BASELINE_METHOD: lambda row: (
            row[f"{BASELINE_METHOD}_rel_error"], row["war_rel_error"], row["weight_sum"]
        ),
    }
    ranking_dir = task_dir / "rankings"
    ranking_dir.mkdir(parents=True, exist_ok=True)
    for name, key in ranking_specs.items():
        ranked = sorted(paired, key=key)
        atomic_write_json(ranking_dir / f"ranking_{name}.json", ranked)
        write_csv(ranking_dir / f"ranking_{name}.csv", ranked)
    write_csv(task_dir / "run_matrix.csv", matrix)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "task_id": task.task_id,
        "expected_candidates": len(candidate_vectors(task)),
        "expected_method_runs": len(candidate_vectors(task)) * len(METHODS),
        "complete_runs": complete_runs,
        "failed_runs": failed_runs,
        "paired_complete_candidates": len(paired),
        "complete": len(paired) == len(candidate_vectors(task)),
        "updated_at": utc_now(),
    }
    atomic_write_json(task_dir / "summary.json", summary)
    return summary


def summarize_root(root: Path, tasks: tuple[str, ...]) -> dict[str, Any]:
    task_summaries = [summarize_task(TASKS[name], root / name) for name in tasks]
    summary = {
        "protocol_id": PROTOCOL_ID,
        "updated_at": utc_now(),
        "tasks": list(tasks),
        "complete": all(value["complete"] for value in task_summaries),
        "task_summaries": task_summaries,
    }
    atomic_write_json(root / "summary.json", summary)
    return summary


def write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS" and ".tmp." not in path.name
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
        for path in paths
    ]
    atomic_write_text(root / "SHA256SUMS", "\n".join(lines) + "\n")


def _run_subprocess(command: list[str], log: Path, timeout: float) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"\n# started_at={utc_now()}\n")
        # Flush the parent-side buffer before the child writes directly to the
        # shared descriptor; otherwise the metadata line can be flushed after
        # the child's final JSON record and cease to be the log's last line.
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    return completed.returncode


def build_smoke_conclusion(
    root: Path,
    tasks: tuple[str, ...],
    *,
    seconds: float,
    failures: int,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for task_name in tasks:
        for method in METHODS:
            result = load_result(root / task_name / f"{method}.json") or {}
            cell: dict[str, Any] = {
                "task_id": task_name,
                "method": method,
                "status": result.get("status", "missing"),
            }
            for key in ("steps", "loss", "rel_error", "peak_mb", "ms_per_step"):
                if key in result:
                    cell[key] = result[key]
            metrics = result.get("metrics")
            if isinstance(metrics, dict):
                for key in ("mass_drift_rms", "mass_drift_max_abs"):
                    if key in metrics:
                        cell[key] = metrics[key]
            for key in ("error_type", "error"):
                if key in result:
                    cell[key] = result[key]
            cells.append(cell)
    raw_manifest = load_result(root / "manifest.json") or {}
    return {
        "protocol_id": PROTOCOL_ID,
        "kind": "ephemeral_cuda_smoke_conclusion",
        "completed_at": utc_now(),
        "seconds_per_cell": seconds,
        "tasks": list(tasks),
        "methods": list(METHODS),
        "cell_count": len(cells),
        "failure_count": failures,
        "passed": failures == 0 and all(
            cell.get("status") == "complete" for cell in cells
        ),
        "cells": cells,
        "hardware": raw_manifest.get("hardware"),
        "git": raw_manifest.get("git"),
        "raw_artifacts_retained": False,
        "interpretation": (
            "startup/finite-gradient/data-pipeline gate only; not a formal "
            "accuracy result and not eligible for paper statistics"
        ),
    }


def run_smoke(args: argparse.Namespace) -> int:
    tasks = selected_tasks(args.tasks)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.ephemeral_conclusion is not None:
        temporary = tempfile.TemporaryDirectory(
            prefix=f"apolarity-{PROBLEM_FAMILY}-smoke-"
        )
        root = Path(temporary.name).resolve()
    else:
        root = (args.output_root or DEFAULT_ROOT / "_smoke").resolve()
        root.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_json(root / "manifest.json", manifest(
            tasks,
            seconds=args.seconds,
            smoke=True,
            n_int=args.n_int,
            n_ic=args.n_ic,
            n_bc=args.n_bc,
            n_eval=args.n_eval,
            history_eval_n=args.history_eval_n,
        ))
        failures = 0
        for task_name in tasks:
            task = TASKS[task_name]
            for method in METHODS:
                output = root / task_name / f"{method}.json"
                log = root / task_name / f"{method}.log"
                command = worker_command(
                    task=task_name,
                    method=method,
                    weights=task.center_weights,
                    seconds=args.seconds,
                    output=output,
                    smoke=True,
                    train_seed=args.train_seed,
                    eval_seed=args.eval_seed,
                    n_int=args.n_int,
                    n_ic=args.n_ic,
                    n_bc=args.n_bc,
                    n_eval=args.n_eval,
                    history_eval_n=args.history_eval_n,
                )
                print(f"[smoke] {task_name} {method}", flush=True)
                return_code = _run_subprocess(
                    command, log, timeout=max(300.0, args.seconds * 30.0)
                )
                if return_code != 0 or not result_complete(
                    output,
                    task_id=task_name,
                    method=method,
                    weights=task.center_weights,
                    seconds=args.seconds,
                    smoke=True,
                    train_seed=args.train_seed,
                    eval_seed=args.eval_seed,
                ):
                    failures += 1
                    print(
                        f"[smoke-failed] {task_name} {method}", flush=True
                    )
                else:
                    result = load_result(output) or {}
                    print(
                        f"[smoke-ok] loss={result.get('loss'):.4e} "
                        f"rel_error={result.get('rel_error'):.4e} "
                        f"peak_mb={result.get('peak_mb'):.1f}",
                        flush=True,
                    )
        write_checksums(root)
        atomic_write_text(
            root / ("SMOKE_COMPLETE" if failures == 0 else "SMOKE_FAILED"),
            f"failures={failures}\n",
        )
        if args.ephemeral_conclusion is not None:
            conclusion = build_smoke_conclusion(
                root, tasks, seconds=args.seconds, failures=failures
            )
            atomic_write_json(args.ephemeral_conclusion.resolve(), conclusion)
        return 0 if failures == 0 else 1
    finally:
        if temporary is not None:
            temporary.cleanup()


def archive_incomplete(output: Path, point_dir: Path, method: str) -> None:
    if not output.exists():
        return
    attempts = point_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    shutil.copy2(output, attempts / f"{method}.{stamp}.json")


def run_orchestrate(args: argparse.Namespace) -> int:
    tasks = selected_tasks(args.tasks)
    root = (args.output_root or DEFAULT_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    expected_manifest = manifest(
        tasks,
        seconds=args.seconds,
        smoke=False,
        n_int=args.n_int,
        n_ic=args.n_ic,
        n_bc=args.n_bc,
        n_eval=args.n_eval,
        history_eval_n=args.history_eval_n,
    )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        compatibility_keys = [
            "protocol_id",
            "tasks",
            "methods",
            "seconds_per_method_run",
            "sample_overrides",
        ]
        strict_binding = bool(
            getattr(_problem, "STRICT_MANIFEST_BINDING", False)
        )
        if strict_binding:
            compatibility_keys.extend([
                "problem_metadata",
                "method_order_policy",
            ])
        for key in compatibility_keys:
            if existing.get(key) != expected_manifest.get(key):
                raise ValueError(f"incompatible manifest field {key!r} at {manifest_path}")
        if (
            strict_binding
            and existing.get("git", {}).get("sha")
            != expected_manifest.get("git", {}).get("sha")
        ):
            raise ValueError(f"incompatible git SHA at {manifest_path}")
    else:
        atomic_write_json(manifest_path, expected_manifest)

    total = expected_manifest["method_run_count"]
    processed = 0
    attempted = 0
    failures = 0
    started = time.time()
    expected_reference_sha = (
        expected_manifest.get("problem_metadata", {})
        .get("reference", {})
        .get("sha256")
    )
    for task_name in tasks:
        task = TASKS[task_name]
        task_dir = root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        for index, weights in enumerate(candidate_vectors(task)):
            candidate = candidate_record(task, index, weights)
            point_dir = task_dir / "points" / candidate["candidate_id"]
            point_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(point_dir / "candidate.json", candidate)
            method_order = (
                METHODS
                if not bool(getattr(_problem, "ALTERNATE_METHOD_ORDER", False))
                or index % 2 == 0
                else tuple(reversed(METHODS))
            )
            for method in method_order:
                output = point_dir / f"{method}.json"
                log = point_dir / f"{method}.log"
                expected_result = {
                    "task_id": task_name,
                    "method": method,
                    "weights": weights,
                    "seconds": args.seconds,
                    "smoke": False,
                    "train_seed": args.train_seed,
                    "eval_seed": args.eval_seed,
                    "reference_sha256": expected_reference_sha,
                }
                if args.resume and result_complete(output, **expected_result):
                    processed += 1
                    continue
                if output.exists():
                    archive_incomplete(output, point_dir, method)
                success = False
                for attempt in range(1, args.retries + 2):
                    attempted += 1
                    command = worker_command(
                        task=task_name,
                        method=method,
                        weights=weights,
                        seconds=args.seconds,
                        output=output,
                        smoke=False,
                        train_seed=args.train_seed,
                        eval_seed=args.eval_seed,
                        n_int=args.n_int,
                        n_ic=args.n_ic,
                        n_bc=args.n_bc,
                        n_eval=args.n_eval,
                        history_eval_n=args.history_eval_n,
                    )
                    return_code = _run_subprocess(
                        command,
                        log,
                        timeout=max(600.0, args.seconds * 30.0),
                    )
                    if return_code == 0 and result_complete(
                        output, **expected_result
                    ):
                        success = True
                        break
                    if output.exists():
                        archive_incomplete(output, point_dir, method)
                    with log.open("a") as handle:
                        handle.write(f"# retry_after_attempt={attempt}\n")
                processed += 1
                if not success:
                    failures += 1
                elapsed = time.time() - started
                rate = processed / max(elapsed, 1e-9)
                eta_seconds = (total - processed) / max(rate, 1e-9)
                progress = {
                    "protocol_id": PROTOCOL_ID,
                    "updated_at": utc_now(),
                    "processed_runs": processed,
                    "total_runs": total,
                    "attempted_subprocesses": attempted,
                    "failures": failures,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta_seconds,
                    "current": {
                        "task": task_name,
                        "candidate_id": candidate["candidate_id"],
                        "method": method,
                    },
                }
                atomic_write_json(root / "progress.json", progress)
                print(
                    f"[{processed}/{total}] {task_name} {candidate['candidate_id']} "
                    f"{method} success={success} eta={eta_seconds/3600:.2f}h",
                    flush=True,
                )
            summarize_task(task, task_dir)
    summary = summarize_root(root, tasks)
    write_checksums(root)
    if summary["complete"] and failures == 0:
        atomic_write_text(root / "SEARCH_COMPLETE", f"completed_at={utc_now()}\n")
        return 0
    atomic_write_text(
        root / "SEARCH_INCOMPLETE",
        f"completed_at={utc_now()} failures={failures}\n",
    )
    return 1


def run_summarize(args: argparse.Namespace) -> int:
    tasks = selected_tasks(args.tasks)
    root = (args.output_root or DEFAULT_ROOT).resolve()
    summary = summarize_root(root, tasks)
    write_checksums(root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["complete"] else 1


def add_sample_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-int", type=int)
    parser.add_argument("--n-ic", type=int)
    parser.add_argument("--n-bc", type=int)
    parser.add_argument("--n-eval", type=int)
    parser.add_argument("--history-eval-n", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--task", choices=TASK_ORDER, required=True)
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--weights", required=True)
    worker.add_argument("--seconds", type=float, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--smoke", action="store_true")
    worker.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    worker.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    add_sample_arguments(worker)
    worker.set_defaults(func=run_worker)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--tasks", default="all")
    smoke.add_argument("--seconds", type=float, default=3.0)
    smoke_destination = smoke.add_mutually_exclusive_group()
    smoke_destination.add_argument("--output-root", type=Path)
    smoke_destination.add_argument("--ephemeral-conclusion", type=Path)
    smoke.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    smoke.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    add_sample_arguments(smoke)
    smoke.set_defaults(func=run_smoke)

    orchestrate = subparsers.add_parser("orchestrate")
    orchestrate.add_argument("--tasks", default="all")
    orchestrate.add_argument("--seconds", type=float, default=60.0)
    orchestrate.add_argument("--output-root", type=Path)
    orchestrate.add_argument("--resume", action="store_true")
    orchestrate.add_argument("--retries", type=int, default=1)
    orchestrate.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    orchestrate.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    add_sample_arguments(orchestrate)
    orchestrate.set_defaults(func=run_orchestrate)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--tasks", default="all")
    summarize.add_argument("--output-root", type=Path)
    summarize.set_defaults(func=run_summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
