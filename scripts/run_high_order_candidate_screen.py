#!/usr/bin/env python3
"""Smoke, screen, resume, and summarize high-order PDE candidates.

The approved screening stage is four tasks x two methods x three seeds x 600
training seconds.  A selected task can later be rerun with ``--stage formal``
under five seeds x 1200 seconds without mixing the two result protocols.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
CANDIDATES = ROOT / "experiments" / "high_order_candidates"
SRC = ROOT / "src"
for path in (str(COMMON), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

_PROBLEM_SPEC = importlib.util.spec_from_file_location(
    "apolarity_high_order_candidate_problem", CANDIDATES / "problem.py"
)
if _PROBLEM_SPEC is None or _PROBLEM_SPEC.loader is None:  # pragma: no cover
    raise ImportError("cannot load high-order candidate problem module")
_PROBLEM = importlib.util.module_from_spec(_PROBLEM_SPEC)
sys.modules[_PROBLEM_SPEC.name] = _PROBLEM
_PROBLEM_SPEC.loader.exec_module(_PROBLEM)

COMPLEX_DTYPE = _PROBLEM.COMPLEX_DTYPE
DEPTH = _PROBLEM.DEPTH
EVAL_SEED = _PROBLEM.EVAL_SEED
HIDDEN = _PROBLEM.HIDDEN
HISTORY_INTERVAL_SECONDS = _PROBLEM.HISTORY_INTERVAL_SECONDS
LEARNING_RATE = _PROBLEM.LEARNING_RATE
LEARNING_RATE_FINAL = _PROBLEM.LEARNING_RATE_FINAL
METHODS = _PROBLEM.METHODS
ENGINE_PROTOCOL_ID = _PROBLEM.PROTOCOL_ID
REAL_DTYPE = _PROBLEM.REAL_DTYPE
TASKS = _PROBLEM.TASKS
TASK_ORDER = _PROBLEM.TASK_ORDER
CandidateTask = _PROBLEM.CandidateTask
build_model = _PROBLEM.build_model
make_loss_bundle = _PROBLEM.make_loss_bundle
model_metadata = _PROBLEM.model_metadata
tensor_components_to_float = _PROBLEM.tensor_components_to_float


PROTOCOL_ID = "high_order_candidate_screen_v1"
DEFAULT_ROOT = ROOT / "outputs" / "search" / "high-order-candidate-pilot-v1"
GRAD_CLIP = 10.0
DEFAULT_SAMPLES = {
    "n_int": 2048,
    "n_ic": 512,
    "n_bc": 1024,
    "n_eval": 16384,
    "history_eval_n": 2048,
}


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


def write_checksums(root: Path) -> None:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS", "run.pid"}
        and ".tmp." not in path.name
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
        for path in paths
    ]
    atomic_write_text(root / "SHA256SUMS", "\n".join(lines) + "\n")


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
    result: dict[str, Any] = {
        "hostname": platform.node(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update({
            "cuda_device_index": index,
            "cuda_device_name": properties.name,
            "cuda_total_memory_bytes": properties.total_memory,
        })
    return result


def selected_tasks(value: str) -> tuple[str, ...]:
    names = TASK_ORDER if value == "all" else tuple(
        part.strip() for part in value.split(",") if part.strip()
    )
    if not names:
        raise ValueError("at least one task is required")
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    return names


def seed_list(count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("--seeds must be positive")
    return tuple(range(count))


def sample_counts(args: argparse.Namespace) -> dict[str, int]:
    result = {
        "n_int": int(args.n_int),
        "n_ic": int(args.n_ic),
        "n_bc": int(args.n_bc),
        "n_eval": int(args.n_eval),
        "history_eval_n": int(args.history_eval_n),
    }
    if min(result.values()) <= 0:
        raise ValueError("all sample counts must be positive")
    if result["history_eval_n"] > result["n_eval"]:
        raise ValueError("history_eval_n cannot exceed n_eval")
    return result


def finite_mapping(values: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in values.values())


def _metrics_finite(values: dict[str, object]) -> bool:
    for value in values.values():
        if isinstance(value, dict):
            if not _metrics_finite(value):
                return False
        elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def train_one(
    task: CandidateTask,
    method: str,
    *,
    stage: str,
    seconds: float,
    samples: dict[str, int],
    train_seed: int,
    eval_seed: int,
    smoke: bool,
) -> dict[str, Any]:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("candidate training requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    started_at = utc_now()
    model, dtype, backend = build_model(task, method, device)
    bundle = make_loss_bundle(
        task,
        model,
        dtype,
        backend,
        device,
        **samples,
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
    print(json.dumps({"event": "HISTORY", **history[-1]}, sort_keys=True), flush=True)
    del initial_loss, initial_components

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    evaluation_seconds = 0.0
    next_history = min(HISTORY_INTERVAL_SECONDS, max(1.0, seconds / 3.0))
    history_interval = next_history
    steps = 0
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
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if not bool(torch.isfinite(grad_norm).item()):
            nan_hit = True
            break
        latest_grad_norm = float(grad_norm.detach().item())
        optimizer.step()
        steps += 1
        torch.cuda.synchronize(device)
        elapsed_after = time.perf_counter() - started - evaluation_seconds
        if elapsed_after >= next_history:
            component_values = tensor_components_to_float(components)
            evaluation_started = time.perf_counter()
            metrics = bundle.history_metrics_fn()
            torch.cuda.synchronize(device)
            evaluation_seconds += time.perf_counter() - evaluation_started
            elapsed_after = time.perf_counter() - started - evaluation_seconds
            point = {
                "elapsed_seconds": round(elapsed_after, 6),
                "step": steps,
                "learning_rate": current_lr,
                "grad_norm": latest_grad_norm,
                **component_values,
                **metrics,
            }
            history.append(point)
            print(json.dumps({"event": "HISTORY", **point}, sort_keys=True), flush=True)
            while next_history <= elapsed_after:
                next_history += history_interval
        del loss, components

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started - evaluation_seconds
    final_loss, final_components = bundle.loss_fn()
    final_component_values = tensor_components_to_float(final_components)
    final_metrics = bundle.eval_metrics_fn()
    final_lr = float(optimizer.param_groups[0]["lr"])
    peak_mb = torch.cuda.max_memory_allocated(device) / (2**20)
    if history[-1]["step"] != steps:
        point = {
            "elapsed_seconds": round(training_seconds, 6),
            "step": steps,
            "learning_rate": final_lr,
            "grad_norm": latest_grad_norm,
            **final_component_values,
            **final_metrics,
        }
        history.append(point)
        print(json.dumps({"event": "HISTORY", **point}, sort_keys=True), flush=True)

    status = "complete"
    if (
        nan_hit
        or not bool(torch.isfinite(final_loss).item())
        or not finite_mapping(final_component_values)
        or not _metrics_finite(final_metrics)
    ):
        status = "failed_nonfinite"
    result = {
        "protocol_id": ENGINE_PROTOCOL_ID,
        "screen_protocol_id": PROTOCOL_ID,
        "stage": stage,
        "status": status,
        "task_id": task.task_id,
        "family": task.family,
        "spatial_dim": task.spatial_dim,
        "order": task.order,
        "method": method,
        "weights": list(task.weights),
        "weight_map": dict(zip(task.weight_names, task.weights)),
        "budget_seconds": seconds,
        "smoke": smoke,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "started_at": started_at,
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
                "L_BC",
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
    task: str,
    method: str,
    stage: str,
    seed: int,
    seconds: float,
    samples: dict[str, int],
) -> bool:
    result = load_result(path)
    problem = result.get("problem", {}) if result else {}
    return bool(
        result
        and result.get("protocol_id") == ENGINE_PROTOCOL_ID
        and result.get("screen_protocol_id") == PROTOCOL_ID
        and result.get("stage") == stage
        and result.get("status") == "complete"
        and result.get("task_id") == task
        and result.get("method") == method
        and int(result.get("train_seed", -1)) == seed
        and math.isclose(float(result.get("budget_seconds", -1)), seconds)
        and math.isfinite(float(result.get("loss", math.inf)))
        and math.isfinite(float(result.get("rel_error", math.inf)))
        and int(problem.get("n_int", -1)) == samples["n_int"]
        and int(problem.get("n_bc", -1)) == samples["n_bc"]
        and int(problem.get("n_eval", -1)) == samples["n_eval"]
        and int(problem.get("history_eval_n", -1)) == samples["history_eval_n"]
    )


def worker_command(
    *,
    task: str,
    method: str,
    stage: str,
    seed: int,
    eval_seed: int,
    seconds: float,
    samples: dict[str, int],
    output: Path,
    smoke: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--task", task,
        "--method", method,
        "--stage", stage,
        "--seed", str(seed),
        "--eval-seed", str(eval_seed),
        "--seconds", str(seconds),
        "--output", str(output),
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
    samples = sample_counts(args)
    output = args.output.resolve()
    base = {
        "protocol_id": ENGINE_PROTOCOL_ID,
        "screen_protocol_id": PROTOCOL_ID,
        "stage": args.stage,
        "status": "running",
        "task_id": args.task,
        "method": args.method,
        "train_seed": args.seed,
        "eval_seed": args.eval_seed,
        "budget_seconds": args.seconds,
        "sample_counts": samples,
        "smoke": args.smoke,
        "started_at": utc_now(),
    }
    print(json.dumps({
        "event": "CELL_START",
        **base,
        "python_executable": sys.executable,
        "git": git_state(),
        "hardware": hardware_metadata(),
    }, sort_keys=True), flush=True)
    try:
        result = train_one(
            task,
            args.method,
            stage=args.stage,
            seconds=args.seconds,
            samples=samples,
            train_seed=args.seed,
            eval_seed=args.eval_seed,
            smoke=args.smoke,
        )
        result.update({"git": git_state(), "hardware": hardware_metadata()})
        atomic_write_json(output, result)
        print(json.dumps({
            "event": "CELL_FINAL",
            "status": result["status"],
            "task_id": args.task,
            "method": args.method,
            "seed": args.seed,
            "steps": result.get("steps"),
            "loss": result.get("loss"),
            "rel_error": result.get("rel_error"),
            "ms_per_step": result.get("ms_per_step"),
            "peak_mb": result.get("peak_mb"),
        }, sort_keys=True), flush=True)
        return 0 if result["status"] == "complete" else 2
    except BaseException as error:  # noqa: BLE001 - persist every failure
        failure = {
            **base,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
            "traceback": traceback.format_exc(limit=50),
            "completed_at": utc_now(),
        }
        try:
            failure.update({"git": git_state(), "hardware": hardware_metadata()})
        except Exception as metadata_error:  # noqa: BLE001
            failure["metadata_error"] = repr(metadata_error)[:1000]
        atomic_write_json(output, failure)
        print(json.dumps({
            "event": "CELL_FINAL",
            "status": "failed",
            "task_id": args.task,
            "method": args.method,
            "seed": args.seed,
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        }, sort_keys=True), flush=True)
        return 1


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
        row["history_points"] = len(result.get("history", []))
        row["final_history_has_loss"] = bool(
            result.get("history") and "loss" in result["history"][-1]
        )
        row["final_history_has_rel_error"] = bool(
            result.get("history") and "rel_error" in result["history"][-1]
        )
    return row


def build_summary(
    root: Path,
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for task in tasks:
        paired: list[dict[str, Any]] = []
        for seed in seeds:
            pair: dict[str, Any] = {"task_id": task, "seed": seed}
            complete = True
            for method in METHODS:
                result = load_result(
                    root / task / f"seed_{seed:03d}" / f"{method}.json"
                )
                all_rows.append(flatten_result(task, method, seed, result))
                if not result or result.get("status") != "complete":
                    complete = False
                    continue
                pair[f"{method}_rel_error"] = float(result["rel_error"])
                pair[f"{method}_loss"] = float(result["loss"])
                pair[f"{method}_ms_per_step"] = float(result["ms_per_step"])
                pair[f"{method}_peak_mb"] = float(result["peak_mb"])
            if complete:
                war_error = pair["war_rel_error"]
                ad_error = pair["real_tanh_autodiff_rel_error"]
                pair.update({
                    "status": "complete",
                    "winner": "war" if war_error < ad_error else "real_tanh_autodiff",
                    "geometric_mean_error": math.sqrt(war_error * ad_error),
                    "max_error": max(war_error, ad_error),
                    "ad_over_war_step_time": (
                        pair["real_tanh_autodiff_ms_per_step"]
                        / pair["war_ms_per_step"]
                    ),
                    "ad_over_war_peak_memory": (
                        pair["real_tanh_autodiff_peak_mb"]
                        / pair["war_peak_mb"]
                    ),
                })
            else:
                pair["status"] = "incomplete"
            paired.append(pair)
        complete_pairs = [row for row in paired if row["status"] == "complete"]
        war_errors = [row["war_rel_error"] for row in complete_pairs]
        ad_errors = [row["real_tanh_autodiff_rel_error"] for row in complete_pairs]
        max_errors = [row["max_error"] for row in complete_pairs]
        geometric = [row["geometric_mean_error"] for row in complete_pairs]
        war_median = statistics.median(war_errors) if war_errors else math.inf
        ad_median = statistics.median(ad_errors) if ad_errors else math.inf
        screen_pass = bool(
            len(complete_pairs) == len(seeds)
            and math.isfinite(war_median)
            and math.isfinite(ad_median)
            and min(war_median, ad_median) < 0.2
            and max(war_median, ad_median) < 0.75
        )
        summary = {
            "task_id": task,
            "family": TASKS[task].family,
            "spatial_dim": TASKS[task].spatial_dim,
            "order": TASKS[task].order,
            "paired_complete_seed_count": len(complete_pairs),
            "expected_seed_count": len(seeds),
            "screen_pass": screen_pass,
            "screen_rule": (
                "all pairs complete; best method median rel_error < 0.2; "
                "worst method median rel_error < 0.75"
            ),
            "war_rel_error": _distribution(war_errors),
            "real_tanh_autodiff_rel_error": _distribution(ad_errors),
            "shared_geometric_mean": _distribution(geometric),
            "shared_minimax": _distribution(max_errors),
            "ad_over_war_step_time": _distribution([
                row["ad_over_war_step_time"] for row in complete_pairs
            ]),
            "ad_over_war_peak_memory": _distribution([
                row["ad_over_war_peak_memory"] for row in complete_pairs
            ]),
            "war_seed_wins": sum(row.get("winner") == "war" for row in complete_pairs),
            "ad_seed_wins": sum(
                row.get("winner") == "real_tanh_autodiff" for row in complete_pairs
            ),
            "seed_metrics": paired,
        }
        atomic_write_json(root / task / "summary.json", summary)
        write_csv(root / task / "paired.csv", paired)
        task_summaries.append(summary)

    ranking = sorted(
        task_summaries,
        key=lambda item: (
            not bool(item["screen_pass"]),
            float(item["shared_minimax"]["median"] or math.inf),
            float(item["shared_geometric_mean"]["median"] or math.inf),
            item["task_id"],
        ),
    )
    for index, item in enumerate(ranking, start=1):
        item["screen_rank"] = index
    expected = len(tasks) * len(seeds) * len(METHODS)
    complete = sum(row["status"] == "complete" for row in all_rows)
    result = {
        "protocol_id": PROTOCOL_ID,
        "engine_protocol_id": ENGINE_PROTOCOL_ID,
        "updated_at": utc_now(),
        "tasks": list(tasks),
        "seeds": list(seeds),
        "expected_runs": expected,
        "complete_runs": complete,
        "all_complete": complete == expected,
        "task_summaries": task_summaries,
        "candidate_ranking": [item["task_id"] for item in ranking],
    }
    write_csv(root / "runs.csv", all_rows)
    atomic_write_json(root / "runs.json", all_rows)
    atomic_write_json(root / "candidate_ranking.json", ranking)
    atomic_write_json(root / "summary.json", result)
    return result


def manifest(
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
    stage: str,
    seconds: float,
    samples: dict[str, int],
) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "engine_protocol_id": ENGINE_PROTOCOL_ID,
        "stage": stage,
        "tasks": list(tasks),
        "task_specs": [
            {
                "task_id": name,
                "family": TASKS[name].family,
                "spatial_dim": TASKS[name].spatial_dim,
                "input_dim": TASKS[name].input_dim,
                "order": TASKS[name].order,
                "weights": dict(zip(TASKS[name].weight_names, TASKS[name].weights)),
                "uniqueness_basis": TASKS[name].uniqueness,
            }
            for name in tasks
        ],
        "methods": list(METHODS),
        "architecture": {
            "shared": {
                "hidden": HIDDEN,
                "depth": DEPTH,
                "init_mode": "common_xavier",
                "input": "affine-normalized raw coordinates",
                "trigonometric_input_features": False,
                "frequency_initialization": False,
            },
            "war": {
                "dtype": str(COMPLEX_DTYPE),
                "activation": "sinh",
                "backend": "waring_complex_jet",
            },
            "real_tanh_autodiff": {
                "dtype": str(REAL_DTYPE),
                "activation": "tanh",
                "backend": "direct_autodiff",
            },
        },
        "seeds": list(seeds),
        "seconds_per_method_seed": seconds,
        "sample_counts": samples,
        "method_order_policy": "war-first on even seeds; AD-first on odd seeds",
        "serial_single_gpu": True,
        "expected_runs": len(tasks) * len(seeds) * len(METHODS),
        "nominal_training_seconds": len(tasks) * len(seeds) * len(METHODS) * seconds,
        "learning_rate": LEARNING_RATE,
        "learning_rate_final": LEARNING_RATE_FINAL,
        "gradient_clip": GRAD_CLIP,
        "history_interval_seconds": HISTORY_INTERVAL_SECONDS,
        "git": git_state(),
        "hardware": hardware_metadata(),
    }


def _archive_incomplete(path: Path, cell_dir: Path, method: str) -> None:
    if not path.exists():
        return
    attempts = cell_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    shutil.copy2(path, attempts / f"{method}.{stamp}.json")
    log = cell_dir / f"{method}.log"
    if log.exists():
        shutil.copy2(log, attempts / f"{method}.{stamp}.log")


def run_smoke(args: argparse.Namespace) -> int:
    tasks = selected_tasks(args.tasks)
    samples = sample_counts(args)
    conclusion = (args.conclusion or DEFAULT_ROOT / "SMOKE_CONCLUSION.json").resolve()
    cells: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="apolarity-highorder-smoke-") as raw:
        raw_root = Path(raw)
        for task in tasks:
            for method in METHODS:
                cell_dir = raw_root / task
                output = cell_dir / f"{method}.json"
                log = cell_dir / f"{method}.log"
                cell_dir.mkdir(parents=True, exist_ok=True)
                command = worker_command(
                    task=task,
                    method=method,
                    stage="smoke",
                    seed=0,
                    eval_seed=args.eval_seed,
                    seconds=args.seconds,
                    samples=samples,
                    output=output,
                    smoke=True,
                )
                started = time.perf_counter()
                with log.open("w") as handle:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        timeout=max(900.0, args.seconds * 60.0),
                        check=False,
                    )
                elapsed = time.perf_counter() - started
                result = load_result(output) or {}
                passed = completed.returncode == 0 and result_complete(
                    output,
                    task=task,
                    method=method,
                    stage="smoke",
                    seed=0,
                    seconds=args.seconds,
                    samples=samples,
                )
                cell = {
                    "task_id": task,
                    "method": method,
                    "passed": passed,
                    "status": result.get("status", "missing"),
                    "returncode": completed.returncode,
                    "wall_seconds": elapsed,
                    "steps": result.get("steps"),
                    "loss": result.get("loss"),
                    "rel_error": result.get("rel_error"),
                    "ms_per_step": result.get("ms_per_step"),
                    "peak_mb": result.get("peak_mb"),
                }
                if not passed:
                    cell["error_type"] = result.get("error_type")
                    cell["error"] = result.get("error")
                    cell["log_tail"] = "\n".join(
                        log.read_text(errors="replace").splitlines()[-30:]
                    )[-8000:]
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
        "methods": list(METHODS),
        "tasks": list(tasks),
        "raw_artifacts_retained": False,
        "cells": cells,
        "git": git_state(),
        "hardware": hardware_metadata(),
        "conclusion": (
            "full-size CUDA startup, finite-gradient, loss and data-pipeline "
            "gate only; not an accuracy result"
        ),
    }
    atomic_write_json(conclusion, payload)
    return 0 if passed else 1


def run_orchestrate(args: argparse.Namespace) -> int:
    tasks = selected_tasks(args.tasks)
    seeds = seed_list(args.seeds)
    samples = sample_counts(args)
    root = (args.output_root or DEFAULT_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    expected_manifest = manifest(
        tasks, seeds, args.stage, args.seconds, samples
    )
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in (
            "protocol_id",
            "engine_protocol_id",
            "stage",
            "tasks",
            "methods",
            "seeds",
            "seconds_per_method_seed",
            "sample_counts",
        ):
            if existing.get(key) != expected_manifest.get(key):
                raise ValueError(f"incompatible manifest field {key!r}")
    else:
        atomic_write_json(manifest_path, expected_manifest)

    print(json.dumps({
        "event": "ORCHESTRATOR_START_OR_RESUME",
        "root": str(root),
        **expected_manifest,
    }, sort_keys=True), flush=True)
    total = len(tasks) * len(seeds) * len(METHODS)
    completed_count = 0
    failures = 0
    started = time.time()
    for task in tasks:
        task_dir = root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(task_dir / "manifest.json", {
            "protocol_id": PROTOCOL_ID,
            "engine_protocol_id": ENGINE_PROTOCOL_ID,
            "stage": args.stage,
            "task": task,
            "task_spec": expected_manifest["task_specs"][list(tasks).index(task)],
            "methods": list(METHODS),
            "seeds": list(seeds),
            "seconds_per_method_seed": args.seconds,
            "sample_counts": samples,
        })
        for seed in seeds:
            cell_dir = task_dir / f"seed_{seed:03d}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(cell_dir / "config.json", {
                "protocol_id": PROTOCOL_ID,
                "engine_protocol_id": ENGINE_PROTOCOL_ID,
                "stage": args.stage,
                "task": task,
                "seed": seed,
                "methods": list(METHODS),
                "method_execution_order": (
                    list(METHODS) if seed % 2 == 0 else list(reversed(METHODS))
                ),
                "seconds_per_method": args.seconds,
                "sample_counts": samples,
            })
            method_order = METHODS if seed % 2 == 0 else tuple(reversed(METHODS))
            for method in method_order:
                output = cell_dir / f"{method}.json"
                log = cell_dir / f"{method}.log"
                if args.resume and result_complete(
                    output,
                    task=task,
                    method=method,
                    stage=args.stage,
                    seed=seed,
                    seconds=args.seconds,
                    samples=samples,
                ):
                    completed_count += 1
                    print(json.dumps({
                        "event": "SKIP_COMPLETE",
                        "task": task,
                        "method": method,
                        "seed": seed,
                        "completed": completed_count,
                        "total": total,
                    }, sort_keys=True), flush=True)
                    continue
                _archive_incomplete(output, cell_dir, method)
                command = worker_command(
                    task=task,
                    method=method,
                    stage=args.stage,
                    seed=seed,
                    eval_seed=args.eval_seed,
                    seconds=args.seconds,
                    samples=samples,
                    output=output,
                    smoke=False,
                )
                progress = {
                    "protocol_id": PROTOCOL_ID,
                    "stage": args.stage,
                    "status": "running",
                    "current": {"task": task, "method": method, "seed": seed},
                    "complete_runs": completed_count,
                    "expected_runs": total,
                    "failures": failures,
                    "elapsed_wall_seconds": time.time() - started,
                    "updated_at": utc_now(),
                }
                atomic_write_json(root / "progress.json", progress)
                print(json.dumps({"event": "LAUNCH_CELL", **progress}, sort_keys=True), flush=True)
                with log.open("w") as handle:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        timeout=max(1800.0, args.seconds * 5.0),
                        check=False,
                    )
                valid = completed.returncode == 0 and result_complete(
                    output,
                    task=task,
                    method=method,
                    stage=args.stage,
                    seed=seed,
                    seconds=args.seconds,
                    samples=samples,
                )
                result = load_result(output) or {}
                if valid:
                    completed_count += 1
                else:
                    failures += 1
                last = {
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "valid": valid,
                    "returncode": completed.returncode,
                    "loss": result.get("loss"),
                    "rel_error": result.get("rel_error"),
                    "steps": result.get("steps"),
                    "ms_per_step": result.get("ms_per_step"),
                    "peak_mb": result.get("peak_mb"),
                }
                atomic_write_json(root / "progress.json", {
                    "protocol_id": PROTOCOL_ID,
                    "stage": args.stage,
                    "status": "running" if completed_count + failures < total else "summarizing",
                    "last_completed": last,
                    "complete_runs": completed_count,
                    "expected_runs": total,
                    "failures": failures,
                    "elapsed_wall_seconds": time.time() - started,
                    "updated_at": utc_now(),
                })
                print(json.dumps({"event": "CELL_RETURN", **last}, sort_keys=True), flush=True)
            if all(
                result_complete(
                    cell_dir / f"{method}.json",
                    task=task,
                    method=method,
                    stage=args.stage,
                    seed=seed,
                    seconds=args.seconds,
                    samples=samples,
                )
                for method in METHODS
            ):
                atomic_write_text(cell_dir / "DONE", utc_now() + "\n")

    summary = build_summary(root, tasks, seeds)
    all_complete = bool(summary["all_complete"])
    marker = "FORMAL_COMPLETE" if args.stage == "formal" else "PILOT_COMPLETE"
    if all_complete:
        atomic_write_text(root / marker, utc_now() + "\n")
    atomic_write_json(root / "progress.json", {
        "protocol_id": PROTOCOL_ID,
        "stage": args.stage,
        "status": "complete" if all_complete else "incomplete",
        "complete_runs": summary["complete_runs"],
        "expected_runs": summary["expected_runs"],
        "failures": failures,
        "elapsed_wall_seconds": time.time() - started,
        "updated_at": utc_now(),
    })
    write_checksums(root)
    print(json.dumps({
        "event": "ORCHESTRATOR_FINAL",
        "all_complete": all_complete,
        "complete_runs": summary["complete_runs"],
        "expected_runs": summary["expected_runs"],
        "failures": failures,
        "marker": marker if all_complete else None,
    }, sort_keys=True), flush=True)
    return 0 if all_complete else 2


def run_summarize(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    manifest_value = json.loads((root / "manifest.json").read_text())
    tasks = tuple(manifest_value["tasks"])
    seeds = tuple(int(seed) for seed in manifest_value["seeds"])
    summary = build_summary(root, tasks, seeds)
    write_checksums(root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_complete"] else 2


def add_sample_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-int", type=int, default=DEFAULT_SAMPLES["n_int"])
    parser.add_argument("--n-ic", type=int, default=DEFAULT_SAMPLES["n_ic"])
    parser.add_argument("--n-bc", type=int, default=DEFAULT_SAMPLES["n_bc"])
    parser.add_argument("--n-eval", type=int, default=DEFAULT_SAMPLES["n_eval"])
    parser.add_argument(
        "--history-eval-n",
        type=int,
        default=DEFAULT_SAMPLES["history_eval_n"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--task", choices=TASK_ORDER, required=True)
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--stage", choices=("smoke", "pilot", "formal"), required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    worker.add_argument("--seconds", type=float, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--smoke", action="store_true")
    add_sample_arguments(worker)
    worker.set_defaults(handler=run_worker)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--tasks", default="all")
    smoke.add_argument("--seconds", type=float, default=3.0)
    smoke.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    smoke.add_argument("--conclusion", type=Path)
    add_sample_arguments(smoke)
    smoke.set_defaults(handler=run_smoke)

    orchestrate = subparsers.add_parser("orchestrate")
    orchestrate.add_argument("--stage", choices=("pilot", "formal"), required=True)
    orchestrate.add_argument("--tasks", default="all")
    orchestrate.add_argument("--seeds", type=int, required=True)
    orchestrate.add_argument("--seconds", type=float, required=True)
    orchestrate.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    orchestrate.add_argument("--output-root", type=Path)
    orchestrate.add_argument("--resume", action="store_true")
    add_sample_arguments(orchestrate)
    orchestrate.set_defaults(handler=run_orchestrate)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-root", type=Path, required=True)
    summarize.set_defaults(handler=run_summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
