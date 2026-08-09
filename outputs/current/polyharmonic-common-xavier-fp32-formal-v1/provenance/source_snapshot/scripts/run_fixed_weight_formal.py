#!/usr/bin/env python3
"""Run the approved fixed-weight WAR/real-autodiff formal experiment.

The formal protocol is intentionally separate from the exhaustive 60-second
weight search.  It evaluates the five selected task/weight pairs with five
training seeds, running WAR and the width-matched real tanh autodiff method
serially on one GPU.  Every cell is written atomically and can be resumed.

Examples
--------
Smoke the complete task/method matrix::

    python scripts/run_fixed_weight_formal.py smoke --seconds 5

Run or resume the approved 1200-second experiment::

    python scripts/run_fixed_weight_formal.py orchestrate --seconds 1200 \
        --seeds 5 --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from run_weight_search import (
    COMPLEX_DTYPE,
    EVAL_SEED,
    METHODS,
    TASKS,
    TASK_ORDER as _SEARCH_TASK_ORDER,
    atomic_write_json,
    atomic_write_text,
    git_state,
    hardware_metadata,
    load_result,
    REAL_DTYPE,
    result_complete,
    train_one,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "war_realad_fixed_weights_common_xavier_fp32_v1"
ENGINE_PROTOCOL_ID = "war_realad_weight_grid_common_xavier_fp32_v1"
DEFAULT_ROOT = ROOT / "outputs" / PROTOCOL_ID
TASK_ORDER = (
    "poly_d2_o2",
    "poly_d2_o4",
    "poly_d2_o6",
    "cahn_hilliard_o4",
    "cahn_hilliard_o6",
)
FIXED_WEIGHTS: dict[str, tuple[float, ...]] = {
    "poly_d2_o2": (1e0,),
    "poly_d2_o4": (1e0, 1e0),
    "poly_d2_o6": (1e1, 1e0, 1e0),
    "cahn_hilliard_o4": (1e1, 1e1),
    "cahn_hilliard_o6": (1e1, 1e1),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_name(path.name + f".tmp.{__import__('os').getpid()}")
    with temporary.open("w", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def selected_tasks(text: str) -> tuple[str, ...]:
    names = TASK_ORDER if text == "all" else tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [name for name in names if name not in TASKS]
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    return names


def seed_list(count: int) -> tuple[int, ...]:
    if count <= 0:
        raise ValueError("--seeds must be positive")
    return tuple(range(count))


def complete_formal_result(path: Path, *, task: str, method: str, seed: int, seconds: float) -> bool:
    if not result_complete(path):
        return False
    result = load_result(path) or {}
    return (
        result.get("formal_protocol_id") == PROTOCOL_ID
        and result.get("task_id") == task
        and result.get("method") == method
        and int(result.get("seed", -1)) == seed
        and math.isclose(float(result.get("budget_seconds", -1.0)), seconds)
    )


def manifest(tasks: Iterable[str], seconds: float, seeds: tuple[int, ...], *, smoke: bool) -> dict[str, Any]:
    task_list = list(tasks)
    return {
        "protocol_id": PROTOCOL_ID,
        "engine_protocol_id": ENGINE_PROTOCOL_ID,
        "created_at": utc_now(),
        "smoke": smoke,
        "tasks": [
            {
                "task_id": task,
                "family": TASKS[task].family,
                "order": TASKS[task].order,
                "weight_names": list(TASKS[task].weight_names),
                "weights": list(FIXED_WEIGHTS[task]),
                "weight_labels": [f"1e{round(math.log10(value)):+d}" for value in FIXED_WEIGHTS[task]],
            }
            for task in task_list
        ],
        "methods": list(METHODS),
        "architecture": {
            "war": {
                "representation": "native_complex",
                "activation": "sinh",
                "backend": "waring_complex_jet",
                "hidden": 128,
                "depth": 4,
                "init_mode": "common_xavier",
                "frequency_initialization": "disabled",
                "parameter_dtype": str(COMPLEX_DTYPE),
            },
            "real_tanh_autodiff": {
                "representation": "real",
                "activation": "tanh",
                "backend": "direct_autodiff",
                "hidden": 128,
                "depth": 4,
                "init_mode": "common_xavier",
                "frequency_initialization": "disabled",
                "parameter_dtype": str(REAL_DTYPE),
            },
        },
        "seconds_per_method_seed": seconds,
        "seeds": list(seeds),
        "eval_seed": EVAL_SEED,
        "task_count": len(task_list),
        "method_seed_run_count": len(task_list) * len(METHODS) * len(seeds),
        "nominal_training_seconds": len(task_list) * len(METHODS) * len(seeds) * seconds,
        "serial_single_gpu": True,
        "history_required_fields": ["elapsed_seconds", "step", "rel_error", "loss"],
        "git": git_state(),
        "hardware": hardware_metadata(),
    }


def worker_command(task: str, method: str, weights: tuple[float, ...], seed: int,
                   eval_seed: int, seconds: float, output: Path, *, smoke: bool) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--task", task,
        "--method", method,
        "--weights", ",".join(f"{value:.16g}" for value in weights),
        "--seed", str(seed),
        "--eval-seed", str(eval_seed),
        "--seconds", str(seconds),
        "--output", str(output),
    ]
    if smoke:
        command.append("--smoke")
    return command


def run_worker(args: argparse.Namespace) -> int:
    task = TASKS[args.task]
    weights = FIXED_WEIGHTS[args.task]
    output = args.output.resolve()
    base: dict[str, Any] = {
        "protocol_id": ENGINE_PROTOCOL_ID,
        "formal_protocol_id": PROTOCOL_ID,
        "status": "running",
        "task_id": args.task,
        "method": args.method,
        "weights": list(weights),
        "seed": args.seed,
        "eval_seed": args.eval_seed,
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
            train_seed=args.seed,
            eval_seed=args.eval_seed,
        )
        result.update({
            "formal_protocol_id": PROTOCOL_ID,
            "seed": args.seed,
            "eval_seed": args.eval_seed,
            "git": git_state(),
            "hardware": hardware_metadata(),
        })
        atomic_write_json(output, result)
        # Keep loss and rel_error on the final line for machine and human audit.
        print(json.dumps({
            "status": result["status"],
            "task_id": args.task,
            "method": args.method,
            "seed": args.seed,
            "loss": result.get("loss"),
            "rel_error": result.get("rel_error"),
        }, sort_keys=True), flush=True)
        return 0 if result["status"] == "complete" else 2
    except BaseException as error:  # noqa: BLE001 - persist every worker failure
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
            failure["metadata_error"] = repr(metadata_error)[:1000]
        atomic_write_json(output, failure)
        print(json.dumps({
            "status": "failed",
            "task_id": args.task,
            "method": args.method,
            "seed": args.seed,
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        }, sort_keys=True), flush=True)
        return 1


def archive_incomplete(output: Path, point_dir: Path, method: str) -> None:
    if not output.exists():
        return
    attempts = point_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    shutil.copy2(output, attempts / f"{method}.{stamp}.json")


def flatten_result(task: str, method: str, seed: int, result: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task_id": task,
        "method": method,
        "seed": seed,
        "weights": json.dumps(FIXED_WEIGHTS[task]),
        "status": "missing" if result is None else result.get("status", "unknown"),
    }
    if result:
        for key in ("loss", "rel_error", "L2_err", "steps", "training_seconds", "evaluation_seconds", "ms_per_step", "peak_mb", "started_at", "completed_at"):
            if key in result:
                row[key] = result[key]
        row["history_points"] = len(result.get("history", []))
        row["final_history_has_loss"] = bool(result.get("history") and "loss" in result["history"][-1])
        row["final_history_has_rel_error"] = bool(result.get("history") and "rel_error" in result["history"][-1])
    return row


def build_summary(root: Path, tasks: tuple[str, ...], seeds: tuple[int, ...]) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for task in tasks:
        seed_rows: list[dict[str, Any]] = []
        for seed in seeds:
            pair: dict[str, Any] = {"task_id": task, "seed": seed, "weights": list(FIXED_WEIGHTS[task])}
            results: dict[str, dict[str, Any] | None] = {}
            for method in METHODS:
                output = root / task / f"seed_{seed:03d}" / f"{method}.json"
                result = load_result(output)
                results[method] = result
                all_rows.append(flatten_result(task, method, seed, result))
                if result is not None and result.get("status") == "complete":
                    pair[f"{method}_loss"] = float(result["loss"])
                    pair[f"{method}_rel_error"] = float(result["rel_error"])
            if all(results.get(method) and results[method].get("status") == "complete" for method in METHODS):
                war_error = float(results["war"]["rel_error"])
                ad_error = float(results["real_tanh_autodiff"]["rel_error"])
                pair["geometric_mean"] = math.sqrt(max(0.0, war_error * ad_error))
                pair["max_error"] = max(war_error, ad_error)
                pair["mean_error"] = 0.5 * (war_error + ad_error)
                pair["status"] = "complete"
            else:
                pair["status"] = "incomplete"
            seed_rows.append(pair)
        ranking_dir = root / task / "rankings"
        ranking_dir.mkdir(parents=True, exist_ok=True)
        complete_pairs = [row for row in seed_rows if row["status"] == "complete"]
        for name, key in {
            "war": lambda row: row.get("war_rel_error", math.inf),
            "real_tanh_autodiff": lambda row: row.get("real_tanh_autodiff_rel_error", math.inf),
            "geometric_mean": lambda row: row.get("geometric_mean", math.inf),
            "minimax": lambda row: row.get("max_error", math.inf),
        }.items():
            ranked = sorted(complete_pairs, key=lambda row: (key(row), row["seed"]))
            atomic_write_json(ranking_dir / f"{name}.json", ranked)
            write_csv(ranking_dir / f"{name}.csv", ranked)
        war_values = [row["war_rel_error"] for row in complete_pairs]
        ad_values = [row["real_tanh_autodiff_rel_error"] for row in complete_pairs]
        gm_values = [row["geometric_mean"] for row in complete_pairs]
        mm_values = [row["max_error"] for row in complete_pairs]
        summary = {
            "task_id": task,
            "weights": list(FIXED_WEIGHTS[task]),
            "expected_seed_count": len(seeds),
            "paired_complete_seed_count": len(complete_pairs),
            "war_complete": sum(1 for row in all_rows if row["task_id"] == task and row["method"] == "war" and row["status"] == "complete"),
            "real_tanh_autodiff_complete": sum(1 for row in all_rows if row["task_id"] == task and row["method"] == "real_tanh_autodiff" and row["status"] == "complete"),
            "war_mean_rel_error": sum(war_values) / len(war_values) if war_values else None,
            "real_tanh_autodiff_mean_rel_error": sum(ad_values) / len(ad_values) if ad_values else None,
            "geometric_mean_mean": sum(gm_values) / len(gm_values) if gm_values else None,
            "minimax_worst_seed": max(mm_values) if mm_values else None,
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


def run_smoke(args: argparse.Namespace) -> int:
    tasks = selected_tasks(args.tasks)
    seeds = (0,)
    root = (args.output_root or DEFAULT_ROOT / "_smoke").resolve()
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "manifest.json", manifest(tasks, args.seconds, seeds, smoke=True))
    failures = 0
    for task in tasks:
        for method in METHODS:
            output = root / task / "seed_000" / f"{method}.json"
            log = root / task / "seed_000" / f"{method}.log"
            output.parent.mkdir(parents=True, exist_ok=True)
            command = worker_command(task, method, FIXED_WEIGHTS[task], 0, args.eval_seed, args.seconds, output, smoke=True)
            print(f"[smoke] {task} {method} weights={FIXED_WEIGHTS[task]}", flush=True)
            with log.open("w") as handle:
                completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=max(300.0, args.seconds * 20.0), check=False)
            if completed.returncode != 0 or not complete_formal_result(output, task=task, method=method, seed=0, seconds=args.seconds):
                failures += 1
                print(f"[smoke-failed] see {log}", flush=True)
            else:
                result = load_result(output) or {}
                print(f"[smoke-ok] loss={result.get('loss')} rel_error={result.get('rel_error')}", flush=True)
    build_summary(root, tasks, seeds)
    write_checksums(root)
    marker = "SMOKE_COMPLETE" if failures == 0 else "SMOKE_COMPLETE_WITH_FAILURES"
    atomic_write_text(root / marker, f"failures={failures}\n")
    return 0 if failures == 0 else 1


def run_orchestrator(args: argparse.Namespace) -> int:
    tasks = selected_tasks(args.tasks)
    seeds = seed_list(args.seeds)
    root = (args.output_root or DEFAULT_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    expected_manifest = manifest(tasks, args.seconds, seeds, smoke=False)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in ("protocol_id", "seconds_per_method_seed", "seeds", "tasks", "methods"):
            if existing.get(key) != expected_manifest.get(key):
                raise ValueError(f"incompatible manifest field {key!r} at {manifest_path}")
    else:
        atomic_write_json(manifest_path, expected_manifest)

    total = len(tasks) * len(seeds) * len(METHODS)
    processed = 0
    attempted = 0
    failures = 0
    started = time.time()
    for task_name in tasks:
        task_dir = root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(task_dir / "manifest.json", {
            "protocol_id": PROTOCOL_ID,
            "task_id": task_name,
            "weights": list(FIXED_WEIGHTS[task_name]),
            "weight_names": list(TASKS[task_name].weight_names),
            "methods": list(METHODS),
            "seeds": list(seeds),
            "seconds_per_method_seed": args.seconds,
        })
        for seed in seeds:
            point_dir = task_dir / f"seed_{seed:03d}"
            point_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(point_dir / "config.json", {
                "task_id": task_name,
                "weights": list(FIXED_WEIGHTS[task_name]),
                "seed": seed,
                "eval_seed": args.eval_seed,
                "seconds": args.seconds,
                "methods": list(METHODS),
            })
            for method in METHODS:
                output = point_dir / f"{method}.json"
                log = point_dir / f"{method}.log"
                if args.resume and complete_formal_result(output, task=task_name, method=method, seed=seed, seconds=args.seconds):
                    processed += 1
                    continue
                if output.exists():
                    archive_incomplete(output, point_dir, method)
                for attempt in range(1, args.retries + 2):
                    attempted += 1
                    command = worker_command(task_name, method, FIXED_WEIGHTS[task_name], seed, args.eval_seed, args.seconds, output, smoke=False)
                    with log.open("a") as handle:
                        handle.write(f"\n# attempt={attempt} started_at={utc_now()}\n")
                        handle.flush()
                        completed_proc = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, timeout=max(1800.0, args.seconds * 2.0 + 300.0), check=False)
                    if completed_proc.returncode == 0 and complete_formal_result(output, task=task_name, method=method, seed=seed, seconds=args.seconds):
                        break
                    failures += 1
                    if attempt <= args.retries:
                        archive_incomplete(output, point_dir, method)
                        print(f"[retry] {task_name} seed={seed} {method} attempt={attempt}", flush=True)
                processed += 1
                elapsed = time.time() - started
                eta = elapsed / max(1, processed) * max(0, total - processed)
                result = load_result(output) or {}
                atomic_write_json(root / "progress.json", {
                    "protocol_id": PROTOCOL_ID,
                    "updated_at": utc_now(),
                    "total_runs": total,
                    "processed_runs": processed,
                    "attempted_runs": attempted,
                    "failures_seen_this_process": failures,
                    "current_task": task_name,
                    "current_seed": seed,
                    "current_method": method,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": eta,
                    "last_status": result.get("status"),
                    "last_loss": result.get("loss"),
                    "last_rel_error": result.get("rel_error"),
                })
                print(f"[run {processed}/{total}] {task_name} seed={seed} {method} status={result.get('status')} loss={result.get('loss')} rel_error={result.get('rel_error')}", flush=True)

    final = build_summary(root, tasks, seeds)
    final.update({"attempted_runs": attempted, "failures_seen_this_process": failures, "completed_at": utc_now()})
    atomic_write_json(root / "summary.json", final)
    write_checksums(root)
    marker = "FORMAL_COMPLETE" if final["all_complete"] else "FORMAL_COMPLETE_WITH_FAILURES"
    atomic_write_text(root / marker, json.dumps(final, indent=2) + "\n")
    return 0 if final["all_complete"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--task", choices=TASK_ORDER, required=True)
    worker.add_argument("--method", choices=METHODS, required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    worker.add_argument("--weights", required=False)
    worker.add_argument("--seconds", type=float, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--smoke", action="store_true")
    worker.set_defaults(func=run_worker)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--tasks", default="all")
    smoke.add_argument("--seconds", type=float, default=5.0)
    smoke.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    smoke.add_argument("--output-root", type=Path)
    smoke.set_defaults(func=run_smoke)
    orchestrate = sub.add_parser("orchestrate")
    orchestrate.add_argument("--tasks", default="all")
    orchestrate.add_argument("--seconds", type=float, default=1200.0)
    orchestrate.add_argument("--seeds", type=int, default=5)
    orchestrate.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    orchestrate.add_argument("--output-root", type=Path)
    orchestrate.add_argument("--resume", action="store_true")
    orchestrate.add_argument("--retries", type=int, default=1)
    orchestrate.set_defaults(func=run_orchestrator)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "worker" and args.weights:
        parsed = tuple(float(value) for value in args.weights.split(",") if value)
        if parsed != FIXED_WEIGHTS[args.task]:
            raise ValueError(f"weights for {args.task} are fixed at {FIXED_WEIGHTS[args.task]}")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
