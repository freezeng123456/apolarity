#!/usr/bin/env python3
"""Run the gated HO-04 hyper-NS search, pilot, and formal workflow.

The workflow is intentionally fail-closed.  It never changes the equation,
precision, architecture, sample counts, weights grid, or wall-clock budgets in
response to a bad result.  A failed gate writes an auditable STOP marker and
does not launch later stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK = "hyperviscous_ns_2d_o4"
METHODS = ("war", "real_tanh_autodiff")
SAMPLES = {
    "n_int": 2048,
    "n_ic": 512,
    "n_bc": 1024,
    "n_eval": 16384,
    "history_eval_n": 2048,
}
SENTINEL_WEIGHTS = ((1.0, 1.0), (10.0, 1.0), (10.0, 10.0))
SENTINEL_SECONDS = 180.0
SEARCH_SECONDS = 60.0
PILOT_SECONDS = 600.0
FORMAL_SECONDS = 1200.0
PILOT_SEEDS = 3
FORMAL_SEEDS = 5
EVAL_SEED = 68421
GRID_VALUES = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


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


def gpu_processes() -> list[dict[str, str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr[-1000:]}")
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            rows.append({
                "pid": parts[0],
                "process_name": parts[1],
                "used_memory_mb": parts[2],
            })
    return rows


def sample_flags() -> list[str]:
    return [
        "--n-int", str(SAMPLES["n_int"]),
        "--n-ic", str(SAMPLES["n_ic"]),
        "--n-bc", str(SAMPLES["n_bc"]),
        "--n-eval", str(SAMPLES["n_eval"]),
        "--history-eval-n", str(SAMPLES["history_eval_n"]),
    ]


def run_logged(
    command: list[str],
    log: Path,
    *,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"\n# launch={utc_now()}\n")
        handle.write("# command=" + json.dumps(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    return completed.returncode


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def result_valid(
    result: dict[str, Any] | None,
    *,
    method: str,
    weights: tuple[float, float],
    seconds: float,
    sha: str,
) -> bool:
    if not isinstance(result, dict):
        return False
    found_weights = tuple(float(value) for value in result.get("weights", ()))
    return bool(
        result.get("status") == "complete"
        and result.get("task_id") == TASK
        and result.get("method") == method
        and found_weights == weights
        and math.isclose(float(result.get("budget_seconds", -1.0)), seconds)
        and result.get("git", {}).get("sha") == sha
        and finite_tree(result.get("history", []))
        and finite_tree(result.get("metrics", {}))
        and math.isfinite(float(result.get("loss", math.inf)))
        and math.isfinite(float(result.get("rel_error", math.inf)))
    )


def physical_metrics_pass(result: dict[str, Any]) -> bool:
    metrics = result.get("metrics", {})
    return bool(
        float(metrics.get("divergence_rms", math.inf)) < 0.5
        and float(metrics.get("pressure_mean_max_abs", math.inf)) < 0.5
        and float(metrics.get("energy_relative_rmse", math.inf)) < 1.0
        and float(metrics.get("pressure_rel_error", math.inf)) < 2.0
    )


def write_pipeline_status(root: Path, stage: str, **extra: Any) -> None:
    atomic_write_json(root / "pipeline_status.json", {
        "protocol": "hyperns_ho04_gated_pipeline_v1",
        "stage": stage,
        "updated_at": utc_now(),
        **extra,
    })


def stop_pipeline(root: Path, stage: str, reason: str, details: Any) -> int:
    payload = {
        "decision": "STOP",
        "stage": stage,
        "reason": reason,
        "details": details,
        "created_at": utc_now(),
    }
    atomic_write_json(root / f"{stage}_GATE.json", payload)
    atomic_write_text(root / "PIPELINE_STOPPED", f"{stage}: {reason}\n")
    write_pipeline_status(root, "stopped", stop_stage=stage, reason=reason)
    return 3


def ensure_smoke(
    root: Path,
    name: str,
    python: str,
    sha: str,
    *,
    full_size: bool,
) -> None:
    conclusion = root / "smoke" / f"{name}.json"
    existing = load_json(conclusion)
    if (
        isinstance(existing, dict)
        and existing.get("passed") is True
        and existing.get("git", {}).get("sha") == sha
    ):
        return
    command = [
        python,
        str(ROOT / "scripts" / "run_hyperns_weight_search.py"),
        "smoke",
        "--seconds", "3",
        "--ephemeral-conclusion", str(conclusion),
    ]
    if full_size:
        command.extend(sample_flags())
    return_code = run_logged(
        command,
        root / "smoke" / f"{name}.log",
        timeout=3600.0,
    )
    value = load_json(conclusion)
    if return_code != 0 or not isinstance(value, dict) or not value.get("passed"):
        raise RuntimeError(f"{name} CUDA smoke failed")


def run_sentinel(root: Path, python: str, sha: str) -> dict[str, Any]:
    sentinel_root = root / "sentinel"
    cells: list[dict[str, Any]] = []
    for index, weights in enumerate(SENTINEL_WEIGHTS):
        point_dir = sentinel_root / f"point_{index:03d}"
        atomic_write_json(point_dir / "candidate.json", {
            "candidate_id": f"point_{index:03d}",
            "weights": list(weights),
            "weight_map": {"lambda_ic": weights[0], "lambda_bc": weights[1]},
        })
        method_order = METHODS if index % 2 == 0 else tuple(reversed(METHODS))
        for method in method_order:
            output = point_dir / f"{method}.json"
            result = load_json(output)
            if not result_valid(
                result if isinstance(result, dict) else None,
                method=method,
                weights=weights,
                seconds=SENTINEL_SECONDS,
                sha=sha,
            ):
                command = [
                    python,
                    str(ROOT / "scripts" / "run_hyperns_weight_search.py"),
                    "worker",
                    "--task", TASK,
                    "--method", method,
                    "--weights", f"{weights[0]},{weights[1]}",
                    "--seconds", str(SENTINEL_SECONDS),
                    "--output", str(output),
                    "--train-seed", "42",
                    "--eval-seed", str(EVAL_SEED),
                    *sample_flags(),
                ]
                return_code = run_logged(
                    command,
                    point_dir / f"{method}.log",
                    timeout=3600.0,
                )
                result = load_json(output)
                if return_code != 0 or not result_valid(
                    result if isinstance(result, dict) else None,
                    method=method,
                    weights=weights,
                    seconds=SENTINEL_SECONDS,
                    sha=sha,
                ):
                    raise RuntimeError(
                        f"sentinel cell failed: {weights} {method}"
                    )
            assert isinstance(result, dict)
            cells.append({
                "candidate_id": f"point_{index:03d}",
                "weights": list(weights),
                "method": method,
                "loss": result["loss"],
                "rel_error": result["rel_error"],
                "metrics": result["metrics"],
                "steps": result["steps"],
                "ms_per_step": result["ms_per_step"],
                "peak_mb": result["peak_mb"],
            })
        write_pipeline_status(
            root,
            "sentinel",
            complete_cells=len(cells),
            expected_cells=len(SENTINEL_WEIGHTS) * len(METHODS),
        )

    candidates: list[dict[str, Any]] = []
    for index, weights in enumerate(SENTINEL_WEIGHTS):
        results = {
            method: load_json(
                sentinel_root / f"point_{index:03d}" / f"{method}.json"
            )
            for method in METHODS
        }
        assert all(isinstance(result, dict) for result in results.values())
        typed = {key: value for key, value in results.items() if isinstance(value, dict)}
        errors = [float(typed[method]["rel_error"]) for method in METHODS]
        eligible = bool(
            min(errors) < 0.2
            and max(errors) < 0.75
            and all(physical_metrics_pass(typed[method]) for method in METHODS)
        )
        candidates.append({
            "candidate_id": f"point_{index:03d}",
            "weights": list(weights),
            "war_rel_error": errors[0],
            "real_tanh_autodiff_rel_error": errors[1],
            "max_error": max(errors),
            "geometric_mean": math.sqrt(errors[0] * errors[1]),
            "eligible": eligible,
        })
    gate = {
        "decision": "GO_SEARCH" if any(row["eligible"] for row in candidates) else "STOP",
        "rule": (
            "at least one shared sentinel weight: best method rel_error<0.2, "
            "other<0.75, and both pass registered divergence/gauge/energy/pressure gates"
        ),
        "candidates": candidates,
        "cells": cells,
        "created_at": utc_now(),
    }
    atomic_write_json(root / "sentinel_GATE.json", gate)
    return gate


def run_full_search(root: Path, python: str) -> Path:
    search_root = root / "search"
    command = [
        python,
        str(ROOT / "scripts" / "run_hyperns_weight_search.py"),
        "orchestrate",
        "--seconds", str(SEARCH_SECONDS),
        "--output-root", str(search_root),
        "--resume",
        "--retries", "1",
        *sample_flags(),
    ]
    return_code = run_logged(
        command,
        root / "search.log",
        timeout=24 * 3600.0,
    )
    if return_code != 0 or not (search_root / "SEARCH_COMPLETE").is_file():
        raise RuntimeError("complete 7x7 search did not finish cleanly")
    return search_root


def select_shared_weight(root: Path, search_root: Path) -> tuple[float, float]:
    ranking_path = (
        search_root / TASK / "rankings" / "ranking_shared_minimax.json"
    )
    ranking = load_json(ranking_path)
    if not isinstance(ranking, list) or len(ranking) != 49:
        raise RuntimeError("shared minimax ranking is incomplete")
    selected: dict[str, Any] | None = None
    for row in ranking:
        if not isinstance(row, dict):
            continue
        weights = tuple(float(value) for value in row.get("weights", ()))
        if len(weights) != 2 or any(value not in GRID_VALUES for value in weights):
            continue
        if not math.isfinite(float(row.get("max_error", math.inf))):
            continue
        if float(row["max_error"]) >= 1.25:
            continue
        selected = row
        break
    if selected is None:
        raise RuntimeError("no finite shared search candidate passed max_error<1.25")
    weights = tuple(float(value) for value in selected["weights"])
    assert len(weights) == 2
    payload = {
        "selection_rule": (
            "first complete shared-minimax ranking entry with max_error<1.25; "
            "ties in the ranking are geomean then total weight"
        ),
        "candidate_id": selected["candidate_id"],
        "weights": list(weights),
        "weight_map": {"lambda_ic": weights[0], "lambda_bc": weights[1]},
        "search_metrics": selected,
        "created_at": utc_now(),
    }
    atomic_write_json(root / "selected_weight.json", payload)
    return weights


def fixed_env(weights: tuple[float, float]) -> dict[str, str]:
    env = dict(os.environ)
    env["APOLARITY_HYPERNS_FIXED_WEIGHTS"] = f"{weights[0]},{weights[1]}"
    return env


def run_fixed_stage(
    root: Path,
    python: str,
    weights: tuple[float, float],
    *,
    stage: str,
    seeds: int,
    seconds: float,
) -> Path:
    stage_root = root / stage
    command = [
        python,
        str(ROOT / "scripts" / "run_hyperns_fixed.py"),
        "orchestrate",
        "--stage", stage,
        "--seeds", str(seeds),
        "--seconds", str(seconds),
        "--output-root", str(stage_root),
        "--resume",
        *sample_flags(),
    ]
    return_code = run_logged(
        command,
        root / f"{stage}.log",
        env=fixed_env(weights),
        timeout=48 * 3600.0,
    )
    marker = "PILOT_COMPLETE" if stage == "pilot" else "FORMAL_COMPLETE"
    if return_code != 0 or not (stage_root / marker).is_file():
        raise RuntimeError(f"{stage} did not finish cleanly")
    return stage_root


def pilot_gate(root: Path, pilot_root: Path) -> dict[str, Any]:
    summary = load_json(pilot_root / "summary.json")
    if not isinstance(summary, dict):
        raise RuntimeError("pilot summary missing")
    task_summaries = summary.get("task_summaries", [])
    if len(task_summaries) != 1 or not isinstance(task_summaries[0], dict):
        raise RuntimeError("pilot task summary malformed")
    task_summary = task_summaries[0]
    seed_metrics = task_summary.get("seed_metrics", [])
    individual_errors: list[float] = []
    physical: dict[str, dict[str, float]] = {}
    for method in METHODS:
        values: dict[str, list[float]] = {
            "pressure_rel_error": [],
            "divergence_rms": [],
            "pressure_mean_max_abs": [],
            "energy_relative_rmse": [],
        }
        for seed in range(PILOT_SEEDS):
            result = load_json(
                pilot_root / TASK / f"seed_{seed:03d}" / f"{method}.json"
            )
            if not isinstance(result, dict) or result.get("status") != "complete":
                raise RuntimeError(f"pilot result missing: seed={seed} method={method}")
            individual_errors.append(float(result["rel_error"]))
            for key in values:
                values[key].append(float(result["metrics"][key]))
        physical[method] = {
            key: statistics.median(metric_values)
            for key, metric_values in values.items()
        }
    physics_pass = all(
        item["divergence_rms"] < 0.25
        and item["pressure_mean_max_abs"] < 0.25
        and item["energy_relative_rmse"] < 0.5
        and item["pressure_rel_error"] < 1.5
        for item in physical.values()
    )
    no_degenerate_seed = max(individual_errors) < 0.95
    decision = bool(
        summary.get("all_complete")
        and task_summary.get("screen_pass")
        and physics_pass
        and no_degenerate_seed
    )
    gate = {
        "decision": "GO_FORMAL" if decision else "STOP",
        "accuracy_gate": task_summary.get("screen_pass"),
        "physics_gate": physics_pass,
        "no_degenerate_seed_gate": no_degenerate_seed,
        "individual_velocity_rel_errors": individual_errors,
        "median_physical_metrics": physical,
        "rule": (
            "6/6 complete; best method median velocity error<0.2; other<0.75; "
            "every seed<0.95; per-method median divergence<0.25, pressure mean<0.25, "
            "energy error<0.5 and pressure error<1.5"
        ),
        "created_at": utc_now(),
    }
    atomic_write_json(root / "pilot_GATE.json", gate)
    return gate


def write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS", "run.pid"}
        and ".tmp." not in path.name
    )
    atomic_write_text(
        root / "SHA256SUMS",
        "\n".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}"
            for path in paths
        ) + "\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(root / "run.pid", str(os.getpid()) + "\n")
    state = git_state()
    if state["dirty"]:
        raise RuntimeError("pipeline requires a clean immutable git worktree")
    manifest = {
        "protocol": "hyperns_ho04_gated_pipeline_v1",
        "created_at": utc_now(),
        "git": state,
        "task": TASK,
        "methods": list(METHODS),
        "samples": SAMPLES,
        "sentinel_weights": [list(value) for value in SENTINEL_WEIGHTS],
        "sentinel_seconds": SENTINEL_SECONDS,
        "search_grid": list(GRID_VALUES),
        "search_seconds": SEARCH_SECONDS,
        "pilot": {"seeds": PILOT_SEEDS, "seconds": PILOT_SECONDS},
        "formal": {"seeds": FORMAL_SEEDS, "seconds": FORMAL_SECONDS},
        "strictly_serial_single_gpu": True,
        "automatic_parameter_changes": False,
    }
    manifest_path = root / "pipeline_manifest.json"
    existing = load_json(manifest_path)
    if isinstance(existing, dict):
        for key in (
            "protocol", "git", "task", "methods", "samples",
            "sentinel_weights", "sentinel_seconds", "search_grid",
            "search_seconds", "pilot", "formal",
        ):
            if existing.get(key) != manifest.get(key):
                raise RuntimeError(f"incompatible pipeline manifest field {key}")
    else:
        atomic_write_json(manifest_path, manifest)

    active = gpu_processes()
    if active:
        return stop_pipeline(root, "gpu_preflight", "GPU is not exclusive", active)

    try:
        write_pipeline_status(root, "basic_smoke")
        ensure_smoke(root, "basic", args.python, state["sha"], full_size=False)
        write_pipeline_status(root, "full_size_smoke")
        ensure_smoke(root, "full_size", args.python, state["sha"], full_size=True)

        write_pipeline_status(root, "sentinel")
        sentinel = run_sentinel(root, args.python, state["sha"])
        if sentinel["decision"] != "GO_SEARCH":
            return stop_pipeline(
                root, "sentinel", "no sentinel candidate passed", sentinel
            )

        write_pipeline_status(root, "search")
        search_root = run_full_search(root, args.python)
        try:
            weights = select_shared_weight(root, search_root)
        except RuntimeError as error:
            return stop_pipeline(root, "search", str(error), {})

        write_pipeline_status(root, "pilot", selected_weights=list(weights))
        pilot_root = run_fixed_stage(
            root,
            args.python,
            weights,
            stage="pilot",
            seeds=PILOT_SEEDS,
            seconds=PILOT_SECONDS,
        )
        gate = pilot_gate(root, pilot_root)
        if gate["decision"] != "GO_FORMAL":
            return stop_pipeline(root, "pilot", "pilot gate did not pass", gate)

        write_pipeline_status(root, "formal", selected_weights=list(weights))
        formal_root = run_fixed_stage(
            root,
            args.python,
            weights,
            stage="formal",
            seeds=FORMAL_SEEDS,
            seconds=FORMAL_SECONDS,
        )
        write_pipeline_status(root, "plotting", selected_weights=list(weights))
        plot_return = run_logged(
            [
                args.python,
                str(ROOT / "scripts" / "plot_hyperns_results.py"),
                "--formal-root", str(formal_root),
                "--output-dir", str(root / "analysis"),
            ],
            root / "plot.log",
            timeout=3600.0,
        )
        if plot_return != 0:
            raise RuntimeError("server-side result plotting failed")
        write_checksums(root)
        atomic_write_text(root / "PIPELINE_COMPLETE", utc_now() + "\n")
        write_pipeline_status(
            root, "complete", selected_weights=list(weights), formal_complete=True
        )
        return 0
    except BaseException as error:  # noqa: BLE001 - preserve exact stop evidence
        payload = {
            "decision": "STOP",
            "stage": (load_json(root / "pipeline_status.json") or {}).get(
                "stage", "unknown"
            ),
            "error_type": type(error).__name__,
            "error": str(error)[:4000],
            "created_at": utc_now(),
        }
        atomic_write_json(root / "PIPELINE_FAILURE.json", payload)
        atomic_write_text(
            root / "PIPELINE_STOPPED",
            f"{payload['stage']}: {payload['error_type']}: {payload['error']}\n",
        )
        write_pipeline_status(
            root,
            "stopped",
            stop_stage=payload["stage"],
            reason=payload["error"],
        )
        write_checksums(root)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

