#!/usr/bin/env python3
"""Run the approved 2D Cahn--Hilliard search strictly after Poly completes.

Queue order:

1. wait for Poly's successful ``FORMAL_COMPLETE`` marker;
2. require the predecessor PID to remain alive while waiting;
3. require an empty GPU compute-process list for consecutive polls;
4. run an ephemeral minimal CUDA smoke;
5. run an ephemeral search-batch-sized CUDA smoke;
6. run the 60-second, 196-cell CH4/CH6 weight search serially.

This queue intentionally has no command for a 1200-second CH formal run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEARCH_RUNNER = ROOT / "scripts" / "run_cahn2d_weight_search.py"
PROTOCOL_ID = "cahn_hilliard_2d_after_poly_queue_v1"
SEARCH_PROTOCOL_ID = "cahn_hilliard_2d_natural_bc_common_sinh_fp32_v1"
EXPECTED_SEARCH_RUNS = 196
EXPECTED_CELLS_PER_SMOKE = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpu_compute_pids() -> tuple[int, ...]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pids = []
    for line in completed.stdout.splitlines():
        value = line.strip()
        if value:
            pids.append(int(value))
    return tuple(pids)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_smoke(path: Path) -> dict[str, Any]:
    conclusion = load_json(path)
    cells = conclusion.get("cells")
    if conclusion.get("protocol_id") != SEARCH_PROTOCOL_ID:
        raise ValueError(f"unexpected smoke protocol at {path}")
    if conclusion.get("passed") is not True:
        raise RuntimeError(f"smoke did not pass: {path}")
    if conclusion.get("raw_artifacts_retained") is not False:
        raise ValueError(f"smoke raw-artifact policy not satisfied: {path}")
    if not isinstance(cells, list) or len(cells) != EXPECTED_CELLS_PER_SMOKE:
        raise ValueError(f"expected four smoke cells at {path}")
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("status") != "complete":
            raise ValueError(f"incomplete smoke cell at {path}")
        if not all(finite(cell.get(key)) for key in ("loss", "rel_error", "peak_mb")):
            raise ValueError(f"non-finite smoke metric at {path}")
    return conclusion


def validate_search(search_root: Path) -> dict[str, Any]:
    if not (search_root / "SEARCH_COMPLETE").is_file():
        raise ValueError("search completion marker is missing")
    summary = load_json(search_root / "summary.json")
    if summary.get("protocol_id") != SEARCH_PROTOCOL_ID or summary.get("complete") is not True:
        raise ValueError("search summary is not complete")
    task_summaries = summary.get("task_summaries")
    if not isinstance(task_summaries, list) or len(task_summaries) != 2:
        raise ValueError("expected two complete task summaries")
    completed = 0
    for task_summary in task_summaries:
        if not isinstance(task_summary, dict) or task_summary.get("complete") is not True:
            raise ValueError("incomplete task summary")
        if task_summary.get("paired_complete_candidates") != 49:
            raise ValueError("task does not contain 49 paired candidates")
        completed += int(task_summary.get("complete_runs", 0))
    if completed != EXPECTED_SEARCH_RUNS:
        raise ValueError(f"expected {EXPECTED_SEARCH_RUNS} complete method runs; got {completed}")
    return summary


def run_checked(command: list[str]) -> None:
    print(json.dumps({"event": "command_start", "at": utc_now(), "command": command}), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    print(
        json.dumps({
            "event": "command_end",
            "at": utc_now(),
            "returncode": completed.returncode,
        }),
        flush=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"subprocess exited with code {completed.returncode}: {command}"
        )


def status_payload(args: argparse.Namespace, state: str, **extra: Any) -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "updated_at": utc_now(),
        "state": state,
        "queue_pid": os.getpid(),
        "poly_pid": args.poly_pid,
        "poly_root": str(args.poly_root),
        "search_root": str(args.search_root),
        "seconds_per_search_cell": args.search_seconds,
        "search_sample_counts": {
            "n_int": args.n_int,
            "n_ic": args.n_ic,
            "n_bc": args.n_bc,
            "n_eval": args.n_eval,
            "history_eval_n": args.history_eval_n,
        },
        "automatic_formal_ch_run": False,
        **extra,
    }


def build_smoke_command(
    conclusion: Path,
    *,
    seconds: float,
    sample_counts: dict[str, int] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(SEARCH_RUNNER),
        "smoke",
        "--tasks", "all",
        "--seconds", str(seconds),
        "--ephemeral-conclusion", str(conclusion),
    ]
    if sample_counts is not None:
        for option, value in sample_counts.items():
            command.extend([f"--{option.replace('_', '-')}", str(value)])
    return command


def run_queue(args: argparse.Namespace) -> int:
    args.poly_root = args.poly_root.resolve()
    args.queue_root = args.queue_root.resolve()
    args.search_root = args.search_root.resolve()
    args.queue_root.mkdir(parents=True, exist_ok=True)
    status_path = args.queue_root / "status.json"
    basic_conclusion = args.queue_root / "basic_smoke_conclusion.json"
    sized_conclusion = args.queue_root / "search_sized_smoke_conclusion.json"
    started_at = utc_now()
    try:
        if (args.search_root / "SEARCH_COMPLETE").is_file():
            summary = validate_search(args.search_root)
            atomic_write_json(
                status_path,
                status_payload(
                    args,
                    "complete",
                    started_at=started_at,
                    resumed_complete_search=True,
                    summary=summary,
                ),
            )
            atomic_write_text(args.queue_root / "QUEUE_COMPLETE", f"completed_at={utc_now()}\n")
            return 0

        deadline = time.monotonic() + args.max_wait_seconds
        while not (args.poly_root / "FORMAL_COMPLETE").is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for Poly FORMAL_COMPLETE")
            if not process_alive(args.poly_pid):
                raise RuntimeError(
                    "Poly orchestrator exited without FORMAL_COMPLETE"
                )
            atomic_write_json(
                status_path,
                status_payload(args, "waiting_for_poly", started_at=started_at),
            )
            time.sleep(args.poll_seconds)

        idle_polls = 0
        while idle_polls < args.required_idle_polls:
            pids = gpu_compute_pids()
            idle_polls = idle_polls + 1 if not pids else 0
            atomic_write_json(
                status_path,
                status_payload(
                    args,
                    "waiting_for_gpu_idle",
                    started_at=started_at,
                    idle_polls=idle_polls,
                    required_idle_polls=args.required_idle_polls,
                    observed_compute_pids=list(pids),
                ),
            )
            if idle_polls < args.required_idle_polls:
                time.sleep(args.poll_seconds)

        if basic_conclusion.is_file():
            validate_smoke(basic_conclusion)
        else:
            atomic_write_json(
                status_path,
                status_payload(args, "basic_smoke", started_at=started_at),
            )
            run_checked(build_smoke_command(basic_conclusion, seconds=args.smoke_seconds))
            validate_smoke(basic_conclusion)

        search_counts = {
            "n_int": args.n_int,
            "n_ic": args.n_ic,
            "n_bc": args.n_bc,
            "n_eval": args.n_eval,
            "history_eval_n": args.history_eval_n,
        }
        if sized_conclusion.is_file():
            validate_smoke(sized_conclusion)
        else:
            atomic_write_json(
                status_path,
                status_payload(args, "search_sized_smoke", started_at=started_at),
            )
            run_checked(
                build_smoke_command(
                    sized_conclusion,
                    seconds=args.smoke_seconds,
                    sample_counts=search_counts,
                )
            )
            validate_smoke(sized_conclusion)

        atomic_write_json(
            status_path,
            status_payload(args, "weight_search", started_at=started_at),
        )
        search_command = [
            sys.executable,
            str(SEARCH_RUNNER),
            "orchestrate",
            "--tasks", "all",
            "--seconds", str(args.search_seconds),
            "--output-root", str(args.search_root),
            "--resume",
            "--retries", str(args.retries),
        ]
        for option, value in search_counts.items():
            search_command.extend([f"--{option.replace('_', '-')}", str(value)])
        run_checked(search_command)
        summary = validate_search(args.search_root)
        atomic_write_json(
            status_path,
            status_payload(
                args,
                "complete",
                started_at=started_at,
                completed_at=utc_now(),
                summary=summary,
            ),
        )
        atomic_write_text(args.queue_root / "QUEUE_COMPLETE", f"completed_at={utc_now()}\n")
        return 0
    except BaseException as error:  # noqa: BLE001 - preserve queue failure
        failure = status_payload(
            args,
            "failed",
            started_at=started_at,
            failed_at=utc_now(),
            error_type=type(error).__name__,
            error=str(error)[:2000],
            traceback=traceback.format_exc(limit=50),
        )
        atomic_write_json(status_path, failure)
        atomic_write_text(
            args.queue_root / "QUEUE_FAILED",
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
        )
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poly-root", type=Path, required=True)
    parser.add_argument("--poly-pid", type=int, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--max-wait-seconds", type=float, default=86400.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--required-idle-polls", type=int, default=3)
    parser.add_argument("--smoke-seconds", type=float, default=3.0)
    parser.add_argument("--search-seconds", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--n-int", type=int, default=512)
    parser.add_argument("--n-ic", type=int, default=256)
    parser.add_argument("--n-bc", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=4096)
    parser.add_argument("--history-eval-n", type=int, default=1024)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.poly_pid <= 0:
        raise ValueError("poly PID must be positive")
    if args.poll_seconds <= 0 or args.max_wait_seconds <= 0:
        raise ValueError("wait durations must be positive")
    if args.required_idle_polls < 1:
        raise ValueError("required idle polls must be positive")
    if min(args.n_int, args.n_ic, args.n_bc, args.n_eval, args.history_eval_n) <= 0:
        raise ValueError("all sample counts must be positive")
    return run_queue(args)


if __name__ == "__main__":
    raise SystemExit(main())
