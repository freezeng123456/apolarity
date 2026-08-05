#!/usr/bin/env python3
"""Deadline-aware sequential scheduler for the approved one-H20 experiment set."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


BASE = Path("/root/apolarity_overnight_20260804")
REPO = BASE / "src" / "apolarity"
RESULTS = BASE / "results"
LOGS = BASE / "logs"
PYTHON = BASE / "miniforge3" / "envs" / "apolarity" / "bin" / "python"
STATUS_PATH = RESULTS / "suite_status.json"
DISPATCH_DEADLINE = datetime.fromisoformat("2026-08-05T09:00:00+08:00").timestamp()
HARD_DEADLINE = datetime.fromisoformat("2026-08-05T09:15:00+08:00").timestamp()
STOP = False


def request_stop(_signum, _frame) -> None:
    global STOP
    STOP = True


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def main() -> None:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "created_at": now_text(),
        "dispatch_deadline": "2026-08-05T09:00:00+08:00",
        "hard_deadline": "2026-08-05T09:15:00+08:00",
        "source_parent_sha": "e1bbf7e36892bd4efa317e95b250c540812fe00e",
        "jobs": [],
        "phase": "waiting_for_core_benchmark",
        "status": "running",
    }
    atomic_json(STATUS_PATH, status)

    def checkpoint() -> None:
        status["updated_at"] = now_text()
        status["remaining_to_dispatch_seconds"] = max(0.0, DISPATCH_DEADLINE - time.time())
        atomic_json(STATUS_PATH, status)

    def run_job(label: str, args: list[str], expected: Path,
                expected_seconds: float, timeout_seconds: float,
                required: bool = True) -> bool:
        if STOP or time.time() + expected_seconds > DISPATCH_DEADLINE:
            status["jobs"].append({
                "label": label,
                "status": "skipped_deadline",
                "required": required,
                "at": now_text(),
            })
            checkpoint()
            return False
        log_path = LOGS / f"{label}.log"
        record = {
            "label": label,
            "status": "running",
            "required": required,
            "started_at": now_text(),
            "command": [str(PYTHON), *args],
            "log": str(log_path),
            "expected_output": str(expected),
        }
        status["jobs"].append(record)
        status["phase"] = label
        checkpoint()
        started = time.perf_counter()
        try:
            with log_path.open("w") as log:
                completed = subprocess.run(
                    [str(PYTHON), *args], cwd=REPO, stdout=log,
                    stderr=subprocess.STDOUT, timeout=timeout_seconds,
                    check=False,
                )
            record["exit_code"] = completed.returncode
            record["status"] = "complete" if completed.returncode == 0 and expected.exists() else "failed"
        except subprocess.TimeoutExpired:
            record["status"] = "timeout"
        except Exception as exc:
            record["status"] = "scheduler_error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = time.perf_counter() - started
        record["completed_at"] = now_text()
        checkpoint()
        return record["status"] == "complete"

    core_pid_path = LOGS / "core_full.pid"
    if not core_pid_path.exists():
        status["status"] = "blocked"
        status["error"] = "missing core_full.pid"
        checkpoint()
        return
    core_pid = int(core_pid_path.read_text().strip())
    while pid_alive(core_pid) and not STOP and time.time() < HARD_DEADLINE:
        checkpoint()
        time.sleep(20)

    core_path = RESULTS / "core_full.json"
    core_gate = {"available": core_path.exists(), "critical_rows": 0, "critical_failures": []}
    if core_path.exists():
        payload = json.loads(core_path.read_text())
        critical = [
            row for row in payload.get("rows", [])
            if row.get("dtype") == "complex128"
            and row.get("batch") == 8
            and row.get("pattern") == [4, 2]
        ]
        core_gate["critical_rows"] = len(critical)
        core_gate["critical_failures"] = [
            row for row in critical
            if row.get("status") != "ok"
            or row.get("value_allclose") is False
            or row.get("grad_allclose") is False
        ]
        core_gate["total_rows"] = len(payload.get("rows", []))
    core_gate["passed"] = bool(
        core_gate["available"]
        and core_gate["critical_rows"] >= 42
        and not core_gate["critical_failures"]
    )
    status["core_gate"] = core_gate
    checkpoint()

    if STOP:
        status["status"] = "stopped"
        checkpoint()
        return

    # A four-method, three-second smoke test catches baseline-runner defects
    # before the long backend series consumes the night.
    status["phase"] = "risk_baseline_smoke"
    smoke_specs = [
        ("chirp_a2", "vanilla"),
        ("chirp_a2", "complex_sinh"),
        ("maxwell_a4", "pwnn"),
        ("maxwell_a4", "complex_sinh"),
    ]
    for problem, method in smoke_specs:
        out = RESULTS / "risk_smoke" / f"{problem}_{method}.csv"
        run_job(
            f"risk_smoke_{problem}_{method}",
            [
                "scripts/run_risk_baseline_overnight.py",
                "--problem", problem, "--method", method, "--seed", "10",
                "--seconds", "3", "--n-int", "64", "--n-bc", "32",
                "--eval-n", "1024", "--history-eval-n", "256",
                "--out", str(out),
            ],
            out, expected_seconds=10, timeout_seconds=90,
        )

    if not core_gate["passed"]:
        status["status"] = "blocked_by_core_correctness_gate"
        status["phase"] = "complete_without_backend_training"
        checkpoint()
        return

    # Fixed-step control: identical initialization, batches and step-based LR.
    status["phase"] = "backend_fixed_step_control"
    control_dir = RESULTS / "pinn_fixed_step"
    for backend in ("direct_autodiff", "polarization_jet", "waring_complex_jet"):
        out = control_dir / f"seed0_{backend}.json"
        ckpt = control_dir / f"seed0_{backend}.pt"
        run_job(
            f"pinn_control_seed0_{backend}",
            [
                "scripts/run_backend_pinn_overnight.py",
                "--backend", backend, "--seed", "0",
                "--seconds", "900", "--max-steps", "500", "--lr-basis", "steps",
                "--probe-seconds", "30", "--probe-eval-n", "2048",
                "--final-eval-n", "8192", "--boundary-eval-n", "4096",
                "--out", str(out), "--checkpoint", str(ckpt),
            ],
            out, expected_seconds=180, timeout_seconds=900,
        )

    # Compare parameter snapshots to direct AD at predeclared steps.
    snapshot_report: dict[str, Any] = {"reference": "direct_autodiff", "comparisons": []}
    direct_path = control_dir / "seed0_direct_autodiff.pt"
    if direct_path.exists():
        reference = torch.load(direct_path, map_location="cpu", weights_only=False)["snapshots"]
        for backend in ("polarization_jet", "waring_complex_jet"):
            path = control_dir / f"seed0_{backend}.pt"
            if not path.exists():
                continue
            candidate = torch.load(path, map_location="cpu", weights_only=False)["snapshots"]
            for step in (0, 1, 10, 100, 500):
                if step not in reference or step not in candidate:
                    continue
                max_abs = 0.0
                ref_sq = 0.0
                diff_sq = 0.0
                for name, ref in reference[step].items():
                    cand = candidate[step][name]
                    diff = (cand - ref).abs()
                    max_abs = max(max_abs, float(diff.max().item()))
                    ref_sq += float(ref.abs().square().sum().item())
                    diff_sq += float(diff.square().sum().item())
                snapshot_report["comparisons"].append({
                    "backend": backend,
                    "step": step,
                    "max_abs": max_abs,
                    "rel_l2": (diff_sq ** 0.5) / (ref_sq ** 0.5 + 1.0e-300),
                })
    atomic_json(control_dir / "snapshot_comparison.json", snapshot_report)

    # Five paired seeds, balanced backend order, 1200 wall-clock seconds each.
    status["phase"] = "backend_fixed_time"
    backend_order = ("direct_autodiff", "polarization_jet", "waring_complex_jet")
    fixed_time_dir = RESULTS / "pinn_fixed_time"
    for seed in range(5):
        rotated = backend_order[seed % 3:] + backend_order[:seed % 3]
        for backend in rotated:
            out = fixed_time_dir / f"seed{seed}_{backend}.json"
            ckpt = fixed_time_dir / f"seed{seed}_{backend}.pt"
            run_job(
                f"pinn_time_seed{seed}_{backend}",
                [
                    "scripts/run_backend_pinn_overnight.py",
                    "--backend", backend, "--seed", str(seed),
                    "--seconds", "1200", "--lr-basis", "time",
                    "--probe-seconds", "10", "--probe-eval-n", "8192",
                    "--final-eval-n", str(2 ** 16),
                    "--boundary-eval-n", str(2 ** 14),
                    "--out", str(out), "--checkpoint", str(ckpt),
                ],
                out, expected_seconds=1270, timeout_seconds=1500,
            )

    # Independent final seeds for the two known ranking reversals.  Seeds
    # 10--12 are required by the one-GPU plan; 13--14 run only if time remains.
    status["phase"] = "risk_baseline_final"
    final_specs = [
        ("chirp_a2", "vanilla"),
        ("chirp_a2", "complex_sinh"),
        ("maxwell_a4", "pwnn"),
        ("maxwell_a4", "complex_sinh"),
    ]
    for seed in range(10, 15):
        rotated = final_specs[(seed - 10) % 4:] + final_specs[:(seed - 10) % 4]
        for problem, method in rotated:
            out = RESULTS / "risk_baselines" / f"{problem}_{method}_seed{seed}.csv"
            run_job(
                f"risk_{problem}_{method}_seed{seed}",
                [
                    "scripts/run_risk_baseline_overnight.py",
                    "--problem", problem, "--method", method,
                    "--seed", str(seed), "--seconds", "600",
                    "--eval-n", str(2 ** 16), "--history-eval-n", "4096",
                    "--out", str(out),
                ],
                out, expected_seconds=640, timeout_seconds=780,
                required=seed <= 12,
            )

    status["status"] = "complete" if not STOP else "stopped"
    status["phase"] = "finished"
    status["completed_at"] = now_text()
    checkpoint()


if __name__ == "__main__":
    main()
