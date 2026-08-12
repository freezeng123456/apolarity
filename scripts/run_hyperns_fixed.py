#!/usr/bin/env python3
"""Run fixed-weight HO-04 hyper-NS pilot or formal experiments.

The shared ``(lambda_ic,lambda_bc)`` vector is supplied through
``APOLARITY_HYPERNS_FIXED_WEIGHTS`` and is inherited unchanged by every worker
subprocess.  The pipeline writes the selected vector before invoking this
wrapper, which keeps resume manifests bound to the same weights and code SHA.
"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_high_order_candidate_screen as runner  # noqa: E402
from experiments.hyperviscous_ns_2d import problem  # noqa: E402


FORMAL_ENGINE_PROTOCOL_ID = "hyperns_2d_fixed_common_xavier_fp32_v1"
FORMAL_RUNNER_PROTOCOL_ID = "hyperns_2d_pilot_formal_v1"


def _parse_weights() -> tuple[float, float]:
    raw = os.environ.get("APOLARITY_HYPERNS_FIXED_WEIGHTS", "1,1")
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 2 or any(value not in problem.GRID_VALUES for value in values):
        raise ValueError(
            "APOLARITY_HYPERNS_FIXED_WEIGHTS must contain two registered "
            "power-of-ten grid values"
        )
    return values


FIXED_WEIGHTS = _parse_weights()
SOURCE_TASK = problem.TASKS["hyperviscous_ns_2d_o4"]
FIXED_TASK = replace(SOURCE_TASK, weights=FIXED_WEIGHTS)


def _make_loss_bundle(
    task: problem.HyperNSTask,
    model,
    dtype,
    backend: str,
    device,
    **samples: Any,
):
    return problem.make_loss_bundle(
        task,
        model,
        dtype,
        backend,
        task.weights,
        device,
        smoke=False,
        **samples,
    )


runner.TASKS = {FIXED_TASK.task_id: FIXED_TASK}
runner.TASK_ORDER = tuple(runner.TASKS)
runner.CandidateTask = problem.HyperNSTask
runner.ENGINE_PROTOCOL_ID = FORMAL_ENGINE_PROTOCOL_ID
runner.PROTOCOL_ID = FORMAL_RUNNER_PROTOCOL_ID
runner.COMPLEX_DTYPE = problem.COMPLEX_DTYPE
runner.REAL_DTYPE = problem.REAL_DTYPE
runner.HIDDEN = problem.HIDDEN
runner.DEPTH = problem.DEPTH
runner.HISTORY_INTERVAL_SECONDS = problem.HISTORY_INTERVAL_SECONDS
runner.LEARNING_RATE = problem.LEARNING_RATE
runner.LEARNING_RATE_FINAL = problem.LEARNING_RATE_FINAL
runner.METHODS = problem.METHODS
runner.EVAL_SEED = problem.EVAL_SEED
runner.SUMMARY_METRIC_KEYS = problem.SUMMARY_METRIC_KEYS
runner.HISTORY_REQUIRED_METRICS = problem.HISTORY_REQUIRED_METRICS
runner.STRICT_MANIFEST_BINDING = True
runner.build_model = problem.build_model
runner.make_loss_bundle = _make_loss_bundle
runner.model_metadata = problem.model_metadata
runner.tensor_components_to_float = problem.tensor_components_to_float
runner.DEFAULT_ROOT = ROOT / "outputs" / "hyperns-fixed-v1"

_base_worker_command = runner.worker_command


def _worker_command(**kwargs: Any) -> list[str]:
    command = _base_worker_command(**kwargs)
    command[1] = str(Path(__file__).resolve())
    return command


runner.worker_command = _worker_command


if __name__ == "__main__":
    raise SystemExit(runner.main())

