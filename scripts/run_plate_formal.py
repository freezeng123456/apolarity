#!/usr/bin/env python3
"""Run fixed-weight formal plate experiments with the audited runner.

This wrapper keeps the high-order candidate orchestrator and worker protocol,
but freezes the weights selected from the completed 7x7 search:

* ``dynamic_plate_2d_o4``: ``(lambda_ic, lambda_bc) = (0.1, 1.0)``
* ``strain_gradient_plate_2d_o6``: ``(lambda_ic, lambda_bc) = (10.0, 10.0)``

The caller selects one task with ``--tasks``.  The wrapper is also used by
worker subprocesses so every cell re-enters the same task-specific globals.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_high_order_candidate_screen as runner  # noqa: E402
from experiments.dynamic_plate_weight_search import problem  # noqa: E402


FORMAL_PROTOCOL_ID = "dynamic_plate_o4_o6_fixed_formal_common_xavier_fp32_v1"


def _fixed_task(task_id: str, weights: tuple[float, float]):
    source = problem.TASKS[task_id]
    return replace(source, weights=weights)


# The formal run has two selectable task entries, each with a pre-registered
# shared weight.  The orchestrator is invoked with --tasks to keep the jobs
# strictly serial and to give each result root an unambiguous protocol.
runner.TASKS = {
    "dynamic_plate_2d_o4": _fixed_task(
        "dynamic_plate_2d_o4", (0.1, 1.0)
    ),
    "strain_gradient_plate_2d_o6": _fixed_task(
        "strain_gradient_plate_2d_o6", (10.0, 10.0)
    ),
}
runner.TASK_ORDER = tuple(runner.TASKS)
runner.CandidateTask = problem.PlateSearchTask
problem.PROTOCOL_ID = FORMAL_PROTOCOL_ID
runner.ENGINE_PROTOCOL_ID = FORMAL_PROTOCOL_ID
runner.PROTOCOL_ID = FORMAL_PROTOCOL_ID
runner.COMPLEX_DTYPE = problem.COMPLEX_DTYPE
runner.REAL_DTYPE = problem.REAL_DTYPE
runner.HIDDEN = problem.HIDDEN
runner.DEPTH = problem.DEPTH
runner.HISTORY_INTERVAL_SECONDS = problem.HISTORY_INTERVAL_SECONDS
runner.LEARNING_RATE = problem.LEARNING_RATE
runner.LEARNING_RATE_FINAL = problem.LEARNING_RATE_FINAL
runner.METHODS = problem.METHODS
runner.EVAL_SEED = problem.EVAL_SEED
runner.GRAD_CLIP = problem.GRAD_CLIP
runner.build_model = problem.build_model
runner.make_loss_bundle = problem.make_loss_bundle
runner.model_metadata = problem.model_metadata
runner.tensor_components_to_float = problem.tensor_components_to_float
runner.DEFAULT_ROOT = ROOT / "outputs" / "current" / "plate-formal-v1"


_base_worker_command = runner.worker_command


def _worker_command(**kwargs: Any) -> list[str]:
    command = _base_worker_command(**kwargs)
    command[1] = str(Path(__file__).resolve())
    return command


runner.worker_command = _worker_command


if __name__ == "__main__":
    raise SystemExit(runner.main())
