#!/usr/bin/env python3
"""Run the sixth-order strain-gradient Kirchhoff plate benchmark.

This thin entry point reuses the audited, resumable high-order orchestrator
while replacing its frozen four-task pilot table with one independent
sixth-order plate task. The original high-order candidate protocol and its
published results remain unchanged.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_high_order_candidate_screen as runner

from experiments.sixth_order_plate import problem


runner.TASKS = problem.TASKS
runner.TASK_ORDER = problem.TASK_ORDER
runner.CandidateTask = problem.CandidateTask
runner.COMPLEX_DTYPE = problem.COMPLEX_DTYPE
runner.REAL_DTYPE = problem.REAL_DTYPE
runner.HIDDEN = problem.HIDDEN
runner.DEPTH = problem.DEPTH
runner.HISTORY_INTERVAL_SECONDS = problem.HISTORY_INTERVAL_SECONDS
runner.LEARNING_RATE = problem.LEARNING_RATE
runner.LEARNING_RATE_FINAL = problem.LEARNING_RATE_FINAL
runner.METHODS = problem.METHODS
runner.ENGINE_PROTOCOL_ID = problem.PROTOCOL_ID
runner.PROTOCOL_ID = "strain_gradient_plate_2d_o6_screen_v1"
runner.DEFAULT_ROOT = (
    ROOT / "outputs" / "search" / "strain-gradient-plate-2d-o6-pilot-v1"
)
runner.build_model = problem.build_model
runner.make_loss_bundle = problem.make_loss_bundle
runner.model_metadata = problem.model_metadata
runner.tensor_components_to_float = problem.tensor_components_to_float


# ``worker_command`` is defined in the reused module, so its ``__file__``
# otherwise points back to ``run_high_order_candidate_screen.py``.  Keep all
# of the audited orchestration logic but make every child process re-enter
# this task-specific wrapper, where the sixth-order globals are installed.
_base_worker_command = runner.worker_command


def _sixth_order_worker_command(**kwargs: Any) -> list[str]:
    command = _base_worker_command(**kwargs)
    command[1] = str(Path(__file__).resolve())
    return command


runner.worker_command = _sixth_order_worker_command


if __name__ == "__main__":
    raise SystemExit(runner.main())
