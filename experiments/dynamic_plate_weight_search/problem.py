"""Problem adapter for the fourth-/sixth-order dynamic-plate weight grid.

Both tasks retain their audited PDE definitions.  This module only supplies a
common two-weight interface so that ``lambda_ic`` and ``lambda_bc`` can be
varied over the same Cartesian grid for WAR and the real-tanh AD baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from torch import nn

from experiments.high_order_candidates import problem as fourth_order
from experiments.sixth_order_plate import problem as sixth_order


PROTOCOL_ID = "dynamic_plate_o4_o6_shared_weight_grid_fp32_v1"
GRID_VALUES = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
REAL_DTYPE = fourth_order.REAL_DTYPE
COMPLEX_DTYPE = fourth_order.COMPLEX_DTYPE
METHODS = fourth_order.METHODS
HIDDEN = fourth_order.HIDDEN
DEPTH = fourth_order.DEPTH
LEARNING_RATE = fourth_order.LEARNING_RATE
LEARNING_RATE_FINAL = fourth_order.LEARNING_RATE_FINAL
HISTORY_INTERVAL_SECONDS = fourth_order.HISTORY_INTERVAL_SECONDS
TRAIN_SEED = 42
EVAL_SEED = fourth_order.EVAL_SEED
INIT_MODE = "common_xavier"
GRAD_CLIP = 10.0
SAMPLE_COUNTS = {
    "n_int": 2048,
    "n_ic": 512,
    "n_bc": 1024,
    "n_eval": 16384,
    "history_eval_n": 2048,
}


@dataclass(frozen=True)
class PlateSearchTask(fourth_order.CandidateTask):
    @property
    def weight_count(self) -> int:
        return len(self.weight_names)

    @property
    def center_weights(self) -> tuple[float, ...]:
        return self.weights


def _copy_task(source: fourth_order.CandidateTask) -> PlateSearchTask:
    values = {
        field.name: getattr(source, field.name)
        for field in fields(fourth_order.CandidateTask)
    }
    return PlateSearchTask(**values)


TASKS: dict[str, PlateSearchTask] = {
    "dynamic_plate_2d_o4": _copy_task(
        fourth_order.TASKS["dynamic_plate_2d_o4"]
    ),
    "strain_gradient_plate_2d_o6": _copy_task(sixth_order.TASK),
}
TASK_ORDER = tuple(TASKS)


def problem_protocol_id(task: PlateSearchTask) -> str:
    if task.task_id == "dynamic_plate_2d_o4":
        return fourth_order.PROTOCOL_ID
    if task.task_id == "strain_gradient_plate_2d_o6":
        return sixth_order.PROTOCOL_ID
    raise ValueError(task.task_id)


def with_weights(
    task: PlateSearchTask, weights: tuple[float, ...]
) -> PlateSearchTask:
    if len(weights) != task.weight_count:
        raise ValueError(
            f"{task.task_id} expects {task.weight_count} weights, got {weights}"
        )
    numeric = tuple(float(value) for value in weights)
    if any(value <= 0 for value in numeric):
        raise ValueError("all loss weights must be positive")
    return replace(task, weights=numeric)


def build_model(
    task: PlateSearchTask,
    method: str,
    device: torch.device,
    *,
    hidden: int = HIDDEN,
    depth: int = DEPTH,
) -> tuple[nn.Module, torch.dtype, str]:
    return fourth_order.build_model(
        task, method, device, hidden=hidden, depth=depth
    )


def model_metadata(model: nn.Module, method: str) -> dict[str, object]:
    metadata = fourth_order.model_metadata(model, method)
    metadata.update({
        "input_transform": "affine_only",
        "frequency_initialization": "disabled",
        "trigonometric_input_features": False,
    })
    return metadata


def make_loss_bundle(
    task: PlateSearchTask,
    model: nn.Module,
    dtype: torch.dtype,
    backend: str,
    device: torch.device,
    *,
    n_int: int,
    n_ic: int,
    n_bc: int,
    n_eval: int,
    history_eval_n: int,
    train_seed: int,
    eval_seed: int,
):
    arguments = dict(
        n_int=n_int,
        n_ic=n_ic,
        n_bc=n_bc,
        n_eval=n_eval,
        history_eval_n=history_eval_n,
        train_seed=train_seed,
        eval_seed=eval_seed,
    )
    if task.task_id == "dynamic_plate_2d_o4":
        bundle = fourth_order.make_loss_bundle(
            task, model, dtype, backend, device, **arguments
        )
    elif task.task_id == "strain_gradient_plate_2d_o6":
        bundle = sixth_order.make_loss_bundle(
            task, model, dtype, backend, device, **arguments
        )
    else:  # pragma: no cover - guarded by the frozen task table
        raise ValueError(task.task_id)
    bundle.metadata.update({
        "weight_search_protocol_id": PROTOCOL_ID,
        "underlying_problem_protocol_id": problem_protocol_id(task),
        "weights": dict(zip(task.weight_names, task.weights)),
    })
    return bundle


def tensor_components_to_float(
    values: dict[str, torch.Tensor],
) -> dict[str, float]:
    return fourth_order.tensor_components_to_float(values)


__all__ = [
    "COMPLEX_DTYPE",
    "DEPTH",
    "EVAL_SEED",
    "GRAD_CLIP",
    "GRID_VALUES",
    "HIDDEN",
    "HISTORY_INTERVAL_SECONDS",
    "INIT_MODE",
    "LEARNING_RATE",
    "LEARNING_RATE_FINAL",
    "METHODS",
    "PROTOCOL_ID",
    "PlateSearchTask",
    "REAL_DTYPE",
    "SAMPLE_COUNTS",
    "TASKS",
    "TASK_ORDER",
    "TRAIN_SEED",
    "build_model",
    "make_loss_bundle",
    "model_metadata",
    "problem_protocol_id",
    "tensor_components_to_float",
    "with_weights",
]
