"""Two-dimensional sixth-order strain-gradient plate benchmark.

The benchmark is a damped vibration problem for a gradient-elastic
Kirchhoff plate,

    u_tt + c u_t + Delta^2 u - ell^2 Delta^3 u = f,

on (-1, 1)^2 x (0, 1). Gradient-elastic Kirchhoff plate models are
sixth-order in space. We use a smooth manufactured solution with homogeneous
third-order essential traces and displacement/velocity initial data. The
network receives only affine-scaled (x, y, t) coordinates.

For the difference of two solutions, the homogeneous initial data and the
coercive H^3 plate energy give uniqueness by the standard energy estimate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from experiments.high_order_candidates.problem import (
    COMPLEX_DTYPE,
    DEPTH,
    EVAL_SEED,
    HIDDEN,
    HISTORY_INTERVAL_SECONDS,
    LEARNING_RATE,
    LEARNING_RATE_FINAL,
    METHODS,
    REAL_DTYPE,
    CandidateTask,
    _chunked_prediction,
    _partial,
    _predict_real,
    _relative_l2,
    build_model,
    model_metadata,
    sample_face_pairs,
    sample_initial,
    sample_interior,
)


PROTOCOL_ID = "strain_gradient_plate_2d_o6_common_xavier_fp32_v1"
TASK_ID = "strain_gradient_plate_2d_o6"
PLATE_DAMPING = 0.1
INTERNAL_LENGTH = 0.5
TIME_OMEGA = math.pi
PDE_RESIDUAL_SCALE = 500.0
BOUNDARY_FIRST_SCALE = 2.0
BOUNDARY_SECOND_SCALE = 6.0


TASK = CandidateTask(
    task_id=TASK_ID,
    family="dynamic_strain_gradient_kirchhoff_plate",
    spatial_dim=2,
    order=6,
    coordinate_names=("x", "y", "t"),
    lows=(-1.0, -1.0, 0.0),
    highs=(1.0, 1.0, 1.0),
    weight_names=("lambda_ic", "lambda_bc"),
    weights=(10.0, 10.0),
    residual_scale=PDE_RESIDUAL_SCALE,
    uniqueness=(
        "linear damped gradient-elastic Kirchhoff plate with homogeneous "
        "u, normal-derivative and second-normal-derivative traces plus "
        "displacement/velocity initial data; the coercive H3 plate energy "
        "and damping give uniqueness by the standard energy estimate"
    ),
)
TASKS = {TASK_ID: TASK}
TASK_ORDER = (TASK_ID,)


def _a0(value: Tensor) -> Tensor:
    return (1.0 - value.square()).pow(3)


def _a2(value: Tensor) -> Tensor:
    return -6.0 + 36.0 * value.square() - 30.0 * value.pow(4)


def _a4(value: Tensor) -> Tensor:
    return 72.0 - 360.0 * value.square()


def _a6(value: Tensor) -> Tensor:
    return torch.full_like(value, -720.0)


def exact_components(points: Tensor) -> dict[str, Tensor]:
    x = points[..., 0]
    y = points[..., 1]
    t = points[..., 2]
    ax0, ay0 = _a0(x), _a0(y)
    ax2, ay2 = _a2(x), _a2(y)
    ax4, ay4 = _a4(x), _a4(y)
    ax6, ay6 = _a6(x), _a6(y)
    phi = ax0 * ay0
    biharmonic_phi = ax4 * ay0 + 2.0 * ax2 * ay2 + ax0 * ay4
    triharmonic_phi = (
        ax6 * ay0
        + 3.0 * ax4 * ay2
        + 3.0 * ax2 * ay4
        + ax0 * ay6
    )
    cos_t = torch.cos(TIME_OMEGA * t)
    sin_t = torch.sin(TIME_OMEGA * t)
    u = phi * cos_t
    return {
        "u": u,
        "u_t": -TIME_OMEGA * phi * sin_t,
        "u_tt": -(TIME_OMEGA**2) * u,
        "biharmonic": biharmonic_phi * cos_t,
        "triharmonic": triharmonic_phi * cos_t,
    }


def exact_solution(points: Tensor, task: CandidateTask = TASK) -> Tensor:
    if task.task_id != TASK_ID:
        raise ValueError(task.task_id)
    return exact_components(points)["u"]


def manufactured_source(points: Tensor, task: CandidateTask = TASK) -> Tensor:
    if task.task_id != TASK_ID:
        raise ValueError(task.task_id)
    values = exact_components(points)
    return (
        values["u_tt"]
        + PLATE_DAMPING * values["u_t"]
        + values["biharmonic"]
        - INTERNAL_LENGTH**2 * values["triharmonic"]
    )


def spatial_laplacian_power(
    model: nn.Module,
    points: Tensor,
    power: int,
    backend: str,
) -> Tensor:
    total: Tensor | None = None
    for x_order in range(power + 1):
        y_order = power - x_order
        coefficient = math.comb(power, x_order)
        alpha = (0,) * (2 * x_order) + (1,) * (2 * y_order)
        value = _partial(model, points, alpha, backend)
        term = coefficient * value
        total = term if total is None else total + term
    if total is None:  # pragma: no cover
        raise RuntimeError("empty Laplacian expansion")
    return total


def _plate_boundary_loss(
    task: CandidateTask,
    model: nn.Module,
    dtype: torch.dtype,
    backend: str,
    count: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, dict[str, Tensor]]:
    per_face = max(8, count // (2 * task.spatial_dim))
    value_losses: list[Tensor] = []
    first_losses: list[Tensor] = []
    second_losses: list[Tensor] = []
    for coordinate in range(task.spatial_dim):
        lower, upper = sample_face_pairs(
            task,
            per_face,
            coordinate,
            device=device,
            generator=generator,
        )
        for physical in (lower, upper):
            points = physical.to(dtype=dtype)
            value_losses.append(_predict_real(model, points).square().mean())
            first = _partial(model, points, (coordinate,), backend)
            second = _partial(model, points, (coordinate, coordinate), backend)
            first_losses.append(
                (first / BOUNDARY_FIRST_SCALE).square().mean()
            )
            second_losses.append(
                (second / BOUNDARY_SECOND_SCALE).square().mean()
            )
    l_value = torch.stack(value_losses).mean()
    l_first = torch.stack(first_losses).mean()
    l_second = torch.stack(second_losses).mean()
    total = (l_value + l_first + l_second) / 3.0
    return total, {
        "L_BC_value": l_value,
        "L_BC_normal_first": l_first,
        "L_BC_normal_second": l_second,
        "L_BC": total,
    }


@dataclass
class LossBundle:
    loss_fn: Callable[[], tuple[Tensor, dict[str, Tensor]]]
    eval_metrics_fn: Callable[[], dict[str, float]]
    history_metrics_fn: Callable[[], dict[str, float]]
    metadata: dict[str, object]


def make_loss_bundle(
    task: CandidateTask,
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
) -> LossBundle:
    if task.task_id != TASK_ID:
        raise ValueError(task.task_id)
    if min(n_int, n_ic, n_bc, n_eval, history_eval_n) <= 0:
        raise ValueError("sample counts must be positive")
    if history_eval_n > n_eval:
        raise ValueError("history_eval_n cannot exceed n_eval")

    train_generator = torch.Generator(device=device).manual_seed(train_seed)
    eval_generator = torch.Generator(device=device).manual_seed(eval_seed)
    eval_points = sample_interior(
        task, n_eval, device=device, generator=eval_generator
    )
    history_points = eval_points[:history_eval_n]
    eval_target = exact_solution(eval_points, task).detach()
    history_target = exact_solution(history_points, task).detach()

    def pde_loss(interior: Tensor) -> Tensor:
        points = interior.to(dtype=dtype)
        source = manufactured_source(interior, task).detach()
        u_t = _partial(model, points, (2,), backend)
        u_tt = _partial(model, points, (2, 2), backend)
        biharmonic = spatial_laplacian_power(model, points, 2, backend)
        triharmonic = spatial_laplacian_power(model, points, 3, backend)
        residual = (
            u_tt
            + PLATE_DAMPING * u_t
            + biharmonic
            - INTERNAL_LENGTH**2 * triharmonic
            - source
        )
        return (residual / PDE_RESIDUAL_SCALE).square().mean()

    def loss_fn() -> tuple[Tensor, dict[str, Tensor]]:
        interior = sample_interior(
            task, n_int, device=device, generator=train_generator
        )
        l_pde = pde_loss(interior)
        l_bc, boundary = _plate_boundary_loss(
            task, model, dtype, backend, n_bc, device, train_generator
        )
        initial = sample_initial(
            task, n_ic, device=device, generator=train_generator
        )
        points_ic = initial.to(dtype=dtype)
        target_ic = exact_solution(initial, task).detach()
        l_ic_value = (
            _predict_real(model, points_ic) - target_ic
        ).square().mean()
        target_velocity = exact_components(initial)["u_t"].detach()
        velocity = _partial(model, points_ic, (2,), backend)
        l_ic_velocity = (
            (velocity - target_velocity) / TIME_OMEGA
        ).square().mean()
        l_ic = 0.5 * (l_ic_value + l_ic_velocity)
        lambda_ic, lambda_bc = task.weights
        weighted_ic = lambda_ic * l_ic
        weighted_bc = lambda_bc * l_bc
        total = l_pde + weighted_ic + weighted_bc
        components = {
            "L_PDE": l_pde,
            **boundary,
            "L_IC_value": l_ic_value,
            "L_IC_velocity": l_ic_velocity,
            "L_IC": l_ic,
            "weighted_L_IC": weighted_ic,
            "weighted_L_BC": weighted_bc,
            "loss": total,
        }
        return total, components

    def evaluate(points: Tensor, target: Tensor) -> dict[str, float]:
        prediction = _chunked_prediction(model, points, dtype)
        return {"rel_error": _relative_l2(prediction, target)}

    return LossBundle(
        loss_fn=loss_fn,
        eval_metrics_fn=lambda: evaluate(eval_points, eval_target),
        history_metrics_fn=lambda: evaluate(history_points, history_target),
        metadata={
            "task_id": task.task_id,
            "family": task.family,
            "order": task.order,
            "spatial_dim": task.spatial_dim,
            "physical_input_dim": task.input_dim,
            "coordinates": list(task.coordinate_names),
            "domain_lows": list(task.lows),
            "domain_highs": list(task.highs),
            "equation": (
                "u_tt + 0.1*u_t + Delta_xy^2*u "
                "- 0.25*Delta_xy^3*u = f"
            ),
            "boundary_type": (
                "homogeneous u, first normal derivative and second normal "
                "derivative on every spatial face"
            ),
            "uniqueness_basis": task.uniqueness,
            "manufactured_solution": True,
            "manufactured_spatial_profile": (
                "(1-x^2)^3*(1-y^2)^3"
            ),
            "input_transform": "affine_only",
            "trigonometric_input_features": False,
            "frequency_initialization": False,
            "weights": dict(zip(task.weight_names, task.weights)),
            "residual_scale": PDE_RESIDUAL_SCALE,
            "internal_length": INTERNAL_LENGTH,
            "damping": PLATE_DAMPING,
            "time_omega": TIME_OMEGA,
            "n_int": n_int,
            "n_ic": n_ic,
            "n_bc": n_bc,
            "n_eval": n_eval,
            "history_eval_n": history_eval_n,
            "sample_policy": "resample_each_training_step",
        },
    )


def tensor_components_to_float(values: dict[str, Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().real.item())
        for key, value in values.items()
    }


__all__ = [
    "COMPLEX_DTYPE",
    "DEPTH",
    "EVAL_SEED",
    "HIDDEN",
    "HISTORY_INTERVAL_SECONDS",
    "LEARNING_RATE",
    "LEARNING_RATE_FINAL",
    "METHODS",
    "PROTOCOL_ID",
    "REAL_DTYPE",
    "TASK",
    "TASKS",
    "TASK_ID",
    "TASK_ORDER",
    "build_model",
    "exact_components",
    "exact_solution",
    "make_loss_bundle",
    "manufactured_source",
    "model_metadata",
    "spatial_laplacian_power",
    "tensor_components_to_float",
]
