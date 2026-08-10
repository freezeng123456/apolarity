"""Well-posed high-order PDE candidates for a WAR versus real-AD screen.

The screen deliberately spans distinct mathematical types while keeping the
network protocol fixed:

* two- and three-spatial-dimensional Zakharov--Kuznetsov equations (order 3),
* a two-dimensional damped dynamic Kirchhoff--Love plate (space order 4), and
* a coercive stationary Swift--Hohenberg equation in two dimensions (order 4).

Every task has a smooth manufactured solution and a uniqueness argument that
does not rely on the neural network.  The networks receive only affine-scaled
raw coordinates.  Trigonometric functions occur in analytic targets/sources,
never in the model input or initialisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from osc_common import build_plain, deriv_alpha, laplacian_power_terms, n_params
from torch import Tensor, nn

from apolarity.taylor_jet import TaylorJet, jet_forward_sequential


PROTOCOL_ID = "high_order_candidate_common_xavier_fp32_v1"
REAL_DTYPE = torch.float32
COMPLEX_DTYPE = torch.complex64
METHODS = ("war", "real_tanh_autodiff")
HIDDEN = 128
DEPTH = 4
LEARNING_RATE = 1e-3
LEARNING_RATE_FINAL = 1e-4
HISTORY_INTERVAL_SECONDS = 20.0
TRAIN_SEED = 42
EVAL_SEED = 68421
T_MAX = 1.0
TWO_PI = 2.0 * math.pi
PLATE_OMEGA = math.pi
PLATE_DAMPING = 0.1


@dataclass(frozen=True)
class CandidateTask:
    task_id: str
    family: str
    spatial_dim: int
    order: int
    coordinate_names: tuple[str, ...]
    lows: tuple[float, ...]
    highs: tuple[float, ...]
    weight_names: tuple[str, ...]
    weights: tuple[float, ...]
    residual_scale: float
    uniqueness: str

    @property
    def input_dim(self) -> int:
        return len(self.coordinate_names)

    @property
    def has_time(self) -> bool:
        return self.coordinate_names[-1] == "t"


TASKS: dict[str, CandidateTask] = {
    "zk_2d_o3": CandidateTask(
        task_id="zk_2d_o3",
        family="zakharov_kuznetsov",
        spatial_dim=2,
        order=3,
        coordinate_names=("x", "y", "t"),
        lows=(0.0, 0.0, 0.0),
        highs=(TWO_PI, TWO_PI, T_MAX),
        weight_names=("lambda_ic", "lambda_bc"),
        weights=(10.0, 10.0),
        residual_scale=20.0,
        uniqueness=(
            "smooth periodic forced ZK initial-value problem; uniqueness follows "
            "from the established well-posedness theory in the smooth class"
        ),
    ),
    "zk_3d_o3": CandidateTask(
        task_id="zk_3d_o3",
        family="zakharov_kuznetsov",
        spatial_dim=3,
        order=3,
        coordinate_names=("x", "y", "z", "t"),
        lows=(0.0, 0.0, 0.0, 0.0),
        highs=(TWO_PI, TWO_PI, TWO_PI, T_MAX),
        weight_names=("lambda_ic", "lambda_bc"),
        weights=(10.0, 10.0),
        residual_scale=24.0,
        uniqueness=(
            "smooth periodic forced 3D ZK initial-value problem on a finite "
            "time interval; uniqueness is taken in its local smooth solution class"
        ),
    ),
    "dynamic_plate_2d_o4": CandidateTask(
        task_id="dynamic_plate_2d_o4",
        family="dynamic_kirchhoff_love_plate",
        spatial_dim=2,
        order=4,
        coordinate_names=("x", "y", "t"),
        lows=(-1.0, -1.0, 0.0),
        highs=(1.0, 1.0, T_MAX),
        weight_names=("lambda_ic", "lambda_bc"),
        weights=(10.0, 10.0),
        residual_scale=100.0,
        uniqueness=(
            "linear damped plate with clamped boundary and displacement/velocity "
            "initial data; the standard energy estimate gives uniqueness"
        ),
    ),
    "swift_hohenberg_2d_o4": CandidateTask(
        task_id="swift_hohenberg_2d_o4",
        family="coercive_stationary_swift_hohenberg",
        spatial_dim=2,
        order=4,
        coordinate_names=("x", "y"),
        lows=(0.0, 0.0),
        highs=(TWO_PI, TWO_PI),
        weight_names=("lambda_bc",),
        weights=(10.0,),
        residual_scale=20.0,
        uniqueness=(
            "on the periodic torus, (1+Delta)^2 + I is coercive and u^3 is "
            "monotone; testing the difference of two solutions proves uniqueness"
        ),
    ),
}

TASK_ORDER = tuple(TASKS)


ZK2_MODES: tuple[tuple[float, ...], ...] = (
    (0.50, 1.0, 1.0),
    (0.25, 2.0, 1.0),
    (0.25, 1.0, 2.0),
)
ZK3_MODES: tuple[tuple[float, ...], ...] = (
    (0.40, 1.0, 1.0, 1.0),
    (0.20, 2.0, 1.0, 1.0),
    (0.20, 1.0, 2.0, 1.0),
    (0.20, 1.0, 1.0, 2.0),
)
SH_MODES: tuple[tuple[float, float, float], ...] = (
    (0.50, 1.0, 1.0),
    (0.25, 2.0, 1.0),
    (0.25, 1.0, 2.0),
)


def _common_xavier_init_(net: nn.Sequential) -> None:
    """Variance-match real and native-complex Xavier initialisation."""

    with torch.no_grad():
        for layer in net:
            if not isinstance(layer, nn.Linear):
                continue
            if layer.weight.dtype.is_complex:
                real = torch.empty_like(layer.weight.real)
                imag = torch.empty_like(layer.weight.real)
                nn.init.xavier_uniform_(real)
                nn.init.xavier_uniform_(imag)
                layer.weight.copy_(torch.complex(real, imag) / math.sqrt(2.0))
            else:
                nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.zero_()


class AffineBoxMLP(nn.Module):
    """Apply a physical-box-to-[-1,1] affine map before the plain MLP."""

    def __init__(
        self,
        net: nn.Sequential,
        lows: tuple[float, ...],
        highs: tuple[float, ...],
    ) -> None:
        super().__init__()
        if len(lows) != len(highs) or any(hi <= lo for lo, hi in zip(lows, highs)):
            raise ValueError("invalid affine box")
        self.net = net
        scale = [2.0 / (hi - lo) for lo, hi in zip(lows, highs)]
        shift = [-(hi + lo) / (hi - lo) for lo, hi in zip(lows, highs)]
        self.register_buffer("input_scale", torch.tensor(scale, dtype=REAL_DTYPE))
        self.register_buffer("input_shift", torch.tensor(shift, dtype=REAL_DTYPE))

    def _scale_for(self, value: Tensor) -> Tensor:
        return self.input_scale.to(device=value.device, dtype=value.dtype)

    def _shift_for(self, value: Tensor) -> Tensor:
        return self.input_shift.to(device=value.device, dtype=value.dtype)

    def forward(self, points: Tensor) -> Tensor:
        if points.shape[-1] != self.input_scale.numel():
            raise ValueError("physical input dimension does not match affine box")
        return self.net(
            points * self._scale_for(points) + self._shift_for(points)
        )

    def jet_forward(self, jet: TaylorJet) -> TaylorJet:
        if jet.terms[0].shape[-1] != self.input_scale.numel():
            raise ValueError("Taylor jet dimension does not match affine box")
        scale = self._scale_for(jet.terms[0])
        shift = self._shift_for(jet.terms[0])
        transformed = TaylorJet([
            jet.terms[0] * scale + shift,
            *[term * scale for term in jet.terms[1:]],
        ])
        return jet_forward_sequential(self.net, transformed)


def build_model(
    task: CandidateTask,
    method: str,
    device: torch.device,
    *,
    hidden: int = HIDDEN,
    depth: int = DEPTH,
) -> tuple[nn.Module, torch.dtype, str]:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    is_war = method == "war"
    dtype = COMPLEX_DTYPE if is_war else REAL_DTYPE
    activation = "sinh" if is_war else "tanh"
    net = build_plain(task.input_dim, hidden, depth, dtype, activation, out=1)
    _common_xavier_init_(net)
    model = AffineBoxMLP(net, task.lows, task.highs).to(device=device)
    backend = "waring_complex_jet" if is_war else "direct_autodiff"
    return model, dtype, backend


def model_metadata(model: nn.Module, method: str) -> dict[str, object]:
    parameter = next(model.parameters())
    parameter_elements = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "method": method,
        "representation": "native_complex" if method == "war" else "real",
        "activation": "sinh" if method == "war" else "tanh",
        "derivative_backend": (
            "waring_complex_jet" if method == "war" else "direct_autodiff"
        ),
        "hidden": HIDDEN,
        "depth": DEPTH,
        "parameter_elements": parameter_elements,
        "real_dof": n_params(model),
        "literal_layer_shape_matched": True,
        "init_mode": "common_xavier",
        "frequency_initialization": "disabled",
        "input_transform": "affine_only",
        "parameter_dtype": str(parameter.dtype),
    }


def _predict_real(model: nn.Module, points: Tensor) -> Tensor:
    return model(points).real.squeeze(-1)


def _partial(
    model: nn.Module,
    points: Tensor,
    alpha: tuple[int, ...],
    backend: str,
) -> Tensor:
    if not alpha:
        return _predict_real(model, points)
    return deriv_alpha(model, points, alpha, backend=backend).real.squeeze(-1)


def spatial_laplacian_power(
    model: nn.Module,
    points: Tensor,
    spatial_dim: int,
    power: int,
    backend: str,
) -> Tensor:
    total: Tensor | None = None
    for coefficient, alpha in laplacian_power_terms(spatial_dim, power):
        value = _partial(model, points, alpha, backend)
        term = coefficient * value
        total = term if total is None else total + term
    if total is None:
        raise RuntimeError("empty Laplacian expansion")
    return total


def _zk_components(points: Tensor, spatial_dim: int) -> dict[str, Tensor]:
    modes = ZK2_MODES if spatial_dim == 2 else ZK3_MODES
    spatial = points[..., :spatial_dim]
    time = points[..., spatial_dim]
    decay = torch.exp(-time)
    u = torch.zeros_like(time)
    u_x = torch.zeros_like(time)
    dispersive = torch.zeros_like(time)
    for raw_mode in modes:
        amplitude, *frequencies = raw_mode
        kx = frequencies[0]
        sin_x = torch.sin(kx * spatial[..., 0])
        cos_x = torch.cos(kx * spatial[..., 0])
        transverse = torch.ones_like(time)
        for coordinate, frequency in enumerate(frequencies[1:], start=1):
            transverse = transverse * torch.cos(frequency * spatial[..., coordinate])
        factor = float(amplitude) * decay * transverse
        u = u + factor * sin_x
        u_x = u_x + factor * kx * cos_x
        k_squared = sum(float(value) ** 2 for value in frequencies)
        dispersive = dispersive - factor * kx * k_squared * cos_x
    return {
        "u": u,
        "u_t": -u,
        "u_x": u_x,
        "dispersive": dispersive,
    }


def _plate_components(points: Tensor) -> dict[str, Tensor]:
    x = points[..., 0]
    y = points[..., 1]
    t = points[..., 2]
    ax = (1.0 - x.square()).square()
    ay = (1.0 - y.square()).square()
    ax2 = -4.0 + 12.0 * x.square()
    ay2 = -4.0 + 12.0 * y.square()
    phi = ax * ay
    biharm_phi = 24.0 * ay + 2.0 * ax2 * ay2 + 24.0 * ax
    cos_t = torch.cos(PLATE_OMEGA * t)
    sin_t = torch.sin(PLATE_OMEGA * t)
    u = phi * cos_t
    u_t = -PLATE_OMEGA * phi * sin_t
    return {
        "u": u,
        "u_t": u_t,
        "u_tt": -(PLATE_OMEGA**2) * u,
        "biharmonic": biharm_phi * cos_t,
    }


def _sh_components(points: Tensor) -> dict[str, Tensor]:
    x = points[..., 0]
    y = points[..., 1]
    u = torch.zeros_like(x)
    lap = torch.zeros_like(x)
    biharm = torch.zeros_like(x)
    for amplitude, kx, ky in SH_MODES:
        mode = float(amplitude) * torch.cos(kx * x) * torch.cos(ky * y)
        eigenvalue = kx * kx + ky * ky
        u = u + mode
        lap = lap - eigenvalue * mode
        biharm = biharm + eigenvalue * eigenvalue * mode
    return {"u": u, "laplacian": lap, "biharmonic": biharm}


def exact_solution(points: Tensor, task: CandidateTask) -> Tensor:
    if task.family == "zakharov_kuznetsov":
        return _zk_components(points, task.spatial_dim)["u"]
    if task.family == "dynamic_kirchhoff_love_plate":
        return _plate_components(points)["u"]
    if task.family == "coercive_stationary_swift_hohenberg":
        return _sh_components(points)["u"]
    raise ValueError(task.family)


def manufactured_source(points: Tensor, task: CandidateTask) -> Tensor:
    if task.family == "zakharov_kuznetsov":
        values = _zk_components(points, task.spatial_dim)
        return (
            values["u_t"]
            + values["u"] * values["u_x"]
            + values["dispersive"]
        )
    if task.family == "dynamic_kirchhoff_love_plate":
        values = _plate_components(points)
        return (
            values["u_tt"]
            + PLATE_DAMPING * values["u_t"]
            + values["biharmonic"]
        )
    if task.family == "coercive_stationary_swift_hohenberg":
        values = _sh_components(points)
        u = values["u"]
        return values["biharmonic"] + 2.0 * values["laplacian"] + 2.0 * u + u**3
    raise ValueError(task.family)


def sample_interior(
    task: CandidateTask,
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    unit = torch.rand(
        count, task.input_dim, device=device, dtype=REAL_DTYPE, generator=generator
    )
    lows = torch.tensor(task.lows, device=device, dtype=REAL_DTYPE)
    widths = torch.tensor(
        [hi - lo for lo, hi in zip(task.lows, task.highs)],
        device=device,
        dtype=REAL_DTYPE,
    )
    return unit * widths + lows


def sample_initial(
    task: CandidateTask,
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    if not task.has_time:
        raise ValueError("stationary task has no initial surface")
    points = sample_interior(task, count, device=device, generator=generator)
    points[:, -1] = task.lows[-1]
    return points


def sample_face_pairs(
    task: CandidateTask,
    count: int,
    coordinate: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    lower = sample_interior(task, count, device=device, generator=generator)
    upper = lower.clone()
    lower[:, coordinate] = task.lows[coordinate]
    upper[:, coordinate] = task.highs[coordinate]
    return lower, upper


def _relative_l2(prediction: Tensor, target: Tensor) -> float:
    numerator = torch.mean((prediction - target).square()).sqrt()
    denominator = torch.mean(target.square()).sqrt().clamp_min(1e-12)
    return float((numerator / denominator).item())


def _chunked_prediction(
    model: nn.Module,
    points: Tensor,
    dtype: torch.dtype,
    *,
    chunk_size: int = 8192,
) -> Tensor:
    outputs: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, points.shape[0], chunk_size):
            chunk = points[start:start + chunk_size].to(dtype=dtype)
            outputs.append(_predict_real(model, chunk).to(dtype=REAL_DTYPE))
    return torch.cat(outputs, dim=0)


def _periodic_boundary_loss(
    task: CandidateTask,
    model: nn.Module,
    dtype: torch.dtype,
    backend: str,
    count: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, dict[str, Tensor]]:
    per_axis = max(8, count // task.spatial_dim)
    losses: list[Tensor] = []
    components: dict[str, Tensor] = {}
    for coordinate in range(task.spatial_dim):
        lower, upper = sample_face_pairs(
            task,
            per_axis,
            coordinate,
            device=device,
            generator=generator,
        )
        lower = lower.to(dtype=dtype)
        upper = upper.to(dtype=dtype)
        if task.family == "zakharov_kuznetsov":
            orders = (0, 1, 2) if coordinate == 0 else (0, 1)
        else:
            orders = (0, 1)
        for derivative_order in orders:
            alpha = (coordinate,) * derivative_order
            difference = (
                _partial(model, lower, alpha, backend)
                - _partial(model, upper, alpha, backend)
            )
            normalizer = 2.0**derivative_order
            value = (difference / normalizer).square().mean()
            components[
                f"L_BC_{task.coordinate_names[coordinate]}_order{derivative_order}"
            ] = value
            losses.append(value)
    total = torch.stack(losses).mean()
    components["L_BC"] = total
    return total, components


def _plate_boundary_loss(
    task: CandidateTask,
    model: nn.Module,
    dtype: torch.dtype,
    backend: str,
    count: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, dict[str, Tensor]]:
    per_axis = max(8, count // (2 * task.spatial_dim))
    value_losses: list[Tensor] = []
    normal_losses: list[Tensor] = []
    for coordinate in range(task.spatial_dim):
        lower, upper = sample_face_pairs(
            task,
            per_axis,
            coordinate,
            device=device,
            generator=generator,
        )
        for physical in (lower, upper):
            points = physical.to(dtype=dtype)
            value_losses.append(_predict_real(model, points).square().mean())
            normal_losses.append(
                _partial(model, points, (coordinate,), backend).square().mean()
            )
    l_value = torch.stack(value_losses).mean()
    l_normal = torch.stack(normal_losses).mean()
    total = 0.5 * (l_value + l_normal)
    return total, {
        "L_BC_value": l_value,
        "L_BC_normal": l_normal,
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
    if min(n_int, n_bc, n_eval, history_eval_n) <= 0:
        raise ValueError("sample counts must be positive")
    if task.has_time and n_ic <= 0:
        raise ValueError("time-dependent tasks require initial samples")
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
        if task.family == "zakharov_kuznetsov":
            time_coordinate = task.spatial_dim
            u = _predict_real(model, points)
            u_t = _partial(model, points, (time_coordinate,), backend)
            u_x = _partial(model, points, (0,), backend)
            dispersive = _partial(model, points, (0, 0, 0), backend)
            for coordinate in range(1, task.spatial_dim):
                dispersive = dispersive + _partial(
                    model, points, (0, coordinate, coordinate), backend
                )
            residual = u_t + u * u_x + dispersive - source
        elif task.family == "dynamic_kirchhoff_love_plate":
            u_t = _partial(model, points, (2,), backend)
            u_tt = _partial(model, points, (2, 2), backend)
            biharmonic = spatial_laplacian_power(model, points, 2, 2, backend)
            residual = u_tt + PLATE_DAMPING * u_t + biharmonic - source
        elif task.family == "coercive_stationary_swift_hohenberg":
            u = _predict_real(model, points)
            laplacian = spatial_laplacian_power(model, points, 2, 1, backend)
            biharmonic = spatial_laplacian_power(model, points, 2, 2, backend)
            residual = biharmonic + 2.0 * laplacian + 2.0 * u + u**3 - source
        else:  # pragma: no cover - frozen task table
            raise ValueError(task.family)
        return (residual / task.residual_scale).square().mean()

    def loss_fn() -> tuple[Tensor, dict[str, Tensor]]:
        interior = sample_interior(
            task, n_int, device=device, generator=train_generator
        )
        l_pde = pde_loss(interior)
        components: dict[str, Tensor] = {"L_PDE": l_pde}

        if task.family == "dynamic_kirchhoff_love_plate":
            l_bc, boundary_components = _plate_boundary_loss(
                task, model, dtype, backend, n_bc, device, train_generator
            )
        else:
            l_bc, boundary_components = _periodic_boundary_loss(
                task, model, dtype, backend, n_bc, device, train_generator
            )
        components.update(boundary_components)

        if task.has_time:
            initial = sample_initial(
                task, n_ic, device=device, generator=train_generator
            )
            points_ic = initial.to(dtype=dtype)
            target_ic = exact_solution(initial, task).detach()
            l_ic_value = (
                _predict_real(model, points_ic) - target_ic
            ).square().mean()
            if task.family == "dynamic_kirchhoff_love_plate":
                target_velocity = _plate_components(initial)["u_t"].detach()
                velocity = _partial(model, points_ic, (2,), backend)
                l_ic_velocity = (
                    (velocity - target_velocity) / PLATE_OMEGA
                ).square().mean()
                l_ic = 0.5 * (l_ic_value + l_ic_velocity)
                components["L_IC_velocity"] = l_ic_velocity
            else:
                l_ic = l_ic_value
            components["L_IC_value"] = l_ic_value
            components["L_IC"] = l_ic
            lambda_ic, lambda_bc = task.weights
            weighted_ic = lambda_ic * l_ic
            weighted_bc = lambda_bc * l_bc
            total = l_pde + weighted_ic + weighted_bc
            components["weighted_L_IC"] = weighted_ic
            components["weighted_L_BC"] = weighted_bc
        else:
            (lambda_bc,) = task.weights
            weighted_bc = lambda_bc * l_bc
            total = l_pde + weighted_bc
            components["weighted_L_BC"] = weighted_bc
        components["loss"] = total
        return total, components

    def evaluate(points: Tensor, target: Tensor) -> dict[str, float]:
        prediction = _chunked_prediction(model, points, dtype)
        return {"rel_error": _relative_l2(prediction, target)}

    def history_metrics_fn() -> dict[str, float]:
        return evaluate(history_points, history_target)

    def eval_metrics_fn() -> dict[str, float]:
        return evaluate(eval_points, eval_target)

    if task.family == "zakharov_kuznetsov":
        equation = "u_t + u*u_x + u_xxx + sum_j u_xjj = f"
        boundary_type = "periodic traces through derivative order required by ZK"
    elif task.family == "dynamic_kirchhoff_love_plate":
        equation = "u_tt + 0.1*u_t + Delta_xy^2 u = f"
        boundary_type = "clamped u=0 and d_n u=0"
    else:
        equation = "Delta^2 u + 2*Delta u + 2*u + u^3 = f"
        boundary_type = "periodic u and first normal derivative"
    return LossBundle(
        loss_fn=loss_fn,
        eval_metrics_fn=eval_metrics_fn,
        history_metrics_fn=history_metrics_fn,
        metadata={
            "task_id": task.task_id,
            "family": task.family,
            "order": task.order,
            "spatial_dim": task.spatial_dim,
            "physical_input_dim": task.input_dim,
            "coordinates": list(task.coordinate_names),
            "domain_lows": list(task.lows),
            "domain_highs": list(task.highs),
            "equation": equation,
            "boundary_type": boundary_type,
            "uniqueness_basis": task.uniqueness,
            "manufactured_solution": True,
            "input_transform": "affine_only",
            "trigonometric_input_features": False,
            "frequency_initialization": False,
            "weights": dict(zip(task.weight_names, task.weights)),
            "residual_scale": task.residual_scale,
            "n_int": n_int,
            "n_ic": n_ic if task.has_time else 0,
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
    "CandidateTask",
    "DEPTH",
    "EVAL_SEED",
    "HIDDEN",
    "HISTORY_INTERVAL_SECONDS",
    "LEARNING_RATE",
    "LEARNING_RATE_FINAL",
    "METHODS",
    "PROTOCOL_ID",
    "REAL_DTYPE",
    "TASKS",
    "TASK_ORDER",
    "TRAIN_SEED",
    "build_model",
    "exact_solution",
    "make_loss_bundle",
    "manufactured_source",
    "model_metadata",
    "spatial_laplacian_power",
    "tensor_components_to_float",
]

