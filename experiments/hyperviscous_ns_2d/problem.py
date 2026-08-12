"""Two-dimensional fourth-order hyperviscous Navier--Stokes benchmark.

The problem is posed on the periodic torus ``[0,2*pi]^2`` for ``t in [0,1]``:

    U_t + (U . grad) U + grad p - nu Delta U + eta Delta^2 U = 0,
    div U = 0,

with ``nu=0.05`` and ``eta=0.01``.  A mean-zero-pressure Taylor--Green vortex
provides an exact unforced solution, so no numerical reference or manufactured
forcing is needed.  Both compared networks output ``(u,v,p)`` directly.

Only an affine normalization of raw ``(x,y,t)`` is passed to the networks.
Periodic traces are enforced explicitly; there are no trigonometric input
features, periodic embeddings, or frequency-aware initializations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from osc_common import build_plain, deriv_alpha, n_params
from torch import Tensor, nn

from apolarity.taylor_jet import TaylorJet, jet_forward_sequential


PROTOCOL_ID = "hyperns_2d_taylor_green_common_xavier_fp32_v1"
RUNNER_FAMILY_NAME = "hyperviscous_navier_stokes_2d"
BASELINE_ACTIVATION = "tanh"
ALTERNATE_METHOD_ORDER = True
STRICT_MANIFEST_BINDING = True
INPUT_DIM = 3
OUTPUT_DIM = 3
REAL_DTYPE = torch.float32
COMPLEX_DTYPE = torch.complex64
METHODS = ("war", "real_tanh_autodiff")
GRID_VALUES = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
HIDDEN = 128
DEPTH = 4
LEARNING_RATE = 1e-3
LEARNING_RATE_FINAL = 1e-4
HISTORY_INTERVAL_SECONDS = 10.0
TRAIN_SEED = 42
EVAL_SEED = 68421
DOMAIN_MAX = 2.0 * math.pi
T_MAX = 1.0
NU = 0.05
ETA = 0.01
DECAY_RATE = 2.0 * NU + 4.0 * ETA
DIV_WEIGHT = 1.0
GAUGE_WEIGHT = 1.0
MOMENTUM_SCALE = 1.0
MAX_SPATIAL_WAVENUMBER = 1.0
DIAGNOSTIC_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)

SUMMARY_METRIC_KEYS = (
    "velocity_rel_error",
    "pressure_rel_error",
    "divergence_rms",
    "pressure_mean_max_abs",
    "energy_relative_rmse",
)
SMOKE_METRIC_KEYS = SUMMARY_METRIC_KEYS
HISTORY_REQUIRED_METRICS = (
    "velocity_rel_error",
    "pressure_rel_error",
)


@dataclass(frozen=True)
class HyperNSTask:
    task_id: str = "hyperviscous_ns_2d_o4"
    family: str = RUNNER_FAMILY_NAME
    spatial_dim: int = 2
    order: int = 4
    q: int = 2
    eta: float = ETA
    coordinate_names: tuple[str, ...] = ("x", "y", "t")
    lows: tuple[float, ...] = (0.0, 0.0, 0.0)
    highs: tuple[float, ...] = (DOMAIN_MAX, DOMAIN_MAX, T_MAX)
    weight_names: tuple[str, str] = ("lambda_ic", "lambda_bc")
    weights: tuple[float, float] = (1.0, 1.0)
    uniqueness: str = (
        "global unique smooth periodic 2D hyperviscous Navier--Stokes "
        "velocity for smooth divergence-free data; pressure is unique after "
        "the zero-spatial-mean gauge is fixed"
    )

    @property
    def input_dim(self) -> int:
        return len(self.coordinate_names)

    @property
    def has_time(self) -> bool:
        return True

    @property
    def weight_count(self) -> int:
        return len(self.weight_names)

    @property
    def center_weights(self) -> tuple[float, float]:
        return self.weights


# Compatibility alias used by the audited two-weight grid runner.
Cahn2DTask = HyperNSTask
TASKS: dict[str, HyperNSTask] = {
    "hyperviscous_ns_2d_o4": HyperNSTask(),
}
TASK_ORDER = tuple(TASKS)


RUNNER_MANIFEST_METADATA = {
    "equation": (
        "U_t + (U.grad)U + grad(p) - nu*Delta(U) + eta*Delta^2(U) = 0; "
        "div(U)=0"
    ),
    "nu": NU,
    "eta": ETA,
    "domain": "[0,2*pi]^2 x [0,1]",
    "exact_solution": (
        "A=exp(-(2*nu+4*eta)t); u=A*sin(x)*cos(y); "
        "v=-A*cos(x)*sin(y); p=A^2*(cos(2x)+cos(2y))/4"
    ),
    "boundary": (
        "explicit periodic velocity trace matching at normal orders 0..3; "
        "pressure order 0; per-time zero-mean pressure gauge"
    ),
    "training_precision": {"war": "complex64", "real_ad": "float32"},
    "network_output": ["u", "v", "p"],
    "primary_metric": "combined velocity relative L2",
    "direct_fourth_order_residual": True,
    "analytic_unforced_reference": True,
}


def runner_manifest_metadata(*, smoke: bool) -> dict[str, object]:
    metadata = dict(RUNNER_MANIFEST_METADATA)
    metadata["smoke"] = bool(smoke)
    return metadata


def _common_xavier_init_(net: nn.Sequential) -> None:
    """Variance-match real and native-complex Xavier initialization."""

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


class AffineCoordinateMLP(nn.Module):
    """Plain MLP on affine-normalized physical ``(x,y,t)`` coordinates."""

    def __init__(self, net: nn.Sequential) -> None:
        super().__init__()
        self.net = net
        self.register_buffer(
            "input_scale",
            torch.tensor(
                [2.0 / DOMAIN_MAX, 2.0 / DOMAIN_MAX, 2.0 / T_MAX],
                dtype=REAL_DTYPE,
            ),
        )
        self.register_buffer(
            "input_shift",
            torch.tensor([-1.0, -1.0, -1.0], dtype=REAL_DTYPE),
        )

    def _scale_for(self, value: Tensor) -> Tensor:
        return self.input_scale.to(device=value.device, dtype=value.dtype)

    def _shift_for(self, value: Tensor) -> Tensor:
        return self.input_shift.to(device=value.device, dtype=value.dtype)

    def forward(self, points: Tensor) -> Tensor:
        if points.shape[-1] != INPUT_DIM:
            raise ValueError("hyper-NS model expects physical inputs (x,y,t)")
        return self.net(
            points * self._scale_for(points) + self._shift_for(points)
        )

    def jet_forward(self, jet: TaylorJet) -> TaylorJet:
        if jet.terms[0].shape[-1] != INPUT_DIM:
            raise ValueError("hyper-NS jet expects physical inputs (x,y,t)")
        scale = self._scale_for(jet.terms[0])
        shift = self._shift_for(jet.terms[0])
        transformed = TaylorJet([
            jet.terms[0] * scale + shift,
            *[term * scale for term in jet.terms[1:]],
        ])
        return jet_forward_sequential(self.net, transformed)


def build_model(
    task: HyperNSTask,
    method: str,
    device: torch.device,
    *,
    hidden: int = HIDDEN,
    depth: int = DEPTH,
) -> tuple[nn.Module, torch.dtype, str]:
    del task
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    is_war = method == "war"
    dtype = COMPLEX_DTYPE if is_war else REAL_DTYPE
    activation = "sinh" if is_war else BASELINE_ACTIVATION
    net = build_plain(
        INPUT_DIM, hidden, depth, dtype, activation, out=OUTPUT_DIM
    )
    _common_xavier_init_(net)
    model = AffineCoordinateMLP(net).to(device=device)
    backend = "waring_complex_jet" if is_war else "direct_autodiff"
    return model, dtype, backend


def model_metadata(model: nn.Module, method: str) -> dict[str, object]:
    parameter = next(model.parameters())
    parameter_elements = sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )
    return {
        "method": method,
        "representation": "native_complex" if method == "war" else "real",
        "activation": "sinh" if method == "war" else BASELINE_ACTIVATION,
        "derivative_backend": (
            "waring_complex_jet" if method == "war" else "direct_autodiff"
        ),
        "hidden": HIDDEN,
        "depth": DEPTH,
        "output_dim": OUTPUT_DIM,
        "output_names": ["u", "v", "p"],
        "parameter_elements": parameter_elements,
        "real_dof": n_params(model),
        "literal_layer_shape_matched": True,
        "init_mode": "common_xavier",
        "frequency_initialization": "disabled",
        "input_transform": "affine_only",
        "trigonometric_input_features": False,
        "parameter_dtype": str(parameter.dtype),
    }


def _predict_all(model: nn.Module, points: Tensor) -> Tensor:
    prediction = model(points).real
    if prediction.ndim != 2 or prediction.shape[1] != OUTPUT_DIM:
        raise ValueError(
            f"hyper-NS network must output (batch,{OUTPUT_DIM}); "
            f"got {tuple(prediction.shape)}"
        )
    return prediction


def _partial_all(
    model: nn.Module,
    points: Tensor,
    alpha: tuple[int, ...],
    backend: str,
) -> Tensor:
    value = deriv_alpha(model, points, alpha, backend=backend).real
    if value.ndim != 2 or value.shape[1] != OUTPUT_DIM:
        raise ValueError(
            f"hyper-NS derivative must have shape (batch,{OUTPUT_DIM}); "
            f"got {tuple(value.shape)}"
        )
    return value


def exact_solution(points: Tensor) -> Tensor:
    """Return the mean-zero Taylor--Green state ``(u,v,p)``."""

    if points.shape[-1] != INPUT_DIM:
        raise ValueError("Taylor--Green solution expects (x,y,t)")
    x, y, time = points.unbind(dim=-1)
    amplitude = torch.exp(-DECAY_RATE * time)
    u = amplitude * torch.sin(x) * torch.cos(y)
    v = -amplitude * torch.cos(x) * torch.sin(y)
    pressure = 0.25 * amplitude.square() * (
        torch.cos(2.0 * x) + torch.cos(2.0 * y)
    )
    return torch.stack([u, v, pressure], dim=-1)


def exact_energy(time: Tensor) -> Tensor:
    """Spatially averaged kinetic energy ``0.5*mean(u^2+v^2)``."""

    return 0.25 * torch.exp(-2.0 * DECAY_RATE * time)


def sample_interior(
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    points = torch.empty(count, INPUT_DIM, device=device, dtype=REAL_DTYPE)
    points[:, :2].uniform_(0.0, DOMAIN_MAX, generator=generator)
    points[:, 2].uniform_(0.0, T_MAX, generator=generator)
    return points


def sample_initial(
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    points = sample_interior(count, device=device, generator=generator)
    points[:, 2] = 0.0
    return points


def sample_face_pairs(
    count: int,
    coordinate: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    if coordinate not in (0, 1):
        raise ValueError("periodic face coordinate must be x or y")
    lower = sample_interior(count, device=device, generator=generator)
    upper = lower.clone()
    lower[:, coordinate] = 0.0
    upper[:, coordinate] = DOMAIN_MAX
    return lower, upper


def sample_pressure_gauge_groups(
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, int, int]:
    """Sample spatial groups sharing time so each pressure mean is pinned."""

    group_count = max(4, min(16, int(math.sqrt(max(16, count)))))
    per_group = max(8, count // group_count)
    points = torch.empty(
        group_count, per_group, INPUT_DIM, device=device, dtype=REAL_DTYPE
    )
    points[..., :2].uniform_(0.0, DOMAIN_MAX, generator=generator)
    times = torch.rand(
        group_count, 1, 1, device=device, dtype=REAL_DTYPE,
        generator=generator,
    ) * T_MAX
    points[..., 2:3] = times
    return points.reshape(-1, INPUT_DIM), group_count, per_group


def momentum_and_divergence(
    model: nn.Module,
    points: Tensor,
    backend: str,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Evaluate both momentum components and incompressibility directly."""

    state = _predict_all(model, points)
    d_t = _partial_all(model, points, (2,), backend)
    d_x = _partial_all(model, points, (0,), backend)
    d_y = _partial_all(model, points, (1,), backend)
    d_xx = _partial_all(model, points, (0, 0), backend)
    d_yy = _partial_all(model, points, (1, 1), backend)
    d_xxxx = _partial_all(model, points, (0, 0, 0, 0), backend)
    d_xxyy = _partial_all(model, points, (0, 0, 1, 1), backend)
    d_yyyy = _partial_all(model, points, (1, 1, 1, 1), backend)

    u, v = state[:, 0], state[:, 1]
    laplacian = d_xx + d_yy
    biharmonic = d_xxxx + 2.0 * d_xxyy + d_yyyy
    residual_u = (
        d_t[:, 0]
        + u * d_x[:, 0]
        + v * d_y[:, 0]
        + d_x[:, 2]
        - NU * laplacian[:, 0]
        + ETA * biharmonic[:, 0]
    )
    residual_v = (
        d_t[:, 1]
        + u * d_x[:, 1]
        + v * d_y[:, 1]
        + d_y[:, 2]
        - NU * laplacian[:, 1]
        + ETA * biharmonic[:, 1]
    )
    divergence = d_x[:, 0] + d_y[:, 1]
    return residual_u, residual_v, {
        "divergence": divergence,
        "laplacian_u": laplacian[:, 0],
        "laplacian_v": laplacian[:, 1],
        "biharmonic_u": biharmonic[:, 0],
        "biharmonic_v": biharmonic[:, 1],
    }


def _relative_l2(prediction: Tensor, target: Tensor) -> float:
    numerator = torch.mean((prediction - target).square()).sqrt()
    denominator = torch.mean(target.square()).sqrt().clamp_min(1e-12)
    return float((numerator / denominator).item())


def _velocity_relative_l2(prediction: Tensor, target: Tensor) -> float:
    numerator = torch.mean(
        (prediction[:, :2] - target[:, :2]).square().sum(dim=1)
    ).sqrt()
    denominator = torch.mean(target[:, :2].square().sum(dim=1)).sqrt()
    return float((numerator / denominator.clamp_min(1e-12)).item())


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
            outputs.append(_predict_all(model, chunk).to(REAL_DTYPE))
    return torch.cat(outputs, dim=0)


def _chunked_partial_all(
    model: nn.Module,
    points: Tensor,
    dtype: torch.dtype,
    alpha: tuple[int, ...],
    backend: str,
    *,
    chunk_size: int = 512,
) -> Tensor:
    outputs: list[Tensor] = []
    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start:start + chunk_size].to(dtype=dtype)
        outputs.append(
            _partial_all(model, chunk, alpha, backend).detach().to(REAL_DTYPE)
        )
    return torch.cat(outputs, dim=0)


def _periodic_boundary_loss(
    model: nn.Module,
    dtype: torch.dtype,
    backend: str,
    count: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, dict[str, Tensor]]:
    per_axis = max(8, count // 2)
    velocity_losses: list[Tensor] = []
    pressure_losses: list[Tensor] = []
    components: dict[str, Tensor] = {}
    for coordinate, label in ((0, "x"), (1, "y")):
        lower, upper = sample_face_pairs(
            per_axis,
            coordinate,
            device=device,
            generator=generator,
        )
        lower = lower.to(dtype=dtype)
        upper = upper.to(dtype=dtype)
        for order in range(4):
            alpha = (coordinate,) * order
            if order == 0:
                difference = _predict_all(model, lower) - _predict_all(model, upper)
            else:
                difference = (
                    _partial_all(model, lower, alpha, backend)
                    - _partial_all(model, upper, alpha, backend)
                )
            normalizer = MAX_SPATIAL_WAVENUMBER**order
            for component, name in ((0, "u"), (1, "v")):
                value = (difference[:, component] / normalizer).square().mean()
                components[f"L_BC_{name}_{label}_order{order}"] = value
                velocity_losses.append(value)
            if order == 0:
                pressure_value = difference[:, 2].square().mean()
                components[f"L_BC_p_{label}_order0"] = pressure_value
                pressure_losses.append(pressure_value)
    l_velocity = torch.stack(velocity_losses).mean()
    l_pressure = torch.stack(pressure_losses).mean()
    total = 0.5 * (l_velocity + l_pressure)
    components.update({
        "L_BC_velocity": l_velocity,
        "L_BC_pressure": l_pressure,
        "L_BC": total,
    })
    return total, components


def _diagnostic_grid(side: int, device: torch.device) -> tuple[Tensor, int]:
    axis = torch.arange(side, device=device, dtype=REAL_DTYPE) * (
        DOMAIN_MAX / side
    )
    x_grid, y_grid = torch.meshgrid(axis, axis, indexing="ij")
    spatial = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)
    groups: list[Tensor] = []
    for value in DIAGNOSTIC_TIMES:
        time_column = torch.full(
            (spatial.shape[0], 1), value, device=device, dtype=REAL_DTYPE
        )
        groups.append(torch.cat([spatial, time_column], dim=-1))
    return torch.cat(groups, dim=0), spatial.shape[0]


@dataclass
class HyperNSLossBundle:
    loss_fn: Callable[[], tuple[Tensor, dict[str, Tensor]]]
    eval_metrics_fn: Callable[[], dict[str, object]]
    history_metrics_fn: Callable[[], dict[str, object]]
    metadata: dict[str, object]


def make_loss_bundle(
    task: HyperNSTask,
    model: nn.Module,
    dtype: torch.dtype,
    backend: str,
    weights: tuple[float, ...],
    device: torch.device,
    *,
    smoke: bool,
    n_int: int | None = None,
    n_ic: int | None = None,
    n_bc: int | None = None,
    n_eval: int | None = None,
    history_eval_n: int | None = None,
    train_seed: int = TRAIN_SEED,
    eval_seed: int = EVAL_SEED,
) -> HyperNSLossBundle:
    if len(weights) != 2:
        raise ValueError("hyper-NS expects [lambda_ic, lambda_bc]")
    lambda_ic, lambda_bc = (float(value) for value in weights)
    defaults = (16, 16, 16, 128, 64) if smoke else (
        2048,
        512,
        1024,
        16384,
        2048,
    )
    n_int = defaults[0] if n_int is None else int(n_int)
    n_ic = defaults[1] if n_ic is None else int(n_ic)
    n_bc = defaults[2] if n_bc is None else int(n_bc)
    n_eval = defaults[3] if n_eval is None else int(n_eval)
    history_eval_n = defaults[4] if history_eval_n is None else int(history_eval_n)
    if min(n_int, n_ic, n_bc, n_eval, history_eval_n) <= 0:
        raise ValueError("all hyper-NS sample counts must be positive")
    if history_eval_n > n_eval:
        raise ValueError("history_eval_n cannot exceed n_eval")

    train_generator = torch.Generator(device=device).manual_seed(train_seed)
    eval_generator = torch.Generator(device=device).manual_seed(eval_seed)
    eval_points = sample_interior(
        n_eval, device=device, generator=eval_generator
    )
    history_points = eval_points[:history_eval_n]
    eval_target = exact_solution(eval_points).detach()
    history_target = exact_solution(history_points).detach()
    diagnostic_side = 6 if smoke else 24
    diagnostic_points, diagnostic_group = _diagnostic_grid(
        diagnostic_side, device
    )

    def loss_fn() -> tuple[Tensor, dict[str, Tensor]]:
        interior = sample_interior(
            n_int, device=device, generator=train_generator
        )
        points = interior.to(dtype=dtype)
        residual_u, residual_v, differential = momentum_and_divergence(
            model, points, backend
        )
        l_momentum_u = (residual_u / MOMENTUM_SCALE).square().mean()
        l_momentum_v = (residual_v / MOMENTUM_SCALE).square().mean()
        l_momentum = 0.5 * (l_momentum_u + l_momentum_v)
        l_div = differential["divergence"].square().mean()
        l_pde = l_momentum + DIV_WEIGHT * l_div

        initial = sample_initial(
            n_ic, device=device, generator=train_generator
        )
        initial_prediction = _predict_all(model, initial.to(dtype=dtype))[:, :2]
        initial_target = exact_solution(initial).detach()[:, :2]
        l_ic_u = (initial_prediction[:, 0] - initial_target[:, 0]).square().mean()
        l_ic_v = (initial_prediction[:, 1] - initial_target[:, 1]).square().mean()
        l_ic = 0.5 * (l_ic_u + l_ic_v)

        l_bc, boundary_components = _periodic_boundary_loss(
            model, dtype, backend, n_bc, device, train_generator
        )
        gauge_points, gauge_groups, gauge_per_group = sample_pressure_gauge_groups(
            max(64, n_ic), device=device, generator=train_generator
        )
        pressure = _predict_all(
            model, gauge_points.to(dtype=dtype)
        )[:, 2].reshape(gauge_groups, gauge_per_group)
        l_gauge = pressure.mean(dim=1).square().mean()

        weighted_ic = lambda_ic * l_ic
        weighted_bc = lambda_bc * l_bc
        weighted_gauge = GAUGE_WEIGHT * l_gauge
        total = l_pde + weighted_ic + weighted_bc + weighted_gauge
        components: dict[str, Tensor] = {
            "L_momentum_u": l_momentum_u,
            "L_momentum_v": l_momentum_v,
            "L_momentum": l_momentum,
            "L_div": l_div,
            "L_PDE": l_pde,
            "L_IC_u": l_ic_u,
            "L_IC_v": l_ic_v,
            "L_IC": l_ic,
            **boundary_components,
            "L_gauge": l_gauge,
            "weighted_L_IC": weighted_ic,
            "weighted_L_BC": weighted_bc,
            "weighted_L_gauge": weighted_gauge,
            "weighted_L_constraints": weighted_ic + weighted_bc + weighted_gauge,
            "loss": total,
        }
        return total, components

    def basic_metrics(points: Tensor, target: Tensor) -> dict[str, object]:
        prediction = _chunked_prediction(model, points, dtype)
        velocity_error = _velocity_relative_l2(prediction, target)
        return {
            "rel_error": velocity_error,
            "velocity_rel_error": velocity_error,
            "u_rel_error": _relative_l2(prediction[:, 0], target[:, 0]),
            "v_rel_error": _relative_l2(prediction[:, 1], target[:, 1]),
            "pressure_rel_error": _relative_l2(
                prediction[:, 2], target[:, 2]
            ),
        }

    def history_metrics_fn() -> dict[str, object]:
        return basic_metrics(history_points, history_target)

    def eval_metrics_fn() -> dict[str, object]:
        metrics = basic_metrics(eval_points, eval_target)
        physics_points = eval_points[:min(2048, n_eval)]
        d_x = _chunked_partial_all(model, physics_points, dtype, (0,), backend)
        d_y = _chunked_partial_all(model, physics_points, dtype, (1,), backend)
        divergence = d_x[:, 0] + d_y[:, 1]

        diagnostic_prediction = _chunked_prediction(
            model, diagnostic_points, dtype
        ).reshape(len(DIAGNOSTIC_TIMES), diagnostic_group, OUTPUT_DIM)
        energy = 0.5 * diagnostic_prediction[:, :, :2].square().sum(dim=2).mean(dim=1)
        time_tensor = torch.tensor(
            DIAGNOSTIC_TIMES, device=device, dtype=REAL_DTYPE
        )
        target_energy = exact_energy(time_tensor)
        energy_relative_rmse = (
            (energy - target_energy).square().mean().sqrt()
            / target_energy.square().mean().sqrt().clamp_min(1e-12)
        )
        energy_changes = energy[1:] - energy[:-1]
        pressure_means = diagnostic_prediction[:, :, 2].mean(dim=1)
        metrics.update({
            "divergence_rms": float(divergence.square().mean().sqrt().item()),
            "divergence_max_abs": float(divergence.abs().max().item()),
            "pressure_mean_rms": float(
                pressure_means.square().mean().sqrt().item()
            ),
            "pressure_mean_max_abs": float(pressure_means.abs().max().item()),
            "energy_relative_rmse": float(energy_relative_rmse.item()),
            "energy_increase_max": float(
                torch.clamp_min(energy_changes, 0.0).max().item()
            ),
            "energy_violation_fraction": float(
                (energy_changes > 1e-6).to(REAL_DTYPE).mean().item()
            ),
            "energy_by_time": {
                f"{value:.2f}": float(item.item())
                for value, item in zip(DIAGNOSTIC_TIMES, energy, strict=True)
            },
            "exact_energy_by_time": {
                f"{value:.2f}": float(item.item())
                for value, item in zip(
                    DIAGNOSTIC_TIMES, target_energy, strict=True
                )
            },
            "pressure_mean_by_time": {
                f"{value:.2f}": float(item.item())
                for value, item in zip(
                    DIAGNOSTIC_TIMES, pressure_means, strict=True
                )
            },
        })
        return metrics

    return HyperNSLossBundle(
        loss_fn=loss_fn,
        eval_metrics_fn=eval_metrics_fn,
        history_metrics_fn=history_metrics_fn,
        metadata={
            "task_id": task.task_id,
            "family": task.family,
            "order": task.order,
            "spatial_dim": task.spatial_dim,
            "physical_input_dim": INPUT_DIM,
            "output_dim": OUTPUT_DIM,
            "output_names": ["u", "v", "p"],
            "coordinates": list(task.coordinate_names),
            "domain": {
                "x": [0.0, DOMAIN_MAX],
                "y": [0.0, DOMAIN_MAX],
                "t": [0.0, T_MAX],
            },
            "equation": RUNNER_MANIFEST_METADATA["equation"],
            "nu": NU,
            "eta": ETA,
            "decay_rate": DECAY_RATE,
            "boundary_type": RUNNER_MANIFEST_METADATA["boundary"],
            "uniqueness_basis": task.uniqueness,
            "analytic_unforced_reference": True,
            "manufactured_solution": False,
            "smoke": bool(smoke),
            "input_transform": "affine_only",
            "trigonometric_input_features": False,
            "frequency_initialization": False,
            "weights": dict(zip(task.weight_names, weights)),
            "fixed_weights": {
                "lambda_div": DIV_WEIGHT,
                "lambda_gauge": GAUGE_WEIGHT,
            },
            "primary_metric": "velocity_rel_error",
            "n_int": n_int,
            "n_ic": n_ic,
            "n_bc": n_bc,
            "n_eval": n_eval,
            "history_eval_n": history_eval_n,
            "sample_policy": "resample_each_training_step",
            "diagnostic_times": list(DIAGNOSTIC_TIMES),
            "diagnostic_grid_side": diagnostic_side,
        },
    )


def tensor_components_to_float(values: dict[str, Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().real.item()) for key, value in values.items()
    }


__all__ = [
    "BASELINE_ACTIVATION",
    "COMPLEX_DTYPE",
    "Cahn2DTask",
    "DECAY_RATE",
    "DEPTH",
    "ETA",
    "EVAL_SEED",
    "GRID_VALUES",
    "HIDDEN",
    "HISTORY_INTERVAL_SECONDS",
    "HyperNSTask",
    "LEARNING_RATE",
    "LEARNING_RATE_FINAL",
    "METHODS",
    "NU",
    "OUTPUT_DIM",
    "PROTOCOL_ID",
    "REAL_DTYPE",
    "TASKS",
    "TASK_ORDER",
    "TRAIN_SEED",
    "build_model",
    "exact_energy",
    "exact_solution",
    "make_loss_bundle",
    "model_metadata",
    "momentum_and_divergence",
    "runner_manifest_metadata",
    "tensor_components_to_float",
]

