"""Two-spatial-dimensional Cahn--Hilliard benchmark with natural boundaries.

The active physical coordinates are ``(x, y, t)`` on
``(0, pi)^2 x [0, 1]``.  Both comparison methods receive only an affine
normalisation of these raw coordinates; there is no Fourier/periodic feature
map and no task-frequency initialisation.

For ``q in {2, 3}`` the manufactured problem is

    u_t - Delta(u^3 - u) + eta_q Delta^q u = f,

with ``eta_2 = +1e-2`` and ``eta_3 = -1e-2``.  These signs make the leading
Fourier symbol positive on the residual's left-hand side.  Natural no-flux
boundary conditions are imposed as

    d_n Delta^ell u = 0,  ell = 0, ..., q - 1.

The formal manufactured profile is a symmetric three-mode cosine field.  The
cosines occur only in the analytic target/source; they are never input
features of either neural network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from osc_common import build_plain, deriv_alpha, laplacian_power_terms, n_params
from torch import Tensor, nn

from apolarity.taylor_jet import TaylorJet, jet_forward_sequential


PROTOCOL_ID = "cahn_hilliard_2d_natural_bc_common_sinh_fp32_v1"
REAL_DTYPE = torch.float32
COMPLEX_DTYPE = torch.complex64
METHODS = ("war", "real_sinh_autodiff")
GRID_VALUES = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
HIDDEN = 128
DEPTH = 4
LEARNING_RATE = 1e-3
LEARNING_RATE_FINAL = 1e-4
HISTORY_INTERVAL_SECONDS = 5.0
TRAIN_SEED = 42
EVAL_SEED = 54321
DOMAIN_MAX = math.pi
T_MAX = 1.0
GAMMA_NONLINEAR = 1.0
ETA_MAGNITUDE = 1e-2
KAPPA_MAX_SQUARED = 5.0


@dataclass(frozen=True)
class Cahn2DTask:
    task_id: str
    order: int
    q: int
    eta: float
    weight_names: tuple[str, str] = ("lambda_ic", "lambda_bc")
    center_weights: tuple[float, float] = (1.0, 1.0)

    @property
    def weight_count(self) -> int:
        return len(self.weight_names)


TASKS: dict[str, Cahn2DTask] = {
    "cahn_hilliard_2d_o4": Cahn2DTask(
        "cahn_hilliard_2d_o4", order=4, q=2, eta=+ETA_MAGNITUDE
    ),
    "cahn_hilliard_2d_o6": Cahn2DTask(
        "cahn_hilliard_2d_o6", order=6, q=3, eta=-ETA_MAGNITUDE
    ),
}


# (amplitude, k_x, k_y).  Sum of absolute amplitudes is one.
MANUFACTURED_MODES: tuple[tuple[float, int, int], ...] = (
    (0.50, 1, 1),
    (0.25, 2, 1),
    (0.25, 1, 2),
)


def _common_xavier_init_(net: nn.Sequential) -> None:
    """Use one variance-matched Xavier family for real and complex weights."""

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
    """MLP on an affine normalisation of raw physical ``(x, y, t)``.

    Spatial coordinates in ``[0, pi]`` and time in ``[0, 1]`` are mapped to
    ``[-1, 1]``.  The jet rule applies the same affine map to every Taylor
    coefficient, preserving derivatives with respect to physical coordinates.
    """

    def __init__(self, net: nn.Sequential):
        super().__init__()
        self.net = net
        self.register_buffer(
            "input_scale",
            torch.tensor([2.0 / math.pi, 2.0 / math.pi, 2.0], dtype=REAL_DTYPE),
        )
        self.register_buffer(
            "input_shift",
            torch.tensor([-1.0, -1.0, -1.0], dtype=REAL_DTYPE),
        )

    def _scale_for(self, value: Tensor) -> Tensor:
        return self.input_scale.to(device=value.device, dtype=value.dtype)

    def _shift_for(self, value: Tensor) -> Tensor:
        return self.input_shift.to(device=value.device, dtype=value.dtype)

    def forward(self, xyt: Tensor) -> Tensor:
        if xyt.shape[-1] != 3:
            raise ValueError(
                f"AffineCoordinateMLP expects (..., 3) physical inputs; got {tuple(xyt.shape)}"
            )
        normalised = xyt * self._scale_for(xyt) + self._shift_for(xyt)
        return self.net(normalised)

    def jet_forward(self, jet: TaylorJet) -> TaylorJet:
        if jet.terms[0].shape[-1] != 3:
            raise ValueError("AffineCoordinateMLP expects physical inputs (x,y,t)")
        scale = self._scale_for(jet.terms[0])
        shift = self._shift_for(jet.terms[0])
        normalised = TaylorJet(
            [
                jet.terms[0] * scale + shift,
                *[term * scale for term in jet.terms[1:]],
            ]
        )
        return jet_forward_sequential(self.net, normalised)


def build_model(
    task: Cahn2DTask,
    method: str,
    device: torch.device,
    *,
    hidden: int = HIDDEN,
    depth: int = DEPTH,
) -> tuple[nn.Module, torch.dtype, str]:
    """Build the architecture-matched WAR or real-autodiff model."""

    del task  # Architecture is intentionally identical for CH4 and CH6.
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    is_war = method == "war"
    dtype = COMPLEX_DTYPE if is_war else REAL_DTYPE
    net = build_plain(3, hidden, depth, dtype, "sinh", out=1)
    _common_xavier_init_(net)
    model = AffineCoordinateMLP(net).to(device=device)
    backend = "waring_complex_jet" if is_war else "direct_autodiff"
    return model, dtype, backend


def model_metadata(model: nn.Module, method: str) -> dict[str, object]:
    parameter_dtype = next(model.parameters()).dtype
    parameter_elements = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "method": method,
        "representation": "native_complex" if method == "war" else "real",
        "activation": "sinh",
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
        "parameter_dtype": str(parameter_dtype),
    }


def manufactured_components(points: Tensor, q: int) -> dict[str, Tensor]:
    """Return analytic manufactured values without autograd or FFT."""

    if points.shape[-1] != 3:
        raise ValueError("manufactured solution expects physical (x,y,t) inputs")
    if q not in (2, 3):
        raise ValueError(f"q must be 2 or 3; got {q}")
    x = points[..., 0]
    y = points[..., 1]
    t = points[..., 2]
    decay = torch.exp(-t)
    spatial = torch.zeros_like(x)
    spatial_x = torch.zeros_like(x)
    spatial_y = torch.zeros_like(x)
    spatial_lap = torch.zeros_like(x)
    spatial_high = torch.zeros_like(x)
    for amplitude, kx, ky in MANUFACTURED_MODES:
        kx_f = float(kx)
        ky_f = float(ky)
        eigenvalue = kx_f * kx_f + ky_f * ky_f
        cos_x = torch.cos(kx_f * x)
        cos_y = torch.cos(ky_f * y)
        mode = float(amplitude) * cos_x * cos_y
        spatial = spatial + mode
        spatial_x = spatial_x - float(amplitude) * kx_f * torch.sin(kx_f * x) * cos_y
        spatial_y = spatial_y - float(amplitude) * ky_f * cos_x * torch.sin(ky_f * y)
        spatial_lap = spatial_lap - eigenvalue * mode
        spatial_high = spatial_high + ((-eigenvalue) ** q) * mode
    u = decay * spatial
    return {
        "u": u,
        "u_t": -u,
        "u_x": decay * spatial_x,
        "u_y": decay * spatial_y,
        "lap_u": decay * spatial_lap,
        "lap_q_u": decay * spatial_high,
    }


def exact_solution(points: Tensor) -> Tensor:
    return manufactured_components(points, q=2)["u"]


def manufactured_source(points: Tensor, task: Cahn2DTask) -> Tensor:
    values = manufactured_components(points, task.q)
    u = values["u"]
    lap_nonlinear = (
        (3.0 * u.square() - 1.0) * values["lap_u"]
        + 6.0 * u * (values["u_x"].square() + values["u_y"].square())
    )
    return (
        values["u_t"]
        - GAMMA_NONLINEAR * lap_nonlinear
        + task.eta * values["lap_q_u"]
    )


def _predict_real(model: nn.Module, points: Tensor) -> Tensor:
    return model(points).real.squeeze(-1)


def laplacian_power(
    model: nn.Module,
    points: Tensor,
    power: int,
    backend: str,
) -> Tensor:
    """Evaluate ``Delta_xy^power u`` as an explicit mixed-partial sum."""

    if power == 0:
        return _predict_real(model, points)
    total: Tensor | None = None
    for coefficient, alpha in laplacian_power_terms(2, power):
        value = deriv_alpha(model, points, alpha, backend=backend).real.squeeze(-1)
        term = coefficient * value
        total = term if total is None else total + term
    if total is None:  # pragma: no cover - guarded by power >= 1
        raise RuntimeError("empty Laplacian expansion")
    return total


def normal_laplacian_power(
    model: nn.Module,
    points: Tensor,
    power: int,
    normal_coordinate: int,
    backend: str,
) -> Tensor:
    """Evaluate ``partial_n Delta_xy^power u`` on an axis-aligned face.

    The outward sign is immaterial for the squared homogeneous boundary loss,
    so ``normal_coordinate`` selects x (0) or y (1) only.
    """

    if normal_coordinate not in (0, 1):
        raise ValueError("normal_coordinate must be 0 (x-face) or 1 (y-face)")
    terms = (
        [(1.0, ())]
        if power == 0
        else laplacian_power_terms(2, power)
    )
    total: Tensor | None = None
    for coefficient, alpha in terms:
        value = deriv_alpha(
            model,
            points,
            tuple(alpha) + (normal_coordinate,),
            backend=backend,
        ).real.squeeze(-1)
        term = coefficient * value
        total = term if total is None else total + term
    if total is None:  # pragma: no cover - terms is always non-empty
        raise RuntimeError("empty normal-Laplacian expansion")
    return total


def sample_interior(
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    points = torch.empty(count, 3, device=device, dtype=REAL_DTYPE)
    points[:, 0:2].uniform_(0.0, DOMAIN_MAX, generator=generator)
    points[:, 2].uniform_(0.0, T_MAX, generator=generator)
    return points


def sample_initial(
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    points = torch.empty(count, 3, device=device, dtype=REAL_DTYPE)
    points[:, 0:2].uniform_(0.0, DOMAIN_MAX, generator=generator)
    points[:, 2] = 0.0
    return points


def _sample_axis_faces(
    count: int,
    normal_coordinate: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    points = torch.empty(count, 3, device=device, dtype=REAL_DTYPE)
    points[:, 0:2].uniform_(0.0, DOMAIN_MAX, generator=generator)
    points[:, 2].uniform_(0.0, T_MAX, generator=generator)
    half = count // 2
    points[:half, normal_coordinate] = 0.0
    points[half:, normal_coordinate] = DOMAIN_MAX
    return points


def sample_boundary_groups(
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    if count < 4:
        raise ValueError("boundary sample count must be at least four")
    count_x = count // 2
    count_y = count - count_x
    return (
        _sample_axis_faces(
            count_x, 0, device=device, generator=generator
        ),
        _sample_axis_faces(
            count_y, 1, device=device, generator=generator
        ),
    )


def _fixed_random_points(
    count: int,
    *,
    device: torch.device,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return sample_interior(count, device=device, generator=generator)


def _mass_grid(
    n_t: int,
    n_xy: int,
    *,
    device: torch.device,
) -> Tensor:
    # Midpoints avoid duplicating square-boundary coordinates.
    xy = (torch.arange(n_xy, device=device, dtype=REAL_DTYPE) + 0.5) * (
        DOMAIN_MAX / n_xy
    )
    times = torch.linspace(0.0, T_MAX, n_t, device=device, dtype=REAL_DTYPE)
    x_grid, y_grid, t_grid = torch.meshgrid(xy, xy, times, indexing="ij")
    return torch.stack(
        [x_grid.permute(2, 0, 1).reshape(-1),
         y_grid.permute(2, 0, 1).reshape(-1),
         t_grid.permute(2, 0, 1).reshape(-1)],
        dim=-1,
    )


def _time_slice_grid(
    times: tuple[float, ...],
    n_xy: int,
    *,
    device: torch.device,
) -> tuple[Tensor, int]:
    xy = (torch.arange(n_xy, device=device, dtype=REAL_DTYPE) + 0.5) * (
        DOMAIN_MAX / n_xy
    )
    x_grid, y_grid = torch.meshgrid(xy, xy, indexing="ij")
    spatial = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)
    groups = []
    for time_value in times:
        t = torch.full(
            (spatial.shape[0], 1), time_value, device=device, dtype=REAL_DTYPE
        )
        groups.append(torch.cat([spatial, t], dim=-1))
    return torch.cat(groups, dim=0), spatial.shape[0]


def _relative_l2_from_values(prediction: Tensor, target: Tensor) -> float:
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
    outputs = []
    with torch.no_grad():
        for start in range(0, points.shape[0], chunk_size):
            chunk = points[start:start + chunk_size].to(dtype=dtype)
            outputs.append(_predict_real(model, chunk).to(dtype=REAL_DTYPE))
    return torch.cat(outputs, dim=0)


@dataclass
class Cahn2DLossBundle:
    loss_fn: Callable[[], tuple[Tensor, dict[str, Tensor]]]
    eval_metrics_fn: Callable[[], dict[str, object]]
    history_metrics_fn: Callable[[], dict[str, object]]
    metadata: dict[str, object]


def make_loss_bundle(
    task: Cahn2DTask,
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
) -> Cahn2DLossBundle:
    if len(weights) != 2:
        raise ValueError("2D Cahn--Hilliard expects [lambda_ic, lambda_bc]")
    lambda_ic, lambda_bc = (float(value) for value in weights)
    if smoke:
        default_sizes = (32, 16, 16, 256, 128)
        mass_shape = (4, 8)
        slice_side = 8
    else:
        # These are exposed as runner arguments and may be reduced after the
        # H20 memory smoke.  Resampling each step provides broad domain coverage.
        default_sizes = (4096, 1024, 2048, 32768, 4096)
        mass_shape = (8, 16)
        slice_side = 64
    n_int = default_sizes[0] if n_int is None else int(n_int)
    n_ic = default_sizes[1] if n_ic is None else int(n_ic)
    n_bc = default_sizes[2] if n_bc is None else int(n_bc)
    n_eval = default_sizes[3] if n_eval is None else int(n_eval)
    history_eval_n = (
        default_sizes[4] if history_eval_n is None else int(history_eval_n)
    )
    if min(n_int, n_ic, n_bc, n_eval, history_eval_n) <= 0:
        raise ValueError("all sample counts must be positive")

    train_generator = torch.Generator(device=device).manual_seed(train_seed)
    eval_points = _fixed_random_points(n_eval, device=device, seed=eval_seed)
    history_points = eval_points[: min(history_eval_n, n_eval)]
    target_eval = exact_solution(eval_points).detach().to(REAL_DTYPE)
    target_history = exact_solution(history_points).detach().to(REAL_DTYPE)
    mass_points = _mass_grid(*mass_shape, device=device)
    formal_times = (0.0, 0.25, 0.5, 0.75, 1.0)
    time_slice_points, time_slice_group = _time_slice_grid(
        formal_times, slice_side, device=device
    )
    time_slice_targets = exact_solution(time_slice_points).detach().to(REAL_DTYPE)

    # Characteristic scaling for derivative orders 1, 3, and 5, based on the
    # maximum manufactured |k|^2=5.  This is loss normalisation only; it never
    # enters the network input or initialisation.
    boundary_scales = [
        KAPPA_MAX_SQUARED ** (0.5 * (2 * ell + 1))
        for ell in range(task.q)
    ]

    def loss_fn() -> tuple[Tensor, dict[str, Tensor]]:
        interior = sample_interior(
            n_int, device=device, generator=train_generator
        )
        initial = sample_initial(n_ic, device=device, generator=train_generator)
        boundary_x, boundary_y = sample_boundary_groups(
            n_bc, device=device, generator=train_generator
        )
        xi = interior.to(dtype=dtype)
        xic = initial.to(dtype=dtype)
        xbx = boundary_x.to(dtype=dtype)
        xby = boundary_y.to(dtype=dtype)

        u = _predict_real(model, xi)
        u_x = deriv_alpha(model, xi, (0,), backend=backend).real.squeeze(-1)
        u_y = deriv_alpha(model, xi, (1,), backend=backend).real.squeeze(-1)
        u_t = deriv_alpha(model, xi, (2,), backend=backend).real.squeeze(-1)
        lap_u = laplacian_power(model, xi, 1, backend)
        lap_nonlinear = (
            (3.0 * u.square() - 1.0) * lap_u
            + 6.0 * u * (u_x.square() + u_y.square())
        )
        high = laplacian_power(model, xi, task.q, backend)
        source = manufactured_source(interior, task).detach()
        residual = (
            u_t
            - GAMMA_NONLINEAR * lap_nonlinear
            + task.eta * high
            - source
        )
        l_pde = residual.square().mean()

        target_ic = exact_solution(initial).detach()
        prediction_ic = _predict_real(model, xic)
        l_ic = (prediction_ic - target_ic).square().mean()

        raw_bc: list[Tensor] = []
        normalised_bc: list[Tensor] = []
        components: dict[str, Tensor] = {
            "L_PDE": l_pde,
            "L_IC": l_ic,
        }
        for ell, scale in enumerate(boundary_scales):
            residual_x = normal_laplacian_power(model, xbx, ell, 0, backend)
            residual_y = normal_laplacian_power(model, xby, ell, 1, backend)
            raw = 0.5 * (residual_x.square().mean() + residual_y.square().mean())
            normalised = raw / (scale * scale)
            derivative_order = 2 * ell + 1
            components[f"L_BC_order{derivative_order}_raw"] = raw
            components[f"L_BC_order{derivative_order}"] = normalised
            raw_bc.append(raw)
            normalised_bc.append(normalised)
        l_bc = torch.stack(normalised_bc).mean()
        weighted_ic = lambda_ic * l_ic
        weighted_bc = lambda_bc * l_bc
        # Both methods optimize exactly the same physical objective.  In
        # particular, WAR receives no method-specific parameter regularizer.
        total = l_pde + weighted_ic + weighted_bc
        components.update({
            "L_BC": l_bc,
            "weighted_L_IC": weighted_ic,
            "weighted_L_BC": weighted_bc,
            "weighted_L_constraints": weighted_ic + weighted_bc,
            "loss": total,
        })
        return total, components

    def evaluation(points: Tensor, targets: Tensor) -> dict[str, object]:
        prediction = _chunked_prediction(model, points, dtype)
        return {"rel_error": _relative_l2_from_values(prediction, targets)}

    def history_metrics_fn() -> dict[str, object]:
        metrics = evaluation(history_points, target_history)
        mass_prediction = _chunked_prediction(model, mass_points, dtype)
        mass_by_time = mass_prediction.reshape(mass_shape[0], -1).mean(dim=1)
        metrics.update({
            "mass_drift_rms": float(mass_by_time.square().mean().sqrt().item()),
            "mass_drift_max_abs": float(mass_by_time.abs().max().item()),
        })
        return metrics

    def eval_metrics_fn() -> dict[str, object]:
        metrics = evaluation(eval_points, target_eval)
        mass_prediction = _chunked_prediction(model, mass_points, dtype)
        mass_by_time = mass_prediction.reshape(mass_shape[0], -1).mean(dim=1)
        metrics.update({
            "mass_drift_rms": float(mass_by_time.square().mean().sqrt().item()),
            "mass_drift_max_abs": float(mass_by_time.abs().max().item()),
        })
        slice_prediction = _chunked_prediction(model, time_slice_points, dtype)
        slice_errors: dict[str, float] = {}
        for index, time_value in enumerate(formal_times):
            start = index * time_slice_group
            stop = start + time_slice_group
            slice_errors[f"t={time_value:g}"] = _relative_l2_from_values(
                slice_prediction[start:stop], time_slice_targets[start:stop]
            )
        metrics["time_slice_rel_errors"] = slice_errors
        return metrics

    return Cahn2DLossBundle(
        loss_fn=loss_fn,
        eval_metrics_fn=eval_metrics_fn,
        history_metrics_fn=history_metrics_fn,
        metadata={
            "family": "cahn_hilliard_2d",
            "domain": "(x,y) in (0,pi)^2, t in [0,1]",
            "physical_input_dim": 3,
            "network_input_dim": 3,
            "input_features": ["affine(x)", "affine(y)", "affine(t)"],
            "trigonometric_input_features": False,
            "periodic_embedding": False,
            "exact_solution": (
                "exp(-t)*(0.5*cos(x)*cos(y) + 0.25*cos(2*x)*cos(y) "
                "+ 0.25*cos(x)*cos(2*y))"
            ),
            "manufactured_modes": [list(mode) for mode in MANUFACTURED_MODES],
            "equation": "u_t - Delta(u^3-u) + eta_q*Delta^q(u) = f",
            "q": task.q,
            "eta_q": task.eta,
            "gamma_nonlinear": GAMMA_NONLINEAR,
            "well_posed_leading_symbol": True,
            "boundary_type": "homogeneous_natural_no_flux",
            "boundary_conditions": [
                f"d_n_Delta^{ell}_u=0" for ell in range(task.q)
            ],
            "boundary_derivative_orders": [
                2 * ell + 1 for ell in range(task.q)
            ],
            "boundary_normalization": boundary_scales,
            "mass_constraint_in_training": False,
            "mass_used_as_diagnostic": True,
            "sample_policy": "resample_each_training_step",
            "n_int": n_int,
            "n_ic": n_ic,
            "n_bc_total": n_bc,
            "n_eval": n_eval,
            "history_eval_n": min(history_eval_n, n_eval),
            "training_real_dtype": str(REAL_DTYPE),
            "war_parameter_dtype": str(COMPLEX_DTYPE),
            "autodiff_parameter_dtype": str(REAL_DTYPE),
            "activation_shared_by_methods": "sinh",
            "lambda_bulk": 1.0,
            "lambda_ic": lambda_ic,
            "lambda_bc": lambda_bc,
        },
    )


def tensor_components_to_float(
    components: dict[str, Tensor],
) -> dict[str, float]:
    return {
        name: float(value.detach().real.item())
        for name, value in components.items()
    }


__all__ = [
    "AffineCoordinateMLP",
    "COMPLEX_DTYPE",
    "Cahn2DLossBundle",
    "Cahn2DTask",
    "DEPTH",
    "DOMAIN_MAX",
    "EVAL_SEED",
    "ETA_MAGNITUDE",
    "GAMMA_NONLINEAR",
    "GRID_VALUES",
    "HIDDEN",
    "HISTORY_INTERVAL_SECONDS",
    "LEARNING_RATE",
    "LEARNING_RATE_FINAL",
    "MANUFACTURED_MODES",
    "METHODS",
    "PROTOCOL_ID",
    "REAL_DTYPE",
    "TASKS",
    "TRAIN_SEED",
    "T_MAX",
    "build_model",
    "exact_solution",
    "laplacian_power",
    "make_loss_bundle",
    "manufactured_components",
    "manufactured_source",
    "model_metadata",
    "normal_laplacian_power",
    "sample_boundary_groups",
    "sample_initial",
    "sample_interior",
    "tensor_components_to_float",
]
