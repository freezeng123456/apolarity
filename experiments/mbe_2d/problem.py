"""Two-dimensional fourth-order slope-selection MBE benchmark.

The physical problem is posed on ``T^2 x [0, 1]`` with ``T=[0,2*pi]``:

    h_t = div((|grad h|^2 - 1) grad h) - nu Delta^2 h,  nu=0.05.

Both networks receive only an affine normalization of raw ``(x,y,t)``.  The
periodic boundary is imposed by matching the value and normal derivatives of
orders one through three on opposite faces.  No periodic embedding, Fourier
feature, or frequency-aware initialization is used.

The ephemeral CUDA smoke uses a smooth manufactured solution and analytic
source.  Non-smoke runs require an independently generated, convergence-
checked pseudospectral reference for the unforced equation; the path is passed
through ``APOLARITY_MBE_REFERENCE_PATH``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from osc_common import build_plain, deriv_alpha, laplacian_power_terms, n_params
from torch import Tensor, nn

from apolarity.taylor_jet import TaylorJet, jet_forward_sequential


PROTOCOL_ID = "mbe_2d_slope_selection_spectral_reference_fp32_v1"
REFERENCE_PROTOCOL_ID = "mbe_2d_etdrk4_reference_v1"
REFERENCE_ENV = "APOLARITY_MBE_REFERENCE_PATH"
RUNNER_FAMILY_NAME = "mbe_2d_slope_selection"
BASELINE_ACTIVATION = "tanh"
ALTERNATE_METHOD_ORDER = True
STRICT_MANIFEST_BINDING = True
INPUT_DIM = 3
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
RESIDUAL_SCALE = 1.0
MAX_INITIAL_WAVENUMBER = 2.0
DIAGNOSTIC_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class MBETask:
    task_id: str
    family: str = RUNNER_FAMILY_NAME
    order: int = 4
    q: int = 2
    eta: float = -NU
    weight_names: tuple[str, str] = ("lambda_ic", "lambda_bc")
    center_weights: tuple[float, float] = (1.0, 1.0)

    @property
    def weight_count(self) -> int:
        return len(self.weight_names)


# Compatibility alias used by the shared audited grid runner.
Cahn2DTask = MBETask
TASKS: dict[str, MBETask] = {"mbe_2d_o4": MBETask("mbe_2d_o4")}
TASK_ORDER = tuple(TASKS)


RUNNER_MANIFEST_METADATA = {
    "equation": (
        "h_t = div((|grad h|^2-1)grad h) - nu*Delta_xy^2 h"
    ),
    "nu": NU,
    "domain": "[0,2*pi]^2 x [0,1]",
    "initial_condition": (
        "0.2*cos(x)*cos(y) + 0.1*cos(2*x)*cos(y)"
    ),
    "boundary": "explicit periodic trace matching, normal orders 0..3",
    "formal_source": "unforced",
    "smoke_source": "analytic manufactured forcing",
    "reference_protocol_id": REFERENCE_PROTOCOL_ID,
    "training_precision": {"war": "complex64", "real_ad": "float32"},
    "direct_fourth_order_residual": True,
}


def runner_manifest_metadata(*, smoke: bool) -> dict[str, object]:
    """Bind a non-smoke run manifest to the audited reference bytes."""

    metadata = dict(RUNNER_MANIFEST_METADATA)
    if smoke:
        metadata["reference"] = {
            "protocol_id": "analytic_manufactured_smoke_only"
        }
        return metadata
    raw_path = os.environ.get(REFERENCE_ENV)
    if not raw_path:
        raise RuntimeError(f"non-smoke MBE runs require {REFERENCE_ENV}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MBE reference does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        reference = json.loads(str(payload["metadata_json"].item()))
    if reference.get("protocol_id") != REFERENCE_PROTOCOL_ID:
        raise ValueError("MBE reference protocol mismatch")
    if not bool(reference.get("convergence_passed")):
        raise ValueError("MBE reference convergence gate did not pass")
    metadata["reference"] = {
        "protocol_id": reference["protocol_id"],
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "eval_seed": int(reference["eval_seed"]),
        "n_eval": int(reference["n_eval"]),
        "levels": reference["levels"],
        "coarse_medium_relative_difference": reference[
            "coarse_medium_relative_difference"
        ],
        "medium_fine_relative_difference": reference[
            "medium_fine_relative_difference"
        ],
        "tolerance": reference["tolerance"],
        "energy_increase_max": reference["energy_increase_max"],
        "convergence_passed": True,
    }
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
    """Plain MLP on affine-normalized raw physical coordinates."""

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
            "input_shift", torch.tensor([-1.0, -1.0, -1.0], dtype=REAL_DTYPE)
        )

    def _scale_for(self, value: Tensor) -> Tensor:
        return self.input_scale.to(device=value.device, dtype=value.dtype)

    def _shift_for(self, value: Tensor) -> Tensor:
        return self.input_shift.to(device=value.device, dtype=value.dtype)

    def forward(self, points: Tensor) -> Tensor:
        if points.shape[-1] != INPUT_DIM:
            raise ValueError("MBE model expects physical inputs (x,y,t)")
        return self.net(
            points * self._scale_for(points) + self._shift_for(points)
        )

    def jet_forward(self, jet: TaylorJet) -> TaylorJet:
        if jet.terms[0].shape[-1] != INPUT_DIM:
            raise ValueError("MBE jet expects physical inputs (x,y,t)")
        scale = self._scale_for(jet.terms[0])
        shift = self._shift_for(jet.terms[0])
        transformed = TaylorJet([
            jet.terms[0] * scale + shift,
            *[term * scale for term in jet.terms[1:]],
        ])
        return jet_forward_sequential(self.net, transformed)


def build_model(
    task: MBETask,
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
    net = build_plain(INPUT_DIM, hidden, depth, dtype, activation, out=1)
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
        "parameter_elements": parameter_elements,
        "real_dof": n_params(model),
        "literal_layer_shape_matched": True,
        "init_mode": "common_xavier",
        "frequency_initialization": "disabled",
        "input_transform": "affine_only",
        "trigonometric_input_features": False,
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
    power: int,
    backend: str,
) -> Tensor:
    total: Tensor | None = None
    for coefficient, alpha in laplacian_power_terms(2, power):
        term = coefficient * _partial(model, points, alpha, backend)
        total = term if total is None else total + term
    if total is None:  # pragma: no cover - positive powers only
        raise RuntimeError("empty Laplacian expansion")
    return total


MANUFACTURED_MODES: tuple[tuple[float, int, int], ...] = (
    (0.2, 1, 1),
    (0.1, 2, 1),
)


def manufactured_components(points: Tensor) -> dict[str, Tensor]:
    """Analytic fields for ``exp(-t) h_0`` used only by smoke."""

    if points.shape[-1] != INPUT_DIM:
        raise ValueError("manufactured MBE solution expects (x,y,t)")
    x, y, t = points.unbind(dim=-1)
    decay = torch.exp(-t)
    fields = {
        name: torch.zeros_like(x)
        for name in ("h", "h_x", "h_y", "h_xx", "h_yy", "h_xy", "biharm")
    }
    for amplitude, kx_raw, ky_raw in MANUFACTURED_MODES:
        kx = float(kx_raw)
        ky = float(ky_raw)
        cos_x = torch.cos(kx * x)
        sin_x = torch.sin(kx * x)
        cos_y = torch.cos(ky * y)
        sin_y = torch.sin(ky * y)
        mode = float(amplitude) * decay * cos_x * cos_y
        fields["h"] = fields["h"] + mode
        fields["h_x"] = fields["h_x"] - kx * float(amplitude) * decay * sin_x * cos_y
        fields["h_y"] = fields["h_y"] - ky * float(amplitude) * decay * cos_x * sin_y
        fields["h_xx"] = fields["h_xx"] - (kx**2) * mode
        fields["h_yy"] = fields["h_yy"] - (ky**2) * mode
        fields["h_xy"] = fields["h_xy"] + (
            kx * ky * float(amplitude) * decay * sin_x * sin_y
        )
        fields["biharm"] = fields["biharm"] + (
            (kx**2 + ky**2) ** 2 * mode
        )
    fields["h_t"] = -fields["h"]
    return fields


def slope_divergence_from_components(
    h_x: Tensor,
    h_y: Tensor,
    h_xx: Tensor,
    h_yy: Tensor,
    h_xy: Tensor,
) -> Tensor:
    """Expand ``div((|grad h|^2-1)grad h)`` without product AD."""

    return (
        (3.0 * h_x.square() + h_y.square() - 1.0) * h_xx
        + (h_x.square() + 3.0 * h_y.square() - 1.0) * h_yy
        + 4.0 * h_x * h_y * h_xy
    )


def manufactured_source(points: Tensor) -> Tensor:
    values = manufactured_components(points)
    divergence = slope_divergence_from_components(
        values["h_x"],
        values["h_y"],
        values["h_xx"],
        values["h_yy"],
        values["h_xy"],
    )
    return values["h_t"] - divergence + NU * values["biharm"]


def initial_condition(points: Tensor) -> Tensor:
    x = points[..., 0]
    y = points[..., 1]
    return 0.2 * torch.cos(x) * torch.cos(y) + 0.1 * torch.cos(2.0 * x) * torch.cos(y)


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
            chunk = points[start : start + chunk_size].to(dtype=dtype)
            outputs.append(_predict_real(model, chunk).to(REAL_DTYPE))
    return torch.cat(outputs, dim=0)


def _chunked_partial(
    model: nn.Module,
    points: Tensor,
    dtype: torch.dtype,
    alpha: tuple[int, ...],
    backend: str,
    *,
    chunk_size: int = 1024,
) -> Tensor:
    outputs: list[Tensor] = []
    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start : start + chunk_size].to(dtype=dtype)
        outputs.append(
            _partial(model, chunk, alpha, backend).detach().to(REAL_DTYPE)
        )
    return torch.cat(outputs, dim=0)


def _diagnostic_grid(side: int, device: torch.device) -> tuple[Tensor, int]:
    xy = torch.arange(side, device=device, dtype=REAL_DTYPE) * (
        DOMAIN_MAX / side
    )
    x_grid, y_grid = torch.meshgrid(xy, xy, indexing="ij")
    spatial = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=-1)
    groups = []
    for value in DIAGNOSTIC_TIMES:
        time_column = torch.full(
            (spatial.shape[0], 1), value, device=device, dtype=REAL_DTYPE
        )
        groups.append(torch.cat([spatial, time_column], dim=-1))
    return torch.cat(groups, dim=0), spatial.shape[0]


def _load_reference(
    n_eval: int,
    history_eval_n: int,
    eval_seed: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, object]]:
    raw_path = os.environ.get(REFERENCE_ENV)
    if not raw_path:
        raise RuntimeError(
            f"non-smoke MBE runs require {REFERENCE_ENV}"
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MBE reference does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        points_np = np.asarray(payload["points"], dtype=np.float32)
        values_np = np.asarray(payload["values"], dtype=np.float32)
    if metadata.get("protocol_id") != REFERENCE_PROTOCOL_ID:
        raise ValueError("MBE reference protocol mismatch")
    if int(metadata.get("eval_seed", -1)) != int(eval_seed):
        raise ValueError("MBE reference eval seed mismatch")
    if not bool(metadata.get("convergence_passed")):
        raise ValueError("MBE reference convergence gate did not pass")
    if points_np.shape != (values_np.shape[0], INPUT_DIM):
        raise ValueError("invalid MBE reference array shapes")
    if n_eval > points_np.shape[0] or history_eval_n > n_eval:
        raise ValueError("requested evaluation count exceeds MBE reference")
    points = torch.from_numpy(points_np[:n_eval]).to(device=device)
    values = torch.from_numpy(values_np[:n_eval]).to(device=device)
    history_points = points[:history_eval_n]
    history_values = values[:history_eval_n]
    public_metadata = {
        **metadata,
        "reference_file": path.name,
        "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return points, values, history_points, history_values, public_metadata


def _periodic_boundary_loss(
    model: nn.Module,
    dtype: torch.dtype,
    backend: str,
    count: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, dict[str, Tensor]]:
    per_axis = max(8, count // 2)
    values: list[Tensor] = []
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
            mismatch = (
                _partial(model, lower, alpha, backend)
                - _partial(model, upper, alpha, backend)
            )
            normalizer = MAX_INITIAL_WAVENUMBER**order
            component = (mismatch / normalizer).square().mean()
            components[f"L_BC_{label}_order{order}"] = component
            values.append(component)
    total = torch.stack(values).mean()
    components["L_BC"] = total
    return total, components


@dataclass
class MBELossBundle:
    loss_fn: Callable[[], tuple[Tensor, dict[str, Tensor]]]
    eval_metrics_fn: Callable[[], dict[str, object]]
    history_metrics_fn: Callable[[], dict[str, object]]
    metadata: dict[str, object]


def make_loss_bundle(
    task: MBETask,
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
) -> MBELossBundle:
    if len(weights) != 2:
        raise ValueError("MBE expects [lambda_ic, lambda_bc]")
    lambda_ic, lambda_bc = (float(value) for value in weights)
    defaults = (32, 16, 16, 256, 128) if smoke else (
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
        raise ValueError("all MBE sample counts must be positive")
    if history_eval_n > n_eval:
        raise ValueError("history_eval_n cannot exceed n_eval")

    train_generator = torch.Generator(device=device).manual_seed(train_seed)
    reference_metadata: dict[str, object]
    if smoke:
        eval_generator = torch.Generator(device=device).manual_seed(eval_seed)
        eval_points = sample_interior(
            n_eval, device=device, generator=eval_generator
        )
        history_points = eval_points[:history_eval_n]
        eval_target = manufactured_components(eval_points)["h"].detach()
        history_target = manufactured_components(history_points)["h"].detach()
        reference_metadata = {
            "protocol_id": "analytic_manufactured_smoke_only",
            "convergence_passed": True,
        }
    else:
        (
            eval_points,
            eval_target,
            history_points,
            history_target,
            reference_metadata,
        ) = _load_reference(n_eval, history_eval_n, eval_seed, device)

    diagnostic_side = 8 if smoke else 24
    diagnostic_points, diagnostic_group = _diagnostic_grid(
        diagnostic_side, device
    )

    def loss_fn() -> tuple[Tensor, dict[str, Tensor]]:
        interior = sample_interior(
            n_int, device=device, generator=train_generator
        )
        initial = sample_initial(
            n_ic, device=device, generator=train_generator
        )
        points = interior.to(dtype=dtype)
        h_x = _partial(model, points, (0,), backend)
        h_y = _partial(model, points, (1,), backend)
        h_xx = _partial(model, points, (0, 0), backend)
        h_yy = _partial(model, points, (1, 1), backend)
        h_xy = _partial(model, points, (0, 1), backend)
        h_t = _partial(model, points, (2,), backend)
        biharmonic = spatial_laplacian_power(model, points, 2, backend)
        divergence = slope_divergence_from_components(
            h_x, h_y, h_xx, h_yy, h_xy
        )
        source = (
            manufactured_source(interior).detach()
            if smoke
            else torch.zeros_like(interior[:, 0])
        )
        residual = h_t - divergence + NU * biharmonic - source
        l_pde_raw = residual.square().mean()
        l_pde = l_pde_raw / (RESIDUAL_SCALE**2)

        initial_points = initial.to(dtype=dtype)
        target_ic = initial_condition(initial).detach()
        l_ic = (
            _predict_real(model, initial_points) - target_ic
        ).square().mean()
        l_bc, boundary_components = _periodic_boundary_loss(
            model,
            dtype,
            backend,
            n_bc,
            device,
            train_generator,
        )
        weighted_ic = lambda_ic * l_ic
        weighted_bc = lambda_bc * l_bc
        total = l_pde + weighted_ic + weighted_bc
        components: dict[str, Tensor] = {
            "L_PDE_raw": l_pde_raw,
            "L_PDE": l_pde,
            "L_IC": l_ic,
            **boundary_components,
            "weighted_L_IC": weighted_ic,
            "weighted_L_BC": weighted_bc,
            "weighted_L_constraints": weighted_ic + weighted_bc,
            "loss": total,
        }
        return total, components

    def basic_metrics(points: Tensor, target: Tensor) -> dict[str, object]:
        prediction = _chunked_prediction(model, points, dtype)
        diagnostic_prediction = _chunked_prediction(
            model, diagnostic_points, dtype
        ).reshape(len(DIAGNOSTIC_TIMES), diagnostic_group)
        masses = diagnostic_prediction.mean(dim=1)
        mass_drift = masses - masses[0]
        return {
            "rel_error": _relative_l2(prediction, target),
            "mass_drift_rms": float(mass_drift.square().mean().sqrt().item()),
            "mass_drift_max_abs": float(mass_drift.abs().max().item()),
            "mass_by_time": {
                f"{value:.2f}": float(mass.item())
                for value, mass in zip(DIAGNOSTIC_TIMES, masses, strict=True)
            },
        }

    def history_metrics_fn() -> dict[str, object]:
        return basic_metrics(history_points, history_target)

    def eval_metrics_fn() -> dict[str, object]:
        metrics = basic_metrics(eval_points, eval_target)
        h_x = _chunked_partial(
            model, diagnostic_points, dtype, (0,), backend
        )
        h_y = _chunked_partial(
            model, diagnostic_points, dtype, (1,), backend
        )
        h_xx = _chunked_partial(
            model, diagnostic_points, dtype, (0, 0), backend
        )
        h_yy = _chunked_partial(
            model, diagnostic_points, dtype, (1, 1), backend
        )
        slope_squared = (h_x.square() + h_y.square()).reshape(
            len(DIAGNOSTIC_TIMES), diagnostic_group
        )
        laplacian = (h_xx + h_yy).reshape(
            len(DIAGNOSTIC_TIMES), diagnostic_group
        )
        energies = (
            0.25 * (slope_squared - 1.0).square()
            + 0.5 * NU * laplacian.square()
        ).mean(dim=1)
        energy_changes = energies[1:] - energies[:-1]
        metrics.update({
            "energy_by_time": {
                f"{value:.2f}": float(energy.item())
                for value, energy in zip(
                    DIAGNOSTIC_TIMES, energies, strict=True
                )
            },
            "energy_initial": float(energies[0].item()),
            "energy_final": float(energies[-1].item()),
            "energy_increase_max": float(
                torch.clamp_min(energy_changes, 0.0).max().item()
            ),
            "energy_violation_fraction": float(
                (energy_changes > 1e-6).to(REAL_DTYPE).mean().item()
            ),
            "slope_rms_final": float(
                slope_squared[-1].mean().sqrt().item()
            ),
            "energy_monotonicity_applicable": not smoke,
        })
        return metrics

    return MBELossBundle(
        loss_fn=loss_fn,
        eval_metrics_fn=eval_metrics_fn,
        history_metrics_fn=history_metrics_fn,
        metadata={
            "task_id": task.task_id,
            "family": task.family,
            "order": task.order,
            "spatial_dim": 2,
            "physical_input_dim": INPUT_DIM,
            "coordinates": ["x", "y", "t"],
            "domain": {"x": [0.0, DOMAIN_MAX], "y": [0.0, DOMAIN_MAX], "t": [0.0, T_MAX]},
            "equation": RUNNER_MANIFEST_METADATA["equation"],
            "nu": NU,
            "boundary_type": (
                "explicit opposite-face periodic trace matching for normal "
                "derivative orders 0,1,2,3"
            ),
            "uniqueness_basis": (
                "global well-posedness on the 2*pi-periodic torus for d<=3, "
                "nu>0, smooth mean-zero initial data"
            ),
            "source_mode": "manufactured_smoke" if smoke else "unforced",
            "manufactured_solution": smoke,
            "reference": reference_metadata,
            "input_transform": "affine_only",
            "trigonometric_input_features": False,
            "frequency_initialization": False,
            "weights": dict(zip(task.weight_names, weights)),
            "residual_scale": RESIDUAL_SCALE,
            "boundary_component_normalizers": {
                f"order_{order}": MAX_INITIAL_WAVENUMBER**order
                for order in range(4)
            },
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
        key: float(value.detach().real.item())
        for key, value in values.items()
    }


__all__ = [
    "BASELINE_ACTIVATION",
    "ALTERNATE_METHOD_ORDER",
    "COMPLEX_DTYPE",
    "Cahn2DTask",
    "DEPTH",
    "EVAL_SEED",
    "GRID_VALUES",
    "HIDDEN",
    "HISTORY_INTERVAL_SECONDS",
    "INPUT_DIM",
    "LEARNING_RATE",
    "LEARNING_RATE_FINAL",
    "MBETask",
    "METHODS",
    "NU",
    "PROTOCOL_ID",
    "REAL_DTYPE",
    "REFERENCE_PROTOCOL_ID",
    "RUNNER_FAMILY_NAME",
    "RUNNER_MANIFEST_METADATA",
    "STRICT_MANIFEST_BINDING",
    "TASKS",
    "TASK_ORDER",
    "TRAIN_SEED",
    "build_model",
    "initial_condition",
    "make_loss_bundle",
    "manufactured_components",
    "manufactured_source",
    "model_metadata",
    "runner_manifest_metadata",
    "sample_face_pairs",
    "slope_divergence_from_components",
    "spatial_laplacian_power",
    "tensor_components_to_float",
]
