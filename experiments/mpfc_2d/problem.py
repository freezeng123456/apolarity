"""二维六阶 Modified Phase-Field Crystal (MPFC) PINN 算例。

The benchmark keeps the sixth-order spatial operator explicit.  With

    beta * phi_tt + phi_t = M * Delta(mu),
    mu = Delta**2(phi) + 2*Delta(phi) + (1-epsilon)*phi + phi**3,

the training residual is

    beta*phi_tt + phi_t
      - M*(Delta**3(phi) + 2*Delta**2(phi)
          + (1-epsilon)*Delta(phi) + Delta(phi**3)).

The network receives only affine-normalized raw ``(x,y,t)`` coordinates.  The
periodic boundary loss matches normal derivatives 0 through 5, so the PINN
protocol exposes the full sixth-order nature of the equation rather than
introducing an auxiliary chemical-potential network.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import torch
from osc_common import build_plain, deriv_alpha, laplacian_power_terms, n_params
from torch import Tensor, nn

from apolarity.taylor_jet import TaylorJet, jet_forward_sequential


PROTOCOL_ID = "mpfc_2d_o6_common_xavier_fp32_v1"
RUNNER_FAMILY_NAME = "modified_phase_field_crystal"
TASK_ID = "mpfc_2d_o6"
FAMILY = "modified_phase_field_crystal"
REAL_DTYPE = torch.float32
COMPLEX_DTYPE = torch.complex64
METHODS = ("war", "real_tanh_autodiff")
BASELINE_ACTIVATION = "tanh"
ALTERNATE_METHOD_ORDER = True
STRICT_MANIFEST_BINDING = True
GRID_VALUES = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
HIDDEN = 128
DEPTH = 4
LEARNING_RATE = 1e-3
LEARNING_RATE_FINAL = 1e-4
HISTORY_INTERVAL_SECONDS = 10.0
DOMAIN_MAX = 2.0 * math.pi
T_MAX = 1.0
MOBILITY = 1.0
BETA = 0.1
EPSILON = 0.25
RESIDUAL_SCALE = 64.0
TRAIN_SEED = 42
EVAL_SEED = 68421
INPUT_DIM = 3


@dataclass(frozen=True)
class MPFCTask:
    task_id: str = TASK_ID
    family: str = FAMILY
    spatial_dim: int = 2
    order: int = 6
    q: int = 3
    eta: float = 1.0
    coordinate_names: tuple[str, ...] = ("x", "y", "t")
    lows: tuple[float, ...] = (0.0, 0.0, 0.0)
    highs: tuple[float, ...] = (DOMAIN_MAX, DOMAIN_MAX, T_MAX)
    weight_names: tuple[str, str] = ("lambda_ic", "lambda_bc")
    weights: tuple[float, float] = (1.0, 1.0)
    uniqueness: str = (
        "periodic smooth/energy solution of the MPFC initial-value problem; "
        "for zero-mean initial velocity, mass is conserved and the pseudo-energy "
        "is non-increasing"
    )

    @property
    def input_dim(self) -> int:
        return len(self.coordinate_names)

    @property
    def weight_count(self) -> int:
        return len(self.weight_names)

    @property
    def center_weights(self) -> tuple[float, float]:
        return self.weights


TASK = MPFCTask()
Cahn2DTask = MPFCTask
TASKS = {TASK_ID: TASK}
TASK_ORDER = (TASK_ID,)

# The generic audited two-weight runner consumes the same metadata/metric
# contract as the existing HO-04 and CH runners.  Keeping these declarations
# here makes the MPFC experiment self-describing in its manifest.
OUTPUT_DIM = 1
SUMMARY_METRIC_KEYS = ("rel_error",)
SMOKE_METRIC_KEYS = SUMMARY_METRIC_KEYS
HISTORY_REQUIRED_METRICS: tuple[str, ...] = ()
_REFERENCE_ROOT_RAW = os.environ.get("APOLARITY_MPFC_REFERENCE_ROOT")
REFERENCE_ROOT = (
    Path(_REFERENCE_ROOT_RAW).expanduser() if _REFERENCE_ROOT_RAW else None
)


def _reference_sha256(root: Path | None = None) -> str | None:
    base = root or REFERENCE_ROOT
    if base is None:
        return None
    path = base / "REFERENCE_SHA256"
    try:
        return path.read_text().split()[0]
    except (OSError, IndexError):
        return None


RUNNER_MANIFEST_METADATA = {
    "equation": (
        "beta*phi_tt + phi_t - M*(Delta^3(phi) + 2*Delta^2(phi) "
        "+ (1-epsilon)*Delta(phi) + Delta(phi^3)) = 0"
    ),
    "M": MOBILITY,
    "beta": BETA,
    "epsilon": EPSILON,
    "domain": "[0,2*pi]^2 x [0,1]",
    "initial_condition": (
        "0.1 + 0.15*cos(x)*cos(y) + 0.05*cos(2*x)*cos(y)"
    ),
    "initial_velocity": "0",
    "boundary": "explicit periodic traces of normal derivatives 0..5",
    "training_precision": {"war": "complex64", "real_ad": "float32"},
    "primary_metric": "relative L2 error against converged Fourier reference",
    "direct_sixth_order_residual": True,
    "frequency_initialization": False,
    "trigonometric_input_features": False,
}


def runner_manifest_metadata(*, smoke: bool) -> dict[str, object]:
    metadata = dict(RUNNER_MANIFEST_METADATA)
    metadata["smoke"] = bool(smoke)
    reference_sha = _reference_sha256()
    metadata["reference"] = {
        "root": str(REFERENCE_ROOT) if REFERENCE_ROOT else None,
        "sha256": reference_sha,
        "protocol": "mpfc_2d_imex_fourier_reference_v1",
    }
    return metadata


def _common_xavier_init_(net: nn.Sequential) -> None:
    """Use the same variance-matched Xavier law for real and complex nets."""

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
    """Plain scalar MLP on affine-normalized physical ``(x,y,t)``."""

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
        if points.ndim != 2 or points.shape[-1] != INPUT_DIM:
            raise ValueError("MPFC model expects points with shape (batch,3)")
        return self.net(
            points * self._scale_for(points) + self._shift_for(points)
        )

    def jet_forward(self, jet: TaylorJet) -> TaylorJet:
        if jet.terms[0].shape[-1] != INPUT_DIM:
            raise ValueError("MPFC jet expects physical inputs (x,y,t)")
        scale = self._scale_for(jet.terms[0])
        shift = self._shift_for(jet.terms[0])
        transformed = TaylorJet(
            [jet.terms[0] * scale + shift, *[term * scale for term in jet.terms[1:]]]
        )
        return jet_forward_sequential(self.net, transformed)


def build_model(
    task: MPFCTask,
    method: str,
    device: torch.device,
    *,
    hidden: int = HIDDEN,
    depth: int = DEPTH,
) -> tuple[nn.Module, torch.dtype, str]:
    if task.task_id != TASK_ID:
        raise ValueError(f"unknown MPFC task {task.task_id!r}")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    is_war = method == "war"
    dtype = COMPLEX_DTYPE if is_war else REAL_DTYPE
    activation = "sinh" if is_war else "tanh"
    net = build_plain(INPUT_DIM, hidden, depth, dtype, activation, out=1)
    _common_xavier_init_(net)
    return AffineCoordinateMLP(net).to(device=device), dtype, (
        "waring_complex_jet" if is_war else "direct_autodiff"
    )


def model_metadata(model: nn.Module, method: str) -> dict[str, object]:
    parameter = next(model.parameters())
    return {
        "method": method,
        "representation": "native_complex" if method == "war" else "real",
        "activation": "sinh" if method == "war" else "tanh",
        "derivative_backend": (
            "waring_complex_jet" if method == "war" else "direct_autodiff"
        ),
        "hidden": HIDDEN,
        "depth": DEPTH,
        "output_dim": 1,
        "parameter_elements": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "real_dof": n_params(model),
        "literal_layer_shape_matched": True,
        "init_mode": "common_xavier",
        "frequency_initialization": "disabled",
        "input_transform": "affine_only",
        "trigonometric_input_features": False,
        "parameter_dtype": str(parameter.dtype),
    }


def _predict_real(model: nn.Module, points: Tensor) -> Tensor:
    value = model(points).real
    if value.ndim != 2 or value.shape[-1] != 1:
        raise ValueError("MPFC network must output shape (batch,1)")
    return value[:, 0]


def _partial(
    model: nn.Module,
    points: Tensor,
    alpha: tuple[int, ...],
    backend: str,
) -> Tensor:
    if not alpha:
        return _predict_real(model, points)
    return deriv_alpha(model, points, alpha, backend=backend).real[:, 0]


def spatial_laplacian_power(
    model: nn.Module,
    points: Tensor,
    power: int,
    backend: str,
) -> Tensor:
    if power < 1 or power > 3:
        raise ValueError("MPFC Laplacian power must be in {1,2,3}")
    value: Tensor | None = None
    for coefficient, alpha in laplacian_power_terms(2, power):
        term = coefficient * _partial(model, points, alpha, backend)
        value = term if value is None else value + term
    if value is None:  # pragma: no cover
        raise RuntimeError("empty Laplacian expansion")
    return value


def laplacian_of_cube(
    model: nn.Module,
    points: Tensor,
    backend: str,
) -> Tensor:
    """Evaluate ``Delta(phi**3)`` without hiding the sixth-order residual.

    The product rule gives ``Delta(phi^3)=6 phi |grad phi|^2 + 3 phi^2 Delta phi``;
    this avoids asking the derivative backend to differentiate a detached
    intermediate while remaining exactly equivalent to the expanded PDE.
    """

    phi = _predict_real(model, points)
    grad_sq = sum(
        _partial(model, points, (axis,), backend).square() for axis in (0, 1)
    )
    lap = spatial_laplacian_power(model, points, 1, backend)
    return 6.0 * phi * grad_sq + 3.0 * phi.square() * lap


def mpfc_residual(
    model: nn.Module,
    points: Tensor,
    backend: str,
) -> Tensor:
    """Return the unscaled direct sixth-order MPFC residual."""

    phi_t = _partial(model, points, (2,), backend)
    phi_tt = _partial(model, points, (2, 2), backend)
    lap = spatial_laplacian_power(model, points, 1, backend)
    biharm = spatial_laplacian_power(model, points, 2, backend)
    tri = spatial_laplacian_power(model, points, 3, backend)
    cube_lap = laplacian_of_cube(model, points, backend)
    chemical_lap = tri + 2.0 * biharm + (1.0 - EPSILON) * lap + cube_lap
    return BETA * phi_tt + phi_t - MOBILITY * chemical_lap


def initial_phi(points: Tensor) -> Tensor:
    x, y = points[..., 0], points[..., 1]
    return 0.1 + 0.15 * torch.cos(x) * torch.cos(y) + 0.05 * torch.cos(2.0 * x) * torch.cos(y)


def initial_phi_t(points: Tensor) -> Tensor:
    return torch.zeros_like(points[..., 0])


def sample_interior(
    task: MPFCTask,
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    if count <= 0:
        raise ValueError("sample count must be positive")
    unit = torch.rand(count, task.input_dim, device=device, dtype=REAL_DTYPE, generator=generator)
    lows = torch.tensor(task.lows, device=device, dtype=REAL_DTYPE)
    widths = torch.tensor([hi - lo for lo, hi in zip(task.lows, task.highs)], device=device, dtype=REAL_DTYPE)
    return unit * widths + lows


def sample_initial(
    task: MPFCTask,
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    points = sample_interior(task, count, device=device, generator=generator)
    points[:, -1] = task.lows[-1]
    return points


def sample_face_pairs(
    task: MPFCTask,
    count: int,
    coordinate: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    if coordinate not in (0, 1):
        raise ValueError("periodic MPFC faces are x/y only")
    lower = sample_interior(task, count, device=device, generator=generator)
    upper = lower.clone()
    lower[:, coordinate] = task.lows[coordinate]
    upper[:, coordinate] = task.highs[coordinate]
    return lower, upper


def _periodic_boundary_loss(
    task: MPFCTask,
    model: nn.Module,
    dtype: torch.dtype,
    backend: str,
    count: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, dict[str, Tensor]]:
    per_axis = max(2, count // 2)
    losses: list[Tensor] = []
    components: dict[str, Tensor] = {}
    for coordinate in (0, 1):
        lower, upper = sample_face_pairs(
            task, per_axis, coordinate, device=device, generator=generator
        )
        lower, upper = lower.to(dtype=dtype), upper.to(dtype=dtype)
        for derivative_order in range(6):
            alpha = (coordinate,) * derivative_order
            difference = _partial(model, lower, alpha, backend) - _partial(model, upper, alpha, backend)
            value = (difference / (2.0 ** derivative_order)).square().mean()
            components[f"L_BC_{task.coordinate_names[coordinate]}_order{derivative_order}"] = value
            losses.append(value)
    total = torch.stack(losses).mean()
    components["L_BC"] = total
    return total, components


@dataclass
class LossBundle:
    loss_fn: Callable[[], tuple[Tensor, dict[str, Tensor]]]
    eval_metrics_fn: Callable[[], dict[str, object]]
    history_metrics_fn: Callable[[], dict[str, object]]
    metadata: dict[str, object]


def _load_reference(
    *,
    device: torch.device,
    n_eval: int,
    history_eval_n: int,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    """Load the independently generated Fourier reference and bind its SHA."""

    if REFERENCE_ROOT is None:
        raise FileNotFoundError(
            "MPFC reference.npz is required for non-smoke evaluation; set "
            "APOLARITY_MPFC_REFERENCE_ROOT"
        )
    reference_path = REFERENCE_ROOT / "reference.npz"
    reference_sha = _reference_sha256()
    if not reference_path.is_file():
        raise FileNotFoundError(
            "MPFC reference.npz is required for non-smoke evaluation; set "
            "APOLARITY_MPFC_REFERENCE_ROOT"
        )
    # numpy is deliberately loaded through npz parsing rather than torch
    # serialization; importing it here keeps the training module lightweight.
    import numpy as np
    data = np.load(reference_path)
    points_np = np.asarray(data["points"], dtype=np.float32)
    phi_np = np.asarray(data["phi"], dtype=np.float32)
    if points_np.ndim != 2 or points_np.shape[1] != INPUT_DIM:
        raise ValueError("MPFC reference points must have shape (N,3)")
    if phi_np.ndim != 1 or phi_np.shape[0] != points_np.shape[0]:
        raise ValueError("MPFC reference phi must be a vector aligned to points")
    if n_eval > points_np.shape[0]:
        raise ValueError(
            f"reference has {points_np.shape[0]} points but n_eval={n_eval}"
        )
    if history_eval_n > n_eval:
        raise ValueError("history_eval_n cannot exceed n_eval")
    points = torch.as_tensor(points_np[:n_eval], device=device, dtype=REAL_DTYPE)
    phi = torch.as_tensor(phi_np[:n_eval], device=device, dtype=REAL_DTYPE)
    metadata = {
        "reference_path": str(reference_path),
        "reference_sha256": reference_sha,
        "sha256": reference_sha,
        "reference_count": int(points_np.shape[0]),
        "reference_protocol": "mpfc_2d_imex_fourier_reference_v1",
        "reference_eval_seed": EVAL_SEED,
    }
    return points, phi, metadata


def make_loss_bundle(
    task: MPFCTask,
    model: nn.Module,
    dtype: torch.dtype,
    backend: str,
    weights: tuple[float, ...],
    device: torch.device,
    *,
    smoke: bool = False,
    n_int: int | None = None,
    n_ic: int | None = None,
    n_bc: int | None = None,
    n_eval: int | None = None,
    history_eval_n: int | None = None,
    train_seed: int = TRAIN_SEED,
    eval_seed: int = EVAL_SEED,
) -> LossBundle:
    if len(weights) != 2 or any(value <= 0 for value in weights):
        raise ValueError("MPFC requires positive (lambda_ic, lambda_bc)")
    defaults = (16, 16, 16, 128, 64) if smoke else (
        2048, 512, 1024, 16384, 2048
    )
    n_int = defaults[0] if n_int is None else int(n_int)
    n_ic = defaults[1] if n_ic is None else int(n_ic)
    n_bc = defaults[2] if n_bc is None else int(n_bc)
    n_eval = defaults[3] if n_eval is None else int(n_eval)
    history_eval_n = defaults[4] if history_eval_n is None else int(history_eval_n)
    if min(n_int, n_ic, n_bc, n_eval, history_eval_n) <= 0:
        raise ValueError("all MPFC sample counts must be positive")
    if history_eval_n > n_eval:
        raise ValueError("history_eval_n cannot exceed n_eval")
    train_generator = torch.Generator(device=device).manual_seed(train_seed)
    eval_generator = torch.Generator(device=device).manual_seed(eval_seed)
    if smoke:
        eval_points = sample_interior(
            task, n_eval, device=device, generator=eval_generator
        )
        eval_target = initial_phi(eval_points).detach()
        reference_metadata: dict[str, object] = {
            "reference_protocol": "smoke_initial_condition_only",
            "reference_sha256": None,
            "reference_count": 0,
        }
    else:
        eval_points, eval_target, reference_metadata = _load_reference(
            device=device, n_eval=n_eval, history_eval_n=history_eval_n
        )
    history_points = eval_points[:history_eval_n]
    history_target = eval_target[:history_eval_n]

    def loss_fn() -> tuple[Tensor, dict[str, Tensor]]:
        interior = sample_interior(
            task, n_int, device=device, generator=train_generator
        ).to(dtype=dtype)
        residual = mpfc_residual(model, interior, backend)
        l_pde = (residual / RESIDUAL_SCALE).square().mean()

        boundary, boundary_parts = _periodic_boundary_loss(
            task, model, dtype, backend, n_bc, device, train_generator
        )
        initial_physical = sample_initial(
            task, n_ic, device=device, generator=train_generator
        )
        initial = initial_physical.to(dtype=dtype)
        phi = _predict_real(model, initial)
        phi_t = _partial(model, initial, (2,), backend)
        # Targets are physical real-valued data.  Evaluating them on the
        # complex WAR input would promote the IC loss to complex dtype and
        # make ``loss.backward()`` invalid for a scalar objective.
        target_phi = initial_phi(initial_physical).detach()
        target_phi_t = initial_phi_t(initial_physical).detach()
        l_ic_phi = (phi - target_phi).square().mean()
        l_ic_velocity = (phi_t - target_phi_t).square().mean()
        l_ic = 0.5 * (l_ic_phi + l_ic_velocity)
        lambda_ic, lambda_bc = weights
        total = l_pde + lambda_ic * l_ic + lambda_bc * boundary
        components = {
            "L_PDE": l_pde,
            "L_IC_phi": l_ic_phi,
            "L_IC_phi_t": l_ic_velocity,
            "L_IC": l_ic,
            "L_BC": boundary,
            "weighted_L_IC": lambda_ic * l_ic,
            "weighted_L_BC": lambda_bc * boundary,
            "loss": total,
            **boundary_parts,
        }
        return total, components

    def _relative_l2(prediction: Tensor, target: Tensor) -> float:
        numerator = (prediction - target).square().mean().sqrt()
        denominator = target.square().mean().sqrt().clamp_min(1e-12)
        return float((numerator / denominator).item())

    def _evaluate(points: Tensor, target: Tensor) -> dict[str, object]:
        with torch.no_grad():
            prediction = _predict_real(model, points.to(dtype=dtype))
        return {"rel_error": _relative_l2(prediction, target)}

    def history_metrics_fn() -> dict[str, object]:
        return _evaluate(history_points, history_target)

    def eval_metrics_fn() -> dict[str, object]:
        return _evaluate(eval_points, eval_target)

    return LossBundle(
        loss_fn=loss_fn,
        eval_metrics_fn=eval_metrics_fn,
        history_metrics_fn=history_metrics_fn,
        metadata={
            "task_id": task.task_id,
            "equation": (
                "beta*phi_tt + phi_t - M*(Delta^3(phi) + 2*Delta^2(phi) "
                "+ (1-epsilon)*Delta(phi) + Delta(phi^3)) = 0"
            ),
            "order": 6,
            "spatial_dim": 2,
            "domain": "[0,2*pi]^2 x [0,1]",
            "M": MOBILITY,
            "beta": BETA,
            "epsilon": EPSILON,
            "boundary": "periodic traces of normal derivatives 0..5",
            "weights": {"lambda_ic": weights[0], "lambda_bc": weights[1]},
            "residual_scale": RESIDUAL_SCALE,
            "n_int": n_int,
            "n_ic": n_ic,
            "n_bc": n_bc,
            "sample_policy": "resample_each_training_step",
            "reference": reference_metadata,
            "eval_seed": eval_seed,
            "history_eval_n": history_eval_n,
            "n_eval": n_eval,
            "smoke": bool(smoke),
        },
    )


def tensor_components_to_float(values: dict[str, Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().real.item()) for key, value in values.items()
    }


__all__ = [
    "BASELINE_ACTIVATION",
    "BETA",
    "COMPLEX_DTYPE",
    "Cahn2DTask",
    "DEPTH",
    "EPSILON",
    "EVAL_SEED",
    "GRID_VALUES",
    "HIDDEN",
    "HISTORY_INTERVAL_SECONDS",
    "LEARNING_RATE",
    "LEARNING_RATE_FINAL",
    "METHODS",
    "MPFCTask",
    "MOBILITY",
    "OUTPUT_DIM",
    "PROTOCOL_ID",
    "REAL_DTYPE",
    "RESIDUAL_SCALE",
    "RUNNER_FAMILY_NAME",
    "SMOKE_METRIC_KEYS",
    "SUMMARY_METRIC_KEYS",
    "TASK",
    "TASKS",
    "TASK_ORDER",
    "TRAIN_SEED",
    "build_model",
    "initial_phi",
    "initial_phi_t",
    "laplacian_of_cube",
    "make_loss_bundle",
    "model_metadata",
    "mpfc_residual",
    "runner_manifest_metadata",
    "sample_face_pairs",
    "sample_initial",
    "sample_interior",
    "spatial_laplacian_power",
    "tensor_components_to_float",
]
