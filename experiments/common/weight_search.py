"""Core models and losses for the WAR versus real-autodiff weight search.

The search deliberately lives outside the frozen ``jsc_v3`` protocol.  It
compares the proposed native-complex sinh network using Waring/Taylor jets
against a literal-width-matched real tanh network using nested coordinate
autodiff.  Every loss component is returned separately so the grid runner can
persist auditable real-time traces.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from osc_common import (
    build_plain,
    complex_freq_init_,
    deriv_alpha,
    laplacian_power_terms,
    n_params,
    predict,
    sample_boundary,
    sample_interior,
)
from torch import Tensor, nn

from apolarity.taylor_jet import TaylorJet, jet_forward_sequential, jet_sin

PROTOCOL_ID = "war_realad_weight_grid_v1"
GRID_VALUES = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
METHODS = ("war", "real_tanh_autodiff")
HIDDEN = 128
DEPTH = 4
TRAIN_SEED = 42
EVAL_SEED = 54321
LEARNING_RATE = 1e-3
LEARNING_RATE_FINAL = 1e-4
HISTORY_INTERVAL_SECONDS = 5.0


@dataclass(frozen=True)
class SearchTask:
    task_id: str
    family: str
    order: int
    weight_names: tuple[str, ...]
    center_weights: tuple[float, ...]

    @property
    def weight_count(self) -> int:
        return len(self.weight_names)


TASKS: dict[str, SearchTask] = {
    "poly_d2_o2": SearchTask(
        "poly_d2_o2", "poly", 2, ("w_u",), (1e-1,)
    ),
    "poly_d2_o4": SearchTask(
        "poly_d2_o4", "poly", 4, ("w_u", "w_Delta_u"), (1e-1, 1e1)
    ),
    "poly_d2_o6": SearchTask(
        "poly_d2_o6",
        "poly",
        6,
        ("w_u", "w_Delta_u", "w_Delta2_u"),
        (1e-2, 1.0, 1e1),
    ),
    "cahn_hilliard_o4": SearchTask(
        "cahn_hilliard_o4",
        "cahn_hilliard",
        4,
        ("lambda_ic", "mu_mean"),
        (1e1, 1e-1),
    ),
    "cahn_hilliard_o6": SearchTask(
        "cahn_hilliard_o6",
        "cahn_hilliard",
        6,
        ("lambda_ic", "mu_mean"),
        (1.0, 1e-1),
    ),
}


def _slice_jet(jet: TaylorJet, index: int) -> TaylorJet:
    return TaylorJet([term[..., index:index + 1] for term in jet.terms])


def _shift_jet(jet: TaylorJet, shift: float) -> TaylorJet:
    return TaylorJet([
        jet.terms[0] + shift,
        *jet.terms[1:],
    ])


class PeriodicEmbeddedMLP(nn.Module):
    """MLP on the hard-periodic embedding ``(cos(x), sin(x), t)``.

    ``jet_forward`` propagates a physical ``(x,t)`` Taylor jet through the
    analytic embedding before entering the sequential MLP.  Consequently the
    same wrapper supports both direct coordinate autodiff and exact
    Waring/Taylor-jet derivatives with respect to physical ``x`` and ``t``.
    """

    def __init__(self, net: nn.Sequential):
        super().__init__()
        self.net = net

    @staticmethod
    def embed(xt: Tensor) -> Tensor:
        x = xt[..., 0:1]
        t = xt[..., 1:2]
        return torch.cat([torch.cos(x), torch.sin(x), t], dim=-1)

    def forward(self, xt: Tensor) -> Tensor:
        return self.net(self.embed(xt))

    def jet_forward(self, jet: TaylorJet) -> TaylorJet:
        if jet.terms[0].shape[-1] != 2:
            raise ValueError("PeriodicEmbeddedMLP expects physical inputs (x,t)")
        x_jet = _slice_jet(jet, 0)
        t_jet = _slice_jet(jet, 1)
        sin_jet = jet_sin(x_jet)
        cos_jet = jet_sin(_shift_jet(x_jet, math.pi / 2.0))
        embedded = TaylorJet([
            torch.cat(
                [cos_jet.terms[k], sin_jet.terms[k], t_jet.terms[k]],
                dim=-1,
            )
            for k in range(jet.order + 1)
        ])
        return jet_forward_sequential(self.net, embedded)


def build_search_model(
    task: SearchTask,
    method: str,
    device: torch.device,
    *,
    hidden: int = HIDDEN,
    depth: int = DEPTH,
) -> tuple[nn.Module, torch.dtype, str]:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    is_war = method == "war"
    dtype = torch.complex128 if is_war else torch.float64
    activation = "sinh" if is_war else "tanh"
    input_dim = 2 if task.family == "poly" else 3
    net = build_plain(input_dim, hidden, depth, dtype, activation, out=1)
    if is_war:
        # Poly uses the established frequency-matched initialization.  The CH
        # input is already periodic and contains its fundamental harmonics.
        omega0 = 2.0 * math.pi if task.family == "poly" else 2.0
        complex_freq_init_(net, omega0)
    model: nn.Module = (
        net if task.family == "poly" else PeriodicEmbeddedMLP(net)
    )
    backend = "waring_complex_jet" if is_war else "direct_autodiff"
    return model.to(device), dtype, backend


def model_metadata(model: nn.Module, method: str) -> dict[str, object]:
    return {
        "method": method,
        "representation": "native_complex" if method == "war" else "real",
        "activation": "sinh" if method == "war" else "tanh",
        "derivative_backend": (
            "waring_complex_jet" if method == "war" else "direct_autodiff"
        ),
        "hidden": HIDDEN,
        "depth": DEPTH,
        "real_dof": n_params(model),
    }


def relative_l2(model: nn.Module, points: Tensor, target: Tensor, dtype: torch.dtype) -> float:
    with torch.no_grad():
        pred = predict(model, points.to(dtype)).real.squeeze(-1)
        numerator = torch.mean((pred - target) ** 2).sqrt()
        denominator = torch.mean(target ** 2).sqrt()
        return float((numerator / denominator).item())


def _imaginary_regularizer(model: nn.Module, device: torch.device) -> Tensor:
    terms = [
        parameter.imag.square().mean()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.dtype.is_complex
    ]
    if not terms:
        return torch.zeros((), dtype=torch.float64, device=device)
    return 1e-6 * sum(terms)


def _laplacian_power(
    model: nn.Module,
    x: Tensor,
    power: int,
    backend: str,
) -> Tensor:
    if power == 0:
        return predict(model, x).real.squeeze(-1)
    value: Tensor | None = None
    for coefficient, alpha in laplacian_power_terms(2, power):
        term = deriv_alpha(model, x, alpha, backend=backend).real.squeeze(-1)
        value = coefficient * term if value is None else value + coefficient * term
    if value is None:  # pragma: no cover - power=0 is handled above
        raise AssertionError("empty Laplacian expansion")
    return value


@dataclass
class LossBundle:
    loss_fn: Callable[[], tuple[Tensor, dict[str, Tensor]]]
    eval_fn: Callable[[], float]
    history_eval_fn: Callable[[], float]
    metadata: dict[str, object]


def make_poly_loss(
    task: SearchTask,
    model: nn.Module,
    dtype: torch.dtype,
    backend: str,
    weights: tuple[float, ...],
    device: torch.device,
    *,
    n_int: int,
    n_bc: int,
    n_eval: int,
    history_eval_n: int,
    train_seed: int = TRAIN_SEED,
    eval_seed: int = EVAL_SEED,
) -> LossBundle:
    if len(weights) != task.weight_count:
        raise ValueError(f"{task.task_id} expects {task.weight_count} weights")
    m = task.order // 2
    scale = 2.0 * math.pi**2

    train_gen = torch.Generator(device=device).manual_seed(train_seed)
    eval_gen = torch.Generator(device=device).manual_seed(eval_seed)
    x_int = sample_interior(n_int, 2, device=device, generator=train_gen)
    x_bc = sample_boundary(n_bc, 2, device=device, generator=train_gen)
    x_eval = sample_interior(n_eval, 2, device=device, generator=eval_gen)
    x_hist = x_eval[: min(history_eval_n, n_eval)]

    def exact(x: Tensor) -> Tensor:
        return torch.sin(math.pi * x).prod(dim=-1)

    source = (((-scale) ** m) * exact(x_int)).detach()
    bc_targets = [(((-scale) ** j) * exact(x_bc)).detach() for j in range(m)]
    eval_target = exact(x_eval).detach()
    history_target = exact(x_hist).detach()
    xi = x_int.to(dtype)
    xb = x_bc.to(dtype)

    def loss_fn() -> tuple[Tensor, dict[str, Tensor]]:
        interior = _laplacian_power(model, xi, m, backend)
        l_pde = torch.mean(((interior - source) / (scale**m)) ** 2)
        components: dict[str, Tensor] = {"L_PDE": l_pde}
        weighted_bc = torch.zeros((), dtype=torch.float64, device=device)
        for j, (weight, target) in enumerate(zip(weights, bc_targets)):
            pred = _laplacian_power(model, xb, j, backend)
            denominator = scale**j
            raw = torch.mean(((pred - target) / denominator) ** 2)
            weighted = weight * raw
            components[f"L_bc_j{j}"] = raw
            components[f"weighted_L_bc_j{j}"] = weighted
            weighted_bc = weighted_bc + weighted
        regularizer = _imaginary_regularizer(model, device)
        components["L_imag_regularizer"] = regularizer
        components["weighted_L_constraints"] = weighted_bc
        total = l_pde + weighted_bc + regularizer
        components["loss"] = total
        return total, components

    return LossBundle(
        loss_fn=loss_fn,
        eval_fn=lambda: relative_l2(model, x_eval, eval_target, dtype),
        history_eval_fn=lambda: relative_l2(model, x_hist, history_target, dtype),
        metadata={
            "domain": "[-1,1]^2",
            "exact_solution": "sin(pi*x1)*sin(pi*x2)",
            "boundary_component_order": ["u", "Delta_u", "Delta2_u"][:m],
            "pde_normalization": scale**m,
            "boundary_normalization": [scale**j for j in range(m)],
            "n_int": n_int,
            "n_boundary": n_bc,
            "n_eval": n_eval,
        },
    )


def _ch_exact(points: Tensor) -> Tensor:
    x = points[..., 0]
    t = points[..., 1]
    return torch.exp(-t) * torch.cos(2.0 * x)


def _ch_source(points: Tensor, order: int) -> Tensor:
    x = points[..., 0]
    t = points[..., 1]
    amplitude = torch.exp(-t)
    u = amplitude * torch.cos(2.0 * x)
    dxx_u3_minus_u = (
        -3.0 * amplitude**3 * torch.cos(2.0 * x)
        -9.0 * amplitude**3 * torch.cos(6.0 * x)
        +4.0 * u
    )
    gamma1 = 1e-2 if order == 4 else -1e-2
    high = 16.0 * u if order == 4 else -64.0 * u
    return -u - dxx_u3_minus_u + gamma1 * high


def make_ch_loss(
    task: SearchTask,
    model: nn.Module,
    dtype: torch.dtype,
    backend: str,
    weights: tuple[float, ...],
    device: torch.device,
    *,
    n_int: int,
    n_ic: int,
    n_mean_t: int,
    n_mean_x: int,
    n_eval: int,
    history_eval_n: int,
    train_seed: int = TRAIN_SEED,
    eval_seed: int = EVAL_SEED,
) -> LossBundle:
    if len(weights) != 2:
        raise ValueError("Cahn-Hilliard expects [lambda_ic, mu_mean]")
    lambda_ic, mu_mean = weights
    gamma1 = 1e-2 if task.order == 4 else -1e-2
    gamma2 = 1.0
    q = task.order // 2

    train_gen = torch.Generator(device=device).manual_seed(train_seed)
    eval_gen = torch.Generator(device=device).manual_seed(eval_seed)
    x_int = torch.empty(n_int, 2, device=device, dtype=torch.float64)
    x_int[:, 0].uniform_(0.0, 2.0 * math.pi, generator=train_gen)
    x_int[:, 1].uniform_(0.0, 1.0, generator=train_gen)
    x_ic = torch.empty(n_ic, 2, device=device, dtype=torch.float64)
    x_ic[:, 0].uniform_(0.0, 2.0 * math.pi, generator=train_gen)
    x_ic[:, 1] = 0.0

    mean_t = torch.empty(n_mean_t, device=device, dtype=torch.float64).uniform_(
        0.0, 1.0, generator=train_gen
    )
    mean_x = (
        torch.arange(n_mean_x, device=device, dtype=torch.float64)
        * (2.0 * math.pi / n_mean_x)
    )
    mean_x_grid, mean_t_grid = torch.meshgrid(mean_x, mean_t, indexing="ij")
    x_mean = torch.stack(
        [mean_x_grid.T.reshape(-1), mean_t_grid.T.reshape(-1)], dim=-1
    )

    x_eval = torch.empty(n_eval, 2, device=device, dtype=torch.float64)
    x_eval[:, 0].uniform_(0.0, 2.0 * math.pi, generator=eval_gen)
    x_eval[:, 1].uniform_(0.0, 1.0, generator=eval_gen)
    x_hist = x_eval[: min(history_eval_n, n_eval)]
    source = _ch_source(x_int, task.order).detach()
    target_ic = _ch_exact(x_ic).detach()
    target_eval = _ch_exact(x_eval).detach()
    target_hist = _ch_exact(x_hist).detach()
    xi = x_int.to(dtype)
    xic = x_ic.to(dtype)
    xmean = x_mean.to(dtype)

    def partial(alpha: tuple[int, ...]) -> Tensor:
        return deriv_alpha(model, xi, alpha, backend=backend).real.squeeze(-1)

    def loss_fn() -> tuple[Tensor, dict[str, Tensor]]:
        u = predict(model, xi).real.squeeze(-1)
        u_x = partial((0,))
        u_xx = partial((0, 0))
        u_t = partial((1,))
        high = partial((0,) * (2 * q))
        dxx_u3_minus_u = (3.0 * u.square() - 1.0) * u_xx + 6.0 * u * u_x.square()
        residual = u_t - gamma2 * dxx_u3_minus_u + gamma1 * high - source
        l_pde = residual.square().mean()
        pred_ic = predict(model, xic).real.squeeze(-1)
        l_ic = (pred_ic - target_ic).square().mean()
        pred_mean = predict(model, xmean).real.squeeze(-1).reshape(n_mean_t, n_mean_x)
        l_mean = pred_mean.mean(dim=1).square().mean()
        weighted_ic = lambda_ic * l_ic
        weighted_mean = mu_mean * l_mean
        regularizer = _imaginary_regularizer(model, device)
        total = l_pde + weighted_ic + weighted_mean + regularizer
        return total, {
            "L_PDE": l_pde,
            "L_IC": l_ic,
            "L_mean": l_mean,
            "weighted_L_IC": weighted_ic,
            "weighted_L_mean": weighted_mean,
            "weighted_L_constraints": weighted_ic + weighted_mean,
            "L_imag_regularizer": regularizer,
            "loss": total,
        }

    return LossBundle(
        loss_fn=loss_fn,
        eval_fn=lambda: relative_l2(model, x_eval, target_eval, dtype),
        history_eval_fn=lambda: relative_l2(model, x_hist, target_hist, dtype),
        metadata={
            "domain": "x in [0,2*pi), t in [0,1]",
            "exact_solution": "exp(-t)*cos(2*x)",
            "periodic_embedding": ["cos(x)", "sin(x)", "t"],
            "gamma1": gamma1,
            "gamma2": gamma2,
            "q": q,
            "lambda_bulk": 1.0,
            "pde_normalization": 1.0,
            "n_int": n_int,
            "n_ic": n_ic,
            "n_mean_t": n_mean_t,
            "n_mean_x": n_mean_x,
            "n_eval": n_eval,
        },
    )


def make_loss_bundle(
    task: SearchTask,
    model: nn.Module,
    dtype: torch.dtype,
    backend: str,
    weights: tuple[float, ...],
    device: torch.device,
    *,
    smoke: bool,
    train_seed: int = TRAIN_SEED,
    eval_seed: int = EVAL_SEED,
) -> LossBundle:
    if smoke:
        n_int, n_constraint, n_eval, history_n = 128, 64, 512, 256
    else:
        n_int, n_constraint, n_eval, history_n = 4096, 512, 8192, 4096
    if task.family == "poly":
        return make_poly_loss(
            task,
            model,
            dtype,
            backend,
            weights,
            device,
            n_int=n_int,
            n_bc=n_constraint,
            n_eval=n_eval,
            history_eval_n=history_n,
            train_seed=train_seed,
            eval_seed=eval_seed,
        )
    return make_ch_loss(
        task,
        model,
        dtype,
        backend,
        weights,
        device,
        n_int=n_int,
        n_ic=n_constraint,
        n_mean_t=8 if smoke else 32,
        n_mean_x=16 if smoke else 32,
        n_eval=n_eval,
        history_eval_n=history_n,
        train_seed=train_seed,
        eval_seed=eval_seed,
    )


def tensor_components_to_float(components: dict[str, Tensor]) -> dict[str, float]:
    return {key: float(value.detach().item()) for key, value in components.items()}


__all__ = [
    "DEPTH",
    "EVAL_SEED",
    "GRID_VALUES",
    "HIDDEN",
    "HISTORY_INTERVAL_SECONDS",
    "LEARNING_RATE",
    "LEARNING_RATE_FINAL",
    "METHODS",
    "PROTOCOL_ID",
    "TASKS",
    "TRAIN_SEED",
    "PeriodicEmbeddedMLP",
    "SearchTask",
    "_ch_exact",
    "_ch_source",
    "build_search_model",
    "make_loss_bundle",
    "model_metadata",
    "tensor_components_to_float",
]
