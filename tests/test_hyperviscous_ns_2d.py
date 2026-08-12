from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
SRC = ROOT / "src"
for path in (str(COMMON), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

PROBLEM_PATH = ROOT / "experiments" / "hyperviscous_ns_2d" / "problem.py"
SPEC = importlib.util.spec_from_file_location(
    "apolarity_hyperns_problem_test", PROBLEM_PATH
)
assert SPEC is not None and SPEC.loader is not None
hyperns = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hyperns
SPEC.loader.exec_module(hyperns)


class ExactTaylorGreen(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return hyperns.exact_solution(points)


def _scalar_partial(
    value: torch.Tensor,
    points: torch.Tensor,
    alpha: tuple[int, ...],
) -> torch.Tensor:
    derivative = value
    for coordinate in alpha:
        derivative = torch.autograd.grad(
            derivative.sum(), points, create_graph=True, retain_graph=True
        )[0][:, coordinate]
    return derivative


def test_protocol_is_one_complete_49_point_shared_grid():
    task = hyperns.TASKS["hyperviscous_ns_2d_o4"]
    candidates = tuple(
        itertools.product(hyperns.GRID_VALUES, repeat=task.weight_count)
    )
    assert task.order == 4
    assert task.spatial_dim == 2
    assert len(candidates) == 49
    assert len(candidates) * len(hyperns.METHODS) == 98
    assert hyperns.METHODS == ("war", "real_tanh_autodiff")
    assert hyperns.OUTPUT_DIM == 3


def test_taylor_green_exact_solution_satisfies_unforced_equations():
    points = torch.tensor(
        [
            [0.31, 0.47, 0.13],
            [1.27, 2.11, 0.62],
            [5.73, 0.91, 0.88],
        ],
        dtype=torch.float64,
    )
    residual_u, residual_v, differential = hyperns.momentum_and_divergence(
        ExactTaylorGreen(), points, "direct_autodiff"
    )
    torch.testing.assert_close(
        residual_u, torch.zeros_like(residual_u), rtol=0.0, atol=2e-10
    )
    torch.testing.assert_close(
        residual_v, torch.zeros_like(residual_v), rtol=0.0, atol=2e-10
    )
    torch.testing.assert_close(
        differential["divergence"],
        torch.zeros_like(differential["divergence"]),
        rtol=0.0,
        atol=2e-10,
    )


def test_taylor_green_decay_laplacian_and_pressure_gauge():
    points = torch.tensor(
        [[0.37, 0.83, 0.2], [2.1, 4.7, 0.9]],
        dtype=torch.float64,
        requires_grad=True,
    )
    state = hyperns.exact_solution(points)
    for component in (0, 1):
        laplacian = (
            _scalar_partial(state[:, component], points, (0, 0))
            + _scalar_partial(state[:, component], points, (1, 1))
        )
        biharmonic = (
            _scalar_partial(state[:, component], points, (0, 0, 0, 0))
            + 2.0 * _scalar_partial(
                state[:, component], points, (0, 0, 1, 1)
            )
            + _scalar_partial(state[:, component], points, (1, 1, 1, 1))
        )
        time_derivative = _scalar_partial(
            state[:, component], points, (2,)
        )
        torch.testing.assert_close(laplacian, -2.0 * state[:, component])
        torch.testing.assert_close(biharmonic, 4.0 * state[:, component])
        torch.testing.assert_close(
            time_derivative,
            -hyperns.DECAY_RATE * state[:, component],
        )

    side = 64
    axis = torch.arange(side, dtype=torch.float64) * (2.0 * math.pi / side)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack(
        [x.reshape(-1), y.reshape(-1), torch.full_like(x.reshape(-1), 0.4)],
        dim=-1,
    )
    pressure_mean = hyperns.exact_solution(grid)[:, 2].mean()
    torch.testing.assert_close(
        pressure_mean, torch.zeros_like(pressure_mean), rtol=0.0, atol=2e-16
    )


def test_exact_periodic_traces_match_for_velocity_orders_zero_to_three():
    tangential = torch.tensor([0.37, 1.42, 5.66], dtype=torch.float64)
    times = torch.tensor([0.11, 0.53, 0.91], dtype=torch.float64)
    for coordinate in (0, 1):
        lower = torch.empty(3, 3, dtype=torch.float64)
        upper = torch.empty_like(lower)
        lower[:, coordinate] = 0.0
        upper[:, coordinate] = 2.0 * math.pi
        lower[:, 1 - coordinate] = tangential
        upper[:, 1 - coordinate] = tangential
        lower[:, 2] = times
        upper[:, 2] = times
        lower.requires_grad_(True)
        upper.requires_grad_(True)
        lower_state = hyperns.exact_solution(lower)
        upper_state = hyperns.exact_solution(upper)
        for component in (0, 1):
            for order in range(4):
                alpha = (coordinate,) * order
                lower_trace = (
                    lower_state[:, component]
                    if order == 0
                    else _scalar_partial(
                        lower_state[:, component], lower, alpha
                    )
                )
                upper_trace = (
                    upper_state[:, component]
                    if order == 0
                    else _scalar_partial(
                        upper_state[:, component], upper, alpha
                    )
                )
                torch.testing.assert_close(
                    lower_trace, upper_trace, rtol=0.0, atol=2e-13
                )
        torch.testing.assert_close(
            lower_state[:, 2], upper_state[:, 2], rtol=0.0, atol=2e-13
        )


def test_common_architecture_is_vector_output_raw_affine_and_precision_matched():
    task = hyperns.TASKS["hyperviscous_ns_2d_o4"]
    war, war_dtype, war_backend = hyperns.build_model(
        task, "war", torch.device("cpu"), hidden=8, depth=2
    )
    ad, ad_dtype, ad_backend = hyperns.build_model(
        task, "real_tanh_autodiff", torch.device("cpu"), hidden=8, depth=2
    )
    assert war_dtype == torch.complex64
    assert ad_dtype == torch.float32
    assert war_backend == "waring_complex_jet"
    assert ad_backend == "direct_autodiff"
    assert war.net[0].in_features == ad.net[0].in_features == 3
    assert war.net[-1].out_features == ad.net[-1].out_features == 3
    war_meta = hyperns.model_metadata(war, "war")
    ad_meta = hyperns.model_metadata(ad, "real_tanh_autodiff")
    assert war_meta["activation"] == "sinh"
    assert ad_meta["activation"] == "tanh"
    assert ad_meta["input_transform"] == "affine_only"
    assert ad_meta["trigonometric_input_features"] is False
    assert ad_meta["frequency_initialization"] == "disabled"
    assert war_meta["parameter_elements"] == ad_meta["parameter_elements"]


def test_tiny_loss_and_parameter_gradients_are_finite_for_both_methods():
    task = hyperns.TASKS["hyperviscous_ns_2d_o4"]
    for method in hyperns.METHODS:
        torch.manual_seed(23)
        model, dtype, backend = hyperns.build_model(
            task, method, torch.device("cpu"), hidden=5, depth=1
        )
        bundle = hyperns.make_loss_bundle(
            task,
            model,
            dtype,
            backend,
            (1.0, 1.0),
            torch.device("cpu"),
            smoke=True,
            n_int=2,
            n_ic=2,
            n_bc=4,
            n_eval=8,
            history_eval_n=4,
        )
        loss, components = bundle.loss_fn()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in components.values())
        assert {
            "L_momentum_u",
            "L_momentum_v",
            "L_div",
            "L_PDE",
            "L_IC",
            "L_BC",
            "L_gauge",
            "loss",
        }.issubset(components)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        assert gradients and all(value is not None for value in gradients)
        assert all(
            torch.isfinite(value).all()
            for value in gradients
            if value is not None
        )

