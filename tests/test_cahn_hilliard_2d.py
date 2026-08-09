from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
CAHN2D = ROOT / "experiments" / "cahn_hilliard_2d"
for path in (str(COMMON), str(CAHN2D)):
    if path not in sys.path:
        sys.path.insert(0, path)

from osc_common import deriv_alpha, laplacian_power_terms
from problem import (
    COMPLEX_DTYPE,
    GRID_VALUES,
    METHODS,
    REAL_DTYPE,
    TASKS,
    build_model,
    exact_solution,
    make_loss_bundle,
    manufactured_source,
    model_metadata,
)


def _laplacian(value: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(value)
    for coordinate in (0, 1):
        first = torch.autograd.grad(
            value.sum(), points, create_graph=True, retain_graph=True
        )[0][:, coordinate]
        second = torch.autograd.grad(
            first.sum(), points, create_graph=True, retain_graph=True
        )[0][:, coordinate]
        output = output + second
    return output


def _laplacian_power_exact(
    value: torch.Tensor, points: torch.Tensor, power: int
) -> torch.Tensor:
    result = value
    for _ in range(power):
        result = _laplacian(result, points)
    return result


def _normal_laplacian_exact(
    points: torch.Tensor, normal_coordinate: int, power: int
) -> torch.Tensor:
    xyt = points.detach().clone().requires_grad_(True)
    value = _laplacian_power_exact(exact_solution(xyt), xyt, power)
    return torch.autograd.grad(
        value.sum(), xyt, create_graph=True, retain_graph=True
    )[0][:, normal_coordinate]


def test_task_grid_has_98_vectors_and_196_method_runs():
    counts = {
        task_id: len(tuple(itertools.product(GRID_VALUES, repeat=task.weight_count)))
        for task_id, task in TASKS.items()
    }
    assert counts == {
        "cahn_hilliard_2d_o4": 49,
        "cahn_hilliard_2d_o6": 49,
    }
    assert sum(counts.values()) == 98
    assert sum(counts.values()) * len(METHODS) == 196


def test_well_posed_signs_and_natural_boundary_orders():
    assert TASKS["cahn_hilliard_2d_o4"].eta == 1e-2
    assert TASKS["cahn_hilliard_2d_o6"].eta == -1e-2
    assert [2 * ell + 1 for ell in range(TASKS["cahn_hilliard_2d_o4"].q)] == [1, 3]
    assert [2 * ell + 1 for ell in range(TASKS["cahn_hilliard_2d_o6"].q)] == [1, 3, 5]
    assert laplacian_power_terms(2, 2) == [
        (1.0, (1, 1, 1, 1)),
        (2.0, (0, 0, 1, 1)),
        (1.0, (0, 0, 0, 0)),
    ]
    assert laplacian_power_terms(2, 3) == [
        (1.0, (1, 1, 1, 1, 1, 1)),
        (3.0, (0, 0, 1, 1, 1, 1)),
        (3.0, (0, 0, 0, 0, 1, 1)),
        (1.0, (0, 0, 0, 0, 0, 0)),
    ]


def test_manufactured_source_matches_direct_float32_differentiation():
    points = torch.tensor(
        [
            [0.31, 0.47, 0.13],
            [1.27, 2.11, 0.62],
            [2.73, 0.91, 0.88],
        ],
        dtype=REAL_DTYPE,
        requires_grad=True,
    )
    for task in TASKS.values():
        xyt = points.detach().clone().requires_grad_(True)
        u = exact_solution(xyt)
        u_t = torch.autograd.grad(
            u.sum(), xyt, create_graph=True, retain_graph=True
        )[0][:, 2]
        lap_nonlinear = _laplacian(u**3 - u, xyt)
        high = _laplacian_power_exact(u, xyt, task.q)
        residual = u_t - lap_nonlinear + task.eta * high
        torch.testing.assert_close(
            residual,
            manufactured_source(xyt.detach(), task),
            rtol=5e-4,
            atol=5e-4,
        )


def test_manufactured_solution_satisfies_natural_boundaries_in_float32():
    tangential = torch.tensor([0.37, 1.42, 2.66], dtype=REAL_DTYPE)
    times = torch.tensor([0.11, 0.53, 0.91], dtype=REAL_DTYPE)
    for normal_coordinate in (0, 1):
        for boundary_value in (0.0, math.pi):
            points = torch.empty(3, 3, dtype=REAL_DTYPE)
            points[:, normal_coordinate] = boundary_value
            points[:, 1 - normal_coordinate] = tangential
            points[:, 2] = times
            for ell in range(3):
                residual = _normal_laplacian_exact(points, normal_coordinate, ell)
                torch.testing.assert_close(
                    residual,
                    torch.zeros_like(residual),
                    rtol=0.0,
                    atol=2e-4,
                )


def test_affine_input_has_no_trigonometric_features_and_methods_match_shape():
    task = TASKS["cahn_hilliard_2d_o4"]
    war, war_dtype, _ = build_model(
        task, "war", torch.device("cpu"), hidden=8, depth=2
    )
    real, real_dtype, _ = build_model(
        task, "real_sinh_autodiff", torch.device("cpu"), hidden=8, depth=2
    )
    assert war_dtype == COMPLEX_DTYPE
    assert real_dtype == REAL_DTYPE
    assert war.net[0].in_features == real.net[0].in_features == 3
    war_metadata = model_metadata(war, "war")
    real_metadata = model_metadata(real, "real_sinh_autodiff")
    assert war_metadata["input_transform"] == "affine_only"
    assert real_metadata["activation"] == "sinh"
    assert war_metadata["parameter_elements"] == real_metadata["parameter_elements"]
    assert war_metadata["real_dof"] == 2 * real_metadata["real_dof"]
    points = torch.tensor([[0.2, 1.1, 0.4], [2.0, 0.5, 0.9]], dtype=REAL_DTYPE)
    assert war(points.to(COMPLEX_DTYPE)).shape == (2, 1)
    assert real(points).shape == (2, 1)


def test_mixed_waring_derivatives_match_direct_autodiff_in_complex64():
    torch.manual_seed(19)
    task = TASKS["cahn_hilliard_2d_o6"]
    model, _dtype, _backend = build_model(
        task, "real_sinh_autodiff", torch.device("cpu"), hidden=6, depth=2
    )
    points = torch.tensor(
        [[0.37, 0.71, 0.2], [1.23, 2.07, 0.8]], dtype=REAL_DTYPE
    )
    for alpha in (
        (0,),
        (2,),
        (0, 0, 1, 1),
        (0, 0, 0, 0, 1, 1),
    ):
        war = deriv_alpha(model, points, alpha, backend="waring_complex_jet").real
        direct = deriv_alpha(model, points, alpha, backend="direct_autodiff").real
        torch.testing.assert_close(war, direct, rtol=8e-3, atol=2e-3)


def test_smoke_loss_backward_is_finite_for_both_precisions():
    task = TASKS["cahn_hilliard_2d_o6"]
    for method in METHODS:
        torch.manual_seed(23)
        model, dtype, backend = build_model(
            task, method, torch.device("cpu"), hidden=6, depth=1
        )
        bundle = make_loss_bundle(
            task,
            model,
            dtype,
            backend,
            (1.0, 1.0),
            torch.device("cpu"),
            smoke=True,
            n_int=3,
            n_ic=3,
            n_bc=4,
            n_eval=8,
            history_eval_n=4,
        )
        loss, components = bundle.loss_fn()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in components.values())
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
