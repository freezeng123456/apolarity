from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
SRC = ROOT / "src"
for path in (str(COMMON), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

MODULE_PATH = ROOT / "experiments" / "sixth_order_plate" / "problem.py"
SPEC = importlib.util.spec_from_file_location(
    "test_sixth_order_plate_problem", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
problem = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = problem
SPEC.loader.exec_module(problem)

from osc_common import deriv_alpha  # noqa: E402


def _partial(
    value: torch.Tensor,
    points: torch.Tensor,
    alpha: tuple[int, ...],
) -> torch.Tensor:
    result = value
    for coordinate in alpha:
        result = torch.autograd.grad(
            result.sum(), points, create_graph=True, retain_graph=True
        )[0][:, coordinate]
    return result


def _laplacian(value: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return sum(
        _partial(value, points, (coordinate, coordinate))
        for coordinate in (0, 1)
    )


def test_task_is_raw_input_sixth_order_problem() -> None:
    task = problem.TASK
    assert task.task_id == "strain_gradient_plate_2d_o6"
    assert task.spatial_dim == 2
    assert task.order == 6
    assert task.coordinate_names == ("x", "y", "t")
    assert task.weights == (10.0, 10.0)
    for method in problem.METHODS:
        model, dtype, _ = problem.build_model(
            task, method, torch.device("cpu"), hidden=8, depth=2
        )
        assert model.net[0].in_features == 3
        metadata = problem.model_metadata(model, method)
        assert metadata["input_transform"] == "affine_only"
        assert metadata["frequency_initialization"] == "disabled"
        assert dtype in (torch.float32, torch.complex64)


def test_manufactured_source_matches_direct_sixth_differentiation() -> None:
    points = torch.tensor(
        [[-0.63, 0.21, 0.23], [0.47, -0.54, 0.71]],
        dtype=torch.float64,
        requires_grad=True,
    )
    u = problem.exact_solution(points)
    lap = _laplacian(u, points)
    biharmonic = _laplacian(lap, points)
    triharmonic = _laplacian(biharmonic, points)
    residual = (
        _partial(u, points, (2, 2))
        + problem.PLATE_DAMPING * _partial(u, points, (2,))
        + biharmonic
        - problem.INTERNAL_LENGTH**2 * triharmonic
    )
    torch.testing.assert_close(
        residual,
        problem.manufactured_source(points.detach()),
        rtol=2e-9,
        atol=2e-9,
    )


def test_exact_solution_satisfies_three_homogeneous_spatial_traces() -> None:
    tangential = torch.tensor([-0.72, 0.13, 0.81], dtype=torch.float64)
    times = torch.tensor([0.11, 0.43, 0.88], dtype=torch.float64)
    for coordinate in (0, 1):
        for boundary in (-1.0, 1.0):
            points = torch.empty(3, 3, dtype=torch.float64, requires_grad=True)
            with torch.no_grad():
                points[:, coordinate] = boundary
                points[:, 1 - coordinate] = tangential
                points[:, 2] = times
            value = problem.exact_solution(points)
            first = _partial(value, points, (coordinate,))
            second = _partial(value, points, (coordinate, coordinate))
            for trace in (value, first, second):
                torch.testing.assert_close(
                    trace, torch.zeros_like(trace), atol=2e-11, rtol=0
                )

    initial = torch.tensor(
        [[-0.63, 0.21, 0.0], [0.47, -0.54, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    velocity = _partial(problem.exact_solution(initial), initial, (2,))
    torch.testing.assert_close(
        velocity, torch.zeros_like(velocity), atol=2e-11, rtol=0
    )


def test_complex_waring_sixth_derivatives_match_direct_autodiff() -> None:
    torch.manual_seed(17)
    model, dtype, _ = problem.build_model(
        problem.TASK,
        "war",
        torch.device("cpu"),
        hidden=6,
        depth=2,
    )
    points = torch.tensor(
        [[-0.37, 0.11, 0.23], [0.22, -0.41, 0.71]], dtype=dtype
    )
    for alpha in (
        (0, 0, 0, 0, 0, 0),
        (1, 1, 1, 1, 1, 1),
        (0, 0, 0, 0, 1, 1),
        (0, 0, 1, 1, 1, 1),
    ):
        war = deriv_alpha(
            model, points, alpha, backend="waring_complex_jet"
        )
        direct = deriv_alpha(model, points, alpha, backend="direct_autodiff")
        torch.testing.assert_close(war, direct, rtol=7e-2, atol=2e-2)


def test_tiny_losses_and_gradients_are_finite() -> None:
    for method in problem.METHODS:
        torch.manual_seed(23)
        model, dtype, backend = problem.build_model(
            problem.TASK,
            method,
            torch.device("cpu"),
            hidden=4,
            depth=1,
        )
        bundle = problem.make_loss_bundle(
            problem.TASK,
            model,
            dtype,
            backend,
            torch.device("cpu"),
            n_int=2,
            n_ic=2,
            n_bc=8,
            n_eval=4,
            history_eval_n=2,
            train_seed=23,
            eval_seed=29,
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
        assert gradients and all(value is not None for value in gradients)
        assert all(
            torch.isfinite(value).all()
            for value in gradients
            if value is not None
        )


def test_residual_scale_is_finite_and_uniqueness_is_documented() -> None:
    assert math.isfinite(problem.TASK.residual_scale)
    assert problem.TASK.residual_scale > 0
    assert "uniqueness" in problem.TASK.uniqueness
