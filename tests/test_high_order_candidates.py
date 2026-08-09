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

MODULE_PATH = ROOT / "experiments" / "high_order_candidates" / "problem.py"
SPEC = importlib.util.spec_from_file_location(
    "test_high_order_candidate_problem", MODULE_PATH
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


def _laplacian(
    value: torch.Tensor,
    points: torch.Tensor,
    spatial_dim: int,
) -> torch.Tensor:
    return sum(
        _partial(value, points, (coordinate, coordinate))
        for coordinate in range(spatial_dim)
    )


def test_frozen_candidate_table_spans_orders_and_dimensions():
    assert problem.TASK_ORDER == (
        "zk_2d_o3",
        "zk_3d_o3",
        "dynamic_plate_2d_o4",
        "swift_hohenberg_2d_o4",
    )
    assert {task.order for task in problem.TASKS.values()} == {3, 4}
    assert {task.spatial_dim for task in problem.TASKS.values()} == {2, 3}
    assert problem.METHODS == ("war", "real_tanh_autodiff")
    assert all(task.order >= 3 for task in problem.TASKS.values())


def test_manufactured_sources_match_direct_differentiation():
    samples = {
        "zk_2d_o3": [[0.37, 1.11, 0.23], [4.22, 2.04, 0.71]],
        "zk_3d_o3": [
            [0.37, 1.11, 2.03, 0.23],
            [4.22, 2.04, 5.11, 0.71],
        ],
        "dynamic_plate_2d_o4": [
            [-0.63, 0.21, 0.23],
            [0.47, -0.54, 0.71],
        ],
        "swift_hohenberg_2d_o4": [[0.37, 1.11], [4.22, 2.04]],
    }
    for task_id, raw_points in samples.items():
        task = problem.TASKS[task_id]
        points = torch.tensor(raw_points, dtype=torch.float64, requires_grad=True)
        u = problem.exact_solution(points, task)
        if task.family == "zakharov_kuznetsov":
            time_coordinate = task.spatial_dim
            residual = (
                _partial(u, points, (time_coordinate,))
                + u * _partial(u, points, (0,))
                + _partial(u, points, (0, 0, 0))
            )
            for coordinate in range(1, task.spatial_dim):
                residual = residual + _partial(
                    u, points, (0, coordinate, coordinate)
                )
        elif task.family == "dynamic_kirchhoff_love_plate":
            lap = _laplacian(u, points, 2)
            residual = (
                _partial(u, points, (2, 2))
                + problem.PLATE_DAMPING * _partial(u, points, (2,))
                + _laplacian(lap, points, 2)
            )
        else:
            lap = _laplacian(u, points, 2)
            residual = _laplacian(lap, points, 2) + 2.0 * lap + 2.0 * u + u**3
        torch.testing.assert_close(
            residual,
            problem.manufactured_source(points.detach(), task),
            rtol=2e-9,
            atol=2e-9,
        )


def test_exact_boundary_and_initial_data_are_consistent():
    plate = problem.TASKS["dynamic_plate_2d_o4"]
    tangential = torch.tensor([-0.72, 0.13, 0.81], dtype=torch.float64)
    times = torch.tensor([0.11, 0.43, 0.88], dtype=torch.float64)
    for coordinate in (0, 1):
        for boundary in (-1.0, 1.0):
            points = torch.empty(3, 3, dtype=torch.float64, requires_grad=True)
            with torch.no_grad():
                points[:, coordinate] = boundary
                points[:, 1 - coordinate] = tangential
                points[:, 2] = times
            value = problem.exact_solution(points, plate)
            normal = _partial(value, points, (coordinate,))
            torch.testing.assert_close(value, torch.zeros_like(value), atol=1e-12, rtol=0)
            torch.testing.assert_close(normal, torch.zeros_like(normal), atol=1e-12, rtol=0)

    initial = torch.tensor(
        [[-0.63, 0.21, 0.0], [0.47, -0.54, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    velocity = _partial(problem.exact_solution(initial, plate), initial, (2,))
    torch.testing.assert_close(velocity, torch.zeros_like(velocity), atol=1e-12, rtol=0)


def test_periodic_exact_traces_match_required_derivatives():
    for task_id in ("zk_2d_o3", "zk_3d_o3", "swift_hohenberg_2d_o4"):
        task = problem.TASKS[task_id]
        for coordinate in range(task.spatial_dim):
            midpoint = [0.37 * (hi - lo) + lo for lo, hi in zip(task.lows, task.highs)]
            lower = torch.tensor([midpoint], dtype=torch.float64, requires_grad=True)
            upper = torch.tensor([midpoint], dtype=torch.float64, requires_grad=True)
            with torch.no_grad():
                lower[:, coordinate] = task.lows[coordinate]
                upper[:, coordinate] = task.highs[coordinate]
            lower_u = problem.exact_solution(lower, task)
            upper_u = problem.exact_solution(upper, task)
            orders = (
                (0, 1, 2)
                if task.family == "zakharov_kuznetsov" and coordinate == 0
                else (0, 1)
            )
            for order in orders:
                lower_value = (
                    lower_u if order == 0 else _partial(lower_u, lower, (coordinate,) * order)
                )
                upper_value = (
                    upper_u if order == 0 else _partial(upper_u, upper, (coordinate,) * order)
                )
                torch.testing.assert_close(
                    lower_value, upper_value, rtol=1e-10, atol=1e-10
                )


def test_architectures_keep_real_tanh_baseline_and_raw_inputs():
    for task in problem.TASKS.values():
        war, war_dtype, _ = problem.build_model(
            task, "war", torch.device("cpu"), hidden=8, depth=2
        )
        real, real_dtype, _ = problem.build_model(
            task, "real_tanh_autodiff", torch.device("cpu"), hidden=8, depth=2
        )
        assert war_dtype == problem.COMPLEX_DTYPE
        assert real_dtype == problem.REAL_DTYPE
        assert war.net[0].in_features == real.net[0].in_features == task.input_dim
        war_meta = problem.model_metadata(war, "war")
        real_meta = problem.model_metadata(real, "real_tanh_autodiff")
        assert war_meta["activation"] == "sinh"
        assert real_meta["activation"] == "tanh"
        assert war_meta["input_transform"] == real_meta["input_transform"] == "affine_only"
        assert war_meta["parameter_elements"] == real_meta["parameter_elements"]
        assert war_meta["real_dof"] == 2 * real_meta["real_dof"]


def test_affine_wrapper_waring_jet_matches_direct_autodiff():
    torch.manual_seed(17)
    task = problem.TASKS["zk_3d_o3"]
    model, dtype, _ = problem.build_model(
        task, "war", torch.device("cpu"), hidden=6, depth=2
    )
    points = torch.tensor(
        [[0.37, 1.11, 2.03, 0.23], [4.22, 2.04, 5.11, 0.71]],
        dtype=dtype,
    )
    for alpha in ((0, 1, 1), (0, 2, 2), (0, 0, 0)):
        war = deriv_alpha(model, points, alpha, backend="waring_complex_jet")
        direct = deriv_alpha(model, points, alpha, backend="direct_autodiff")
        torch.testing.assert_close(war, direct, rtol=8e-3, atol=2e-3)


def test_tiny_losses_and_gradients_are_finite_for_every_cell():
    for task in problem.TASKS.values():
        for method in problem.METHODS:
            torch.manual_seed(23)
            model, dtype, backend = problem.build_model(
                task, method, torch.device("cpu"), hidden=4, depth=1
            )
            bundle = problem.make_loss_bundle(
                task,
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


def test_uniqueness_metadata_is_present_and_nonempty():
    for task in problem.TASKS.values():
        assert len(task.uniqueness) > 40
        assert math.isfinite(task.residual_scale) and task.residual_scale > 0
