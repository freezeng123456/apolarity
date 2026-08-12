from __future__ import annotations

import importlib.util
import itertools
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

PROBLEM_PATH = ROOT / "experiments" / "mbe_2d" / "problem.py"
SPEC = importlib.util.spec_from_file_location("apolarity_mbe_problem_test", PROBLEM_PATH)
assert SPEC is not None and SPEC.loader is not None
mbe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mbe
SPEC.loader.exec_module(mbe)


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


def test_protocol_is_one_complete_49_point_shared_grid():
    task = mbe.TASKS["mbe_2d_o4"]
    candidates = tuple(
        itertools.product(mbe.GRID_VALUES, repeat=task.weight_count)
    )
    assert task.order == 4
    assert task.family == "mbe_2d_slope_selection"
    assert len(candidates) == 49
    assert len(candidates) * len(mbe.METHODS) == 98
    assert mbe.METHODS == ("war", "real_tanh_autodiff")
    assert mbe.NU == 0.05


def test_manufactured_source_matches_independent_direct_differentiation():
    points = torch.tensor(
        [
            [0.31, 0.47, 0.13],
            [1.27, 2.11, 0.62],
            [5.73, 0.91, 0.88],
        ],
        dtype=mbe.REAL_DTYPE,
        requires_grad=True,
    )
    h = mbe.manufactured_components(points)["h"]
    h_t = _partial(h, points, (2,))
    h_x = _partial(h, points, (0,))
    h_y = _partial(h, points, (1,))
    h_xx = _partial(h, points, (0, 0))
    h_yy = _partial(h, points, (1, 1))
    h_xy = _partial(h, points, (0, 1))
    biharmonic = (
        _partial(h, points, (0, 0, 0, 0))
        + 2.0 * _partial(h, points, (0, 0, 1, 1))
        + _partial(h, points, (1, 1, 1, 1))
    )
    divergence = mbe.slope_divergence_from_components(
        h_x, h_y, h_xx, h_yy, h_xy
    )
    residual = h_t - divergence + mbe.NU * biharmonic
    torch.testing.assert_close(
        residual,
        mbe.manufactured_source(points.detach()),
        rtol=8e-4,
        atol=3e-4,
    )


def test_manufactured_solution_has_periodic_traces_through_order_three():
    tangential = torch.tensor([0.37, 1.42, 5.66], dtype=mbe.REAL_DTYPE)
    times = torch.tensor([0.11, 0.53, 0.91], dtype=mbe.REAL_DTYPE)
    for coordinate in (0, 1):
        lower = torch.empty(3, 3, dtype=mbe.REAL_DTYPE)
        upper = torch.empty_like(lower)
        lower[:, coordinate] = 0.0
        upper[:, coordinate] = 2.0 * math.pi
        lower[:, 1 - coordinate] = tangential
        upper[:, 1 - coordinate] = tangential
        lower[:, 2] = times
        upper[:, 2] = times
        lower.requires_grad_(True)
        upper.requires_grad_(True)
        lower_value = mbe.manufactured_components(lower)["h"]
        upper_value = mbe.manufactured_components(upper)["h"]
        for order in range(4):
            alpha = (coordinate,) * order
            lower_trace = _partial(lower_value, lower, alpha)
            upper_trace = _partial(upper_value, upper, alpha)
            torch.testing.assert_close(
                lower_trace, upper_trace, rtol=0.0, atol=3e-5
            )


def test_common_architecture_is_raw_affine_and_precision_matched():
    task = mbe.TASKS["mbe_2d_o4"]
    war, war_dtype, war_backend = mbe.build_model(
        task, "war", torch.device("cpu"), hidden=8, depth=2
    )
    ad, ad_dtype, ad_backend = mbe.build_model(
        task, "real_tanh_autodiff", torch.device("cpu"), hidden=8, depth=2
    )
    assert war_dtype == torch.complex64
    assert ad_dtype == torch.float32
    assert war_backend == "waring_complex_jet"
    assert ad_backend == "direct_autodiff"
    assert war.net[0].in_features == ad.net[0].in_features == 3
    assert mbe.model_metadata(war, "war")["activation"] == "sinh"
    ad_metadata = mbe.model_metadata(ad, "real_tanh_autodiff")
    assert ad_metadata["activation"] == "tanh"
    assert ad_metadata["input_transform"] == "affine_only"
    assert ad_metadata["trigonometric_input_features"] is False
    assert ad_metadata["frequency_initialization"] == "disabled"
    assert (
        mbe.model_metadata(war, "war")["parameter_elements"]
        == ad_metadata["parameter_elements"]
    )


def test_face_pair_sampler_holds_tangential_coordinates_fixed():
    generator = torch.Generator().manual_seed(7)
    for coordinate in (0, 1):
        lower, upper = mbe.sample_face_pairs(
            16,
            coordinate,
            device=torch.device("cpu"),
            generator=generator,
        )
        assert torch.all(lower[:, coordinate] == 0.0)
        assert torch.all(upper[:, coordinate] == 2.0 * math.pi)
        keep = [value for value in range(3) if value != coordinate]
        torch.testing.assert_close(lower[:, keep], upper[:, keep])


def test_tiny_smoke_loss_and_parameter_gradients_are_finite():
    task = mbe.TASKS["mbe_2d_o4"]
    for method in mbe.METHODS:
        torch.manual_seed(23)
        model, dtype, backend = mbe.build_model(
            task, method, torch.device("cpu"), hidden=5, depth=1
        )
        bundle = mbe.make_loss_bundle(
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
            "L_PDE",
            "L_IC",
            "L_BC",
            "L_BC_x_order3",
            "L_BC_y_order3",
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


def test_reference_builder_declares_nested_three_level_gate():
    path = ROOT / "scripts" / "build_mbe_spectral_reference.py"
    spec = importlib.util.spec_from_file_location("mbe_reference_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    levels = module.parse_levels(module.DEFAULT_LEVELS)
    assert [(level.side, level.dt) for level in levels] == [
        (32, 0.002),
        (64, 0.001),
        (128, 0.0005),
    ]
    assert all(
        math.isclose(level.steps * level.dt, 1.0) for level in levels
    )
