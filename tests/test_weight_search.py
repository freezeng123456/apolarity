from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"

sys.path.insert(0, str(COMMON))

from osc_common import deriv_alpha
from weight_search import (
    GRID_VALUES,
    METHODS,
    TASKS,
    _ch_exact,
    _ch_source,
    build_search_model,
)


def test_full_grid_has_497_vectors_and_994_method_runs():
    counts = {
        task_id: len(tuple(itertools.product(GRID_VALUES, repeat=task.weight_count)))
        for task_id, task in TASKS.items()
    }
    assert counts == {
        "poly_d2_o2": 7,
        "poly_d2_o4": 49,
        "poly_d2_o6": 343,
        "cahn_hilliard_o4": 49,
        "cahn_hilliard_o6": 49,
    }
    assert sum(counts.values()) == 497
    assert sum(counts.values()) * len(METHODS) == 994


def test_periodic_embedding_waring_matches_direct_physical_derivatives():
    torch.manual_seed(7)
    task = TASKS["cahn_hilliard_o4"]
    model, _dtype, _backend = build_search_model(
        task, "war", torch.device("cpu"), hidden=6, depth=2
    )
    points = torch.tensor(
        [[0.2, 0.1], [1.1, 0.7]], dtype=torch.complex128
    )
    for alpha in ((0,), (1,), (0, 0), (0, 0, 0, 0)):
        war = deriv_alpha(model, points, alpha, backend="waring_complex_jet")
        direct = deriv_alpha(model, points, alpha, backend="direct_autodiff")
        torch.testing.assert_close(war, direct, rtol=2e-8, atol=2e-9)


def _exact_partial(points: torch.Tensor, coordinate: int, repeats: int) -> torch.Tensor:
    x = points.detach().clone().requires_grad_(True)
    value = _ch_exact(x)
    for _ in range(repeats):
        value = torch.autograd.grad(value.sum(), x, create_graph=True)[0][:, coordinate]
    return value


def test_ch_manufactured_source_matches_equation():
    points = torch.tensor(
        [[0.1, 0.2], [1.3, 0.7], [5.8, 0.9]], dtype=torch.float64
    )
    for order, gamma1 in ((4, 1e-2), (6, -1e-2)):
        u = _ch_exact(points)
        u_t = _exact_partial(points, 1, 1)
        u_x = _exact_partial(points, 0, 1)
        u_xx = _exact_partial(points, 0, 2)
        high = _exact_partial(points, 0, order)
        dxx_u3_minus_u = (3.0 * u.square() - 1.0) * u_xx + 6.0 * u * u_x.square()
        residual = u_t - dxx_u3_minus_u + gamma1 * high
        torch.testing.assert_close(
            residual,
            _ch_source(points, order),
            rtol=1e-11,
            atol=1e-11,
        )


def test_runner_module_is_importable_and_task_order_is_complete():
    path = ROOT / "scripts" / "run_weight_search.py"
    spec = importlib.util.spec_from_file_location("run_weight_search", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.TASK_ORDER) == set(TASKS)
    assert math.isclose(497 * len(METHODS) * 60 / 3600, 16.566666666666666)
