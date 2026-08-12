from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "experiments" / "common", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MODULE_PATH = ROOT / "experiments" / "mpfc_2d" / "problem.py"
SPEC = importlib.util.spec_from_file_location("mpfc_problem_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mpfc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mpfc
SPEC.loader.exec_module(mpfc)


def _partial(value: torch.Tensor, points: torch.Tensor, alpha: tuple[int, ...]) -> torch.Tensor:
    result = value
    for coordinate in alpha:
        result = torch.autograd.grad(
            result.sum(), points, create_graph=True, retain_graph=True
        )[0][:, coordinate]
    return result


def test_protocol_is_explicitly_sixth_order_and_periodic():
    assert mpfc.TASK.order == 6
    assert mpfc.TASK.spatial_dim == 2
    assert mpfc.TASK.coordinate_names == ("x", "y", "t")
    assert mpfc.TASK.weights == (1.0, 1.0)
    assert mpfc.METHODS == ("war", "real_tanh_autodiff")
    assert mpfc.PROTOCOL_ID == "mpfc_2d_o6_common_xavier_fp32_v1"


def test_initial_data_and_zero_initial_velocity():
    points = torch.tensor(
        [[0.2, 0.4, 0.0], [2.0, 1.3, 0.0]], dtype=torch.float64
    )
    expected = 0.1 + 0.15 * torch.cos(points[:, 0]) * torch.cos(points[:, 1])
    expected = expected + 0.05 * torch.cos(2.0 * points[:, 0]) * torch.cos(points[:, 1])
    torch.testing.assert_close(mpfc.initial_phi(points), expected)
    torch.testing.assert_close(mpfc.initial_phi_t(points), torch.zeros(2, dtype=torch.float64))


def test_laplacian_cube_product_rule_matches_direct_autodiff():
    class Exact(torch.nn.Module):
        def forward(self, points: torch.Tensor) -> torch.Tensor:
            x, y, t = points.unbind(dim=-1)
            return (torch.sin(x) + 0.25 * torch.cos(2.0 * y) + t.square()).unsqueeze(-1)

    model = Exact()
    points = torch.tensor(
        [[0.2, 0.4, 0.1], [1.2, 2.0, 0.7]], dtype=torch.float64, requires_grad=True
    )
    phi = model(points)[:, 0]
    lap_cube_direct = _partial(phi.square().mul(phi), points, (0, 0)) + _partial(
        phi.square().mul(phi), points, (1, 1)
    )
    grad_sq = _partial(phi, points, (0,)).square() + _partial(phi, points, (1,)).square()
    lap = _partial(phi, points, (0, 0)) + _partial(phi, points, (1, 1))
    expected = 6.0 * phi * grad_sq + 3.0 * phi.square() * lap
    torch.testing.assert_close(lap_cube_direct, expected, rtol=2e-10, atol=2e-10)


def test_periodic_initial_trace_matches_all_orders_zero_to_five():
    tangent = torch.tensor([0.3, 1.1, 0.8], dtype=torch.float64)
    for coordinate in (0, 1):
        lower = torch.zeros(3, 3, dtype=torch.float64)
        upper = torch.zeros_like(lower)
        lower[:, coordinate] = 0.0
        upper[:, coordinate] = 2.0 * math.pi
        lower[:, 1 - coordinate] = tangent
        upper[:, 1 - coordinate] = tangent
        lower[:, 2] = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
        upper[:, 2] = lower[:, 2]
        lower.requires_grad_(True)
        upper.requires_grad_(True)
        for order in range(6):
            alpha = (coordinate,) * order
            left = mpfc.initial_phi(lower) if order == 0 else _partial(mpfc.initial_phi(lower), lower, alpha)
            right = mpfc.initial_phi(upper) if order == 0 else _partial(mpfc.initial_phi(upper), upper, alpha)
            torch.testing.assert_close(left, right, rtol=0.0, atol=5e-12)


def test_tiny_loss_and_gradients_are_finite_for_both_methods():
    for method in mpfc.METHODS:
        torch.manual_seed(23)
        model, dtype, backend = mpfc.build_model(
            mpfc.TASK, method, torch.device("cpu"), hidden=5, depth=1
        )
        bundle = mpfc.make_loss_bundle(
            mpfc.TASK,
            model,
            dtype,
            backend,
            (1.0, 1.0),
            torch.device("cpu"),
            n_int=2,
            n_ic=2,
            n_bc=4,
        )
        loss, components = bundle.loss_fn()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value).all() for value in components.values())
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        assert gradients and all(value is not None for value in gradients)
        assert all(torch.isfinite(value).all() for value in gradients if value is not None)

