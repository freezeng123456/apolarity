from __future__ import annotations

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

from osc_common import deriv_alpha  # noqa: E402
from weight_search import TASKS, build_search_model  # noqa: E402


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
    return _partial(value, points, (0, 0)) + _partial(
        value, points, (1, 1)
    )


def _exact(points: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * points[:, 0]) * torch.sin(
        math.pi * points[:, 1]
    )


def test_poly_o6_exact_solution_matches_triharmonic_source() -> None:
    points = torch.tensor(
        [[-0.63, 0.21], [0.47, -0.54]],
        dtype=torch.float64,
        requires_grad=True,
    )
    value = _exact(points)
    lap = _laplacian(value, points)
    bilap = _laplacian(lap, points)
    trilap = _laplacian(bilap, points)
    source = (-2.0 * math.pi**2) ** 3 * value
    torch.testing.assert_close(trilap, source, rtol=2e-10, atol=2e-10)


def test_poly_o6_exact_solution_matches_all_navier_traces() -> None:
    tangential = torch.tensor([-0.72, 0.13, 0.81], dtype=torch.float64)
    for coordinate in (0, 1):
        for boundary in (-1.0, 1.0):
            points = torch.empty(3, 2, dtype=torch.float64, requires_grad=True)
            with torch.no_grad():
                points[:, coordinate] = boundary
                points[:, 1 - coordinate] = tangential
            value = _exact(points)
            lap = _laplacian(value, points)
            bilap = _laplacian(lap, points)
            for trace in (value, lap, bilap):
                torch.testing.assert_close(
                    trace, torch.zeros_like(trace), atol=2e-10, rtol=0
                )


def test_poly_o6_waring_derivatives_match_direct_autodiff() -> None:
    torch.manual_seed(41)
    task = TASKS["poly_d2_o6"]
    model, dtype, _ = build_search_model(
        task,
        "war",
        torch.device("cpu"),
        hidden=6,
        depth=2,
    )
    points = torch.tensor(
        [[-0.37, 0.11], [0.22, -0.41]], dtype=dtype
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
