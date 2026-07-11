import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

from apolarity import single_monomial_partial

COMMON = Path(__file__).resolve().parents[1] / "experiments" / "common"
POLYHARMONIC = Path(__file__).resolve().parents[1] / "experiments" / "polyharmonic"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(POLYHARMONIC))

from exp_polyharmonic import make_problems  # noqa: E402
from osc_common import laplacian_power_terms  # noqa: E402


def test_laplacian_power_terms_preserves_2d_binomial_expansion():
    assert laplacian_power_terms(2, 3) == [
        (1.0, (1, 1, 1, 1, 1, 1)),
        (3.0, (0, 0, 1, 1, 1, 1)),
        (3.0, (0, 0, 0, 0, 1, 1)),
        (1.0, (0, 0, 0, 0, 0, 0)),
    ]


def test_laplacian_power_terms_3d_counts_and_coefficients():
    for power, expected_terms in ((1, 3), (2, 6), (3, 10)):
        terms = laplacian_power_terms(3, power)
        assert len(terms) == expected_terms
        assert sum(coeff for coeff, _ in terms) == 3**power
        assert all(len(alpha) == 2 * power for _, alpha in terms)


def test_3d_polyharmonic_manufactured_solution():
    problems = make_problems(orders=(2, 4, 6), dim=3)
    x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)

    assert [len(problem.terms) for problem in problems] == [3, 6, 10]
    assert all(problem.d == 3 for problem in problems)
    assert all(math.isclose(problem.S, 3 * math.pi**2) for problem in problems)
    assert all(math.isclose(problem.extra["omega0"], 2 * math.pi) for problem in problems)
    assert torch.isfinite(problems[0].u_exact(x)).all()


class _ProductSine(nn.Module):
    def forward(self, x):
        return torch.sin(math.pi * x).prod(dim=-1, keepdim=True)


def test_3d_polyharmonic_operator_matches_source_through_order_six():
    model = _ProductSine()
    x = torch.tensor(
        [[0.1, 0.2, 0.3], [0.25, -0.15, 0.4]], dtype=torch.float64
    )

    for problem in make_problems(orders=(2, 4, 6), dim=3):
        operator = sum(
            coeff
            * single_monomial_partial(
                model, x, alpha, backend="direct_autodiff"
            )
            for coeff, alpha in problem.terms
        )
        expected = problem.source_f(x).unsqueeze(-1)
        torch.testing.assert_close(operator, expected, rtol=2e-11, atol=2e-11)
