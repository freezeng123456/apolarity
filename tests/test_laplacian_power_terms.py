import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

from apolarity import single_monomial_partial

COMMON = Path(__file__).resolve().parents[1] / "experiments" / "common"
sys.path.insert(0, str(COMMON))

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


class _ProductSine(nn.Module):
    def forward(self, x):
        return torch.sin(math.pi * x).prod(dim=-1, keepdim=True)


def test_polyharmonic_operator_matches_source_through_order_six():
    model = _ProductSine()
    for dimension in (2, 3):
        x = torch.tensor(
            [
                [0.1, 0.2, 0.3][:dimension],
                [0.25, -0.15, 0.4][:dimension],
            ],
            dtype=torch.float64,
        )
        for power in (1, 2, 3):
            operator = sum(
                coefficient
                * single_monomial_partial(
                    model, x, alpha, backend="direct_autodiff"
                )
                for coefficient, alpha in laplacian_power_terms(dimension, power)
            )
            expected = (-dimension * math.pi**2) ** power * model(x)
            torch.testing.assert_close(operator, expected, rtol=2e-11, atol=2e-11)
