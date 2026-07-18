import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "poly_shared_weights", ROOT / "scripts" / "run_poly_shared_weights.py"
)
study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(study)


@pytest.mark.parametrize(
    ("order", "text", "expected"),
    [
        (2, "1", (1.0,)),
        (4, "0.3,1", (0.3, 1.0)),
        (6, "0.1,0.3,1", (0.1, 0.3, 1.0)),
    ],
)
def test_parse_bc_weights(order, text, expected):
    assert study.parse_bc_weights(text, order) == expected


def test_parse_bc_weights_rejects_wrong_count_and_negative_values():
    with pytest.raises(ValueError, match="requires 2"):
        study.parse_bc_weights("1", 4)
    with pytest.raises(ValueError, match="non-negative"):
        study.parse_bc_weights("1,-1", 4)


def test_repeated_laplacians_for_poly_exact_solution():
    x = torch.tensor(
        [[0.2, -0.3], [0.4, 0.1]], dtype=torch.float64, requires_grad=True
    )
    u = study.exact_solution(x)
    powers = study.repeated_laplacians(u, x, 3)
    S = 2.0 * torch.pi**2
    for j, value in enumerate(powers):
        torch.testing.assert_close(value, (-S) ** j * u, rtol=1e-10, atol=1e-10)
