from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


COMMON = Path(__file__).resolve().parents[1] / "experiments" / "common"
sys.path.insert(0, str(COMMON))

from osc_common import (  # noqa: E402
    CauchyNet,
    SplitRealField,
    build_model,
    deriv_alpha,
    sample_boundary,
    sample_interior,
)


def test_cauchy_network_direct_derivatives_are_finite():
    torch.manual_seed(11)
    model = CauchyNet(2, 6, 2)
    x = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float64)

    value = deriv_alpha(model, x, (0, 0, 1))
    assert value.shape == (2, 1)
    assert torch.isfinite(value).all()


def test_split_real_field_combines_component_derivatives():
    re = nn.Sequential(nn.Linear(2, 1, bias=False)).to(torch.float64)
    im = nn.Sequential(nn.Linear(2, 1, bias=False)).to(torch.float64)
    with torch.no_grad():
        re[0].weight.copy_(torch.tensor([[2.0, -1.0]], dtype=torch.float64))
        im[0].weight.copy_(torch.tensor([[0.5, 3.0]], dtype=torch.float64))
    field = SplitRealField(re, im)
    x = torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=torch.float64)

    torch.testing.assert_close(
        field.deriv(x, (0,)),
        torch.full((2,), 2.0 + 0.5j, dtype=torch.complex128),
    )
    torch.testing.assert_close(
        field.deriv(x, (1,)),
        torch.full((2,), -1.0 + 3.0j, dtype=torch.complex128),
    )


def test_seeded_collocation_is_independent_of_model_rng_consumption():
    device = torch.device("cpu")

    generator = torch.Generator(device=device).manual_seed(17)
    expected_int = sample_interior(16, 3, device=device, generator=generator)
    expected_bc = sample_boundary(16, 3, device=device, generator=generator)

    torch.manual_seed(17)
    build_model("tanh", 3, 8, 2)
    build_model("fourier", 3, 8, 2)
    generator = torch.Generator(device=device).manual_seed(17)
    actual_int = sample_interior(16, 3, device=device, generator=generator)
    actual_bc = sample_boundary(16, 3, device=device, generator=generator)

    torch.testing.assert_close(actual_int, expected_int)
    torch.testing.assert_close(actual_bc, expected_bc)
