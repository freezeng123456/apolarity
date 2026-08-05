from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "archived"
    / "other_families"
    / "core_method"
    / "benchmark_single_monomial.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_single_monomial", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Square(nn.Module):
    def forward(self, x):
        return x[:, :1].square()


def test_gaussian_hermite_mc_is_exact_for_quadratic_with_exact_moments():
    x = torch.tensor([[-0.4], [0.7]], dtype=torch.float64)
    nodes = torch.tensor(
        [3.0**0.5, -(3.0**0.5), 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    Z = nodes.view(1, 6, 1).expand(2, -1, -1).clone()

    actual = MODULE.gaussian_hermite_mc(_Square(), x, (0, 0), Z, sigma=0.2)
    torch.testing.assert_close(actual, torch.full_like(actual, 2.0))


def test_gaussian_hermite_mc_validates_scale_and_shapes():
    x = torch.zeros(2, 1, dtype=torch.float64)
    Z = torch.zeros(2, 4, 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="sigma must be positive"):
        MODULE.gaussian_hermite_mc(_Square(), x, (0,), Z, sigma=0.0)
    with pytest.raises(ValueError, match=r"Z=\(B,K,d\)"):
        MODULE.gaussian_hermite_mc(_Square(), x, (0,), Z[0], sigma=0.1)
