"""The operator-level schedule for Delta^m must agree with nested autodiff.

The schedule under test comes from the Waring decomposition of the symbol
(z_1^2 + z_2^2)^m rather than from scheduling each monomial of Delta^m
separately.  These tests pin three things: that it reproduces the operator, that
it is shorter than the term-by-term route, and that it stays differentiable in
the model parameters.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from apolarity import (
    laplacian_power,
    laplacian_power_directions,
    laplacian_power_termwise_rank,
    single_monomial_partial,
)


class Sinh(nn.Module):
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sinh(t)


def _model(width: int = 16, depth: int = 2, seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    dims = [2] + [width] * depth
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b, dtype=torch.float64), Sinh()]
    layers += [nn.Linear(dims[-1], 1, dtype=torch.float64)]
    model = nn.Sequential(*layers)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.mul_(0.35)
                module.bias.mul_(0.35)
    return model


def _nested_laplacian_power(model: nn.Module, x: torch.Tensor, m: int) -> torch.Tensor:
    """Delta^m u by repeated nested autodiff, used as the reference."""
    value = model(x)
    for _ in range(m):
        acc = torch.zeros_like(value)
        for i in range(x.shape[1]):
            g = torch.autograd.grad(value.sum(), x, create_graph=True)[0][:, i:i + 1]
            gg = torch.autograd.grad(g.sum(), x, create_graph=True)[0][:, i:i + 1]
            acc = acc + gg
        value = acc
    return value


@pytest.mark.parametrize("m", [1, 2, 3])
def test_schedule_matches_nested_autodiff(m: int) -> None:
    model = _model()
    x = torch.tensor([[0.13, -0.21], [0.4, 0.05], [-0.3, 0.27]], dtype=torch.float64,
                     requires_grad=True)
    reference = _nested_laplacian_power(model, x, m)
    scheduled = laplacian_power(model, x.detach(), m)
    scale = float(reference.detach().abs().max().clamp_min(1.0))
    assert torch.allclose(scheduled, reference.detach(), rtol=2e-9, atol=2e-9 * scale)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_schedule_is_shorter_than_termwise(m: int) -> None:
    _V, _c, info = laplacian_power_directions(m)
    assert info.rank == m + 1
    assert info.termwise_rank == laplacian_power_termwise_rank(m)
    if m == 1:
        assert info.rank == info.termwise_rank      # nothing to gain at order two
    else:
        assert info.rank < info.termwise_rank
    assert info.directions_are_real


def test_termwise_counts_are_the_published_ones() -> None:
    assert [laplacian_power_termwise_rank(m) for m in (1, 2, 3, 4)] == [2, 5, 12, 21]


@pytest.mark.parametrize("m", [2, 3])
def test_rotation_of_the_direction_set_is_immaterial(m: int) -> None:
    model = _model(seed=1)
    x = torch.tensor([[0.2, 0.1], [-0.15, 0.33]], dtype=torch.float64)
    base = laplacian_power(model, x, m)
    turned = laplacian_power(model, x, m, offset=0.37)
    assert torch.allclose(base, turned, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("m", [1, 2])
def test_schedule_is_differentiable_in_the_parameters(m: int) -> None:
    model = _model(seed=2)
    x = torch.tensor([[0.11, 0.24], [-0.2, 0.08]], dtype=torch.float64)
    loss = laplacian_power(model, x, m).pow(2).sum()
    # The output bias shifts u by a constant and so cannot reach any derivative;
    # every other parameter must receive a finite gradient.
    parameters = list(model.parameters())
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    reached = [g for g in grads if g is not None]
    assert len(reached) == len(parameters) - 1
    assert all(torch.isfinite(g).all() for g in reached)
    assert any(g.abs().max() > 0 for g in reached)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_schedule_agrees_with_the_termwise_schedule(m: int) -> None:
    """Both routes are exact, so they must agree with each other as well."""
    model = _model(seed=3)
    x = torch.tensor([[0.17, -0.09], [0.05, 0.31]], dtype=torch.float64)
    joint = laplacian_power(model, x, m)
    termwise = torch.zeros_like(joint)
    for k in range(m + 1):
        alpha = (0,) * (2 * m - 2 * k) + (1,) * (2 * k)
        termwise = termwise + math.comb(m, k) * single_monomial_partial(
            model, x, alpha, backend="waring_complex_jet"
        ).real
    scale = joint.abs().max().clamp_min(1.0)
    assert torch.allclose(joint, termwise, rtol=1e-8, atol=1e-8 * float(scale))
