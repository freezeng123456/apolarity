"""Polyharmonic schedules in three or more variables must agree with autodiff.

In two variables the symbol of ``Delta^m`` factors into linear forms and the
closed-form schedule of :mod:`apolarity.symbol` is minimal.  For ``d >= 3`` the
quadric is irreducible, so that route is unavailable and
:mod:`apolarity.cubature` builds a spherical cubature rule instead.  These tests
pin that the rule reproduces the operator, that it is real, shorter than the
term-by-term route and never below the catalecticant lower bound, and that it
stays differentiable in the model parameters.
"""
from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.nn as nn

from apolarity import (
    laplacian_power,
    laplacian_power_cubature_directions,
    laplacian_power_lower_bound,
    laplacian_power_termwise_rank,
)


class Sinh(nn.Module):
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sinh(t)


def _model(d: int, width: int = 12, depth: int = 2, seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    dims = [d] + [width] * depth
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b, dtype=torch.float64), Sinh()]
    layers += [nn.Linear(dims[-1], 1, dtype=torch.float64)]
    model = nn.Sequential(*layers)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.mul_(0.3)
                module.bias.mul_(0.3)
    return model


def _points(d: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return 0.4 * torch.randn(3, d, generator=generator, dtype=torch.float64)


def _nested_laplacian_power(model: nn.Module, x: torch.Tensor, m: int) -> torch.Tensor:
    value = model(x)
    for _ in range(m):
        acc = torch.zeros_like(value)
        for i in range(x.shape[1]):
            g = torch.autograd.grad(value.sum(), x, create_graph=True)[0][:, i:i + 1]
            gg = torch.autograd.grad(g.sum(), x, create_graph=True)[0][:, i:i + 1]
            acc = acc + gg
        value = acc
    return value


@pytest.mark.parametrize(("d", "m"), [(3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (5, 2)])
def test_schedule_matches_nested_autodiff(d: int, m: int) -> None:
    model = _model(d)
    x = _points(d).requires_grad_(True)
    reference = _nested_laplacian_power(model, x, m)
    scheduled = laplacian_power(model, x.detach(), m)
    scale = float(reference.detach().abs().max().clamp_min(1.0))
    assert torch.allclose(scheduled, reference.detach(), rtol=2e-8, atol=2e-8 * scale)


@pytest.mark.parametrize("d", [3, 4, 5, 6])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_rule_reproduces_the_symbol_exactly(d: int, m: int) -> None:
    """The polynomial identity behind the schedule, checked coefficient by coefficient."""
    nodes, coeff, _info = laplacian_power_cubature_directions(m, d)
    p = 2 * m
    basis = [
        e for e in itertools.product(range(p + 1), repeat=d) if sum(e) == p
    ]
    quadric = {}
    for ks in itertools.product(range(m + 1), repeat=d):
        if sum(ks) != m:
            continue
        e = tuple(2 * k for k in ks)
        quadric[e] = quadric.get(e, 0.0) + math.factorial(m) / math.prod(
            math.factorial(k) for k in ks
        )

    worst = 0.0
    for e in basis:
        multinomial = math.factorial(p) / math.prod(math.factorial(x) for x in e)
        got = sum(
            float(coeff[r]) * multinomial * math.prod(
                float(nodes[r, j]) ** e[j] for j in range(d)
            )
            for r in range(nodes.shape[0])
        )
        want = math.factorial(p) * quadric.get(e, 0.0)
        worst = max(worst, abs(got - want))
    assert worst < 1e-8


@pytest.mark.parametrize("d", [3, 4, 5, 6])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_rule_is_real_and_respects_the_bounds(d: int, m: int) -> None:
    nodes, coeff, info = laplacian_power_cubature_directions(m, d)
    assert not nodes.is_complex() and not coeff.is_complex()
    assert torch.isfinite(nodes).all() and torch.isfinite(coeff).all()
    # Nodes lie on the unit sphere, as a cubature rule requires.
    assert torch.allclose(nodes.norm(dim=1), torch.ones(info.nodes, dtype=torch.float64))
    assert info.nodes >= info.lower_bound == laplacian_power_lower_bound(m, d)
    assert info.termwise_rank == laplacian_power_termwise_rank(m, d)
    if m > 1:
        assert info.nodes < info.termwise_rank


@pytest.mark.parametrize(("d", "m"), [(3, 1), (3, 2), (4, 1), (5, 1), (6, 1)])
def test_the_known_minimal_cases_are_attained(d: int, m: int) -> None:
    """An orthonormal frame and the icosahedral axes meet the lower bound."""
    _nodes, _coeff, info = laplacian_power_cubature_directions(m, d)
    assert info.meets_lower_bound
    assert info.weights_positive


def test_icosahedral_rule_has_equal_weights() -> None:
    _nodes, coeff, info = laplacian_power_cubature_directions(2, 3)
    assert info.nodes == 6 == info.lower_bound
    assert torch.allclose(coeff, coeff[0].expand_as(coeff))


@pytest.mark.parametrize("d", [3, 4])
def test_schedule_is_differentiable_in_the_parameters(d: int) -> None:
    model = _model(d, seed=2)
    x = _points(d, seed=1)
    loss = laplacian_power(model, x, 2).pow(2).sum()
    parameters = list(model.parameters())
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    # The output bias shifts u by a constant and so cannot reach any derivative.
    reached = [g for g in grads if g is not None]
    assert len(reached) == len(parameters) - 1
    assert all(torch.isfinite(g).all() for g in reached)
    assert any(g.abs().max() > 0 for g in reached)


def test_two_variables_still_use_the_minimal_closed_form() -> None:
    """The cubature path must not regress the case that already had an optimum."""
    for m in (2, 3, 4):
        _nodes, _coeff, info = laplacian_power_cubature_directions(m, 2)
        assert info.nodes == m + 1 == info.lower_bound


def test_unavailable_order_raises_rather_than_approximating() -> None:
    with pytest.raises(NotImplementedError, match="invariant conditions"):
        laplacian_power_cubature_directions(8, 6)


@pytest.mark.parametrize("d", [3, 4])
def test_rule_is_shorter_than_a_coordinate_grid(d: int) -> None:
    """The universal Fourier grid is the fallback; a cubature rule should beat it."""
    for m in (2, 3):
        _nodes, _coeff, info = laplacian_power_cubature_directions(m, d)
        grid = (2 * m + 1) ** (d - 1)
        assert info.nodes < grid
