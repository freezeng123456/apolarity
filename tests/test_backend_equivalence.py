from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from apolarity import single_monomial_partial


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class Sinh(nn.Module):
    def forward(self, x):
        return torch.sinh(x)


def _real_net(activation: nn.Module, out: int = 1) -> nn.Sequential:
    torch.manual_seed(123)
    return nn.Sequential(
        nn.Linear(3, 7),
        activation,
        nn.Linear(7, out),
    ).to(torch.float64)


@pytest.mark.parametrize("activation", [nn.Tanh(), nn.Sigmoid(), Sin(), Sinh()])
@pytest.mark.parametrize("alpha", [(0, 0, 1, 1), (0, 1, 2), (2, 2, 2)])
def test_deterministic_backends_match_direct(activation, alpha):
    model = _real_net(activation)
    x = torch.tensor(
        [[-0.3, 0.2, 0.5], [0.4, -0.1, 0.25]], dtype=torch.float64
    )

    direct = single_monomial_partial(model, x, alpha, backend="direct_autodiff")
    for backend in ("polarization_jet", "waring_complex_jet", "auto"):
        actual = single_monomial_partial(model, x, alpha, backend=backend)
        torch.testing.assert_close(actual.real, direct, rtol=2e-10, atol=2e-11)
        if actual.is_complex():
            torch.testing.assert_close(
                actual.imag, torch.zeros_like(actual.imag), rtol=0.0, atol=2e-11
            )


@pytest.mark.parametrize("backend", ["polarization_jet", "waring_complex_jet"])
def test_real_parameter_gradients_match_direct(backend):
    reference = _real_net(nn.Tanh())
    candidate = copy.deepcopy(reference)
    x = torch.tensor(
        [[-0.25, 0.1, 0.3], [0.2, -0.4, 0.15]], dtype=torch.float64
    )
    alpha = (0, 0, 1, 1)

    direct = single_monomial_partial(
        reference, x, alpha, backend="direct_autodiff"
    )
    direct.real.square().mean().backward()

    actual = single_monomial_partial(candidate, x, alpha, backend=backend)
    actual.real.square().mean().backward()

    for expected_param, actual_param in zip(
        reference.parameters(), candidate.parameters()
    ):
        if expected_param.grad is None:
            assert actual_param.grad is None
        else:
            torch.testing.assert_close(
                actual_param.grad, expected_param.grad, rtol=2e-9, atol=2e-11
            )


@pytest.mark.parametrize("order", [1, 2, 3, 4, 6])
def test_complex_direct_autodiff_matches_analytic_and_jet(order):
    model = nn.Sequential(nn.Linear(1, 1, bias=False), Sinh()).to(torch.complex128)
    weight = 1.0 + 2.0j
    with torch.no_grad():
        model[0].weight.fill_(weight)
    x = torch.tensor([[0.3 + 0.0j]], dtype=torch.complex128)
    alpha = (0,) * order
    z = weight * x
    expected = weight**order * (torch.cosh(z) if order % 2 else torch.sinh(z))

    direct = single_monomial_partial(model, x, alpha, backend="direct_autodiff")
    jet = single_monomial_partial(model, x, alpha, backend="waring_complex_jet")

    torch.testing.assert_close(direct, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(jet, expected, rtol=2e-12, atol=2e-12)


def test_complex_parameter_gradients_match_direct():
    torch.manual_seed(7)
    reference = nn.Sequential(
        nn.Linear(2, 4),
        Sinh(),
        nn.Linear(4, 1),
    ).to(torch.complex128)
    candidate = copy.deepcopy(reference)
    with torch.no_grad():
        for param in reference.parameters():
            param.imag.uniform_(-0.2, 0.2)
        candidate.load_state_dict(reference.state_dict())
    x = torch.tensor(
        [[-0.2 + 0.0j, 0.3 + 0.0j], [0.15 + 0.0j, -0.1 + 0.0j]],
        dtype=torch.complex128,
    )
    alpha = (0, 0, 1)

    direct = single_monomial_partial(
        reference, x, alpha, backend="direct_autodiff"
    )
    (direct.real.square().mean() + 0.3 * direct.imag.square().mean()).backward()

    actual = single_monomial_partial(
        candidate, x, alpha, backend="waring_complex_jet"
    )
    (actual.real.square().mean() + 0.3 * actual.imag.square().mean()).backward()

    for expected_param, actual_param in zip(
        reference.parameters(), candidate.parameters()
    ):
        torch.testing.assert_close(
            actual_param.grad, expected_param.grad, rtol=2e-9, atol=2e-11
        )


def test_request_validation_is_consistent_across_backends():
    model = _real_net(nn.Tanh())
    x = torch.randn(2, 3, dtype=torch.float64)

    for backend in ("direct_autodiff", "polarization_jet", "waring_complex_jet"):
        with pytest.raises(ValueError, match="alpha indices"):
            single_monomial_partial(model, x, (3,), backend=backend)
        with pytest.raises(ValueError, match="alpha indices"):
            single_monomial_partial(model, x, (-1,), backend=backend)

    with pytest.raises(ValueError, match=r"shape \(batch, d\)"):
        single_monomial_partial(model, x[0], (0,), backend="direct_autodiff")


def test_multioutput_models_are_rejected():
    model = _real_net(nn.Tanh(), out=2)
    x = torch.randn(2, 3, dtype=torch.float64)

    with pytest.raises(ValueError, match="scalar model output"):
        single_monomial_partial(model, x, (0,), backend="direct_autodiff")
    with pytest.raises(ValueError, match="scalar model output"):
        single_monomial_partial(model, x, (0,), backend="waring_complex_jet")
