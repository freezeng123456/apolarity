from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from apolarity import single_monomial_partial


COMMON = Path(__file__).resolve().parents[1] / "experiments" / "common"
sys.path.insert(0, str(COMMON))

from osc_common import (  # noqa: E402
    FORMAL_VARIANTS,
    FourierPINN,
    MultiScaleNet,
    ScaledSin,
    build_model,
    build_siren,
    formal_architecture_budgets,
    n_params,
)


def _compare_jet_and_direct(model: nn.Module, x: torch.Tensor, alpha: tuple[int, ...]):
    reference = copy.deepcopy(model)
    candidate = copy.deepcopy(model)
    direct = single_monomial_partial(
        reference, x, alpha, backend="direct_autodiff"
    )
    jet = single_monomial_partial(
        candidate, x, alpha, backend="waring_complex_jet"
    )
    torch.testing.assert_close(jet.real, direct.real, rtol=2e-8, atol=2e-9)

    direct.real.square().mean().backward()
    jet.real.square().mean().backward()
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        if not expected.requires_grad:
            assert actual.grad is None
        elif expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(
                actual.grad, expected.grad, rtol=3e-7, atol=3e-9
            )


def test_formal_method_registry_contains_exactly_four_methods():
    assert FORMAL_VARIANTS == ("complex_sinh", "siren", "fourier", "mscale")
    assert "tanh" not in FORMAL_VARIANTS
    assert "complex_sinh_noinit" not in FORMAL_VARIANTS


def test_siren_matches_upstream_layer_and_initialization_contract():
    torch.manual_seed(3)
    model = build_siren(3, 16, 4)
    linears = [module for module in model if isinstance(module, nn.Linear)]
    activations = [module for module in model if isinstance(module, ScaledSin)]

    assert len(activations) == 4
    assert activations[0].omega0 == 30.0
    assert all(module.omega0 == 30.0 for module in activations[1:])
    assert linears[0].weight.abs().max() <= 1.0 / 3.0
    hidden_bound = math.sqrt(6.0 / 16.0) / 30.0
    for layer in linears[1:]:
        assert layer.weight.abs().max() <= hidden_bound

    x = torch.randn(5, 3, dtype=torch.float64)
    h = x
    for layer, activation in zip(linears[:-1], activations, strict=True):
        h = torch.sin(activation.omega0 * layer(h))
    torch.testing.assert_close(model(x), linears[-1](h))


@pytest.mark.parametrize("order", [1, 2, 4])
def test_scaled_siren_jet_and_parameter_gradients_match_direct(order: int):
    torch.manual_seed(7)
    model = build_siren(2, 8, 2)
    x = torch.tensor(
        [[-0.2, 0.1], [0.15, 0.25]], dtype=torch.float64
    )
    _compare_jet_and_direct(model, x, (0,) * order)


def test_fourier_is_two_frozen_branches_with_shared_four_layer_trunk():
    torch.manual_seed(11)
    model, dtype = build_model(
        "fourier", 2, 12, 4, fourier_sigma=math.pi
    )
    assert dtype == torch.float64
    assert isinstance(model, FourierPINN)
    assert model.branch_sigmas == (1.0, math.pi)
    assert len(model.feature_maps) == 2
    assert len([m for m in model.trunk if isinstance(m, nn.Linear)]) == 4
    assert all(
        not parameter.requires_grad
        for feature in model.feature_maps
        for parameter in feature.parameters()
    )

    x = torch.randn(4, 2, dtype=torch.float64)
    xbar = (x - model.input_mean) / model.input_std
    branches = [model.trunk(feature(xbar)) for feature in model.feature_maps]
    torch.testing.assert_close(
        model(x), model.output(torch.cat(branches, dim=-1))
    )


@pytest.mark.parametrize("alpha", [(0, 1), (0, 0, 1, 1)])
def test_fourier_branch_jet_and_parameter_gradients_match_direct(alpha):
    torch.manual_seed(13)
    model, _ = build_model("fourier", 2, 8, 2, fourier_sigma=2.0)
    x = torch.tensor(
        [[-0.1, 0.2], [0.3, -0.25]], dtype=torch.float64
    )
    _compare_jet_and_direct(model, x, alpha)


def test_mscale_is_explicit_scaled_input_subnet_sum():
    torch.manual_seed(17)
    model, dtype = build_model("mscale", 2, 7, 3)
    assert dtype == torch.float64
    assert isinstance(model, MultiScaleNet)
    assert model.scales == (1.0, 2.0, 4.0)
    x = torch.randn(3, 2, dtype=torch.float64)
    expected = sum(
        subnet(scale * x)
        for scale, subnet in zip(model.scales, model.subnets, strict=True)
    )
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("alpha", [(0, 1), (0, 0, 1, 1)])
def test_mscale_scaled_chain_rule_and_parameter_gradients_match_direct(alpha):
    torch.manual_seed(19)
    model, _ = build_model("mscale", 2, 6, 2)
    x = torch.tensor(
        [[-0.12, 0.08], [0.22, -0.18]], dtype=torch.float64
    )
    _compare_jet_and_direct(model, x, alpha)


def test_complex_sinh_initialization_is_frequency_rich_and_scalar_output():
    torch.manual_seed(23)
    model, dtype = build_model("complex_sinh", 3, 10, 4, omega0=2 * math.pi)
    assert dtype == torch.complex128
    first = next(module for module in model if isinstance(module, nn.Linear))
    assert first.weight.dtype == torch.complex128
    assert first.weight.real.abs().max() <= 1.0 / 3.0
    assert first.weight.imag.abs().max() <= 2 * math.pi / 3.0
    x = torch.randn(5, 3, dtype=torch.complex128)
    assert model(x).shape == (5, 1)


@pytest.mark.parametrize("split_real", [False, True])
def test_four_method_parameter_budgets_are_within_five_percent(split_real):
    budgets = formal_architecture_budgets(
        2,
        depth=4,
        complex_width=128,
        split_real_baselines=split_real,
        omega0=2 * math.pi,
        fourier_sigma=math.pi,
    )
    assert set(budgets) == set(FORMAL_VARIANTS)
    assert budgets["complex_sinh"].width == 128
    assert all(budget.width != 64 for budget in budgets.values())
    assert all(budget.relative_error <= 0.05 for budget in budgets.values())

    for variant, budget in budgets.items():
        model, _ = build_model(
            variant,
            2,
            budget.width,
            4,
            omega0=2 * math.pi,
            fourier_sigma=math.pi,
        )
        multiplier = 2 if budget.representation == "split_real" else 1
        assert multiplier * n_params(model) == budget.real_dof
