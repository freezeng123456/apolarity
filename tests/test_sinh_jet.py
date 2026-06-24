"""Smoke test for the sinh Taylor-jet rule and complex-parameter MLP.

Verifies (within fp64 round-off):
  1. Single-direction T_p via sinh-jet matches reverse-mode nested autograd.
  2. Complex-parameter MLP value-mode partial derivatives match the
     real-parameter reference.
  3. Backward pass through sinh jet gives correct parameter gradients.

Run with:  pytest tests/test_sinh_jet.py -q
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from apolarity.taylor_jet import (
    TaylorJet,
    jet_sinh,
    tp_directional_via_jet,
)
from apolarity import single_monomial_partial


class SinhActivation(nn.Module):
    def forward(self, x):
        return torch.sinh(x)


def reverse_tp(model, x, v, p):
    """Reference T_p via p nested torch.autograd.grad of g(t) = u(x + t v) at t=0."""
    t = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype, requires_grad=True)
    g = model(x + t * v).squeeze(-1)
    deriv = g
    for _ in range(p):
        deriv = torch.autograd.grad(deriv.sum(), t, create_graph=True)[0]
    return deriv / math.factorial(p)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 6])
def test_sinh_jet_matches_reverse(p):
    torch.manual_seed(0)
    d, B, hidden = 5, 3, 16
    net = nn.Sequential(
        nn.Linear(d, hidden),
        SinhActivation(),
        nn.Linear(hidden, hidden),
        SinhActivation(),
        nn.Linear(hidden, 1),
    ).to(dtype=torch.float64)

    x = torch.randn(B, d, dtype=torch.float64)
    v = torch.randn(B, d, dtype=torch.float64)
    Z = v.unsqueeze(1)

    out_jet = tp_directional_via_jet(net, x, Z, p).squeeze()
    out_rev = reverse_tp(net, x, v, p).squeeze()
    rel = (out_jet - out_rev).abs().max() / (out_rev.abs().max() + 1e-30)
    assert rel < 5e-13, f"sinh jet vs reverse mismatch at p={p}: rel={rel:.3e}"


def test_complex_parameter_partial_matches_real():
    """Complex-parameter and real-parameter MLPs with identical real weights
    give the same value-mode partial via single_monomial_partial."""
    torch.manual_seed(1)
    d, B, hidden = 4, 2, 16

    def make(real: bool) -> nn.Sequential:
        net = nn.Sequential(
            nn.Linear(d, hidden),
            SinhActivation(),
            nn.Linear(hidden, hidden),
            SinhActivation(),
            nn.Linear(hidden, 1),
        )
        if not real:
            net = net.to(dtype=torch.complex128)
        else:
            net = net.to(dtype=torch.float64)
        return net

    net_real = make(real=True)
    net_complex = make(real=False)
    # Copy real weights into complex network with zero imaginary part.
    with torch.no_grad():
        for p_real, p_complex in zip(net_real.parameters(), net_complex.parameters()):
            p_complex.data = p_real.data.to(torch.complex128)

    x_real = torch.randn(B, d, dtype=torch.float64)
    x_complex = x_real.to(dtype=torch.complex128)
    alpha = (0, 0, 1, 1)  # u_{x1 x1 x2 x2}

    out_real = single_monomial_partial(net_real, x_real, alpha,
                                        backend="waring_complex_jet")
    out_complex = single_monomial_partial(net_complex, x_complex, alpha,
                                           backend="waring_complex_jet")
    diff = (out_real.to(torch.complex128) - out_complex).abs().max()
    assert diff < 1e-12, f"complex vs real-cast disagreement: {diff:.3e}"


def test_sinh_jet_backward_to_theta():
    """Gradient of sum of T_p w.r.t. parameters via jet equals reverse reference."""
    torch.manual_seed(2)
    d, B, hidden, p = 3, 2, 8, 3
    net = nn.Sequential(
        nn.Linear(d, hidden), SinhActivation(),
        nn.Linear(hidden, 1),
    ).to(dtype=torch.float64)

    x = torch.randn(B, d, dtype=torch.float64)
    v = torch.randn(B, d, dtype=torch.float64)
    Z = v.unsqueeze(1)

    params = list(net.parameters())
    out = tp_directional_via_jet(net, x, Z, p).sum()
    grads_jet = torch.autograd.grad(out, params, allow_unused=True)

    out_rev = reverse_tp(net, x, v, p).sum()
    grads_rev = torch.autograd.grad(out_rev, params, allow_unused=True)

    for param, g_j, g_r in zip(params, grads_jet, grads_rev):
        if g_j is None:
            g_j = torch.zeros_like(param)
        if g_r is None:
            g_r = torch.zeros_like(param)
        rel = (g_j - g_r).norm() / (g_r.norm() + 1e-30)
        assert rel < 5e-12, f"theta grad mismatch: rel={rel:.3e}"
