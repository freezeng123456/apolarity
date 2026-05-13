from __future__ import annotations

import torch

from apolarity.taylor_jet import TaylorJet, jet_sigmoid, jet_sin, jet_tanh


def _poly_eval(coeffs, t):
    out = torch.zeros_like(coeffs[0])
    pow_t = torch.ones_like(t)
    for c in coeffs:
        out = out + c * pow_t
        pow_t = pow_t * t
    return out


def _check_activation(act_name: str, jet_fn, scalar_fn, device="cpu"):
    torch.manual_seed(0)
    p = 5
    n = 4
    dtype = torch.float64
    coeffs = [torch.randn(n, dtype=dtype, device=device, requires_grad=True) * 0.2 for _ in range(p + 1)]
    coeffs = [c.clone().detach().requires_grad_(True) for c in coeffs]
    out = jet_fn(TaylorJet(coeffs)).terms
    loss = sum((k + 1) * y.square().sum() for k, y in enumerate(out))
    grads_custom = torch.autograd.grad(loss, coeffs)

    coeffs_ref = [c.detach().clone().requires_grad_(True) for c in coeffs]
    t = torch.zeros(n, dtype=dtype, device=device, requires_grad=True)
    y = scalar_fn(_poly_eval(coeffs_ref, t))
    terms = [y]
    deriv = y
    fact = 1.0
    for k in range(1, p + 1):
        deriv = torch.autograd.grad(deriv.sum(), t, create_graph=True)[0]
        fact *= k
        terms.append(deriv / fact)
    loss_ref = sum((k + 1) * yk.square().sum() for k, yk in enumerate(terms))
    grads_ref = torch.autograd.grad(loss_ref, coeffs_ref)

    for gc, gr in zip(grads_custom, grads_ref):
        assert torch.allclose(gc, gr, atol=1e-10, rtol=1e-10), act_name


def test_tanh_custom_vjp_matches_autograd():
    _check_activation("tanh", jet_tanh, torch.tanh)


def test_sigmoid_custom_vjp_matches_autograd():
    _check_activation("sigmoid", jet_sigmoid, torch.sigmoid)


def test_sin_custom_vjp_matches_autograd():
    _check_activation("sin", jet_sin, torch.sin)
