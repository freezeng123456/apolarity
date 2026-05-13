from __future__ import annotations

import torch
import torch.nn as nn

from apolarity.operators import single_monomial_partial


def _mlp(d: int) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(d, 8), nn.Tanh(), nn.Linear(8, 1)).to(dtype=torch.float64)


def test_complex_waring_jet_backpropagates_to_real_parameters():
    model = _mlp(3)
    x = torch.randn(4, 3, dtype=torch.float64)
    y = single_monomial_partial(model, x, (0, 0, 1), backend="waring_complex_jet")
    loss = y.real.square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    # Some parameters, e.g. the final bias, can legitimately drop out of a
    # high-order input derivative.  The important property is that nontrivial
    # gradients flow back to the original real-valued model parameters.
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0.0
