"""Single-monomial partial derivative operators.

This module intentionally handles one expanded multi-index at a time.  It does
not implement Laplacian powers, trace contractions, or contractable operator
sums.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Literal

import torch
import torch.nn as nn
from torch import Tensor

from .real_waring import monomial_real_waring_directions
from .taylor_jet import tp_directional_via_jet
from .waring import monomial_waring_directions

Backend = Literal["auto", "direct_autodiff", "polarization_jet", "waring_complex_jet", "waring_real_jet"]


def _alpha_tuple(alpha: Iterable[int]) -> tuple[int, ...]:
    out = tuple(int(i) for i in alpha)
    if len(out) < 1:
        raise ValueError("alpha must have positive order")
    return out


def direct_monomial_autodiff(model: nn.Module, x: Tensor, alpha: Iterable[int], *, create_graph: bool = True) -> Tensor:
    """Direct nested coordinate autodiff reference for one expanded multi-index."""
    alpha_t = _alpha_tuple(alpha)
    x_req = x if x.requires_grad else x.detach().clone().requires_grad_(True)
    y = model(x_req)
    deriv = y
    for k, idx in enumerate(alpha_t):
        cg = create_graph or (k < len(alpha_t) - 1)
        grad = torch.autograd.grad(deriv.sum(), x_req, create_graph=cg, retain_graph=True)[0]
        deriv = grad[:, idx:idx + 1]
    return deriv


def polarization_directions(
    alpha: Iterable[int],
    d: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    antipodal: bool = True,
) -> tuple[Tensor, Tensor]:
    """Real polarization directions for one expanded multi-index."""
    V, coeff, _info = monomial_real_waring_directions(
        alpha, d, device=device, dtype=dtype, strategy="polarization", antipodal=antipodal
    )
    return V, coeff


def _evaluate_direction_formula(model: nn.Module, x: Tensor, V: Tensor, coeff: Tensor, p: int) -> Tensor:
    B, d = x.shape
    Z = V.unsqueeze(0).expand(B, V.shape[0], d).contiguous()
    Tp = tp_directional_via_jet(model, x, Z, p)
    return (Tp * coeff.view(1, -1, 1)).sum(dim=1)


def _complex_model(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    # Caller usually passes a freshly copied model for repeated use.  This helper
    # is deliberately simple for one-off API calls.
    import copy

    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return copy.deepcopy(model).to(dtype=complex_dtype)


def single_monomial_partial(
    model: nn.Module,
    x: Tensor,
    alpha: Iterable[int],
    *,
    backend: Backend = "auto",
    complex_model: nn.Module | None = None,
    create_graph: bool = True,
) -> Tensor:
    """Compute one single-monomial partial derivative.

    Args:
        model: `Linear/Tanh` MLP or wrapper supported by `taylor_jet` for jet backends.
        x: Input tensor `(B, d)`.
        alpha: Expanded zero-based multi-index, e.g. `(0, 0, 1)` for `u112`.
        backend:
            - `direct_autodiff`: nested coordinate autodiff.
            - `polarization_jet`: real polarization + Taylor jet.
            - `waring_complex_jet`: complex Waring directions + Taylor jet.
            - `waring_real_jet`: current real generator + Taylor jet.
            - `auto`: choose complex Waring when it has a clear direction-count advantage,
              otherwise use polarization.
        complex_model: Optional pre-cast complex copy of `model` for repeated complex calls.
        create_graph: Used only by `direct_autodiff`.
    """
    alpha_t = _alpha_tuple(alpha)
    p = len(alpha_t)
    B, d = x.shape

    if backend == "direct_autodiff":
        return direct_monomial_autodiff(model, x, alpha_t, create_graph=create_graph)

    if backend == "auto":
        Vc, _cc, info = monomial_waring_directions(
            alpha_t,
            d,
            device=x.device,
            dtype=torch.complex128 if x.dtype == torch.float64 else torch.complex64,
        )
        Vr, _cr, _ri = monomial_real_waring_directions(alpha_t, d, device=x.device, dtype=x.dtype, strategy="polarization")
        backend = "waring_complex_jet" if info.rank <= 0.7 * Vr.shape[0] else "polarization_jet"

    if backend == "polarization_jet":
        V, coeff = polarization_directions(alpha_t, d, device=x.device, dtype=x.dtype, antipodal=True)
        return _evaluate_direction_formula(model, x, V, coeff, p)

    if backend == "waring_real_jet":
        V, coeff, _info = monomial_real_waring_directions(alpha_t, d, device=x.device, dtype=x.dtype)
        return _evaluate_direction_formula(model, x, V, coeff, p)

    if backend == "waring_complex_jet":
        complex_dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        # If no complex copy is provided, evaluate complex directions through the
        # original real model.  Linear weights are cast inside the Taylor-jet
        # rules, so gradients flow back to the real parameters.
        cm = complex_model if complex_model is not None else model
        cx = x.to(dtype=complex_dtype)
        V, coeff, _info = monomial_waring_directions(alpha_t, d, device=x.device, dtype=complex_dtype)
        return _evaluate_direction_formula(cm, cx, V, coeff, p)

    raise ValueError(f"unknown backend: {backend!r}")
