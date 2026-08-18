"""Directional-schedule operators.

:func:`single_monomial_partial` handles one expanded multi-index at a time.
:func:`laplacian_power` schedules a whole polyharmonic operator from a
decomposition of its symbol, in any dimension, which is strictly shorter than
scheduling its monomials separately.  Trace contractions and general
contractable operator sums are not implemented.
"""
from __future__ import annotations

from typing import Iterable, Literal

import torch
import torch.nn as nn
from torch import Tensor

from .cubature import laplacian_power_cubature_directions
from .polarization import polarization_directions as _polarization_directions
from .symbol import laplacian_power_directions
from .taylor_jet import tp_directional_via_jet
from .waring import monomial_waring_directions

Backend = Literal["auto", "direct_autodiff", "polarization_jet", "waring_complex_jet"]


def _alpha_tuple(alpha: Iterable[int]) -> tuple[int, ...]:
    out = tuple(int(i) for i in alpha)
    if len(out) < 1:
        raise ValueError("alpha must have positive order")
    return out


def _validate_request(x: Tensor, alpha: tuple[int, ...]) -> tuple[int, int]:
    if x.ndim != 2:
        raise ValueError(f"x must have shape (batch, d); got {tuple(x.shape)}")
    batch, d = x.shape
    if any(idx < 0 or idx >= d for idx in alpha):
        raise ValueError(f"alpha indices must lie in [0, {d}); got {alpha}")
    return batch, d


def _real_of(dtype: torch.dtype) -> torch.dtype:
    """Return the underlying real dtype (e.g. complex128 -> float64)."""
    if dtype == torch.complex128:
        return torch.float64
    if dtype == torch.complex64:
        return torch.float32
    return dtype


def direct_monomial_autodiff(model: nn.Module, x: Tensor, alpha: Iterable[int], *, create_graph: bool = True) -> Tensor:
    """Direct nested coordinate autodiff reference for one expanded multi-index."""
    alpha_t = _alpha_tuple(alpha)
    batch, _d = _validate_request(x, alpha_t)
    x_req = x if x.requires_grad else x.detach().clone().requires_grad_(True)
    y = model(x_req)
    if y.ndim != 2 or y.shape != (batch, 1):
        raise ValueError(
            "single_monomial_partial requires scalar model output with shape "
            f"(batch, 1); got {tuple(y.shape)}"
        )
    deriv = y
    for k, idx in enumerate(alpha_t):
        cg = create_graph or (k < len(alpha_t) - 1)
        s = deriv.sum()
        # PyTorch returns the conjugate Wirtinger derivative for a holomorphic
        # complex output. Conjugating it recovers the analytic input derivative,
        # matching directional Taylor jets and real-coordinate finite differences.
        grad = torch.autograd.grad(
            s, x_req, grad_outputs=torch.ones_like(s),
            create_graph=cg, retain_graph=True,
        )[0]
        if grad.is_complex():
            grad = grad.conj()
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
    V, coeff, _info = _polarization_directions(
        alpha, d, device=device, dtype=dtype, antipodal=antipodal
    )
    return V, coeff


def _evaluate_direction_formula(model: nn.Module, x: Tensor, V: Tensor, coeff: Tensor, p: int) -> Tensor:
    B, d = x.shape
    Z = V.unsqueeze(0).expand(B, V.shape[0], d).contiguous()
    Tp = tp_directional_via_jet(model, x, Z, p)
    return (Tp * coeff.view(1, -1, 1)).sum(dim=1)


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
        model: `Linear/Tanh/sinh` MLP supported by `taylor_jet` for jet backends.
        x: Input tensor `(B, d)`.
        alpha: Expanded zero-based multi-index, e.g. `(0, 0, 1)` for `u112`.
        backend:
            - `direct_autodiff`: nested coordinate autodiff (reference).
            - `polarization_jet`: real polarization + Taylor jet.
            - `waring_complex_jet`: complex Waring directions + Taylor jet.
            - `auto`: choose complex Waring when its rank is at least
              20% smaller than the polarization direction count;
              otherwise use polarization.
        complex_model: Optional pre-cast complex copy of `model` for
            repeated complex calls.
        create_graph: Used only by `direct_autodiff`.
    """
    alpha_t = _alpha_tuple(alpha)
    p = len(alpha_t)
    B, d = _validate_request(x, alpha_t)

    if backend == "direct_autodiff":
        return direct_monomial_autodiff(model, x, alpha_t, create_graph=create_graph)

    if backend == "auto":
        complex_dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        _Vc, _cc, info = monomial_waring_directions(
            alpha_t, d, device=x.device, dtype=complex_dtype,
        )
        Vp, _cp, _ip = _polarization_directions(
            alpha_t, d, device=x.device, dtype=_real_of(x.dtype),
        )
        backend = "waring_complex_jet" if info.rank <= 0.8 * Vp.shape[0] else "polarization_jet"

    if backend == "polarization_jet":
        # Polarization yields real directions; if x is complex, generate the
        # directions in the corresponding real dtype and cast to x.dtype so
        # they pass through complex-parameter models cleanly.
        real_dtype = _real_of(x.dtype)
        V, coeff = polarization_directions(alpha_t, d, device=x.device, dtype=real_dtype, antipodal=True)
        if x.dtype.is_complex:
            V = V.to(dtype=x.dtype)
            coeff = coeff.to(dtype=x.dtype)
        return _evaluate_direction_formula(model, x, V, coeff, p)

    if backend == "waring_complex_jet":
        # Pick a complex dtype consistent with the input precision and any
        # complex parameters already present in the model.
        if x.dtype.is_complex:
            complex_dtype = x.dtype
        elif x.dtype == torch.float64:
            complex_dtype = torch.complex128
        else:
            complex_dtype = torch.complex64
        # If no complex copy is provided, evaluate complex directions through the
        # original real model.  Linear weights are cast inside the Taylor-jet
        # rules, so gradients flow back to the real parameters.
        cm = complex_model if complex_model is not None else model
        cx = x.to(dtype=complex_dtype)
        V, coeff, _info = monomial_waring_directions(alpha_t, d, device=x.device, dtype=complex_dtype)
        return _evaluate_direction_formula(cm, cx, V, coeff, p)

    raise ValueError(f"unknown backend: {backend!r}")


def laplacian_power(
    model: nn.Module,
    x: Tensor,
    m: int,
    *,
    offset: float = 0.0,
) -> Tensor:
    """Evaluate ``Delta^m u(x)`` from a directional schedule for the whole symbol.

    In two variables the symbol factors into linear forms, so the closed-form
    schedule of :mod:`apolarity.symbol` applies and uses ``m + 1`` directions,
    the minimum for this operator.  In higher dimensions the quadric is
    irreducible and that formula does not apply; :mod:`apolarity.cubature`
    supplies a spherical cubature rule instead.  Either way the directions are
    real, so no complex arithmetic is introduced, and both are far shorter than
    scheduling each monomial of ``Delta^m`` separately.

    Args:
        model: Scalar model supported by :mod:`apolarity.taylor_jet`.
        x: Input tensor of shape ``(B, d)``.
        m: Power of the Laplacian; the derivative order is ``2m``.
        offset: Rotation of the equally spaced direction set, in radians.  Used
            only when ``d == 2``, where every rotation gives a minimal schedule.
    """
    if x.ndim != 2:
        raise ValueError(f"laplacian_power expects x of shape (B, d), got {tuple(x.shape)}")
    d = x.shape[1]

    if d == 2:
        V, coeff, info = laplacian_power_directions(
            m, 2, device=x.device, dtype=x.dtype, offset=offset
        )
        order = info.order
    else:
        V, coeff, cinfo = laplacian_power_cubature_directions(
            m, d, device=x.device, dtype=x.dtype
        )
        order = cinfo.order
    return _evaluate_direction_formula(model, x, V, coeff, order)
