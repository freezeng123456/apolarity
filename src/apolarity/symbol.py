"""Directional schedules for a whole operator rather than one monomial.

The reduction behind :mod:`apolarity.waring` never uses that the target is a
single monomial.  For any homogeneous constant-coefficient operator

    P = sum_{|beta| = p} b_beta partial^beta

with symbol ``sigma_P(z) = sum_beta b_beta z^beta``, a directional schedule

    P u(x) = sum_r c_r T_p(x; v_r)   for every sufficiently smooth u

exists if and only if ``p! sigma_P(z) = sum_r c_r (v_r . z)^p``.  The minimum
number of directions is therefore the Waring rank of the symbol, and expanding
``P`` into monomials and scheduling each one separately is in general *not*
optimal.

This module implements the first operator for which the optimum is available in
closed form: the polyharmonic operator ``Delta^m`` in two variables.  Its symbol
``(z_1^2 + z_2^2)^m`` becomes the monomial ``zeta^m conj(zeta)^m`` under the
invertible change of variables ``zeta = z_1 + i z_2``, and Waring rank is
invariant under such a change, so the rank is ``m + 1`` by the monomial rank
formula.  A minimizing set of directions is real and equally spaced:

    v_r = (cos(r pi / (m+1)), sin(r pi / (m+1))),   r = 0, ..., m,

with one common coefficient.  Term-by-term scheduling of the same operator costs
1 + sum_{k=1}^{m-1} (max(2m-2k, 2k) + 1) + 1 directions, which is 5 at order
four and 12 at order six against 3 and 4 here.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class SymbolScheduleInfo:
    """Metadata for an operator-level directional schedule."""

    operator: str
    order: int
    rank: int
    termwise_rank: int
    directions_are_real: bool


def _monomial_rank(exponents: Tuple[int, ...]) -> int:
    """Complex Waring rank of a monomial with the given nonzero exponents."""
    active = sorted(e for e in exponents if e > 0)
    if not active:
        raise ValueError("a monomial schedule needs at least one active exponent")
    return math.prod(e + 1 for e in active[1:]) if len(active) > 1 else 1


@lru_cache(maxsize=None)
def quadric_power_coefficients(m: int, d: int) -> dict[Tuple[int, ...], float]:
    """Coefficients of ``(z_1^2 + ... + z_d^2)^m`` in the monomial basis."""
    if m < 1 or d < 1:
        raise ValueError("m and d must be positive")
    out: dict[Tuple[int, ...], float] = {}
    for ks in itertools.product(range(m + 1), repeat=d):
        if sum(ks) != m:
            continue
        exponents = tuple(2 * k for k in ks)
        weight = math.factorial(m) / math.prod(math.factorial(k) for k in ks)
        out[exponents] = out.get(exponents, 0.0) + weight
    return out


def laplacian_power_termwise_rank(m: int, d: int = 2) -> int:
    """Directions used by scheduling every monomial of ``Delta^m`` separately."""
    return sum(
        _monomial_rank(exponents) for exponents in quadric_power_coefficients(m, d)
    )


def laplacian_power_directions(
    m: int,
    d: int = 2,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    offset: float = 0.0,
) -> tuple[Tensor, Tensor, SymbolScheduleInfo]:
    """Rank-optimal schedule for ``Delta^m`` in two variables.

    Args:
        m: Power of the Laplacian; the operator order is ``p = 2m``.
        d: Ambient dimension.  Only ``d = 2`` admits this closed form.
        device, dtype: Tensor placement.  A real dtype is admissible because the
            minimizing directions of this symbol happen to be real.
        offset: Rotation of the direction set, in radians.  Every offset gives a
            valid minimal schedule, since the identity behind the construction
            holds for any rotation of the equally spaced set.

    Returns:
        ``(V, coeff, info)`` with ``V`` of shape ``(m+1, d)`` and ``coeff`` of
        shape ``(m+1,)``, satisfying

            Delta^m u(x) = sum_r coeff[r] * T_{2m}(x; V[r]).
    """
    if m < 1:
        raise ValueError("m must be at least one")
    if d != 2:
        raise NotImplementedError(
            "the closed-form optimum implemented here is specific to d = 2; "
            "in higher dimensions the symbol is not a monomial after any change "
            "of variables and its Waring rank is not known in closed form"
        )

    p = 2 * m
    n = m + 1
    angles = [offset + r * math.pi / n for r in range(n)]
    dirs = [[math.cos(theta), math.sin(theta)] for theta in angles]
    # (z1^2 + z2^2)^m = 2^p / (n binom(p, m)) * sum_r (v_r . z)^p, and the
    # schedule coefficient carries the extra p! of the directional convention.
    weight = math.factorial(p) * 2.0**p / (n * math.comb(p, m))

    V = torch.tensor(dirs, device=device, dtype=dtype)
    coeff = torch.full((n,), weight, device=device, dtype=dtype)
    info = SymbolScheduleInfo(
        operator=f"Delta^{m}",
        order=p,
        rank=n,
        termwise_rank=laplacian_power_termwise_rank(m, d),
        directions_are_real=True,
    )
    return V, coeff, info


__all__ = [
    "SymbolScheduleInfo",
    "laplacian_power_directions",
    "laplacian_power_termwise_rank",
]
