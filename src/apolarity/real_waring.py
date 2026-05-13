"""Real direction formulas for single monomial partial derivatives.

This is the first engineering step toward a real-Waring Taylor-jet backend.
Current coverage:
  - pure powers: rank-1 optimal real formula
  - fallback: real polarization with antipodal merging

Convention:
  T_p(x; v) = (1 / p!) D^p u(x)[v, ..., v]
  returned coeffs satisfy partial^alpha u = sum_r coeff[r] * T_p(x; V[r]).
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class RealWaringInfo:
    order: int
    method: str
    rank: int
    raw_direction_count: int
    active_indices: tuple[int, ...]
    active_exponents: tuple[int, ...]


def _alpha_tuple(alpha: Iterable[int]) -> tuple[int, ...]:
    out = tuple(int(i) for i in alpha)
    if len(out) < 1:
        raise ValueError("alpha must have positive order")
    return out


def _active(alpha: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    counts = Counter(alpha)
    return tuple(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _pure_power_directions(
    alpha: tuple[int, ...],
    d: int,
    *,
    device: torch.device | None,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, RealWaringInfo] | None:
    counts = Counter(alpha)
    if len(counts) != 1:
        return None
    idx, p = next(iter(counts.items()))
    V = torch.zeros(1, d, device=device, dtype=dtype)
    V[0, idx] = 1.0
    coeff = torch.tensor([float(math.factorial(p))], device=device, dtype=dtype)
    info = RealWaringInfo(
        order=p,
        method="pure_power_rank1",
        rank=1,
        raw_direction_count=1,
        active_indices=(idx,),
        active_exponents=(p,),
    )
    return V, coeff, info


def _polarization_directions(
    alpha: tuple[int, ...],
    d: int,
    *,
    device: torch.device | None,
    dtype: torch.dtype,
    antipodal: bool,
) -> tuple[Tensor, Tensor, RealWaringInfo]:
    p = len(alpha)
    table: defaultdict[tuple[int, ...], float] = defaultdict(float)
    raw_count = 0
    for eps in itertools.product((-1, 1), repeat=p):
        raw_count += 1
        c = math.prod(eps) / float(2 ** p)
        v = [0] * d
        for sign, idx in zip(eps, alpha):
            v[idx] += sign
        if all(x == 0 for x in v):
            continue
        if antipodal:
            first = next((x for x in v if x != 0), 0)
            if first < 0:
                v = [-x for x in v]
                c *= (-1.0) ** p
        table[tuple(v)] += c

    items = [(v, c) for v, c in table.items() if abs(c) > 1e-14]
    V = torch.tensor([v for v, _ in items], device=device, dtype=dtype)
    coeff = torch.tensor([c for _, c in items], device=device, dtype=dtype)
    active = _active(alpha)
    info = RealWaringInfo(
        order=p,
        method="polarization_fallback",
        rank=len(items),
        raw_direction_count=raw_count,
        active_indices=tuple(i for i, _ in active),
        active_exponents=tuple(e for _, e in active),
    )
    return V, coeff, info


def monomial_real_waring_directions(
    alpha: Iterable[int],
    d: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    strategy: str = "auto",
    antipodal: bool = True,
) -> tuple[Tensor, Tensor, RealWaringInfo]:
    """Return real directions for one monomial partial derivative.

    Args:
        alpha: Expanded zero-based multi-index.  Example: ``(0, 0, 1)`` means
            ``partial_112``.
        d: Ambient input dimension.
        strategy:
            - ``auto``: use rank-1 pure-power formula, otherwise polarization.
            - ``pure_or_polarization``: alias of ``auto``.
            - ``polarization``: always use polarization fallback.
    """
    if dtype.is_complex:
        raise ValueError(f"real Waring directions require a real dtype, got {dtype}")
    alpha_t = _alpha_tuple(alpha)
    if any(i < 0 or i >= d for i in alpha_t):
        raise ValueError(f"alpha indices {alpha_t} out of range for d={d}")

    if strategy in ("auto", "pure_or_polarization"):
        pure = _pure_power_directions(alpha_t, d, device=device, dtype=dtype)
        if pure is not None:
            return pure
        return _polarization_directions(alpha_t, d, device=device, dtype=dtype, antipodal=antipodal)

    if strategy == "polarization":
        return _polarization_directions(alpha_t, d, device=device, dtype=dtype, antipodal=antipodal)

    raise ValueError(f"unknown real Waring strategy: {strategy!r}")


__all__ = ["RealWaringInfo", "monomial_real_waring_directions"]
