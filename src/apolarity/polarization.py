"""Real polarization directions for one expanded multi-index.

Convention:
  T_p(x; v) = (1 / p!) D^p u(x)[v, ..., v]
  returned coeffs satisfy partial^alpha u = sum_r coeff[r] * T_p(x; V[r]).

The antipodally-merged polarization identity yields at most 2^{p-1} real
directions and is correct for every multi-index of order p; it serves as
the real-arithmetic baseline used elsewhere in the codebase.
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
class PolarizationInfo:
    order: int
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


def polarization_directions(
    alpha: Iterable[int],
    d: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    antipodal: bool = True,
) -> tuple[Tensor, Tensor, PolarizationInfo]:
    """Return antipodally-merged polarization directions and weights.

    Args:
        alpha: Expanded zero-based multi-index. Example: ``(0, 0, 1)`` means
            ``partial_112``.
        d: Ambient input dimension.
        device, dtype: torch placement / scalar type for the returned tensors.
        antipodal: If True (default), merge each direction with its antipode
            so the returned schedule has at most ``2^{p-1}`` directions.
    """
    if dtype.is_complex:
        raise ValueError(f"polarization directions require a real dtype, got {dtype}")
    alpha_t = _alpha_tuple(alpha)
    if any(i < 0 or i >= d for i in alpha_t):
        raise ValueError(f"alpha indices {alpha_t} out of range for d={d}")

    p = len(alpha_t)
    table: defaultdict[tuple[int, ...], float] = defaultdict(float)
    raw_count = 0
    for eps in itertools.product((-1, 1), repeat=p):
        raw_count += 1
        c = math.prod(eps) / float(2 ** p)
        v = [0] * d
        for sign, idx in zip(eps, alpha_t):
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
    active = _active(alpha_t)
    info = PolarizationInfo(
        order=p,
        rank=len(items),
        raw_direction_count=raw_count,
        active_indices=tuple(i for i, _ in active),
        active_exponents=tuple(e for _, e in active),
    )
    return V, coeff, info


__all__ = ["PolarizationInfo", "polarization_directions"]
