"""Closed-form Waring directions for monomial partial derivatives.

For a single monomial differential operator ``partial^alpha`` of order p,
this module constructs a rank-optimal complex Waring decomposition of the
corresponding monomial coefficient functional.

If the active exponents are sorted as ``a0 <= a1 <= ... <= a_{s-1}``, the
classical monomial Waring rank over C is

    R = prod_i (a_i + 1) / (a0 + 1) = prod_{i=1}^{s-1} (a_i + 1).

The construction used here chooses the smallest-exponent active variable as the
base variable and uses roots of unity for the others.  It returns directions
``v_r`` and coefficients ``c_r`` such that, for every smooth u,

    partial^alpha u(x) = sum_r c_r * T_p(x; v_r),
    T_p(x; v) = (1/p!) D^p u(x)[v, ..., v].

The directions and coefficients are generally complex.  For real-valued u, the
final sum is real up to round-off when conjugate directions are kept together.
"""
from __future__ import annotations

import cmath
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class WaringInfo:
    """Metadata for a monomial Waring direction set."""

    order: int
    active_indices: Tuple[int, ...]
    active_exponents: Tuple[int, ...]
    base_index: int
    rank: int
    alpha_factorial: int


def _expanded_to_counts(alpha: Iterable[int]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for idx in alpha:
        counts[int(idx)] += 1
    return counts


def alpha_factorial_from_counts(counts: Counter[int]) -> int:
    """Return alpha! = prod_i alpha_i! for expanded-index representation."""
    out = 1
    for a in counts.values():
        out *= math.factorial(a)
    return out


def monomial_waring_directions(
    alpha: Iterable[int],
    d: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.complex128,
    root_angle_sign: float = 1.0,
) -> tuple[Tensor, Tensor, WaringInfo]:
    """Construct complex rank-optimal directions for ``partial^alpha``.

    Args:
        alpha: Expanded zero-based multi-index.  Example: ``(0, 0, 1)`` means
            ``partial_112``.
        d: Ambient input dimension.
        device, dtype: Tensor placement.  ``dtype`` must be complex.
        root_angle_sign: Use ``+1`` for roots ``exp(+2πik/m)``.  ``-1`` gives
            the conjugate convention and is mathematically equivalent.

    Returns:
        ``(V, coeff, info)`` where ``V`` has shape ``(R, d)`` and ``coeff`` has
        shape ``(R,)``.  The exact identity is

            partial^alpha u = sum_r coeff[r] * T_p(V[r]).
    """
    if not dtype.is_complex:
        raise ValueError(f"monomial_waring_directions requires a complex dtype, got {dtype}")

    alpha_tuple = tuple(int(i) for i in alpha)
    p = len(alpha_tuple)
    if p < 1:
        raise ValueError("alpha must have positive order")
    if any(i < 0 or i >= d for i in alpha_tuple):
        raise ValueError(f"alpha indices {alpha_tuple} out of range for d={d}")

    counts = _expanded_to_counts(alpha_tuple)
    # Sort by exponent first to choose a minimum-exponent base variable; use
    # index as deterministic tie-breaker.
    active = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    base_idx, _base_exp = active[0]
    other = active[1:]

    alpha_fact = alpha_factorial_from_counts(counts)
    rank = 1
    denom = 1
    for _idx, exp in other:
        rank *= exp + 1
        denom *= exp + 1

    dirs: List[List[complex]] = []
    coeffs: List[complex] = []

    if not other:
        v = [0j] * d
        v[base_idx] = 1.0 + 0j
        dirs.append(v)
        coeffs.append(complex(alpha_fact))
    else:
        root_lists: List[List[complex]] = []
        for _idx, exp in other:
            m = exp + 1
            roots = [cmath.exp(root_angle_sign * 2j * math.pi * k / m) for k in range(m)]
            root_lists.append(roots)

        import itertools

        scale = alpha_fact / float(denom)
        for roots in itertools.product(*root_lists):
            v = [0j] * d
            v[base_idx] = 1.0 + 0j
            weight = complex(scale)
            for (idx, _exp), zeta in zip(other, roots):
                v[idx] = zeta
                weight *= zeta
            dirs.append(v)
            coeffs.append(weight)

    V = torch.tensor(dirs, device=device, dtype=dtype)
    c = torch.tensor(coeffs, device=device, dtype=dtype)
    info = WaringInfo(
        order=p,
        active_indices=tuple(idx for idx, _ in active),
        active_exponents=tuple(exp for _, exp in active),
        base_index=base_idx,
        rank=rank,
        alpha_factorial=alpha_fact,
    )
    return V, c, info


__all__ = [
    "WaringInfo",
    "alpha_factorial_from_counts",
    "monomial_waring_directions",
]
