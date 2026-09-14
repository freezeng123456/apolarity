from __future__ import annotations

import cmath
import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class DirectionSchedule:
    alpha: tuple[int, ...]
    order: int
    active_indices: tuple[int, ...]
    active_exponents: tuple[int, ...]
    base_index: int
    theoretical_rank: int
    alpha_factorial: int
    requires_complex: bool
    directions: jnp.ndarray
    weights: jnp.ndarray


def parse_one_based_pattern(pattern: str) -> tuple[int, ...]:
    text = pattern.strip().replace("u_", "").replace("_", "")
    if not text or not text.isdigit() or "0" in text:
        raise ValueError(f"invalid one-based pattern {pattern!r}")
    return tuple(int(ch) - 1 for ch in text)


def exponent_pattern(alpha: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(Counter(int(i) for i in alpha).values(), reverse=True))


def theoretical_waring_rank(alpha: Iterable[int]) -> int:
    alpha_t = tuple(int(i) for i in alpha)
    if not alpha_t:
        raise ValueError("alpha must have positive order")
    exps = sorted(Counter(alpha_t).values())
    return math.prod(e + 1 for e in exps[1:]) if len(exps) > 1 else 1


def rank_optimal_schedule(
    alpha: Iterable[int], d: int, *, real_dtype=jnp.float32
) -> DirectionSchedule:
    alpha_t = tuple(int(i) for i in alpha)
    if not alpha_t:
        raise ValueError("alpha must have positive order")
    if any(i < 0 or i >= d for i in alpha_t):
        raise ValueError(f"alpha indices {alpha_t} out of range for d={d}")
    real_dtype = jnp.dtype(real_dtype)
    if real_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError(f"real_dtype must be float32/float64; got {real_dtype}")
    counts = Counter(alpha_t)
    active = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    base = active[0][0]
    other = active[1:]
    rank = math.prod(exp + 1 for _, exp in other) if other else 1
    alpha_factorial = math.prod(math.factorial(exp) for exp in counts.values())
    requires_complex = any(exp > 1 for _, exp in other)
    dtype = (
        jnp.complex64
        if (requires_complex and real_dtype == jnp.float32)
        else (jnp.complex128 if requires_complex else real_dtype)
    )
    root_lists = []
    for _, exp in other:
        m = exp + 1
        if m == 2 and not requires_complex:
            root_lists.append([1.0, -1.0])
        else:
            root_lists.append([cmath.exp(2j * math.pi * k / m) for k in range(m)])
    products = itertools.product(*root_lists) if root_lists else [()]
    directions, weights = [], []
    scale = alpha_factorial / float(rank)
    for roots in products:
        v = [0j if requires_complex else 0.0 for _ in range(d)]
        v[base] = 1.0
        w = complex(scale) if requires_complex else float(scale)
        for (idx, _), root in zip(other, roots):
            v[idx] = root
            w = w * root
        if not requires_complex:
            v = [float(complex(z).real) for z in v]
            w = float(complex(w).real)
        directions.append(v)
        weights.append(w)
    return DirectionSchedule(
        alpha=alpha_t,
        order=len(alpha_t),
        active_indices=tuple(i for i, _ in active),
        active_exponents=tuple(e for _, e in active),
        base_index=base,
        theoretical_rank=rank,
        alpha_factorial=alpha_factorial,
        requires_complex=requires_complex,
        directions=jnp.asarray(np.asarray(directions), dtype=dtype),
        weights=jnp.asarray(np.asarray(weights), dtype=dtype),
    )
