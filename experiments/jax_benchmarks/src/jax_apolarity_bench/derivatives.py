from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import jax
import jax.numpy as jnp
from jax.experimental import jet

from .directions import rank_optimal_schedule

Method = Literal["nested_ad", "rank_optimal"]


@dataclass(frozen=True)
class DerivativeMeta:
    method: Method
    order: int
    direction_count: int | None
    theoretical_rank: int | None
    direction_dtype: str | None
    requires_complex: bool


def nested_partial_one(fun: Callable, point: jax.Array, alpha: tuple[int, ...]):
    g = fun
    for idx in alpha:
        previous = g
        g = lambda z, previous=previous, idx=idx: jax.grad(previous)(z)[idx]
    return g(point)


def directional_taylor(
    fun: Callable, point: jax.Array, direction: jax.Array, order: int
):
    dtype = jnp.result_type(point.dtype, direction.dtype)
    p = point.astype(dtype)
    v = direction.astype(dtype)
    zero = jnp.zeros_like(v)
    series = (v,) + tuple(zero for _ in range(order - 1))
    _, out = jet.jet(fun, (p,), (series,))
    return out[order - 1] / math.factorial(order)


def rank_partial_one(fun: Callable, point: jax.Array, alpha: tuple[int, ...]):
    schedule = rank_optimal_schedule(
        alpha, point.shape[-1], real_dtype=point.real.dtype
    )
    eval_point = (
        point.astype(schedule.directions.dtype) if schedule.requires_complex else point
    )
    tp = jax.vmap(lambda v: directional_taylor(fun, eval_point, v, len(alpha)))(
        schedule.directions
    )
    return jnp.sum(schedule.weights * tp), schedule


def monomial_partial(
    fun: Callable,
    x: jax.Array,
    alpha: Iterable[int],
    *,
    method: Method,
    return_complex: bool = False,
):
    alpha_t = tuple(int(i) for i in alpha)
    if not alpha_t:
        raise ValueError("alpha must have positive order")
    x = jnp.asarray(x)
    if x.ndim not in (1, 2):
        raise ValueError(f"x must have shape (d,) or (batch,d); got {x.shape}")
    d = x.shape[-1]
    if any(i < 0 or i >= d for i in alpha_t):
        raise ValueError(f"alpha indices {alpha_t} out of range for d={d}")
    if method == "nested_ad":
        one = lambda p: nested_partial_one(fun, p, alpha_t)
        value = one(x) if x.ndim == 1 else jax.vmap(one)(x)
        return value, DerivativeMeta(method, len(alpha_t), None, None, None, False)
    if method == "rank_optimal":
        schedule = rank_optimal_schedule(alpha_t, d, real_dtype=x.real.dtype)

        def one(p):
            eval_p = (
                p.astype(schedule.directions.dtype) if schedule.requires_complex else p
            )
            vals = jax.vmap(lambda v: directional_taylor(fun, eval_p, v, len(alpha_t)))(
                schedule.directions
            )
            return jnp.sum(schedule.weights * vals)

        raw = one(x) if x.ndim == 1 else jax.vmap(one)(x)
        value = (
            raw
            if return_complex or not jnp.issubdtype(raw.dtype, jnp.complexfloating)
            else jnp.real(raw)
        )
        return value, DerivativeMeta(
            method,
            len(alpha_t),
            schedule.theoretical_rank,
            schedule.theoretical_rank,
            str(schedule.directions.dtype),
            schedule.requires_complex,
        )
    raise ValueError(f"unknown method {method!r}")
