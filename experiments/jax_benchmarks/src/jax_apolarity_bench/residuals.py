from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from .derivatives import Method, monomial_partial

KDV2D_TERMS = {
    "u_x": (0,),
    "u_y": (1,),
    "u_xx": (0, 0),
    "u_xy": (0, 1),
    "u_yy": (1, 1),
    "u_ty": (1, 2),
    "u_xxxy": (0, 0, 0, 1),
}
GKDV1D_TERMS = {
    "u_x": (0,),
    "u_t": (1,),
    "u_xx": (0, 0),
    "u_tx": (1, 0),
    "u_tt": (1, 1),
    "u_xxx": (0, 0, 0),
    "u_xxxx": (0, 0, 0, 0),
    "u_txxx": (1, 0, 0, 0),
}
POLYHARMONIC_2D = {
    4: ((1.0, (0, 0, 0, 0)), (2.0, (0, 0, 1, 1)), (1.0, (1, 1, 1, 1))),
    6: (
        (1.0, (0, 0, 0, 0, 0, 0)),
        (3.0, (0, 0, 0, 0, 1, 1)),
        (3.0, (0, 0, 1, 1, 1, 1)),
        (1.0, (1, 1, 1, 1, 1, 1)),
    ),
}


@dataclass(frozen=True)
class ResidualEvaluation:
    value: object
    components: dict
    direction_evaluations: int | None
    max_derivative_order: int


def _partials(fun: Callable, x, terms, method: Method):
    out, total = {}, 0
    for name, alpha in terms.items():
        value, meta = monomial_partial(fun, x, alpha, method=method)
        out[name] = value
        if meta.direction_count is not None:
            total += meta.direction_count
    return out, total if method == "rank_optimal" else None


def kdv2d_residual(fun: Callable, xyt, *, method: Method):
    d, total = _partials(fun, xyt, KDV2D_TERMS, method)
    r = (
        d["u_ty"]
        + d["u_xxxy"]
        + 3 * (d["u_xy"] * d["u_x"] + d["u_y"] * d["u_xx"])
        - d["u_xx"]
        + 2 * d["u_yy"]
    )
    return ResidualEvaluation(r, d, total, 4)


def gkdv1d_gradient_enhanced_loss(
    fun: Callable, xt, *, method: Method, lam1=1e-3, dispersion=0.0025
):
    d, total = _partials(fun, xt, GKDV1D_TERMS, method)
    # Vectorized model values without forcing callers to provide a batch function.
    u = jax.vmap(fun)(xt)
    f = d["u_t"] + u * d["u_x"] + dispersion * d["u_xxx"]
    fx = d["u_tx"] + d["u_x"] ** 2 + u * d["u_xx"] + dispersion * d["u_xxxx"]
    ft = d["u_tt"] + d["u_t"] * d["u_x"] + u * d["u_tx"] + dispersion * d["u_txxx"]
    loss = jnp.mean(f**2) + lam1 * (jnp.mean(fx**2) + jnp.mean(ft**2))
    return ResidualEvaluation(loss, {**d, "f": f, "f_x": fx, "f_t": ft}, total, 4)


def polyharmonic_2d_operator(fun: Callable, xy, *, order: int, method: Method):
    if order not in POLYHARMONIC_2D:
        raise ValueError("order must be 4 or 6")
    components, total, value = {}, 0, None
    for coeff, alpha in POLYHARMONIC_2D[order]:
        partial, meta = monomial_partial(fun, xy, alpha, method=method)
        components["d_" + "".join(str(i + 1) for i in alpha)] = partial
        term = coeff * partial
        value = term if value is None else value + term
        if meta.direction_count is not None:
            total += meta.direction_count
    return ResidualEvaluation(
        value, components, total if method == "rank_optimal" else None, order
    )
