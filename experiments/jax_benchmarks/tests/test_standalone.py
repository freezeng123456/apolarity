from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from jax_apolarity_bench.derivatives import monomial_partial
from jax_apolarity_bench.directions import (
    parse_one_based_pattern,
    rank_optimal_schedule,
)
from jax_apolarity_bench.residuals import (
    gkdv1d_gradient_enhanced_loss,
    kdv2d_residual,
    polyharmonic_2d_operator,
)


def _rel(a, b):
    aa = np.asarray(a)
    bb = np.asarray(b)
    return np.linalg.norm((aa - bb).ravel()) / max(np.linalg.norm(bb.ravel()), 1e-30)


def test_direction_counts_and_square_free_real():
    expected = {"112233": 9, "111222333": 16, "11223344": 27, "123456": 32}
    for name, rank in expected.items():
        s = rank_optimal_schedule(
            parse_one_based_pattern(name), 16, real_dtype=jnp.float32
        )
        assert s.theoretical_rank == rank
    s = rank_optimal_schedule(
        parse_one_based_pattern("123456"), 16, real_dtype=jnp.float32
    )
    assert s.directions.dtype == jnp.float32 and not s.requires_complex


def test_derivative_and_parameter_gradient_order_nine():
    alpha = parse_one_based_pattern("111222333")
    x = jnp.asarray([[0.1, 0.2, 0.3]], dtype=jnp.float32)
    a = jnp.asarray([0.4, 0.5, 0.6], dtype=jnp.float32)

    def value(method, a):
        f = lambda z: jnp.sinh(jnp.dot(a.astype(z.dtype), z) + 0.13)
        return monomial_partial(f, x, alpha, method=method)[0]

    n = value("nested_ad", a)
    r = value("rank_optimal", a)
    assert _rel(r, n) < 1e-4
    gn = jax.grad(lambda q: jnp.mean(value("nested_ad", q) ** 2))(a)
    gr = jax.grad(lambda q: jnp.mean(value("rank_optimal", q) ** 2))(a)
    assert _rel(gr, gn) < 1e-3


def test_pde_operators_match_nested():
    cases = [
        ("kdv", jnp.asarray([[0.1, -0.2, 0.3]], dtype=jnp.float32)),
        ("gkdv", jnp.asarray([[0.1, 0.3]], dtype=jnp.float32)),
        ("poly4", jnp.asarray([[0.1, -0.2]], dtype=jnp.float32)),
        ("poly6", jnp.asarray([[0.1, -0.2]], dtype=jnp.float32)),
    ]
    for kind, x in cases:
        a = jnp.linspace(0.35, 0.45, x.shape[-1], dtype=jnp.float32)
        act = jnp.sin if kind.startswith("poly") else jnp.sinh
        f = lambda z: act(jnp.dot(a.astype(z.dtype), z) + 0.13)
        if kind == "kdv":
            n = kdv2d_residual(f, x, method="nested_ad").value
            r = kdv2d_residual(f, x, method="rank_optimal").value
        elif kind == "gkdv":
            n = jnp.atleast_1d(
                gkdv1d_gradient_enhanced_loss(f, x, method="nested_ad").value
            )
            r = jnp.atleast_1d(
                gkdv1d_gradient_enhanced_loss(f, x, method="rank_optimal").value
            )
        else:
            order = 4 if kind == "poly4" else 6
            n = polyharmonic_2d_operator(f, x, order=order, method="nested_ad").value
            r = polyharmonic_2d_operator(f, x, order=order, method="rank_optimal").value
        assert _rel(r, n) < 3e-3
