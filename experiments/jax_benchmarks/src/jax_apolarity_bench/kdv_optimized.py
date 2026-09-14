"""Two bounded KdV scheduling optimizations; the original baseline is unchanged."""
import jax
import jax.numpy as jnp
from jax.experimental import jet

from .models import mlp_one
from .pde_shared import (
    KDV_DIRECTIONS, KDV_TIME_DIRECTIONS, assemble, jet_batch,
    kdv_components, make_pde_evaluator,
)

METHODS = ('nested_jvp', 'shared_jet', 'shared_jet_fused', 'shared_jet_linear')
DIRECTIONS = KDV_DIRECTIONS + KDV_TIME_DIRECTIONS
# Exact rational reconstruction of unnormalized directional derivatives C_k.
FIRST_NUMERATORS = ((1, 1, 0, 0, 0, 0), (1, -1, 0, 0, 0, 0))
SECOND_NUMERATORS = ((8, 8, -2, -2, 0, 0), (-2, -2, 2, 2, 0, 0),
                     (3, -3, 0, 0, 0, 0), (0, 0, 0, 0, 3, -3))
FOURTH_NUMERATORS = (8, -8, -1, 1, 0, 0)
DENOMINATORS = (2, 12, 48)


def raw_jet_batch(fun, points, directions):
    """Return only required orders as a tuple, without factorial scaling/stacking."""
    def one(x, v):
        zero = jnp.zeros_like(v)
        _, derivatives = jet.jet(fun, (x,), ((v, zero, zero, zero),))
        return derivatives[0], derivatives[1], derivatives[3]
    return jax.vmap(lambda v: jax.vmap(lambda x: one(x, v))(points))(directions)


def make_kdv_evaluator(method, *, components=False):
    if method not in METHODS:
        raise ValueError(method)
    if method in ('nested_jvp', 'shared_jet'):
        return make_pde_evaluator('kdv2d', method, components=components)
    directions = jnp.asarray(DIRECTIONS, dtype=jnp.float32)
    weights = tuple(jnp.asarray(n, dtype=jnp.float32)/d for n, d in zip(
        (FIRST_NUMERATORS, SECOND_NUMERATORS, FOURTH_NUMERATORS), DENOMINATORS))

    def evaluate(params, points):
        fun = lambda x: mlp_one(params, x)
        if method == 'shared_jet_fused':
            q = jet_batch(fun, points, directions, 4)
            d = kdv_components(q, q[4:])
        else:
            c1, c2, c4 = raw_jet_batch(fun, points, directions)
            first, second, fourth = weights[0] @ c1, weights[1] @ c2, weights[2] @ c4
            d = dict(u_x=first[0], u_y=first[1], u_xx=second[0], u_yy=second[1],
                     u_xy=second[2], u_ty=second[3], u_xxxy=fourth)
        value, residual = assemble('kdv2d', d)
        return (value, {**d, **residual}) if components else value
    return evaluate
