"""Fixed linear reconstruction for the existing five-direction g-KdV schedule."""
import jax
import jax.numpy as jnp
from jax.experimental import jet

from .models import mlp_one
from .pde_shared import GKDV_DIRECTIONS, assemble, make_pde_evaluator

METHODS = ('nested_jvp', 'shared_jet', 'shared_jet_linear')
DIRECTIONS = GKDV_DIRECTIONS
FIRST_NUMERATORS = ((2, 0, 0, 0, 0), (0, 1, -1, 0, 0))
SECOND_NUMERATORS = ((4, 0, 0, 0, 0), (0, 1, -1, 0, 0), (-4, 2, 2, 0, 0))
FOURTH_NUMERATORS = ((48, 0, 0, 0, 0), (0, 8, -8, -1, 1))
DENOMINATORS = (2, 4, 48)


def raw_jet_batch(fun, points, directions):
    """Keep primal and raw directional derivatives as a tuple, without rescaling."""
    def one(x, v):
        zero = jnp.zeros_like(v)
        primal, derivatives = jet.jet(fun, (x,), ((v, zero, zero, zero),))
        return primal, derivatives[0], derivatives[1], derivatives[2], derivatives[3]
    return jax.vmap(lambda v: jax.vmap(lambda x: one(x, v))(points))(directions)


def make_gkdv_evaluator(method, *, components=False):
    if method not in METHODS:
        raise ValueError(method)
    if method in ('nested_jvp', 'shared_jet'):
        return make_pde_evaluator('gkdv1d', method, components=components)
    directions = jnp.asarray(DIRECTIONS, dtype=jnp.float32)
    weights = tuple(jnp.asarray(n, dtype=jnp.float32)/d for n, d in zip(
        (FIRST_NUMERATORS, SECOND_NUMERATORS, FOURTH_NUMERATORS), DENOMINATORS))

    def evaluate(params, points):
        fun = lambda x: mlp_one(params, x)
        primal, c1, c2, c3, c4 = raw_jet_batch(fun, points, directions)
        first, second, fourth = weights[0] @ c1, weights[1] @ c2, weights[2] @ c4
        d = dict(u=primal[0], u_x=first[0], u_t=first[1],
                 u_xx=second[0], u_tx=second[1], u_tt=second[2],
                 u_xxx=c3[0], u_xxxx=fourth[0], u_txxx=fourth[1])
        value, residual = assemble('gkdv1d', d)
        return (value, {**d, **residual}) if components else value
    return evaluate
