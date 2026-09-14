"""Real shared Taylor jets and matched AD for the two retained PDE cases.

Only equation definitions are reused from the old benchmark. No outer JIT,
STDE schedule, custom activation rule, or parameter backward is used here.
Taylor coefficients include the primal at index zero and are factorial-normalized.
"""
from math import factorial

import jax
import jax.numpy as jnp

from .models import mlp_one
from .residuals import KDV2D_TERMS, GKDV1D_TERMS

KDV_DIRECTIONS = ((1, 1, 0), (1, -1, 0), (1, 2, 0), (1, -2, 0))
KDV_TIME_DIRECTIONS = ((0, 1, 1), (0, 1, -1))
GKDV_DIRECTIONS = ((1, 0), (1, 1), (1, -1), (1, 2), (1, -2))
METHODS = ("nested_jvp", "nested_grad", "shared_jet", "termwise_jet")


def kdv_components(q, time):
    """Four order-four jets plus two order-two jets; arrays [direction, point, k]."""
    s1, s2 = q[0, :, 2] + q[1, :, 2], q[2, :, 2] + q[3, :, 2]
    return dict(u_x=(q[0, :, 1] + q[1, :, 1]) / 2,
                u_y=(q[0, :, 1] - q[1, :, 1]) / 2,
                u_xx=(4 * s1 - s2) / 3, u_yy=(s2 - s1) / 3,
                u_xy=(q[0, :, 2] - q[1, :, 2]) / 2,
                u_ty=(time[0, :, 2] - time[1, :, 2]) / 2,
                u_xxxy=4 * (q[0, :, 4] - q[1, :, 4])
                       - (q[2, :, 4] - q[3, :, 4]) / 2)


def gkdv_components(q):
    """Five real order-four jets, including the pure-x jet."""
    return dict(u=q[0, :, 0], u_x=q[0, :, 1], u_xx=2*q[0, :, 2],
                u_xxx=6*q[0, :, 3], u_xxxx=24*q[0, :, 4],
                u_t=(q[1, :, 1] - q[2, :, 1]) / 2,
                u_tx=(q[1, :, 2] - q[2, :, 2]) / 2,
                u_tt=q[1, :, 2] + q[2, :, 2] - 2*q[0, :, 2],
                u_txxx=4 * (q[1, :, 4] - q[2, :, 4])
                       - (q[3, :, 4] - q[4, :, 4]) / 2)


def assemble(case, d):
    if case == "kdv2d":
        r = d['u_ty'] + d['u_xxxy'] + 3*(d['u_xy']*d['u_x'] + d['u_y']*d['u_xx']) - d['u_xx'] + 2*d['u_yy']
        return r, dict(residual=r)
    f = d['u_t'] + d['u']*d['u_x'] + .0025*d['u_xxx']
    fx = d['u_tx'] + d['u_x']**2 + d['u']*d['u_xx'] + .0025*d['u_xxxx']
    ft = d['u_tt'] + d['u_t']*d['u_x'] + d['u']*d['u_tx'] + .0025*d['u_txxx']
    return jnp.mean(f*f) + .001*(jnp.mean(fx*fx) + jnp.mean(ft*ft)), dict(f=f, f_x=fx, f_t=ft)


def jet_batch(fun, points, directions, order):
    from jax.experimental import jet

    def one(point, direction):
        zero = jnp.zeros_like(direction)
        primal, derivatives = jet.jet(fun, (point,), ((direction,) + (zero,)*(order-1),))
        return jnp.stack((primal,) + tuple(c/factorial(k) for k, c in enumerate(derivatives, 1)))

    return jax.vmap(lambda v: jax.vmap(lambda x: one(x, v))(points))(directions)


def derivative_chain(fun, basis, axis, count, method):
    """Return highest derivative and lower primal derivatives as auxiliary data.

    has_aux retains the primal values produced by each AD call without
    differentiating every lower output again. No full high-order tensor is built.
    """
    current = lambda x: (fun(x), ())
    for _ in range(count):
        previous = current
        if method == "nested_jvp":
            def current(x, previous=previous):
                value, tangent, aux = jax.jvp(previous, (x,), (basis[axis],), has_aux=True)
                return tangent, (*aux, value)
        else:
            def current(x, previous=previous):
                (value, aux), gradient = jax.value_and_grad(previous, has_aux=True)(x)
                return gradient[axis], (*aux, value)
    return current


def selected_derivatives(fun, point, basis, axes, method):
    if method == "nested_grad":
        (value, aux), gradient = jax.value_and_grad(fun, has_aux=True)(point)
        return value, tuple(gradient[i] for i in axes), aux
    if len(axes) == 1:
        value, tangent, aux = jax.jvp(fun, (point,), (basis[axes[0]],), has_aux=True)
        return value, (tangent,), aux
    directions = jnp.stack(tuple(basis[i] for i in axes))
    values, tangents, aux = jax.vmap(
        lambda v: jax.jvp(fun, (point,), (v,), has_aux=True))(directions)
    return values[0], tuple(tangents[i] for i in range(len(axes))), tuple(a[0] for a in aux)


def ad_components(fun, point, basis, case, method):
    x3 = derivative_chain(fun, basis, 0, 3, method)
    if case == "kdv2d":
        _, (uxxxy,), (_, ux, uxx) = selected_derivatives(x3, point, basis, (1,), method)
        y1 = derivative_chain(fun, basis, 1, 1, method)
        uy, (uxy, uyy, uty), _ = selected_derivatives(y1, point, basis, (0, 1, 2), method)
        return dict(u_x=ux, u_y=uy, u_xx=uxx, u_xy=uxy, u_yy=uyy, u_ty=uty, u_xxxy=uxxxy)
    uxxx, (uxxxx, utxxx), (u, ux, uxx) = selected_derivatives(x3, point, basis, (0, 1), method)
    t1 = derivative_chain(fun, basis, 1, 1, method)
    ut, (utx, utt), _ = selected_derivatives(t1, point, basis, (0, 1), method)
    return dict(u=u, u_x=ux, u_t=ut, u_xx=uxx, u_tx=utx, u_tt=utt,
                u_xxx=uxxx, u_xxxx=uxxxx, u_txxx=utxxx)


def real_term_schedule(alpha, dim):
    """Termwise ablation using the same real quartic formula, not complex roots."""
    if len(set(alpha)) == 1:
        v = tuple(int(i == alpha[0]) for i in range(dim))
        return (v,), (factorial(len(alpha)),)
    if len(alpha) == 2:
        i, j = sorted(alpha)
        return tuple(tuple(int(k == i) + a*int(k == j) for k in range(dim)) for a in (1, -1)), (.5, -.5)
    if len(alpha) == 4 and alpha.count(0) == 3:
        other = next(i for i in alpha if i != 0)
        return tuple(tuple(int(k == 0) + a*int(k == other) for k in range(dim)) for a in (1, -1, 2, -2)), (4., -4., -.5, .5)
    raise ValueError(f"Unsupported PDE term: {alpha}")


def make_pde_evaluator(case, method, *, components=False):
    if case not in ("kdv2d", "gkdv1d") or method not in METHODS:
        raise ValueError((case, method))
    dim = 3 if case == "kdv2d" else 2
    basis = jnp.eye(dim, dtype=jnp.float32)
    qdirs = jnp.asarray(KDV_DIRECTIONS if dim == 3 else GKDV_DIRECTIONS, dtype=jnp.float32)
    tdirs = jnp.asarray(KDV_TIME_DIRECTIONS, dtype=jnp.float32)
    terms = KDV2D_TERMS if dim == 3 else GKDV1D_TERMS
    schedules = {}
    if method == "termwise_jet":
        for name, alpha in terms.items():
            directions, weights = real_term_schedule(alpha, dim)
            schedules[name] = (jnp.asarray(directions, dtype=jnp.float32), jnp.asarray(weights, dtype=jnp.float32), len(alpha))

    def evaluate(params, points):
        fun = lambda x: mlp_one(params, x)
        if method in ("nested_jvp", "nested_grad"):
            d = jax.vmap(lambda x: ad_components(fun, x, basis, case, method))(points)
        elif method == "shared_jet":
            q = jet_batch(fun, points, qdirs, 4)
            d = kdv_components(q, jet_batch(fun, points, tdirs, 2)) if dim == 3 else gkdv_components(q)
        else:
            d = {name: jnp.sum(weights[:, None]*jet_batch(fun, points, directions, order)[:, :, order], axis=0)
                 for name, (directions, weights, order) in schedules.items()}
            if dim == 2:
                d['u'] = jax.vmap(fun)(points)
        value, residual_components = assemble(case, d)
        return (value, {**d, **residual_components}) if components else value
    return evaluate


def schedule_metadata(case, method):
    return dict(direction_calls=(6 if case == 'kdv2d' else 5) if method == 'shared_jet' else (12 if method == 'termwise_jet' else None),
                jets_by_order=({'4': 4, '2': 2} if case == 'kdv2d' else {'4': 5}) if method == 'shared_jet' else None,
                cross_term_jet_reuse=method == 'shared_jet',
                extra_primal_network_call=case == 'gkdv1d' and method == 'termwise_jet',
                ad_primal_reuse=method in ('nested_jvp', 'nested_grad'),
                task='residual_vector' if case == 'kdv2d' else 'gradient_enhanced_objective_value')
