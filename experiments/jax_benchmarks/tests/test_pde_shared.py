from fractions import Fraction
from itertools import product
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_apolarity_bench.models import init_mlp, mlp_one, sample_normal
from jax_apolarity_bench.residuals import KDV2D_TERMS, GKDV1D_TERMS
from jax_apolarity_bench.pde_shared import (
    KDV_DIRECTIONS, KDV_TIME_DIRECTIONS, GKDV_DIRECTIONS,
    METHODS, kdv_components, gkdv_components, make_pde_evaluator,
    schedule_metadata, ad_components,
)


@pytest.mark.parametrize('case', ['kdv2d', 'gkdv1d'])
def test_exact_shared_coefficients(case):
    dim = 3 if case == 'kdv2d' else 2
    targets = dict(KDV2D_TERMS if dim == 3 else GKDV1D_TERMS)
    if dim == 2:
        targets['u'] = ()
    for alpha in product(range(5), repeat=dim):
        if sum(alpha) > 4:
            continue
        def jets(directions, order):
            a = np.full((len(directions), 1, order+1), Fraction(0), dtype=object)
            if sum(alpha) <= order:
                for i, v in enumerate(directions):
                    value = 1
                    for vi, ai in zip(v, alpha):
                        value *= vi**ai
                    a[i, 0, sum(alpha)] = Fraction(value)
            return a
        if dim == 3:
            got = kdv_components(jets(KDV_DIRECTIONS, 4), jets(KDV_TIME_DIRECTIONS, 2))
        else:
            got = gkdv_components(jets(GKDV_DIRECTIONS, 4))
        for name, indices in targets.items():
            powers = tuple(indices.count(i) for i in range(dim))
            expected = np.prod([factorial(i) for i in alpha]) if powers == alpha else 0
            assert got[name][0] == expected, (case, alpha, name, got[name])


def direct_partial(fun, point, indices):
    basis = jnp.eye(point.shape[-1], dtype=point.dtype)
    current = fun
    for index in indices:
        previous = current
        current = lambda x, previous=previous, index=index: jax.jvp(previous, (x,), (basis[index],))[1]
    return current(point)


@pytest.mark.parametrize('case', ['kdv2d', 'gkdv1d'])
def test_network_components_and_assembled_values(case):
    dim = 3 if case == 'kdv2d' else 2
    terms = KDV2D_TERMS if dim == 3 else GKDV1D_TERMS
    points = sample_normal(73, 3, dim)
    for seed in (71, 72):
        params = init_mlp(seed, dim, 4, 2)
        f = lambda z: mlp_one(params, z)
        expected = {name: jax.vmap(lambda z: direct_partial(f, z, alpha))(points) for name, alpha in terms.items()}
        if dim == 3:
            d = expected
            value = d['u_ty'] + d['u_xxxy'] + 3*(d['u_xy']*d['u_x'] + d['u_y']*d['u_xx']) - d['u_xx'] + 2*d['u_yy']
            expected['residual'] = value
        else:
            expected['u'] = jax.vmap(f)(points)
            def residual(z):
                return direct_partial(f, z, (1,)) + f(z)*direct_partial(f, z, (0,)) + .0025*direct_partial(f, z, (0, 0, 0))
            expected['f'] = jax.vmap(residual)(points)
            gradient = jax.vmap(jax.grad(residual))(points)
            expected['f_x'], expected['f_t'] = gradient[:, 0], gradient[:, 1]
            value = jnp.mean(expected['f']**2) + .001*jnp.mean(expected['f_x']**2 + expected['f_t']**2)
        for method in METHODS:
            actual, components = make_pde_evaluator(case, method, components=True)(params, points)
            for name, e in expected.items():
                np.testing.assert_allclose(components[name], e, rtol=1e-3, atol=1e-6, err_msg=f'{case}/{method}/{name}')
            np.testing.assert_allclose(actual, value, rtol=1e-3, atol=1e-6)
            np.testing.assert_allclose(make_pde_evaluator(case, method)(params, points), actual, rtol=1e-6)
            assert np.asarray(actual).dtype == np.float32


def test_metadata_and_no_outer_jit():
    import inspect
    from jax_apolarity_bench import pde_shared
    assert 'jax.jit(' not in inspect.getsource(pde_shared)
    assert schedule_metadata('kdv2d', 'shared_jet')['direction_calls'] == 6
    assert schedule_metadata('gkdv1d', 'shared_jet')['direction_calls'] == 5
    assert schedule_metadata('gkdv1d', 'termwise_jet')['extra_primal_network_call']
    with pytest.raises(ValueError):
        make_pde_evaluator('poly4', 'shared_jet')
