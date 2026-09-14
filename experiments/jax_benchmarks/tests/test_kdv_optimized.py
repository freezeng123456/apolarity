from fractions import Fraction
from itertools import product
from math import factorial, prod

import jax
import numpy as np
import pytest

from jax_apolarity_bench.models import init_mlp, mlp_one, sample_normal
from jax_apolarity_bench.residuals import KDV2D_TERMS
from jax_apolarity_bench.kdv_optimized import (
    DIRECTIONS, FIRST_NUMERATORS, SECOND_NUMERATORS, FOURTH_NUMERATORS,
    DENOMINATORS, METHODS, make_kdv_evaluator,
)
from test_pde_shared import direct_partial


def test_raw_reconstruction_exact():
    targets = ((1, 0, 0), (0, 1, 0), (2, 0, 0), (0, 2, 0),
               (1, 1, 0), (0, 1, 1), (3, 1, 0))
    rows = list(FIRST_NUMERATORS) + list(SECOND_NUMERATORS) + [FOURTH_NUMERATORS]
    divisors = [DENOMINATORS[0]]*2 + [DENOMINATORS[1]]*4 + [DENOMINATORS[2]]
    for alpha in product(range(5), repeat=3):
        if sum(alpha) > 4:
            continue
        for target, row, divisor in zip(targets, rows, divisors):
            order = sum(target)
            values = [factorial(order)*prod(v**a for v, a in zip(direction, alpha))
                      if sum(alpha) == order else 0 for direction in DIRECTIONS]
            actual = sum(Fraction(c, divisor)*v for c, v in zip(row, values))
            expected = prod(factorial(a) for a in alpha) if alpha == target else 0
            assert actual == expected, (alpha, target)


@pytest.mark.parametrize('method', METHODS)
def test_components_and_values(method):
    for seed in (71, 72):
        params = init_mlp(seed, 3, 4, 2)
        points = sample_normal(seed + 1, 3, 3)
        fun = lambda z: mlp_one(params, z)
        expected = {name: jax.vmap(lambda z: direct_partial(fun, z, alpha))(points)
                    for name, alpha in KDV2D_TERMS.items()}
        d = expected
        residual = d['u_ty']+d['u_xxxy']+3*(d['u_xy']*d['u_x']+d['u_y']*d['u_xx'])-d['u_xx']+2*d['u_yy']
        value, components = make_kdv_evaluator(method, components=True)(params, points)
        expected['residual'] = residual
        for name in expected:
            np.testing.assert_allclose(components[name], expected[name], rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(value, residual, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(make_kdv_evaluator(method)(params, points), value, rtol=1e-6)
        assert np.asarray(value).dtype == np.float32


def test_invalid_method_and_no_outer_jit():
    import inspect
    from jax_apolarity_bench import kdv_optimized
    assert 'jax.jit(' not in inspect.getsource(kdv_optimized)
    with pytest.raises(ValueError):
        make_kdv_evaluator('nested_grad')
