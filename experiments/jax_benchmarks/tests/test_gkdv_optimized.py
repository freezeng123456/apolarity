from fractions import Fraction
from itertools import product
from math import factorial, prod
import inspect
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from jax_apolarity_bench import gkdv_optimized
from jax_apolarity_bench.gkdv_optimized import (
    DIRECTIONS, FIRST_NUMERATORS, SECOND_NUMERATORS, FOURTH_NUMERATORS,
    DENOMINATORS, METHODS, make_gkdv_evaluator,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pde_benchmark import independent_reference, inputs


def test_raw_reconstruction_exact():
    targets = ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (4, 0), (3, 1), (3, 0), (0, 0))
    rows = list(FIRST_NUMERATORS) + list(SECOND_NUMERATORS) + list(FOURTH_NUMERATORS) + [(1, 0, 0, 0, 0)]*2
    divisors = [DENOMINATORS[0]]*2 + [DENOMINATORS[1]]*3 + [DENOMINATORS[2]]*2 + [1, 1]
    for alpha in product(range(5), repeat=2):
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
def test_components_and_objective(method):
    for seed in (71, 72):
        params, points = inputs(seed, 'gkdv1d', 3, 4, 2)
        expected, reference = independent_reference('gkdv1d', params, points)
        value, components = make_gkdv_evaluator(method, components=True)(params, points)
        assert set(components) == set(reference)
        for name in reference:
            np.testing.assert_allclose(components[name], reference[name], rtol=1e-3, atol=1e-6)
            assert np.asarray(components[name]).dtype == np.float32
        np.testing.assert_allclose(value, expected, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(make_gkdv_evaluator(method)(params, points), value, rtol=1e-6)
        assert np.asarray(value).shape == () and np.asarray(value).dtype == np.float32


def test_invalid_method_and_no_outer_jit():
    assert 'jax.jit(' not in inspect.getsource(gkdv_optimized)
    with pytest.raises(ValueError):
        make_gkdv_evaluator('nested_grad')
