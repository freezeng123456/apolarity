"""Independent algebra, derivative, and full-objective checks for STDE++ port."""
import copy
from itertools import product
from math import factorial, prod

import pytest
import torch

from torch_pinn import problem, problem_ch2d, problem_kdv2d, stdepp
from torch_pinn.coordinate_jvp import partial as coordinate_partial
from torch_pinn.model import MLP, sample_normal

PATTERNS = ('111111', '111222', '112233', '123456', '11223344', '111222333')


def parse(pattern):
    return tuple(int(c)-1 for c in pattern)


def compositions(total, count):
    if count == 1:
        yield (total,)
    else:
        for first in range(total+1):
            for rest in compositions(total-first, count-1):
                yield (first,)+rest


@pytest.mark.parametrize('pattern', PATTERNS)
def test_exact_integer_polarization_identity_for_every_homogeneous_monomial(pattern):
    alpha = parse(pattern)
    axes, pairs, denominator = stdepp._grouped_schedule(alpha)
    target = tuple(alpha.count(i) for i in axes)
    for beta in compositions(len(alpha), len(axes)):
        numerator = sum(weight*prod(v**power for v, power in zip(direction, beta))
                        for direction, weight in pairs)
        expected = denominator*prod(factorial(a) for a in target) if beta == target else 0
        assert numerator == expected


@pytest.mark.parametrize('pattern,count', zip(PATTERNS, (1, 6, 13, 32, 40, 28)))
def test_grouped_direction_counts_and_real_metadata(pattern, count):
    alpha = parse(pattern)
    directions, weights = stdepp.schedule(alpha, 16, dtype=torch.float64)
    assert directions.shape == (count, 16)
    assert not directions.is_complex() and not weights.is_complex()
    record = stdepp.metadata(alpha)
    assert record['direction_count'] == count
    assert not record['polynomial_prime_construction']


def tanh_derivative(value, order):
    # Independent exact polynomial recurrence P_{n+1}(t)=(1-t²)P_n'(t).
    coefficients = [0, 1]
    for _ in range(order):
        derivative = [i*coefficients[i] for i in range(1, len(coefficients))]
        out = [0]*(len(derivative)+2)
        for i, coefficient in enumerate(derivative):
            out[i] += coefficient
            out[i+2] -= coefficient
        coefficients = out
    result = torch.zeros_like(value)
    for coefficient in reversed(coefficients):
        result = result*value+coefficient
    return result


@pytest.mark.parametrize('pattern', PATTERNS)
def test_one_neuron_analytic_value_and_parameter_gradient(pattern):
    alpha = parse(pattern)
    dim = max(alpha)+1
    model = MLP(dim, 1, 2, 73, dtype=torch.float64)
    points = sample_normal(74, 3, dim, dtype=torch.float64)
    actual = stdepp.partial(model, points, alpha)
    hidden, output = model.layers
    projection = (points @ hidden.weight[0]+hidden.bias[0]).tanh()
    expected = output.weight[0, 0]*tanh_derivative(projection, len(alpha))
    expected = expected*prod(hidden.weight[0, axis] for axis in alpha)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-11)
    parameters = tuple(model.parameters())
    a = torch.autograd.grad(actual.square().sum(), parameters, allow_unused=True)
    b = torch.autograd.grad(expected.square().sum(), parameters, allow_unused=True)
    for parameter, actual_grad, expected_grad in zip(parameters, a, b):
        actual_grad = torch.zeros_like(parameter) if actual_grad is None else actual_grad
        expected_grad = torch.zeros_like(parameter) if expected_grad is None else expected_grad
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-7, atol=1e-10)


def test_curved_jet_normalization_against_independent_scalar_jvp():
    model = MLP(2, 2, 2, 81, dtype=torch.float64)
    points = sample_normal(82, 1, 2, dtype=torch.float64)
    coefficients = stdepp._raw_curves(points, 7, ({2: (1, 0), 3: (0, 1)},))
    q = stdepp.curved_jet(model, points, coefficients)
    def composition(s):
        offset = sum(coefficients[0, i]*s[0]**(i+1) for i in range(7))
        return model(points[0]+offset)
    zero = points.new_zeros(1)
    for order in range(8):
        direct = coordinate_partial(composition, zero, (0,)*order)
        torch.testing.assert_close(q[order][0, 0], direct/factorial(order), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('case,module', (('kdv1d', problem), ('kdv2d', problem_kdv2d), ('ch2d', problem_ch2d)))
def test_full_pde_objective_parameter_gradient_and_adam(case, module):
    dim = 2 if case == 'kdv1d' else 3
    model = MLP(dim, 3, 2, 91, dtype=torch.float64)
    other = copy.deepcopy(model)
    points = sample_normal(92, 2, dim, dtype=torch.float64)
    data = module.constraints(2, dtype=torch.float64)
    actual, parts = stdepp.pde_loss(case, model, points, data)
    expected, expected_parts = module.loss(other, points, data, 'nested_jvp')
    assert parts.keys() == expected_parts.keys()
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
    for key in parts:
        torch.testing.assert_close(parts[key], expected_parts[key], rtol=1e-9, atol=1e-11)
    actual.backward()
    expected.backward()
    for p, q in zip(model.parameters(), other.parameters()):
        a = torch.zeros_like(p) if p.grad is None else p.grad
        b = torch.zeros_like(q) if q.grad is None else q.grad
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
    torch.optim.Adam(model.parameters(), lr=1e-4).step()
    torch.optim.Adam(other.parameters(), lr=1e-4).step()
    for p, q in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-8, atol=1e-10)


def test_invalid_coordinate_schedule():
    for alpha in ((), (-1,), (True,)):
        with pytest.raises(ValueError):
            stdepp.schedule(alpha, 3)
    with pytest.raises(ValueError):
        stdepp.schedule((3,), 3)
