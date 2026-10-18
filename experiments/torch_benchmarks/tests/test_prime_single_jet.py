"""Independent exact checks for the STDE++ prime-single-jet preparation."""
from collections import Counter
from math import factorial, prod

import pytest
import torch

from torch_pinn.prime_single_jet import (
    EXTENDED_PRIMES, PAPER_PRIMES, prime_schedule, search_prime_orders, verify_unique_orders)


def independent_unbounded_count(orders):
    # Deliberately scalar Python, distinct from residue-column NumPy search DP.
    limit = sum(orders)
    ways = [0]*(limit+1)
    ways[0] = 1
    for order in orders:
        for degree in range(order, limit+1):
            ways[degree] = min(2, ways[degree]+ways[degree-order])
    return ways[-1]


@pytest.mark.parametrize('k', range(3, 10))
def test_published_and_extended_primes_have_exact_unique_unbounded_solution(k):
    orders = (PAPER_PRIMES | EXTENDED_PRIMES)[k]
    assert independent_unbounded_count(orders) == 1
    proof = verify_unique_orders(orders)
    assert proof['solution_count'] == 1 and proof['unique_multiplicities'] == [1]*k
    assert proof['subset_sum_count'] == 2**k


def test_unbounded_uniqueness_does_not_only_check_distinct_subset_sums():
    # 2+3+5=2*5 already supplies a second unbounded multiplicity solution.
    proof = verify_unique_orders((2, 3, 5))
    assert proof['subset_sum_count'] < 8
    assert proof['solution_count'] == 2 and not proof['unique']
    # Distinct subset sums alone are not sufficient either: {3,5,7} has eight
    # different subset sums, but total15 is also five3's instead of3+5+7.
    proof = verify_unique_orders((3, 5, 7))
    assert proof['subset_sum_count'] == 8
    assert proof['solution_count'] == 2 and not proof['unique']


@pytest.mark.parametrize('pattern,expected_order', (
    ('111111', 560), ('111222', 560), ('112233', 560), ('123456', 560),
    ('11223344', 12228), ('111222333', 70469)))
def test_every_occurrence_uses_a_distinct_prime_and_sparse_unit_coefficients(pattern, expected_order):
    alpha = tuple(int(c)-1 for c in pattern)
    plan = prime_schedule(alpha, 16)
    assert len(plan.orders) == len(alpha) and len(set(plan.orders)) == len(alpha)
    assert plan.output_order == expected_order
    assert plan.coefficients.shape == (len(alpha), 16)
    assert plan.coefficients.dtype == torch.float32
    assert int(torch.count_nonzero(plan.coefficients)) == len(alpha)
    for row, axis in enumerate(alpha):
        assert plan.coefficients[row, axis] == 1
        assert plan.coefficients[row].sum() == 1
    assert plan.metadata['curve_count'] == 1
    assert not plan.metadata['floating_point_factorials_formed']


def formal_product(first, second, limit):
    result = {}
    for a, x in first.items():
        for b, y in second.items():
            if a+b <= limit:
                result[a+b] = result.get(a+b, 0)+x*y
    return result


@pytest.mark.parametrize('pattern', ('111111', '111222', '112233', '123456', '11223344', '111222333'))
def test_unit_curve_normalization_on_independent_exact_formal_monomials(pattern):
    alpha = tuple(int(c)-1 for c in pattern)
    plan = prime_schedule(alpha, max(alpha)+2, dtype=torch.float64)
    by_axis = {}
    for prime, axis in zip(plan.orders, alpha):
        by_axis.setdefault(axis, {})[prime] = 1
    counts = Counter(alpha)
    coefficient = {0: 1}
    for axis in alpha:
        coefficient = formal_product(coefficient, by_axis[axis], plan.output_order)
    # At the origin u(x)=product(x_j**nu_j) has derivative product(nu_j!).
    # This directly verifies extraction with ordinary input coefficients one,
    # rather than factorial-underflowed raw unit tangent coefficients 1/p!.
    assert coefficient.get(plan.output_order, 0) == prod(factorial(n) for n in counts.values())
    # An unrelated coordinate absent from the input curve cannot contribute.
    wrong = formal_product(coefficient, {}, plan.output_order)
    assert wrong.get(plan.output_order, 0) == 0


def tanh_derivative_polynomial(value, order):
    coefficients = [0, 1]
    for _ in range(order):
        derivative = [i*coefficients[i] for i in range(1, len(coefficients))]
        following = [0]*(len(derivative)+2)
        for i, coefficient in enumerate(derivative):
            following[i] += coefficient
            following[i+2] -= coefficient
        coefficients = following
    result = torch.zeros_like(value)
    for coefficient in reversed(coefficients):
        result = result*value+coefficient
    return result


@pytest.mark.parametrize('pattern', ('111111', '111222', '112233', '123456', '11223344', '111222333'))
def test_common_sparse_backend_prime_values_match_analytic_and_coordinate_jvp(pattern):
    from torch_pinn.coordinate_jvp import partial as coordinate_partial
    from torch_pinn.model import MLP, sample_normal
    from torch_pinn.sparse_jets import prepare_curve_evaluation
    alpha = tuple(int(c)-1 for c in pattern)
    model = MLP(16, 1, 2, 20261018, dtype=torch.float64)
    points = sample_normal(20261019, 1, 16, dtype=torch.float64)
    prime = prime_schedule(alpha, 16, dtype=torch.float64)
    prepared = prepare_curve_evaluation(model, points, prime.orders,
                                        prime.coefficients.unsqueeze(1), prime.output_order)
    with torch.no_grad():
        actual = prepared()
        direct = coordinate_partial(model, points[0], alpha).reshape(1)
        first, last = model.layers
        base = (points @ first.weight[0]+first.bias[0]).tanh()
        analytic = (last.weight[0, 0]*tanh_derivative_polynomial(base, len(alpha))
                    *prod(first.weight[0, axis] for axis in alpha))
    torch.testing.assert_close(actual, analytic, rtol=1e-8, atol=1e-11)
    torch.testing.assert_close(actual, direct, rtol=1e-8, atol=1e-11)
    assert prepared.metadata()['active_order_count'] == 2**len(alpha)
    assert not actual.requires_grad


@pytest.mark.parametrize('k', (8, 9))
def test_seeded_rejection_reproduces_the_frozen_extended_prime_lists(k):
    observed = search_prime_orders(k)
    expected = prime_schedule(tuple(range(k)), k).metadata['precomputation']
    assert tuple(observed['orders']) == EXTENDED_PRIMES[k]
    assert observed['seed'] == expected['seed']
    assert observed['attempts'] == expected['attempts']
    assert observed['search_intervals'] == expected['search_intervals']
    assert observed['precompute_wall_seconds'] >= 0
    assert not observed['globally_minimal']


@pytest.mark.parametrize('orders', ((), (3, 3), (5, 3), (4, 7), (True, 3)))
def test_invalid_prime_lists(orders):
    with pytest.raises(ValueError):
        verify_unique_orders(orders)


@pytest.mark.parametrize('alpha,dim', (((), 2), ((-1,), 2), ((3,), 3), ((True,), 2)))
def test_invalid_partial_indices(alpha, dim):
    with pytest.raises(ValueError):
        prime_schedule(alpha, dim)
