"""Independent correctness and ownership checks for values-only optimization."""
from dataclasses import FrozenInstanceError
from math import factorial, prod

import pytest
import torch

from torch_pinn import stdepp
from torch_pinn.model import MLP, sample_normal
from torch_pinn.monomials import monomial, schedule
from torch_pinn.prepared_partials import prepare_evaluation, reduce_conjugates
from torch_pinn.records import digest

TARGETS = ('111111', '111222', '112233', '123456', '11223344', '111222333')


def alpha(target):
    return tuple(int(c)-1 for c in target)


def tanh_derivative(value, order):
    coefficients = [0, 1]
    for _ in range(order):
        derivative = [i*coefficients[i] for i in range(1, len(coefficients))]
        updated = [0]*(len(derivative)+2)
        for i, entry in enumerate(derivative):
            updated[i] += entry
            updated[i+2] -= entry
        coefficients = updated
    result = torch.zeros_like(value)
    for entry in reversed(coefficients):
        result = result*value+entry
    return result


@pytest.mark.parametrize('target,wdd_count,stde_count', zip(TARGETS, (1,3,5,32,14,10), (1,6,13,32,40,28)))
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_all_partial_values_and_reduced_counts(target, wdd_count, stde_count, dtype):
    model = MLP(16, 2, 2, 73, dtype=dtype)
    points = sample_normal(74, 2, 16, dtype=dtype)
    for name, constructor, expected_count in (('waring_batched', schedule, wdd_count), ('stdepp', stdepp.schedule, stde_count)):
        directions, weights = constructor(alpha(target), 16, dtype=dtype)
        original = stdepp.partial(model, points, alpha(target)) if name == 'stdepp' else monomial(model, points, alpha(target))
        for conjugate in (False, True):
            prepared = prepare_evaluation(model, points, directions, weights, len(target), conjugate_reuse=conjugate)
            observed = prepared()
            torch.testing.assert_close(observed, original, rtol=1e-4 if dtype == torch.float32 else 1e-8,
                                       atol=1e-6 if dtype == torch.float32 else 1e-11)
            assert len(prepared.directions) == (expected_count if conjugate else len(directions))
            assert prepared.metadata()['preconverted'] and not observed.requires_grad
            assert all(not t.requires_grad and t.dtype == prepared.points.dtype for t in prepared.tensors())


@pytest.mark.parametrize('target', TARGETS)
def test_independent_analytic_and_coordinate_jvp_float64(target):
    model = MLP(16, 1, 2, 83, dtype=torch.float64)
    points = sample_normal(84, 1, 16, dtype=torch.float64)
    a = alpha(target)
    hidden, output = model.layers
    projection = (points @ hidden.weight[0]+hidden.bias[0]).tanh()
    expected = output.weight[0,0]*tanh_derivative(projection, len(a))*prod(hidden.weight[0,i] for i in a)
    with torch.no_grad():
        nested = monomial(model, points, a, 'nested_jvp')
    torch.testing.assert_close(nested, expected, rtol=1e-8, atol=1e-11)
    for constructor in (schedule, stdepp.schedule):
        directions, weights = constructor(a, 16, dtype=torch.float64)
        prepared = prepare_evaluation(model, points, directions, weights, len(a))
        torch.testing.assert_close(prepared(), expected, rtol=1e-8, atol=1e-11)


def test_repeated_calls_make_no_fixed_tensor_cast_copies_or_graphs(monkeypatch):
    model = MLP(16, 3, 2, 93)
    points = sample_normal(94, 2, 16)
    directions, weights = schedule(alpha('112233'), 16)
    prepared = prepare_evaluation(model, points, directions, weights, 6)
    identity = prepared.storage_identity()
    expected = prepared()
    original_to, seen = torch.Tensor.to, []
    fixed_pointers = {t.data_ptr() for t in prepared.tensors()}
    def to(tensor, *args, **kwargs):
        observed = original_to(tensor, *args, **kwargs)
        if tensor.data_ptr() in fixed_pointers:
            assert observed.data_ptr() == tensor.data_ptr()
            seen.append(tensor.data_ptr())
        return observed
    monkeypatch.setattr(torch.Tensor, 'to', to)
    for _ in range(3):
        actual = prepared()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not actual.requires_grad and actual.grad_fn is None
        assert prepared.storage_identity() == identity
    assert fixed_pointers-{prepared.weights.data_ptr()} <= set(seen)


def test_snapshot_does_not_mutate_or_share_original_parameter_storage():
    model = MLP(2, 2, 2, 103)
    points = sample_normal(104, 2, 2)
    directions = torch.tensor([[1., 0.]])
    weights = torch.tensor([1.])
    before = tuple(p.detach().clone() for p in model.parameters())
    prepared = prepare_evaluation(model, points, directions, weights, 2)
    expected = prepared()
    snapshot_hash = prepared.metadata()['prepared_tensors_sha256']
    for original, saved, fixed in zip(model.parameters(), before, (t for layer in prepared.model.layers for t in (layer.weight, layer.bias))):
        torch.testing.assert_close(original, saved)
        assert original.data_ptr() != fixed.data_ptr() and original.requires_grad
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
        points.add_(1)
        directions.add_(1)
        weights.add_(1)
    torch.testing.assert_close(prepared(), expected, rtol=0, atol=0)
    assert prepared.metadata()['prepared_tensors_sha256'] == snapshot_hash == digest(prepared.tensors())
    with pytest.raises(FrozenInstanceError):
        prepared.order = 3


def test_rejects_unmatched_misweighted_and_nearby_distinct_complex_directions():
    model = MLP(1, 1, 2, 113, dtype=torch.float64)
    points = torch.zeros(1,1, dtype=torch.float64)
    def attempt(directions, weights):
        return prepare_evaluation(model, points, torch.tensor(directions, dtype=torch.complex128),
                                  torch.tensor(weights, dtype=torch.complex128), 2)
    with pytest.raises(ValueError, match='conjugate partner'):
        attempt([[1j]], [1])
    with pytest.raises(ValueError, match='inconsistent weights'):
        attempt([[1j], [-1j]], [1, 2])
    with pytest.raises(ValueError, match='nonreal weight'):
        attempt([[1]], [1j])
    with pytest.raises(ValueError, match='conjugate partner'):
        attempt([[1+1e-10j]], [1])
    with pytest.raises(ValueError, match='conjugate partner'):
        attempt([[1j], [1e-10-1j]], [1, 1])
    with pytest.raises(ValueError, match='conjugate partner'):
        attempt([[1j], [-1j], [-1j]], [1, 1, 1])


def test_rejects_complex_model_or_points_and_allows_unreduced_ablation():
    model = MLP(1, 1, 2, 123, dtype=torch.float64)
    points = torch.zeros(1,1, dtype=torch.float64)
    direction = torch.tensor([[1j]], dtype=torch.complex128)
    weight = torch.tensor([1.], dtype=torch.complex128)
    unreduced = prepare_evaluation(model, points, direction, weight, 2, conjugate_reuse=False)
    assert unreduced.metadata()['evaluated_count'] == 1
    with pytest.raises(ValueError, match='real floating-point points'):
        prepare_evaluation(model, points.to(torch.complex128), direction, weight, 2)
    with pytest.warns(UserWarning, match='Complex modules'):
        model.to(torch.complex128)
    with pytest.raises(ValueError, match='real model parameters'):
        prepare_evaluation(model, points, direction, weight, 2)


def compositions(total, count):
    if count == 1:
        yield (total,)
    else:
        for first in range(total+1):
            for rest in compositions(total-first, count-1):
                yield (first,)+rest


@pytest.mark.parametrize('target', TARGETS)
def test_reduced_contraction_identity_for_every_homogeneous_monomial(target):
    a = alpha(target)
    axes = sorted(set(a))
    exponents = tuple(a.count(axis) for axis in axes)
    for constructor in (schedule, stdepp.schedule):
        directions, weights = constructor(a, 16, dtype=torch.float64)
        reduced, coefficients, _ = reduce_conjugates(directions, weights)
        for beta in compositions(len(a), len(axes)):
            # At x=0, q_k(s)=prod_j(v_j**beta_j) for f(x)=prod_j(x_j**beta_j).
            # The prescribed mixed derivative is zero unless beta matches alpha.
            value = (coefficients*prod(reduced[:,axis]**power for axis, power in zip(axes, beta))).sum().real
            expected = prod(factorial(power) for power in exponents) if beta == exponents else 0
            torch.testing.assert_close(value, value.new_tensor(expected), rtol=1e-12, atol=1e-9)
