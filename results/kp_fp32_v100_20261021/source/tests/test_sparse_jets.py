"""Structural support, coefficient normalization, and detached sparse evaluation."""
from dataclasses import FrozenInstanceError
from itertools import product

import pytest
import torch

from torch_pinn import stdepp
from torch_pinn.jets import mlp_jet
from torch_pinn.model import MLP, sample_normal
from torch_pinn.monomials import schedule
from torch_pinn.records import digest
from torch_pinn.sparse_jets import _tanh_sparse, compile_support, prepare_curve_evaluation

TARGETS = ('111111', '111222', '112233', '123456', '11223344', '111222333')


@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_linear_schedules_share_sparse_backend_and_match_original_jets(target, dtype):
    alpha = tuple(int(char)-1 for char in target)
    model = MLP(16, 3, 3, 61, dtype=dtype)
    points = sample_normal(62, 2, 16, dtype=dtype)
    for constructor in (schedule, stdepp.schedule):
        directions, weights = constructor(alpha, 16, dtype=dtype)
        prepared = prepare_curve_evaluation(model, points, (1,), directions.unsqueeze(0), len(alpha), weights)
        with torch.no_grad():
            reference = mlp_jet(model, points, directions, len(alpha))[-1]
        tol = 1e-4 if dtype == torch.float32 else 1e-10
        atol = 1e-6 if dtype == torch.float32 else 1e-12
        torch.testing.assert_close(prepared.raw(), reference, rtol=tol, atol=atol)
        torch.testing.assert_close(prepared(), (weights @ reference).real, rtol=tol, atol=atol)
        assert prepared.plan.orders == tuple(range(len(alpha)+1))
        assert not prepared().requires_grad


@pytest.mark.parametrize('input_orders,output_order', (((2,5), 9), ((3,7), 13), ((4,6), 16), ((5,2), 11)))
@pytest.mark.parametrize('complex_curve', (False, True))
def test_irregular_normalized_curves_match_dense_curved_jet(input_orders, output_order, complex_curve):
    model = MLP(2, 3, 3, 71, dtype=torch.float64)
    points = sample_normal(72, 2, 2, dtype=torch.float64)
    coefficients = torch.arange(1,1+len(input_orders)*2*2,dtype=torch.float64).reshape(len(input_orders),2,2)*.01
    if complex_curve:
        coefficients = coefficients.to(torch.complex128)*(1+.3j)
    prepared = prepare_curve_evaluation(model, points, input_orders, coefficients, output_order)
    dense = coefficients.new_zeros((2, output_order, 2))
    for index, order in enumerate(input_orders):
        dense[:,order-1] = coefficients[index]
    with torch.no_grad():
        reference = stdepp.curved_jet(model, points, dense)[output_order]
    torch.testing.assert_close(prepared.raw(), reference, rtol=1e-10, atol=1e-13)
    assert all(order == 0 or any(order-power in prepared.plan.orders for power in input_orders if power <= order)
               for order in prepared.plan.orders)


def test_support_is_structural_even_when_input_coefficients_numerically_zero():
    model = MLP(1, 1, 2, 81, dtype=torch.float64)
    coefficients = torch.zeros((2,1,1),dtype=torch.float64)
    prepared = prepare_curve_evaluation(model, torch.zeros(1,1,dtype=torch.float64), (4,7), coefficients, 15)
    assert prepared.plan.orders == (0,4,7,8,11,15)
    torch.testing.assert_close(prepared(), torch.zeros(1,dtype=torch.float64))
    assert prepared.metadata()['first_affine_order_count'] == 3
    assert prepared.metadata()['active_order_count'] == 6
    assert prepared.metadata()['semigroup_order_count'] == 8


def test_tanh_sparse_matches_independent_normalized_scalar_taylor_coefficients():
    # tanh(z) = z-z^3/3+2*z^5/15-17*z^7/315+62*z^9/2835+...
    # z=t^11+t^17: a non-dense semigroup with a known coefficient at order 51.
    plan = compile_support((11,17), 51)
    a = (torch.zeros(1,dtype=torch.float64), torch.ones(1,dtype=torch.float64), torch.ones(1,dtype=torch.float64))
    observed = _tanh_sparse(a, plan, first_layer=True)[-1]
    # At t^51 only z^3 contributes: choose three t^17 terms, coefficient -1/3.
    torch.testing.assert_close(observed, torch.tensor([-1/3],dtype=torch.float64), rtol=1e-12, atol=1e-13)


def test_high_order_tanh_sparse_has_no_factorial_underflow():
    plan = compile_support((71,73,79,83,103,151), 560)
    a = (torch.tensor([.2],dtype=torch.float64),)+tuple(torch.ones(1,dtype=torch.float64) for _ in range(6))
    observed = _tanh_sparse(a, plan, first_layer=True)[-1]
    # The prime schedule admits exactly one nonnegative exponent solution at 560:
    # each of the six input powers appears once, hence coefficient tanh^(6)(.2).
    value = a[0].tanh()
    polynomial = [0,1]
    for _ in range(6):
        derivative = [i*polynomial[i] for i in range(1,len(polynomial))]
        updated = [0]*(len(derivative)+2)
        for i,c in enumerate(derivative):
            updated[i] += c
            updated[i+2] -= c
        polynomial = updated
    expected = torch.zeros_like(value)
    for coefficient in reversed(polynomial):
        expected = expected*value+coefficient
    torch.testing.assert_close(observed, expected, rtol=1e-10, atol=1e-11)
    assert len(plan.orders) == 64


@pytest.mark.parametrize('powers', ((1093,1291,1321,1367,1451,1693,1973,2039),
                                    (5581,6217,6599,7577,8147,8287,8419,9539,10103)))
def test_large_prime_support_retains_subset_sums_without_dense_order_loops(powers):
    plan = compile_support(powers, sum(powers))
    subset_sums = {sum(power for power, included in zip(powers, bits) if included)
                   for bits in product((False,True),repeat=len(powers))}
    assert set(plan.orders) == subset_sums
    assert len(plan.orders) == 2**len(powers)
    assert plan.semigroup_count > len(plan.orders)
    assert len(plan.q_pairs) == len(plan.orders)-1


def test_target_pruning_preserves_all_four_variable_monomials_that_can_reach_output():
    powers, output = (2,3,5,7), 17
    retained = set(compile_support(powers,output).orders)
    # Enumerate every possible monomial in four input variables through t^17.
    # Multiply its univariate factors independently with and without pruning.
    for exponents in product(*(range(output//power+1) for power in powers)):
        dense, sparse = {0:1}, {0:1}
        for power, exponent in zip(powers,exponents):
            for _ in range(exponent):
                dense = {degree+power:coefficient for degree,coefficient in dense.items()
                         if degree+power <= output}
                sparse = {degree+power:coefficient for degree,coefficient in sparse.items()
                          if degree+power in retained}
        assert dense.get(output,0) == sparse.get(output,0)


def test_fixed_tensors_are_owned_preconverted_and_unchanged_between_calls(monkeypatch):
    model = MLP(2, 2, 2, 91)
    points = sample_normal(92, 2, 2)
    coefficients = torch.tensor([[[1+.2j, -.1j]], [[.1j, .3]]],dtype=torch.complex64)
    weights = torch.ones(1,dtype=torch.complex64)
    original_parameters = tuple(t.detach().clone() for t in model.parameters())
    prepared = prepare_curve_evaluation(model, points, (2,5), coefficients, 9, weights)
    identity, metadata = prepared.storage_identity(), prepared.metadata()
    expected = prepared()
    assert all(t.dtype == torch.complex64 and not t.requires_grad for t in prepared.tensors())
    for original, before, fixed in zip(model.parameters(), original_parameters,
                                      (t for layer in prepared.model.layers for t in (layer.weight,layer.bias))):
        torch.testing.assert_close(original, before)
        assert original.data_ptr() != fixed.data_ptr()
    def forbidden(*args, **kwargs):
        raise AssertionError('Preparation/cast API called during evaluation')
    monkeypatch.setattr(torch.Tensor,'to',forbidden)
    for _ in range(3):
        observed = prepared()
        torch.testing.assert_close(observed,expected,rtol=0,atol=0)
        assert not observed.requires_grad and observed.grad_fn is None
        assert prepared.storage_identity() == identity
    monkeypatch.undo()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
        points.add_(1)
        coefficients.add_(1)
        weights.add_(1)
    torch.testing.assert_close(prepared(), expected, rtol=0,atol=0)
    assert prepared.metadata() == metadata
    assert metadata['prepared_tensors_sha256'] == digest(prepared.tensors())
    with pytest.raises(FrozenInstanceError):
        prepared.plan = None


def test_invalid_support_and_shapes_fail_before_evaluation():
    for orders, output in (((0,),2), ((2,2),4), ((2,),3), ((True,),2), ((4,),3)):
        with pytest.raises(ValueError):
            compile_support(orders,output)
    model = MLP(2,2,2,101)
    points = torch.zeros(1,2)
    with pytest.raises(ValueError,match='Coefficient shape'):
        prepare_curve_evaluation(model,points,(1,2),torch.ones(1,1,2),4)
    with pytest.raises(ValueError,match='One weight'):
        prepare_curve_evaluation(model,points,(1,),torch.ones(1,1,2),4,torch.ones(2))
