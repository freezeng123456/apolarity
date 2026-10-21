"""Values-only Taylor arithmetic that omits only exact structural zeros.

The input curve is x + sum_r c_r t**r, with ordinary normalized coefficients.
Support and all recurrence index pairs are compiled before evaluation. The
same batched affine contractions and tanh recurrence serve linear Waring jets,
polarization jets and prime-polynomial jets; no values determine sparsity.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import torch
from torch.nn import functional as F

from .prepared_partials import _Layer, _Model
from .records import digest


@dataclass(frozen=True)
class SupportPlan:
    input_orders: tuple[int, ...]
    orders: tuple[int, ...]
    first_b_pairs: tuple
    full_b_pairs: tuple
    q_pairs: tuple
    semigroup_count: int
    scales: torch.Tensor
    first_scales: tuple
    full_scales: tuple


def compile_support(input_orders, output_order, *, dtype=torch.float64, device='cpu'):
    """Keep S intersect (L-S), the semigroup degrees that can contribute to L.

    If k contributes and k=i+j with i,j in S, then L-i=(L-k)+j is
    also in S. Thus every retained recurrence dependency is retained too.
    The filter depends only on the explicit powers and requested output.
    """
    input_orders = tuple(input_orders)
    if not isinstance(output_order, int) or isinstance(output_order, bool) or output_order < 1:
        raise ValueError('output_order must be a positive integer')
    if not input_orders or any(not isinstance(i, int) or isinstance(i, bool) or i < 1
                               or i > output_order for i in input_orders):
        raise ValueError('Input orders must be positive integers no greater than output_order')
    if len(set(input_orders)) != len(input_orders):
        raise ValueError('Input orders must be distinct')
    reachable = [False]*(output_order+1)
    reachable[0] = True
    for order in range(1, output_order+1):
        reachable[order] = any(order >= power and reachable[order-power] for power in input_orders)
    if not reachable[output_order]:
        raise ValueError('Requested output order is outside the structural support')
    orders = tuple(order for order, active in enumerate(reachable)
                   if active and reachable[output_order-order])
    lookup = {order: index for index, order in enumerate(orders)}
    first, full, q = [], [], []
    for position, order in enumerate(orders[1:], 1):
        first.append(tuple((index+1, lookup[order-power], power)
                           for index, power in enumerate(input_orders)
                           if power <= order and order-power in lookup))
        full.append(tuple((index, lookup[order-power], power)
                          for index, power in enumerate(orders[1:position+1], 1)
                          if order-power in lookup))
        # q at the final order is never read by a subsequent b coefficient.
        q.append(tuple((index, lookup[order-power])
                       for index, power in enumerate(orders[:position+1])
                       if order-power in lookup) if order < output_order else ())
    scales = torch.tensor([power/order for pairs in (first, full)
                           for order, terms in zip(orders[1:], pairs)
                           for _, _, power in terms], dtype=dtype, device=device)
    split_sizes = [len(terms) for terms in first]+[len(terms) for terms in full]
    views = scales.split(split_sizes)
    count = len(first)
    return SupportPlan(input_orders, orders, tuple(first), tuple(full), tuple(q),
                       sum(reachable), scales, views[:count], views[count:])


def _tanh_sparse(a, plan, first_layer):
    b = [torch.tanh(a[0])]
    q = [1-b[0]*b[0]]
    b_pairs = plan.first_b_pairs if first_layer else plan.full_b_pairs
    scales = plan.first_scales if first_layer else plan.full_scales
    for terms, factors, q_terms in zip(b_pairs, scales, plan.q_pairs):
        left = torch.stack([a[ai] for ai, _, _ in terms])
        right = torch.stack([q[qi] for _, qi, _ in terms])
        b.append((left*right*factors.view((-1,)+(1,)*a[0].ndim)).sum(0))
        if q_terms:
            left = torch.stack([b[index] for index, _ in q_terms])
            right = torch.stack([b[index] for _, index in q_terms])
            q.append(-(left*right).sum(0))
    return tuple(b)


@dataclass(frozen=True)
class SparsePreparedEvaluation:
    model: _Model
    points: torch.Tensor
    curve_coefficients: torch.Tensor
    weights: torch.Tensor
    plan: SupportPlan
    _snapshot_sha256: str

    def raw(self):
        """Return the normalized output coefficient [curve, batch]."""
        curves, batch = len(self.weights), len(self.points)
        a = (self.points.unsqueeze(0).expand(curves, -1, -1),)+tuple(
            coefficient.unsqueeze(1).expand(-1, batch, -1)
            for coefficient in self.curve_coefficients.unbind(0))
        for index, layer in enumerate(self.model.layers):
            affine = F.linear(torch.stack(a), layer.weight).unbind(0)
            a = (affine[0]+layer.bias,)+affine[1:]
            if index < len(self.model.layers)-1:
                a = _tanh_sparse(a, self.plan, first_layer=index == 0)
        return a[-1][..., 0]

    def __call__(self):
        return (self.weights @ self.raw()).real

    def tensors(self):
        return (self.points, self.curve_coefficients, self.weights, self.plan.scales)+tuple(
            tensor for layer in self.model.layers for tensor in (layer.weight, layer.bias))

    def storage_identity(self):
        return tuple((id(t), t.untyped_storage().data_ptr(), t._version,
                      tuple(t.shape), str(t.dtype), str(t.device)) for t in self.tensors())

    def metadata(self):
        return dict(backend='structural_zero_normalized_taylor',
            input_orders=self.plan.input_orders, output_order=self.plan.orders[-1],
            active_orders=self.plan.orders, active_order_count=len(self.plan.orders),
            semigroup_order_count=self.plan.semigroup_count,
            first_affine_order_count=1+len(self.plan.input_orders),
            later_affine_order_count=len(self.plan.orders), curves=len(self.weights),
            first_b_product_count=sum(map(len, self.plan.first_b_pairs)),
            later_b_product_count=sum(map(len, self.plan.full_b_pairs)),
            q_product_count=sum(map(len, self.plan.q_pairs)),
            dtype=str(self.points.dtype), preconverted=True, parameter_backward=False,
            fixed_tensors_detached=True, prepared_tensors_sha256=self._snapshot_sha256,
            structural_zero_rule='S intersect (output_order-S); exact output reachability; no data-based pruning',
            recurrence_contraction='stack paired coefficients, multiply and sum over pairs')


def prepare_curve_evaluation(model, points, input_orders, curve_coefficients,
                             output_order, weights=None):
    """Prepare a detached evaluator; coefficient layout is [input_order,R,dim].

    ``input_orders`` are distinct positive powers. ``curve_coefficients[j]``
    multiplies t**input_orders[j], without raw-tangent factorial conventions.
    Omitted weights mean one per curve. Calls return the real weighted sum;
    ``raw()`` exposes the uncontracted, possibly complex output coefficient.
    """
    input_orders = tuple(input_orders)
    if points.ndim != 2 or len(points) < 1 or curve_coefficients.ndim != 3:
        raise ValueError('Expected nonempty points [batch,dim] and coefficients [input_order,R,dim]')
    if curve_coefficients.shape[0] != len(input_orders) or curve_coefficients.shape[2] != points.shape[1]:
        raise ValueError('Coefficient shape must match input orders and point dimension')
    curves = curve_coefficients.shape[1]
    if curves < 1:
        raise ValueError('At least one curve is required')
    if weights is None:
        weights = curve_coefficients.new_ones(curves)
    if weights.shape != (curves,):
        raise ValueError('One weight per curve is required')
    layers = tuple(getattr(model, 'layers', ()))
    if len(layers) < 2 or any(layer.bias is None for layer in layers):
        raise ValueError('Expected a scalar Linear/tanh MLP with at least two affine layers')
    if layers[0].weight.shape[1] != points.shape[1] or layers[-1].weight.shape[0] != 1:
        raise ValueError('MLP must accept the point dimension and have scalar output')
    parameters = tuple(tensor for layer in layers for tensor in (layer.weight, layer.bias))
    tensors = (points, curve_coefficients, weights)+parameters
    if any(not (t.is_floating_point() or t.is_complex()) for t in tensors):
        raise ValueError('All fixed tensors must have floating or complex dtype')
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError('All fixed tensors must be finite')
    dtype = reduce(torch.promote_types, (tensor.dtype for tensor in tensors))
    plan = compile_support(input_orders, output_order, dtype=dtype, device=points.device)
    snapshot = lambda tensor: tensor.detach().to(device=points.device, dtype=dtype).clone()
    fixed_model = _Model(tuple(_Layer(snapshot(layer.weight), snapshot(layer.bias)) for layer in layers))
    fixed_points, fixed_coefficients, fixed_weights = map(snapshot, (points, curve_coefficients, weights))
    fixed_tensors = (fixed_points, fixed_coefficients, fixed_weights, plan.scales)+tuple(
        tensor for layer in fixed_model.layers for tensor in (layer.weight, layer.bias))
    return SparsePreparedEvaluation(fixed_model, fixed_points, fixed_coefficients, fixed_weights,
                                    plan, digest(fixed_tensors))
