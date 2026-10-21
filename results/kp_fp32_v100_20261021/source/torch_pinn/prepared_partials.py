"""Values-only Taylor contractions with casting and conjugate reuse prepared.

All fixed tensors are owned, detached snapshots. Evaluation uses the unchanged
``mlp_jet`` kernel, including its ordinary dtype checks; those casts become
no-ops because preparation gives every fixed tensor the same promoted dtype.
Conjugate reuse is valid here only for real model parameters and real points.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import torch

from .jets import mlp_jet
from .records import digest

# Schedule roots originate in double-precision cmath, even for complex64.
# A float32-scale tolerance would incorrectly merge distinct nearby directions.
_ROOT_TOL = 32*torch.finfo(torch.float64).eps


def _close(left, right):
    return bool(((left-right).abs() <= _ROOT_TOL*(1+right.abs())).all())


def reduce_conjugates(directions, weights):
    """Keep real rows and one representative of each verified conjugate pair.

    For real Taylor coefficients, Re(c*q(v)+conj(c)*q(conj(v))) is
    Re(2*c*q(v)). Unmatched, ambiguous, or incorrectly weighted pairs fail.
    Pair selection is outside timing and never combines merely nearby rows.
    """
    rows = directions.detach().to(device='cpu', dtype=torch.complex128)
    coefficients = weights.detach().to(device='cpu', dtype=torch.complex128)
    selected, factors, pairs, real_rows, used = [], [], [], [], set()
    max_direction_error = max_weight_error = 0.
    for index, (row, coefficient) in enumerate(zip(rows, coefficients)):
        if index in used:
            continue
        if _close(row, row.conj()):
            if not _close(coefficient, coefficient.conj()):
                raise ValueError('Self-conjugate direction has a nonreal weight')
            selected.append(index)
            factors.append(1)
            real_rows.append(index)
            used.add(index)
            continue
        candidates = [other for other in range(len(rows)) if other != index and other not in used
                      and _close(rows[other], row.conj())]
        if len(candidates) != 1:
            raise ValueError('Nonreal direction needs exactly one unmatched conjugate partner')
        partner = candidates[0]
        if not _close(coefficients[partner], coefficient.conj()):
            raise ValueError('Conjugate directions have inconsistent weights')
        max_direction_error = max(max_direction_error, float((rows[partner]-row.conj()).abs().max()))
        max_weight_error = max(max_weight_error, float((coefficients[partner]-coefficient.conj()).abs()))
        selected.append(index)
        factors.append(2)
        pairs.append((index, partner))
        used.update((index, partner))
    indices = torch.tensor(selected, dtype=torch.long, device=directions.device)
    multipliers = weights.new_tensor(factors)
    return (directions.index_select(0, indices), weights.index_select(0, indices)*multipliers,
            dict(conjugate_pairs=len(pairs), self_conjugate_rows=len(real_rows),
                 retained_indices=selected, pair_indices=[list(pair) for pair in pairs],
                 max_pair_direction_error=max_direction_error, max_pair_weight_error=max_weight_error,
                 pairing_tolerance=_ROOT_TOL))


@dataclass(frozen=True)
class _Layer:
    weight: torch.Tensor
    bias: torch.Tensor


@dataclass(frozen=True)
class _Model:
    layers: tuple[_Layer, ...]


@dataclass(frozen=True)
class PreparedEvaluation:
    model: _Model
    points: torch.Tensor
    directions: torch.Tensor
    weights: torch.Tensor
    order: int
    original_count: int
    conjugate_reuse: bool
    _reduction: tuple
    _snapshot_sha256: str

    def __call__(self):
        coefficients = mlp_jet(self.model, self.points, self.directions, self.order)[self.order]
        return (self.weights @ coefficients).real

    def tensors(self):
        return (self.points, self.directions, self.weights)+tuple(
            tensor for layer in self.model.layers for tensor in (layer.weight, layer.bias))

    def storage_identity(self):
        return tuple((id(t), t.untyped_storage().data_ptr(), t._version,
                      tuple(t.shape), str(t.dtype), str(t.device)) for t in self.tensors())

    def metadata(self):
        return dict(original_count=self.original_count, evaluated_count=len(self.directions),
                    dtype=str(self.points.dtype), preconverted=True,
                    conjugate_reuse=self.conjugate_reuse,
                    parameter_backward=False, fixed_tensors_detached=True,
                    prepared_tensors_sha256=self._snapshot_sha256,
                    **dict(self._reduction))


def prepare_evaluation(model, points, directions, weights, order, *, conjugate_reuse=True):
    """Snapshot real model/points and a batched schedule, then return a callable.

    The callable has no parameter-autograd graph and intentionally cannot be
    used for training. ``conjugate_reuse=False`` retains all original rows for
    the casting-only ablation while keeping the same execution path.
    """
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError('A positive integer order is required')
    if points.ndim != 2 or directions.ndim != 2 or points.shape[1] != directions.shape[1]:
        raise ValueError('Expected compatible points [batch,dim] and directions [row,dim]')
    if len(points) < 1 or len(directions) < 1 or weights.shape != (len(directions),):
        raise ValueError('Expected nonempty points/directions and one weight per direction')
    if points.is_complex() or not points.is_floating_point():
        raise ValueError('Conjugate evaluation requires real floating-point points')
    layers = tuple(getattr(model, 'layers', ()))
    if not layers or any(layer.bias is None for layer in layers):
        raise ValueError('Expected the scalar Linear/tanh MLP with biases')
    parameters = tuple(t for layer in layers for t in (layer.weight, layer.bias))
    if any(t.is_complex() or not t.is_floating_point() for t in parameters):
        raise ValueError('Conjugate evaluation requires real model parameters')
    if hasattr(model, 'parameters') and any(p.is_complex() for p in model.parameters()):
        raise ValueError('Conjugate evaluation requires real model parameters')
    tensors = (points, directions, weights)+parameters
    if any(not (t.is_floating_point() or t.is_complex()) for t in tensors):
        raise ValueError('Floating-point tensors are required')
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError('All fixed tensors must be finite')
    dtype = reduce(torch.promote_types, (t.dtype for t in tensors))
    device = points.device
    snapshot = lambda t: t.detach().to(device=device, dtype=dtype).clone()
    frozen_model = _Model(tuple(_Layer(snapshot(layer.weight), snapshot(layer.bias)) for layer in layers))
    fixed_points, fixed_directions, fixed_weights = map(snapshot, (points, directions, weights))
    original_count = len(fixed_directions)
    reduction = dict(conjugate_pairs=0, self_conjugate_rows=0,
                     retained_indices=list(range(original_count)), pair_indices=[],
                     max_pair_direction_error=0., max_pair_weight_error=0., pairing_tolerance=_ROOT_TOL)
    if conjugate_reuse:
        fixed_directions, fixed_weights, reduction = reduce_conjugates(fixed_directions, fixed_weights)
    # Store nested metadata immutably; metadata() returns independent JSON data.
    reduction['retained_indices'] = tuple(reduction['retained_indices'])
    reduction['pair_indices'] = tuple(tuple(pair) for pair in reduction['pair_indices'])
    fixed_tensors = (fixed_points, fixed_directions, fixed_weights)+tuple(
        tensor for layer in frozen_model.layers for tensor in (layer.weight, layer.bias))
    return PreparedEvaluation(frozen_model, fixed_points, fixed_directions, fixed_weights,
                              order, original_count, conjugate_reuse, tuple(reduction.items()),
                              digest(fixed_tensors))
