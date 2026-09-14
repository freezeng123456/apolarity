"""Coordinate JVP primitives shared by the baseline and boundary derivatives."""
import torch
from torch.func import jvp, vmap


def partial(fun, point, alpha):
    basis = torch.eye(point.shape[-1], dtype=point.dtype, device=point.device)
    current = fun
    for axis in alpha:
        previous = current
        current = lambda x, previous=previous, axis=axis: jvp(previous, (x,), (basis[axis],))[1]
    return current(point)


def chain(fun, basis, axis, count):
    current = lambda x: (fun(x), ())
    for _ in range(count):
        previous = current
        def current(x, previous=previous):
            value, tangent, aux = jvp(previous, (x,), (basis[axis],), has_aux=True)
            return tangent, (*aux, value)
    return current


def selected(fun, point, basis, axes):
    if len(axes) == 1:
        value, tangent, aux = jvp(fun, (point,), (basis[axes[0]],), has_aux=True)
        return value, (tangent,), aux
    values, tangents, aux = vmap(lambda v: jvp(fun, (point,), (v,), has_aux=True))(basis[list(axes)])
    return values[0], tuple(tangents[i] for i in range(len(axes))), tuple(a[0] for a in aux)
