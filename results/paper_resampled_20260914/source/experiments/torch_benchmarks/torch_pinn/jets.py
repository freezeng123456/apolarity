"""Differentiable Taylor arithmetic, using only ordinary PyTorch operations.

Coefficient k is D_v^k u/k!, including the primal at k=0. No repeated JVP,
custom backward, compiler, external derivative package, or detached coefficients.
The tanh rule follows y'=x'(1-y^2). All orders and directions share propagation.
"""
import torch
from torch.nn import functional as F


def tanh_series(a):
    b = [torch.tanh(a[0])]
    q = [1-b[0]*b[0]]
    for k in range(1, len(a)):
        b.append(sum(i*a[i]*q[k-i] for i in range(1, k+1))/k)
        if k < len(a)-1:
            q.append(-sum(b[j]*b[k-j] for j in range(k+1)))
    return tuple(b)


def mlp_jet(model, points, directions, order):
    if order < 1 or points.ndim != 2 or directions.ndim != 2 or points.shape[-1] != directions.shape[-1]:
        raise ValueError('positive order and compatible [batch,dim]/[direction,dim] required')
    if points.device != directions.device:
        raise ValueError('points and directions must share a device')
    dtype = torch.promote_types(points.dtype, directions.dtype)
    x = points.to(dtype).unsqueeze(0).expand(len(directions), -1, -1)
    v = directions.to(dtype).unsqueeze(1).expand(-1, len(points), -1)
    a = (x, v) + (torch.zeros_like(x),)*(order-1)
    for i, layer in enumerate(model.layers):
        w, bias = layer.weight.to(dtype), layer.bias.to(dtype)
        # One affine contraction across order, direction and collocation axes.
        linear = F.linear(torch.stack(a), w).unbind(0)
        a = (linear[0]+bias,) + linear[1:]
        if i < len(model.layers)-1:
            a = tanh_series(a)
    return tuple(c[..., 0] for c in a)
