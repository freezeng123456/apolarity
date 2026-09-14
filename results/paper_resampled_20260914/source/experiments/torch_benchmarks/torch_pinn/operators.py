"""Identical PDE equations with interchangeable coordinate-JVP/shared-jet paths."""
import torch
from torch.func import jvp, vmap
from .jets import mlp_jet

METHODS = ('nested_jvp', 'shared_jet', 'shared_jet_linear')
CASES = ('kdv2d', 'gkdv1d')
KDV_TERMS = dict(u_x=(0,), u_y=(1,), u_xx=(0,0), u_xy=(0,1), u_yy=(1,1), u_ty=(2,1), u_xxxy=(0,0,0,1))
GKDV_TERMS = dict(u_x=(0,), u_t=(1,), u_xx=(0,0), u_tx=(1,0), u_tt=(1,1), u_xxx=(0,0,0), u_xxxx=(0,0,0,0), u_txxx=(1,0,0,0))
KDV_DIRS = ((1,1,0),(1,-1,0),(1,2,0),(1,-2,0),(0,1,1),(0,1,-1))
GKDV_DIRS = ((1,0),(1,1),(1,-1),(1,2),(1,-2))


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


def ad_components(fun, point, basis, case):
    x3 = chain(fun, basis, 0, 3)
    if case == 'kdv2d':
        _, (uxxxy,), (_, ux, uxx) = selected(x3, point, basis, (1,))
        uy, (uxy, uyy, uty), _ = selected(chain(fun, basis, 1, 1), point, basis, (0,1,2))
        return dict(u_x=ux,u_y=uy,u_xx=uxx,u_xy=uxy,u_yy=uyy,u_ty=uty,u_xxxy=uxxxy)
    uxxx, (uxxxx, utxxx), (u, ux, uxx) = selected(x3, point, basis, (0,1))
    ut, (utx, utt), _ = selected(chain(fun, basis, 1, 1), point, basis, (0,1))
    return dict(u=u,u_x=ux,u_t=ut,u_xx=uxx,u_tx=utx,u_tt=utt,u_xxx=uxxx,u_xxxx=uxxxx,u_txxx=utxxx)


def reconstruct(case, q, linear=True):
    """q[k][direction,point] is factorial-normalized; weights absorb factorials."""
    tensor = lambda a: q[0].new_tensor(a)
    if case == 'kdv2d':
        if linear:
            first = tensor(((1,1,0,0,0,0),(1,-1,0,0,0,0))) @ q[1]/2
            second = tensor(((8,8,-2,-2,0,0),(-2,-2,2,2,0,0),(3,-3,0,0,0,0),(0,0,0,0,3,-3))) @ q[2]/6
            fourth = tensor((8,-8,-1,1,0,0)) @ q[4]/2
            return dict(u_x=first[0],u_y=first[1],u_xx=second[0],u_yy=second[1],u_xy=second[2],u_ty=second[3],u_xxxy=fourth)
        s1, s2 = q[2][0]+q[2][1], q[2][2]+q[2][3]
        return dict(u_x=(q[1][0]+q[1][1])/2,u_y=(q[1][0]-q[1][1])/2,
                    u_xx=(4*s1-s2)/3,u_yy=(s2-s1)/3,u_xy=(q[2][0]-q[2][1])/2,
                    u_ty=(q[2][4]-q[2][5])/2,u_xxxy=4*(q[4][0]-q[4][1])-(q[4][2]-q[4][3])/2)
    if linear:
        first = tensor(((2,0,0,0,0),(0,1,-1,0,0))) @ q[1]/2
        second = tensor(((4,0,0,0,0),(0,1,-1,0,0),(-4,2,2,0,0))) @ q[2]/2
        fourth = tensor(((48,0,0,0,0),(0,8,-8,-1,1))) @ q[4]/2
        return dict(u=q[0][0],u_x=first[0],u_t=first[1],u_xx=second[0],u_tx=second[1],u_tt=second[2],
                    u_xxx=6*q[3][0],u_xxxx=fourth[0],u_txxx=fourth[1])
    return dict(u=q[0][0],u_x=q[1][0],u_t=(q[1][1]-q[1][2])/2,u_xx=2*q[2][0],
                u_tx=(q[2][1]-q[2][2])/2,u_tt=q[2][1]+q[2][2]-2*q[2][0],
                u_xxx=6*q[3][0],u_xxxx=24*q[4][0],u_txxx=4*(q[4][1]-q[4][2])-(q[4][3]-q[4][4])/2)


def assemble(case, d):
    if case == 'kdv2d':
        r = d['u_ty']+d['u_xxxy']+3*(d['u_xy']*d['u_x']+d['u_y']*d['u_xx'])-d['u_xx']+2*d['u_yy']
        return (r*r).mean(), dict(residual=r)
    f = d['u_t']+d['u']*d['u_x']+.0025*d['u_xxx']
    fx = d['u_tx']+d['u_x']**2+d['u']*d['u_xx']+.0025*d['u_xxxx']
    ft = d['u_tt']+d['u_t']*d['u_x']+d['u']*d['u_tx']+.0025*d['u_txxx']
    return (f*f).mean()+.001*((fx*fx).mean()+(ft*ft).mean()), dict(f=f,f_x=fx,f_t=ft)


def evaluate(model, points, case='gkdv1d', method='shared_jet_linear', *, components=False):
    if case not in CASES or method not in METHODS:
        raise ValueError((case,method))
    dim = 3 if case == 'kdv2d' else 2
    if points.ndim != 2 or points.shape[1] != dim:
        raise ValueError('invalid point dimension')
    if method == 'nested_jvp':
        basis = torch.eye(dim, dtype=points.dtype, device=points.device)
        d = vmap(lambda x: ad_components(model, x, basis, case))(points)
    else:
        directions = points.new_tensor(KDV_DIRS if dim == 3 else GKDV_DIRS)
        q = mlp_jet(model, points, directions, 4)
        d = reconstruct(case, q, linear=method == 'shared_jet_linear')
    loss, r = assemble(case,d)
    return (loss,{**d,**r}) if components else loss
