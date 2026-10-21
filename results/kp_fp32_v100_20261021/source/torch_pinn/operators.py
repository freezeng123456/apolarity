"""Identical PDE equations with interchangeable coordinate-JVP/shared-jet paths."""
import torch
from torch.func import jvp, vmap
from .jets import mlp_jet
from .coordinate_jvp import partial, chain, selected

METHODS = ('nested_jvp', 'shared_jet_linear')
CASES = ('kdv2d', 'gkdv1d')
KDV_TERMS = dict(u_x=(0,), u_y=(1,), u_xx=(0,0), u_xy=(0,1), u_yy=(1,1), u_ty=(2,1), u_xxxy=(0,0,0,1))
GKDV_TERMS = dict(u_x=(0,), u_t=(1,), u_xx=(0,0), u_tx=(1,0), u_tt=(1,1), u_xxx=(0,0,0), u_xxxx=(0,0,0,0), u_txxx=(1,0,0,0))
KDV_DIRS = ((1,1,0),(1,-1,0),(1,2,0),(1,-2,0),(0,1,1),(0,1,-1))
GKDV_DIRS = ((1,0),(1,1),(1,-1),(1,2),(1,-2))


def ad_components(fun, point, basis, case):
    x3 = chain(fun, basis, 0, 3)
    if case == 'kdv2d':
        _, (uxxxy,), (_, ux, uxx) = selected(x3, point, basis, (1,))
        uy, (uxy, uyy, uty), _ = selected(chain(fun, basis, 1, 1), point, basis, (0,1,2))
        return dict(u_x=ux,u_y=uy,u_xx=uxx,u_xy=uxy,u_yy=uyy,u_ty=uty,u_xxxy=uxxxy)
    uxxx, (uxxxx, utxxx), (u, ux, uxx) = selected(x3, point, basis, (0,1))
    ut, (utx, utt), _ = selected(chain(fun, basis, 1, 1), point, basis, (0,1))
    return dict(u=u,u_x=ux,u_t=ut,u_xx=uxx,u_tx=utx,u_tt=utt,u_xxx=uxxx,u_xxxx=uxxxx,u_txxx=utxxx)


def reconstruct(case, q):
    """Fixed real power-sum identities from the paper; q[k] = D_v^k u / k!."""
    tensor = lambda a: q[0].new_tensor(a)
    if case == 'kdv2d':
        first = tensor(((1,1,0,0,0,0),(1,-1,0,0,0,0))) @ q[1]/2
        second = tensor(((8,8,-2,-2,0,0),(-2,-2,2,2,0,0),(3,-3,0,0,0,0),(0,0,0,0,3,-3))) @ q[2]/6
        fourth = tensor((8,-8,-1,1,0,0)) @ q[4]/2
        return dict(u_x=first[0],u_y=first[1],u_xx=second[0],u_yy=second[1],u_xy=second[2],u_ty=second[3],u_xxxy=fourth)
    if case != 'gkdv1d':
        raise ValueError(case)
    first = tensor(((2,0,0,0,0),(0,1,-1,0,0))) @ q[1]/2
    second = tensor(((4,0,0,0,0),(0,1,-1,0,0),(-4,2,2,0,0))) @ q[2]/2
    fourth = tensor(((48,0,0,0,0),(0,8,-8,-1,1))) @ q[4]/2
    return dict(u=q[0][0],u_x=first[0],u_t=first[1],u_xx=second[0],u_tx=second[1],u_tt=second[2],
                u_xxx=6*q[3][0],u_xxxx=fourth[0],u_txxx=fourth[1])


def require_real_pde(model, points):
    """PDE experiments use real parameters, points and prescribed directions."""
    parameters = model.parameters() if hasattr(model, 'parameters') else ()
    if points.is_complex() or any(p.is_complex() for p in parameters):
        raise ValueError('PDE evaluation requires real points and model parameters')


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
    require_real_pde(model, points)
    dim = 3 if case == 'kdv2d' else 2
    if points.ndim != 2 or points.shape[1] != dim:
        raise ValueError('invalid point dimension')
    if method == 'nested_jvp':
        basis = torch.eye(dim, dtype=points.dtype, device=points.device)
        d = vmap(lambda x: ad_components(model, x, basis, case))(points)
    else:
        directions = points.new_tensor(KDV_DIRS if dim == 3 else GKDV_DIRS)
        q = mlp_jet(model, points, directions, 4)
        d = reconstruct(case, q)
    loss, r = assemble(case,d)
    return (loss,{**d,**r}) if components else loss
