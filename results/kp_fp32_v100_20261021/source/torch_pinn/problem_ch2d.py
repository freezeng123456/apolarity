"""Four-order forced CH, three-mode exact solution, homogeneous natural traces.

Physical coordinates [0,pi]^2 x [0,1]; real tanh MLP without input mapping.
Three unit-direction fourth-order jets recover all spatial interior terms.
Time and boundary derivatives use a shared coordinate-JVP path.
"""
import math
import torch
from torch.func import jvp, vmap
from .jets import mlp_jet
from .operators import require_real_pde, METHODS, partial, chain, selected

MODES = ((.5, 1, 1), (.25, 2, 1), (.25, 1, 2))


def analytic(p):
    x, y, t = p.unbind(-1)
    u, ux, uy, lap, bih = [torch.zeros_like(x) for _ in range(5)]
    for a, k, l in MODES:
        mode = a*torch.cos(k*x)*torch.cos(l*y)*torch.exp(-t)
        u = u+mode
        ux = ux-a*k*torch.sin(k*x)*torch.cos(l*y)*torch.exp(-t)
        uy = uy-a*l*torch.cos(k*x)*torch.sin(l*y)*torch.exp(-t)
        lap = lap-(k*k+l*l)*mode
        bih = bih+(k*k+l*l)**2*mode
    source = -u-(3*u*u-1)*lap-6*u*(ux*ux+uy*uy)+.01*bih
    return dict(u=u, ux=ux, uy=uy, lap=lap, bih=bih, source=source)


def exact(p):
    return analytic(p)['u']


def constraints(n=16, *, dtype=torch.float32, device='cpu'):
    s = torch.linspace(0, math.pi, n, dtype=dtype, device=device)
    t = torch.linspace(0, 1, n, dtype=dtype, device=device)
    x, y = torch.meshgrid(s, s, indexing='ij')
    a, b = torch.meshgrid(s, t, indexing='ij')
    pack = lambda x,y,t: torch.stack((x.flatten(),y.flatten(),t.flatten()),dim=1)
    z, h = torch.zeros_like(a), torch.full_like(a,math.pi)
    return dict(initial=pack(x,y,torch.zeros_like(x)),
                left=pack(z,a,b),right=pack(h,a,b),bottom=pack(a,z,b),top=pack(a,h,b))


def boundary_losses(model, data):
    terms = {'ic': (model(data['initial'])-exact(data['initial'])).square().mean()}
    for name, axis in (('left',0),('right',0),('bottom',1),('top',1)):
        p = data[name]
        first = vmap(lambda z: partial(model,z,(axis,)))(p)
        third = vmap(lambda z: partial(model,z,(0,0,axis))+partial(model,z,(1,1,axis)))(p)
        terms[name+'_n1'] = first.square().mean()
        terms[name+'_n3'] = third.square().mean()
    terms['bc1'] = sum(terms[n+'_n1'] for n in ('left','right','bottom','top'))/4
    terms['bc3'] = sum(terms[n+'_n3'] for n in ('left','right','bottom','top'))/4
    return terms


def components(model, points, method, *, independent=False):
    if method not in METHODS:raise ValueError(method)
    require_real_pde(model, points)
    if method == 'shared_jet_linear' and not independent:
        dirs = points.new_tensor(((1,0,0),(.5,math.sqrt(3)/2,0),(-.5,math.sqrt(3)/2,0)))
        q = mlp_jet(model,points,dirs,4)
        u = q[0][0]
        lap = (4/3)*q[2].sum(0)
        bih = (64/3)*q[4].sum(0)
        grad2 = (2/3)*q[1].square().sum(0)
        time_dir = points.new_tensor((0,0,1))
        ut = vmap(lambda z: jvp(model,(z,),(time_dir,))[1])(points)
    else:
        basis = torch.eye(3,dtype=points.dtype,device=points.device)
        def one(p):
            if independent:
                ds = {a:partial(model,p,a) for a in ((0,),(1,),(2,),(0,0),(1,1),(0,0,0,0),(1,1,1,1),(0,0,1,1))}
                return model(p),ds[(2,)],ds[(0,0)]+ds[(1,1)],ds[(0,0,0,0)]+2*ds[(0,0,1,1)]+ds[(1,1,1,1)],ds[(0,)]**2+ds[(1,)]**2
            xxxx, ax = chain(model,basis,0,4)(p)
            yyyy, ay = chain(model,basis,1,4)(p)
            xxyy = partial(lambda z: chain(model,basis,0,2)(z)[0],p,(1,1))
            ut = jvp(model,(p,),(basis[2],))[1]
            return ax[0],ut,ax[2]+ay[2],xxxx+2*xxyy+yyyy,ax[1]**2+ay[1]**2
        u,ut,lap,bih,grad2 = vmap(one)(points)
    return dict(u=u,ut=ut,lap=lap,bih=bih,grad2=grad2)


def loss(model, points, data, method, *, independent=False):
    d = components(model,points,method,independent=independent)
    r = d['ut']-(3*d['u']**2-1)*d['lap']-6*d['u']*d['grad2']+.01*d['bih']-analytic(points)['source']
    terms = dict(pde=r.square().mean(),**boundary_losses(model,data))
    return terms['pde']+10*(terms['ic']+terms['bc1']+terms['bc3']), terms
