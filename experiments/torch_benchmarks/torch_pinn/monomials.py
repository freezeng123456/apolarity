"""First-group single-partial API using the same PyTorch Taylor engine."""
import cmath
from collections import Counter
from itertools import product
from math import factorial,prod
import torch
from torch.func import vmap
from .jets import mlp_jet
from .coordinate_jvp import partial


def schedule(alpha,dim,*,dtype=torch.float32,device='cpu'):
    if not alpha or any(i<0 or i>=dim for i in alpha):raise ValueError('invalid partial indices')
    # One roots-of-unity construction; real roots do not select another method.
    active=sorted(Counter(alpha).items(),key=lambda kv:(kv[1],kv[0]))
    base,other=active[0][0],active[1:]
    rank=prod(e+1 for _,e in other)
    complex_needed=any(e>1 for _,e in other)
    outdtype=(torch.complex128 if dtype==torch.float64 else torch.complex64) if complex_needed else dtype
    roots=[[cmath.exp(2j*cmath.pi*k/(e+1)) for k in range(e+1)] for _,e in other]
    directions=[];weights=[]
    for selected in product(*roots):
        v=[0j]*dim;v[base]=1
        weight=prod(factorial(e) for _,e in active)/rank
        for (axis,_),r in zip(other,selected):v[axis]=r;weight*=r
        directions.append(v if complex_needed else [complex(a).real for a in v])
        weights.append(weight if complex_needed else complex(weight).real)
    return torch.tensor(directions,dtype=outdtype,device=device),torch.tensor(weights,dtype=outdtype,device=device)


def monomial(model,points,alpha,method='waring_batched',*,return_complex=False):
    alpha=tuple(alpha)
    if not alpha or any(i<0 or i>=points.shape[-1] for i in alpha):raise ValueError('invalid partial indices')
    if method=='nested_jvp':return vmap(lambda p:partial(model,p,alpha))(points)
    if method != 'waring_batched':raise ValueError(method)
    directions,weights=schedule(alpha,points.shape[-1],dtype=points.dtype,device=points.device)
    coefficients=mlp_jet(model,points,directions,len(alpha))[len(alpha)]
    raw=weights @ coefficients
    return raw if return_complex else raw.real
