"""Independent per-term coordinate JVP; unexpanded f differentiated for validation."""
import torch
from torch.func import jvp,vmap
from .operators import KDV_TERMS,GKDV_TERMS,partial,assemble

def reference(model,points,case):
    terms=KDV_TERMS if case=='kdv2d' else GKDV_TERMS
    d={k:vmap(lambda x:partial(model,x,alpha))(points) for k,alpha in terms.items()}
    if case=='gkdv1d':
        d['u']=model(points)
        dispersion=points.new_tensor(.0025)
        def f(p):
            # Explicit tensor constant avoids scalar-JVP dtype promotion in some torch versions.
            return partial(model,p,(1,))+model(p)*partial(model,p,(0,))+dispersion*partial(model,p,(0,0,0))
        fv=vmap(f)(points)
        basis=torch.eye(2,dtype=points.dtype,device=points.device)
        fx=vmap(lambda x:jvp(f,(x,),(basis[0],))[1])(points)
        ft=vmap(lambda x:jvp(f,(x,),(basis[1],))[1])(points)
        return fv.square().mean()+.001*(fx.square().mean()+ft.square().mean()),{**d,'f':fv,'f_x':fx,'f_t':ft}
    val,r=assemble(case,d)
    return val,{**d,**r}
