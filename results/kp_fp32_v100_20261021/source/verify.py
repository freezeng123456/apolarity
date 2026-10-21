"""Full-width derivative, parameter-gradient and matched Adam-update GPU gate."""
import argparse
import copy
from pathlib import Path
import torch
from torch_pinn.model import MLP,sample_points
from torch_pinn.operators import CASES,METHODS,evaluate
from torch_pinn.reference import reference
from torch_pinn.problem import constraints,loss,boundary_losses
from torch_pinn.records import save,check,digest,run_record


def gradients(model):
    return tuple(torch.zeros_like(p) if p.grad is None else p.grad for p in model.parameters())


def verify(args):
    rows=[]
    for case in CASES:
        dim=3 if case=='kdv2d' else 2
        for seed in args.seeds:
            original=MLP(dim,args.width,args.depth,seed+2,device=args.device)
            points=sample_points(seed+1,3,dim,device=args.device)
            expected,components=reference(original,points,case)
            expected.backward()
            refgrads=tuple(g.detach().clone() for g in gradients(original))
            for method in METHODS:
                model=copy.deepcopy(original);model.zero_grad(set_to_none=True)
                actual,d=evaluate(model,points,case,method,components=True)
                row=dict(case=case,seed=seed,method=method,value=check(actual,expected),
                         components={k:check(d[k],v) for k,v in components.items()})
                actual.backward()
                row['gradients']=[check(g,r) for g,r in zip(gradients(model),refgrads)]
                flat=torch.cat([g.flatten() for g in gradients(model)])
                rf=torch.cat([g.flatten() for g in refgrads])
                row['gradient_global']=check(flat,rf)
                if row['gradient_global']['relative_l2']>1e-3:raise AssertionError('global gradient tolerance')
                row['parameter_hash']=digest(model.parameters());row['points_hash']=digest((points,))
                rows.append(row)
                torch.save(dict(value=actual.detach().cpu(),components={k:v.detach().cpu() for k,v in d.items()},
                                gradients=[g.detach().cpu() for g in gradients(model)]),args.out/f'{case}_{method}_{seed}.pt')
                save(args.out/'verification.json',rows)
                print(f'VERIFIED {case} {method} {seed}',flush=True)
    # Identical complete PINN losses, initial states, and batches; update all parameters.
    data=constraints(16,device=args.device)
    updates=[]
    for seed in args.seeds:
        initial=MLP(2,args.width,args.depth,seed+2,device=args.device)
        points=sample_points(seed+1,16,2,device=args.device)
        states={}
        for method in METHODS:
            model=copy.deepcopy(initial)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,foreach=False)
            values=[]
            for step in range(20):
                optimizer.zero_grad(set_to_none=True)
                value,terms=loss(model,points,data,method)
                value.backward()
                if not all(bool(torch.isfinite(g).all()) for g in gradients(model)):raise AssertionError('nonfinite gradient')
                if step==0:
                    independent=MLP(2,args.width,args.depth,seed+2,device=args.device)
                    reference_loss=reference(independent,points,'gkdv1d')[0]+10*sum(boundary_losses(independent,data).values())
                    reference_loss.backward()
                    check(value,reference_loss)
                    for g,r in zip(gradients(model),gradients(independent)):check(g,r)
                optimizer.step();values.append(float(value.detach()))
                if step==0:
                    states[method]=dict(parameters=[p.detach().clone() for p in model.parameters()],
                        optimizer=[{k:v.detach().clone() if torch.is_tensor(v) else v for k,v in optimizer.state[p].items()} for p in model.parameters()])
            if digest(model.parameters())==digest(initial.parameters()):raise AssertionError('parameters did not change')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),losses=values),args.out/f'updates_{method}_{seed}.pt')
            updates.append(dict(method=method,seed=seed,steps=20,initial_loss=values[0],last_loss=values[-1]))
        for method in METHODS[1:]:
            for a,b in zip(states[method]['parameters'],states[METHODS[0]]['parameters']):check(a,b)
            for a,b in zip(states[method]['optimizer'],states[METHODS[0]]['optimizer']):
                for key in a:check(a[key],b[key])
    save(args.out/'updates.json',updates)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--seeds',type=int,nargs='+',default=[20260908,20260909,20260910])
    args=p.parse_args();run_record(args,verify)
