"""Independent full-loss/gradient/Adam gate and complete-objective stage timing."""
import argparse
import copy
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from torch_pinn.model import MLP
from torch_pinn.problem_kdv2d import constraints, loss, sample_numpy
from torch_pinn.records import check, digest, run_record, save, sync

METHODS = ('nested_jvp','shared_jet_linear')
SEEDS = (20260921,20260922,20260923)


def gradients(model):
    return [p.grad.detach().clone() for p in model.parameters()]


def verify(a):
    rows=[]
    for seed in SEEDS:
        base=MLP(3,128,4,seed+2,device=a.device)
        pts=torch.as_tensor(sample_numpy(np.random.default_rng(seed+1),5),device=a.device)
        data=constraints(4,device=a.device)
        ref=copy.deepcopy(base)
        ref_opt=torch.optim.Adam(ref.parameters(),lr=1e-3,foreach=False)
        value,parts=loss(ref,pts,data,'nested_jvp',independent=True)
        value.backward()
        rg=gradients(ref)
        ref_opt.step()
        for method in METHODS:
            model=copy.deepcopy(base)
            opt=torch.optim.Adam(model.parameters(),lr=1e-3,foreach=False)
            result,terms=loss(model,pts,data,method)
            result.backward()
            gg=gradients(model)
            row=dict(seed=seed,method=method,loss=check(result,value),
                     parts={k:check(terms[k],parts[k]) for k in parts},
                     gradients=[check(g,r) for g,r in zip(gg,rg)])
            row['gradient_global']=check(torch.cat([g.flatten() for g in gg]),torch.cat([r.flatten() for r in rg]))
            if row['gradient_global']['relative_l2'] > 1e-3:
                raise AssertionError('global gradient gate failed')
            opt.step()
            row['adam_parameters']=[check(p,r) for p,r in zip(model.parameters(),ref.parameters())]
            row['adam_state']=[{k:check(v,ref_opt.state[r][k]) for k,v in opt.state[p].items()}
                               for p,r in zip(model.parameters(),ref.parameters())]
            for _ in range(19):
                opt.zero_grad(set_to_none=True)
                v,_=loss(model,pts,data,method)
                v.backward()
                if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise AssertionError('nonfinite short-update gradient')
                opt.step()
            row['initial_hash']=digest(base.parameters())
            row['final_hash']=digest(model.parameters())
            assert row['initial_hash'] != row['final_hash']
            row['finite_20_updates']=True
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),gradients=gg),a.out/f'{method}_{seed}.pt')
            rows.append(row)
            save(a.out/'verification.json',rows)
    save(a.out/'acceptance.json',dict(gate='PASS',records=len(rows),seeds=list(SEEDS)))


def benchmark(a):
    rows=[]
    for seed in SEEDS:
        base=MLP(3,128,4,seed+2,device=a.device)
        pts=torch.as_tensor(sample_numpy(np.random.default_rng(seed+1),400),device=a.device)
        data=constraints(16,device=a.device)
        for method in METHODS:
            for phase in ('loss','loss_backward','adam_step'):
                model=copy.deepcopy(base)
                opt=torch.optim.Adam(model.parameters(),lr=1e-3,foreach=False)
                def call():
                    opt.zero_grad(set_to_none=True)
                    value=loss(model,pts,data,method)[0]
                    if phase != 'loss': value.backward()
                    if phase == 'adam_step': opt.step()
                    return value
                sync(a.device)
                start=time.perf_counter()
                call()
                sync(a.device)
                cold=time.perf_counter()-start
                for _ in range(10): call()
                model.load_state_dict(base.state_dict())
                opt=torch.optim.Adam(model.parameters(),lr=1e-3,foreach=False)
                model.zero_grad(set_to_none=True)
                times=[]
                for _ in range(30):
                    sync(a.device)
                    start=time.perf_counter()
                    value=call()
                    sync(a.device)
                    times.append(1000*(time.perf_counter()-start))
                    if not bool(torch.isfinite(value)): raise RuntimeError('nonfinite cost sample')
                rows.append(dict(seed=seed,method=method,phase=phase,times_ms=times,
                            median_ms=statistics.median(times),cold_seconds=cold,
                            initial_parameter_hash=digest(base.parameters()),point_hash=digest((pts,)),
                            task='full_kdv2d_loss_including_initial_and_five_boundary_trace_penalties',
                            warmups=10,repeats=30,parameter_graph=True,
                            timing_scope='Synchronized call; excludes setup, sampling, I/O, finite-gradient checks and evaluation; Adam samples perform real sequential updates from reset initial state.'))
                save(a.out/'results.json',rows)
    save(a.out/'acceptance.json',dict(gate='PASS',records=len(rows),samples=sum(len(r['times_ms']) for r in rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--mode',choices=['verify','cost'],default='verify')
    a=p.parse_args()
    run_record(a,verify if a.mode=='verify' else benchmark)
