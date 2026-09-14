"""Eager loss, loss+backward and Adam-step timings, excluding setup and I/O."""
import argparse
import copy
import statistics
import time
from pathlib import Path
import torch
from torch_pinn.model import MLP,sample_points
from torch_pinn.operators import CASES,METHODS,evaluate
from torch_pinn.problem import constraints,loss
from torch_pinn.records import save,sync,digest,run_record


def benchmark(args):
    rows=[]
    dim=3 if args.case=='kdv2d' else 2
    for seed in args.seeds:
        base=MLP(dim,args.width,args.depth,seed+2,device=args.device)
        points=sample_points(seed+1,args.batch,dim,device=args.device)
        data=constraints(128,device=args.device) if dim==2 else None
        hashes=dict(initial_parameter_hash=digest(base.parameters()),points_hash=digest((points,)))
        for phase in ('loss','loss_backward','adam_step'):
            model=copy.deepcopy(base)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,foreach=False)
            def call():
                optimizer.zero_grad(set_to_none=True)
                value=loss(model,points,data,args.method)[0] if dim==2 else evaluate(model,points,args.case,args.method)
                if phase!='loss':value.backward()
                if phase=='adam_step':optimizer.step()
                return value
            sync(args.device);start=time.perf_counter();call();sync(args.device)
            cold=time.perf_counter()-start
            for _ in range(args.warmups):call()
            sync(args.device)
            # Warmup must not advance the measured initial state.
            model.load_state_dict(base.state_dict());optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,foreach=False)
            model.zero_grad(set_to_none=True)
            if str(args.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(args.device)
            samples=[]
            for _ in range(args.repeats):
                sync(args.device);start=time.perf_counter();value=call();sync(args.device)
                samples.append(1000*(time.perf_counter()-start))
                value=value.detach()
            row=dict(case=args.case,method=args.method,batch=args.batch,seed=seed,phase=phase,
                     task='complete_pinn_with_ic_bc' if dim==2 else 'pde_loss_mean_residual_squared',
                     times_ms=samples,median_ms=statistics.median(samples),cold_seconds=cold,
                     last_loss=float(value.detach()),**hashes,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith('cuda') else None,
                     peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device) if str(args.device).startswith('cuda') else None,
                     memory_scope='stage allocator high-water; reserved cache includes earlier stages')
            if not torch.isfinite(value):raise RuntimeError('nonfinite benchmark')
            rows.append(row);save(args.out/'results.json',rows)
            print(f'{args.case} {args.method} {phase} seed={seed} median_ms={row["median_ms"]:.5f}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--case',choices=CASES,required=True)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--batch',type=int,default=400)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--seeds',nargs='+',type=int,default=[20260908,20260909,20260910])
    p.add_argument('--warmups',type=int,default=5)
    p.add_argument('--repeats',type=int,default=15)
    args=p.parse_args()
    if min(args.batch,args.repeats)<1 or args.warmups<0:p.error('invalid timing sizes')
    run_record(args,benchmark)
