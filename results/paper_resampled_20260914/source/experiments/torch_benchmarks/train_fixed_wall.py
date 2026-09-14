"""Matched fixed-wall-time, constant-lr comparison; diagnostic evaluation is offline."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch_pinn import problem, problem_kdv2d, problem_ch2d
from torch_pinn.model import MLP
from torch_pinn.constraint_sampling import ConstraintSampler, batch_hash, as_tensors
from torch_pinn.records import save, sync, digest, run_record

PROBLEMS = dict(kdv1d=problem,kdv2d=problem_kdv2d,ch2d=problem_ch2d)


def sample(rng,n,case):
    p=rng.random((n,2 if case=='kdv1d' else 3)).astype(np.float32)
    if case=='ch2d':p[:,:-1]*=math.pi
    else:p[:,:-1]=2*p[:,:-1]-1
    return p


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',choices=PROBLEMS,required=True)
    p.add_argument('--method',choices=('nested_jvp','shared_jet_linear'),required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=20260918)
    p.add_argument('--expected-device',choices=('T4','V100','RTX 3080'),default='T4')
    p.add_argument('--eval-seed-base',type=int,default=20261100,help='Independent evaluation stream; offsets 1,2,3,99.')
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--batch',type=int,default=400)
    p.add_argument('--seconds',type=float,default=600)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--checkpoint-seconds',type=float,default=10)
    p.add_argument('--trajectory-seconds',type=float,default=0,
                   help='Optional model-only snapshots for offline error curves; 0 disables.')
    p.add_argument('--eval-points',type=int,default=10000)
    p.add_argument('--constraint-sampling',choices=('fixed','resampled'),default='resampled')
    p.add_argument('--constraint-side',type=int,default=0)
    p.add_argument('--max-steps',type=int,default=0,help='smoke checks only')
    a=p.parse_args()
    if min(a.seconds,a.lr,a.checkpoint_seconds,a.eval_points,a.batch,a.width)<=0: p.error('positive settings required')
    if a.trajectory_seconds < 0 or not math.isfinite(a.trajectory_seconds):p.error('invalid trajectory interval')
    if not 0 <= a.eval_seed_base < 2**32-100:p.error('invalid evaluation seed base')
    return a


def full_loss(module,model,points,data,method,independent=False):
    if module is problem_kdv2d:return module.loss(model,points,data,method,independent=independent)
    if module is problem_ch2d:return module.loss(model,points,data,'independent' if independent else method)
    return module.loss(model,points,data,'nested_jvp' if independent else method)


def prediction_errors(module,model,points):
    with torch.no_grad():
        pred=torch.cat([model(p) for p in points.split(2048)])
        target=module.exact(points)
        error=pred-target
        return dict(relative_l2=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(target)),
                    linf=float(error.abs().max())),pred.cpu().numpy(),target.cpu().numpy()


def train(a):
    module=PROBLEMS[a.case]
    device=torch.device(a.device)
    dim=2 if a.case=='kdv1d' else 3
    model=MLP(dim,a.width,a.depth,a.seed+2,device=device)
    data=module.constraints(a.constraint_side or (128 if dim==2 else 16),device=device)
    constraint_sampler=ConstraintSampler(a.case,a.constraint_side or (128 if dim==2 else 16),a.seed)
    resampled=getattr(a,'constraint_sampling','resampled')=='resampled'
    constraint_hashes=[]
    opt=torch.optim.Adam(model.parameters(),lr=a.lr,betas=(.9,.999),eps=1e-8,foreach=False)
    rng=np.random.default_rng(a.seed+1)
    # Common task-specific, method-independent test streams; never training RNG.
    fixed=sample(np.random.default_rng(20260999 if a.eval_seed_base==20261100 else a.eval_seed_base+99),400,a.case)
    tests=sample(np.random.default_rng(a.eval_seed_base+1),a.eval_points,a.case)
    end=sample(np.random.default_rng(a.eval_seed_base+2),a.eval_points,a.case);end[:,-1]=1
    extra=sample(np.random.default_rng(a.eval_seed_base+3),2*a.eval_points,a.case)
    np.savez(a.out/'evaluation_points.npz',loss=fixed,spacetime=tests,terminal=end,extra=extra)
    initial_hash=digest(model.parameters())
    # Warm only derivative code, without updating parameters or training RNG.
    warm=torch.as_tensor(sample(np.random.default_rng(20260001),min(a.batch,8),a.case),device=device)
    warm_loss,_=full_loss(module,model,warm,data,a.method)
    warm_loss.backward();opt.zero_grad(set_to_none=True)
    sync(device)
    checkpoints=[]
    def checkpoint(step,stamp):
        name=f'checkpoint_{step:07d}.pt'
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,
                        sampler_state=rng.bit_generator.state,constraint_sampler_state=constraint_sampler.state,training_wall_seconds=stamp),a.out/name)
        checkpoints.append(dict(step=step,time=stamp,file=name))
    checkpoint(0,0.)
    trajectories=[dict(checkpoints[-1])] if a.trajectory_seconds else []
    def trajectory(step,stamp):
        # Reuse a full checkpoint when both schedules select the same update.
        if checkpoints[-1]['step']==step:
            item=dict(checkpoints[-1])
        else:
            name=f'trajectory_{step:07d}.pt'
            torch.save(dict(model=model.state_dict(),step=step,
                            training_wall_seconds=stamp),a.out/name)
            item=dict(step=step,time=stamp,file=name)
        if trajectories and trajectories[-1]['step']==step:trajectories[-1]=item
        else:trajectories.append(item)
    records=[];batch_hashes=[];step=0;optimizer_seconds=0.
    next_save=a.checkpoint_seconds
    next_trajectory=a.trajectory_seconds
    sync(device);started=time.perf_counter()
    with (a.out/'steps.jsonl').open('w',buffering=1) as stream:
        while time.perf_counter()-started<a.seconds:
            tick=time.perf_counter()
            pn=sample(rng,a.batch,a.case)
            p=torch.as_tensor(pn,device=device)
            if resampled:
                constraint_np=constraint_sampler.sample_numpy()
                train_data=as_tensors(a.case,constraint_np,device)
            else:train_data=data
            opt.zero_grad(set_to_none=True)
            value,parts=full_loss(module,model,p,train_data,a.method)
            value.backward()
            if not bool(torch.isfinite(value)) or not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError('nonfinite objective/gradient')
            opt.step();sync(device)
            optimizer_seconds+=time.perf_counter()-tick
            step+=1;stamp=time.perf_counter()-started
            row=dict(step=step,time=stamp,loss=float(value.detach()),lr=opt.param_groups[0]['lr'])
            stream.write(json.dumps(row)+'\n');records.append(row)
            batch_hashes.append(hashlib.sha256(pn.tobytes()).hexdigest())
            if resampled:constraint_hashes.append(batch_hash(constraint_np))
            if stamp>=a.seconds or (a.max_steps and step>=a.max_steps):break
            if stamp>=next_save:
                checkpoint(step,stamp)
                print(f'{a.case}/{a.method} step={step} time={stamp:.2f} loss={row["loss"]:.7g}',flush=True)
                next_save=(int((time.perf_counter()-started)/a.checkpoint_seconds)+1)*a.checkpoint_seconds
            if a.trajectory_seconds and stamp>=next_trajectory:
                trajectory(step,stamp)
                next_trajectory=(int((time.perf_counter()-started)/a.trajectory_seconds)+1)*a.trajectory_seconds
    sync(device);wall=time.perf_counter()-started
    checkpoint(step,wall)
    if a.trajectory_seconds:
        trajectory(step,wall)
        save(a.out/'trajectory_checkpoints.json',trajectories)
    save(a.out/'checkpoints.json',checkpoints)
    save(a.out/'batch_hashes.json',batch_hashes)
    save(a.out/'constraint_batch_hashes.json',constraint_hashes)
    final_hash=digest(model.parameters())
    if final_hash==initial_hash:raise AssertionError('unchanged parameters')
    save(a.out/'training_finished.json',dict(steps=step,wall_seconds=wall))
    # Offline diagnostics: exact same coordinate reference path for both methods.
    fixed_tensor=torch.as_tensor(fixed,device=device)
    curve=[]
    for cp in checkpoints:
        state=torch.load(a.out/cp['file'],map_location=device)
        model.load_state_dict(state['model'])
        with torch.no_grad():
            value,parts=full_loss(module,model,fixed_tensor,data,a.method,independent=True)
        curve.append({**cp,'fixed_loss':float(value),'parts':{k:float(v) for k,v in parts.items()}})
    save(a.out/'loss_curve.json',curve)
    metrics={};arrays={}
    for label,pn in (('spacetime',tests),('terminal',end),('extra',extra)):
        metrics[label],arrays[label+'_prediction'],arrays[label+'_target']=prediction_errors(module,model,torch.as_tensor(pn,device=device))
    metrics['relative_estimate_change']=abs(metrics['extra']['relative_l2']/metrics['spacetime']['relative_l2']-1)
    np.savez(a.out/'predictions.npz',**arrays)
    save(a.out/'summary.json',dict(case=a.case,method=a.method,steps=step,wall_seconds=wall,
         steps_per_second=step/wall,optimizer_seconds=optimizer_seconds,lr=a.lr,
         initial_hash=initial_hash,final_hash=final_hash,constraint_sampling="resampled" if resampled else "fixed",errors=metrics,final_fixed_loss=curve[-1]['fixed_loss'],
         timing_scope='Sampling, differentiation, backward, Adam, finite checks, logs, hashes, periodic full/model-only checkpoint IO included; warmup, initial checkpoint, terminal checkpoint and all diagnostic evaluation excluded.',
         terminal=a.max_steps and step>=a.max_steps and 'smoke_step_cap' or 'wall_time_budget'))
    print(f'{a.case}/{a.method} COMPLETE steps={step} wall={wall:.3f}',flush=True)


if __name__=='__main__':run_record(parse_args(),train)
