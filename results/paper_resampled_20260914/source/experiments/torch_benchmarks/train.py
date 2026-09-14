"""One eager PyTorch PINN loop; only the PDE derivative evaluator is selectable."""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch
from torch_pinn.model import MLP
from torch_pinn.problem import constraints,loss,grid,validate
from torch_pinn.records import save,sync,digest,run_record


def train(args):
    device=torch.device(args.device)
    model=MLP(2,args.width,args.depth,args.seed+2,device=device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.999),eps=1e-8,foreach=False)
    rng=np.random.default_rng(args.seed+1)
    validation_rng=np.random.default_rng(20260999)
    data=constraints(args.constraints,device=device)
    validation_points=grid(args.val_nx,args.val_nt,device=device)
    residual_np=validation_rng.random((128,2)).astype(np.float32);residual_np[:,0]=2*residual_np[:,0]-1
    residual_points=torch.as_tensor(residual_np,device=device)
    initial_hash=digest(model.parameters())
    source_hashes=[];metrics=[];validation=[];optimizer_seconds=0.;reached=None;previous_pass=False
    started=time.perf_counter()

    def checkpoint(step,name):
        torch.save(dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler_state=rng.bit_generator.state,optimizer_seconds=optimizer_seconds,
            configuration={**vars(args),'out':str(args.out)}),args.out/name)

    def assess(step):
        nonlocal reached,previous_pass
        record,_,_=validate(model,validation_points,data,residual_points)
        record.update(step=step,optimizer_seconds=optimizer_seconds,wall_seconds=time.perf_counter()-started)
        quality=record['relative_l2']<=.01 and all(record[k]<=.01 for k in ('residual_rms','ic_rms','bc_left_rms','bc_right_rms','bc_dx_rms'))
        if quality and previous_pass and reached is None:reached=validation[-1]['optimizer_seconds']
        previous_pass=quality
        validation.append(record);save(args.out/'validation.json',validation)
        return record

    checkpoint(0,'initial.pt');assess(0)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    step=0
    for step in range(1,args.steps+1):
        sync(device);tick=time.perf_counter()
        batch=rng.random((args.batch,2)).astype(np.float32);batch[:,0]=2*batch[:,0]-1
        points=torch.as_tensor(batch,device=device)
        optimizer.zero_grad(set_to_none=True)
        value,parts=loss(model,points,data,args.method)
        value.backward();optimizer.step();sync(device)
        elapsed=time.perf_counter()-tick;optimizer_seconds+=elapsed
        # Hashing, scalar serialization and validation are outside optimizer-loop time.
        source_hashes.append(digest((points,)))
        row=dict(step=step,loss=float(value.detach()),step_ms=elapsed*1000,optimizer_seconds=optimizer_seconds,
                 **{k:float(v.detach()) for k,v in parts.items()})
        if not all(np.isfinite(v) for v in row.values()):raise RuntimeError('nonfinite training metric')
        metrics.append(row)
        if step%args.eval_every==0 or step==args.steps or optimizer_seconds>=args.max_seconds:
            val=assess(step)
            save(args.out/'metrics.json',metrics)
            save(args.out/'batch_hashes.json',source_hashes)
            checkpoint(step,f'checkpoint_{step:06d}.pt')
            print(f'{args.method} step={step} loss={row["loss"]:.6g} relative_l2={val["relative_l2"]:.6g}',flush=True)
        if optimizer_seconds>=args.max_seconds:break
    checkpoint(step,'final.pt')
    test_points=grid(257,101,device=device)
    test,prediction,target=validate(model,test_points,data,residual_points)
    np.savez(args.out/'predictions.npz',points=test_points.cpu().numpy(),prediction=prediction.cpu().numpy(),target=target.cpu().numpy())
    final_hash=digest(model.parameters())
    if initial_hash==final_hash:raise AssertionError('no parameter update')
    summary=dict(method=args.method,seed=args.seed,steps=step,initial_parameter_hash=initial_hash,final_parameter_hash=final_hash,
                 optimizer_seconds=optimizer_seconds,total_wall_seconds=time.perf_counter()-started,
                 initial_validation=validation[0],final_validation=validation[-1],test=test,
                 time_to_target_seconds=reached,target_reached=reached is not None,
                 termination='step_budget' if step==args.steps else 'time_budget',
                 initial_training_loss=metrics[0]['loss'],final_training_loss=metrics[-1]['loss'],
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
                 memory_scope='process including periodic validation, not isolated optimizer-only peak')
    save(args.out/'summary.json',summary)
    with (args.out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]),lineterminator='\n');writer.writeheader();writer.writerows(metrics)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--method',choices=['nested_jvp','shared_jet_linear'],required=True)
    p.add_argument('--seed',type=int,default=20260918)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--batch',type=int,default=400)
    p.add_argument('--constraints',type=int,default=128)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--max-seconds',type=float,default=600)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--val-nx',type=int,default=128)
    p.add_argument('--val-nt',type=int,default=51)
    args=p.parse_args()
    if min(args.steps,args.batch,args.eval_every)<1 or min(args.constraints,args.val_nx,args.val_nt)<2 or args.max_seconds<=0:p.error('invalid sizes or time budget')
    run_record(args,train)
