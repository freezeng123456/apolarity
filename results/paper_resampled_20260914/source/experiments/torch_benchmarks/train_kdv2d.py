"""Matched fixed-step, complete-loss 2D KdV training, with auditable checkpoints."""
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch
from torch_pinn.model import MLP
from torch_pinn.constraint_sampling import ConstraintSampler, batch_hash, as_tensors
from torch_pinn.problem_kdv2d import PROTOCOL, constraints, grid, loss, predictions, sample_numpy
from torch_pinn.records import digest, run_record, save, sync


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--method', choices=['nested_jvp', 'shared_jet_linear'], required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--expected-device',choices=('T4','V100','RTX 3080'),default='T4')
    p.add_argument('--constraint-sampling',choices=('fixed','resampled'),default='resampled')
    p.add_argument('--seed', type=int, default=20260921)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--batch', type=int, default=400)
    p.add_argument('--constraint-side', type=int, default=16)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--decay-steps', type=int, default=10000)
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val-nxy', type=int, default=33)
    p.add_argument('--val-nt', type=int, default=11)
    p.add_argument('--test-nxy', type=int, default=65)
    p.add_argument('--test-nt', type=int, default=21)
    a = p.parse_args(argv)
    if min(a.steps, a.decay_steps, a.eval_every, a.width, a.batch, a.lr) <= 0:
        p.error('counts and learning rate must be positive')
    if a.steps > a.decay_steps or min(a.depth, a.constraint_side, a.val_nxy, a.val_nt, a.test_nxy, a.test_nt) < 2:
        p.error('invalid grid/depth or decay horizon')
    return a


def train(a):
    device = torch.device(a.device)
    model = MLP(3, a.width, a.depth, a.seed+2, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(.9, .999), eps=1e-8, foreach=False)
    rng = np.random.default_rng(a.seed+1)
    data = constraints(a.constraint_side, device=device)
    constraint_sampler = ConstraintSampler('kdv2d', a.constraint_side, a.seed)
    resampled = getattr(a, 'constraint_sampling', 'resampled') == 'resampled'
    constraint_hashes = []
    fixed = torch.as_tensor(sample_numpy(np.random.default_rng(20260999), 400), device=device)
    val_points = grid(a.val_nxy, a.val_nt, device=device)
    np.savez(a.out/'evaluation_points.npz', loss_points=fixed.cpu().numpy(),
             solution_points=val_points.cpu().numpy(), **{k:v.cpu().numpy() for k,v in data.items()})
    initial_hash = digest(model.parameters())
    metrics, assessments, hashes = [], [], []
    optimizer_seconds = 0.

    def checkpoint(step, name):
        torch.save(dict(step=step, model=model.state_dict(), optimizer=opt.state_dict(),
                        sampler_state=rng.bit_generator.state, constraint_sampler_state=constraint_sampler.state, optimizer_seconds=optimizer_seconds,
                        configuration={**vars(a), 'out':str(a.out)}, protocol=PROTOCOL), a.out/name)

    def assess(step, model_time):
        sync(device)
        tick = time.perf_counter()
        # Independent per-term JVP, not the shared-jet reconstruction.
        with torch.no_grad():
            value, parts = loss(model, fixed, data, 'nested_jvp', independent=True)
        result, _, _ = predictions(model, val_points)
        result.update(step=step, training_wall_seconds=model_time, optimizer_seconds=optimizer_seconds,
                      fixed_loss=float(value), fixed_parts={k:float(v) for k,v in parts.items()})
        if not all(np.isfinite(v) for k,v in result.items() if k != 'fixed_parts'):
            raise RuntimeError('nonfinite assessment')
        sync(device)
        result['assessment_seconds'] = time.perf_counter()-tick
        assessments.append(result)
        save(a.out/'validation.json', assessments)
        return result

    checkpoint(0, 'initial.pt')
    assess(0, 0.)
    sync(device)
    start = time.perf_counter()
    for index in range(a.steps):
        sync(device)
        tick = time.perf_counter()
        points = torch.as_tensor(sample_numpy(rng, a.batch), device=device)
        if resampled:
            constraint_np = constraint_sampler.sample_numpy()
            train_data = as_tensors('kdv2d', constraint_np, device)
        else:
            train_data = data
        opt.zero_grad(set_to_none=True)
        lr_used = a.lr*(1-index/a.decay_steps)
        for group in opt.param_groups:
            group['lr'] = lr_used
        value, parts = loss(model, points, train_data, a.method)
        value.backward()
        # Fail fast before applying an invalid update; identical check in both paths.
        if not bool(torch.isfinite(value)) or not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise RuntimeError('nonfinite loss or parameter gradient')
        opt.step()
        sync(device)
        elapsed = time.perf_counter()-tick
        optimizer_seconds += elapsed
        step = index+1
        for group in opt.param_groups:
            group['lr'] = a.lr*(1-step/a.decay_steps)
        model_time = time.perf_counter()-start
        hashes.append(digest((points,)))
        if resampled: constraint_hashes.append(batch_hash(constraint_np))
        row = dict(step=step, training_wall_seconds=model_time, optimizer_seconds=optimizer_seconds,
                   step_ms=1000*elapsed, loss=float(value.detach()), lr_used=lr_used,
                   next_lr=opt.param_groups[0]['lr'], **{k:float(v.detach()) for k,v in parts.items()})
        if not all(np.isfinite(v) for v in row.values()):
            raise RuntimeError('nonfinite metric')
        metrics.append(row)
        if step % a.eval_every == 0 or step == a.steps:
            result = assess(step, model_time)
            save(a.out/'metrics.json', metrics)
            save(a.out/'batch_hashes.json', hashes)
            checkpoint(step, 'final.pt' if step == a.steps else f'checkpoint_{step:06d}.pt')
            print(f'{a.method} seed={a.seed} step={step} wall={model_time:.3f}s loss={result["fixed_loss"]:.8g} rel_l2={result["relative_l2"]:.6g}', flush=True)
    save(a.out / 'constraint_batch_hashes.json', constraint_hashes)
    sync(device)
    # Includes the final scheduled assessment and checkpoint, consistently for both methods.
    training_wall = time.perf_counter()-start
    test_points = grid(a.test_nxy, a.test_nt, device=device)
    test, pred, target = predictions(model, test_points)
    if not bool(torch.isfinite(pred).all()):
        raise RuntimeError('nonfinite final prediction')
    np.savez(a.out/'predictions.npz', points=test_points.cpu().numpy(), prediction=pred.cpu().numpy(), target=target.cpu().numpy())
    final_hash = digest(model.parameters())
    if initial_hash == final_hash:
        raise AssertionError('parameters unchanged')
    save(a.out/'summary.json', dict(protocol=PROTOCOL, method=a.method, seed=a.seed, steps=a.steps, constraint_sampling="resampled" if resampled else "fixed",
         requested_steps=a.steps, decay_steps=a.decay_steps, termination='fixed_steps',
         initial_parameter_hash=initial_hash, final_parameter_hash=final_hash,
         training_wall_seconds=training_wall, optimizer_seconds=optimizer_seconds,
         training_plus_final_diagnostics_seconds=time.perf_counter()-start,
         initial_validation=assessments[0], final_validation=assessments[-1], test=test,
         evaluation_point_hash=digest((fixed,)), constraint_hash=digest(data.values()),
         final_scheduled_lr=opt.param_groups[0]['lr'],
         timing_scope='Includes sampling, transfer, finite-gradient checks, updates, hashing, logging, all scheduled evaluations and checkpoints including final; excludes initialization/initial assessment and post-training test diagnostics.'))
    with (a.out/'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(metrics)
    print(f'COMPLETE {a.method} seed={a.seed} wall={training_wall:.3f}s test_l2={test["relative_l2"]:.8g}', flush=True)


if __name__ == '__main__':
    run_record(parse_args(), train)
