"""Equal wall-time PINN training; immutable numerical operators are reused."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from torch_pinn.model import MLP
from torch_pinn.constraint_sampling import ConstraintSampler, batch_hash, as_tensors
from torch_pinn.problem import constraints, grid, loss, validate
from torch_pinn.records import digest, run_record, save, sync


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--method', choices=['nested_jvp', 'shared_jet_linear'], required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--expected-device',choices=('T4','V100','RTX 3080'),default='T4')
    p.add_argument('--constraint-sampling',choices=('fixed','resampled'),default='resampled')
    p.add_argument('--seed', type=int, default=20260918)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--batch', type=int, default=400)
    p.add_argument('--constraints', type=int, default=128)
    p.add_argument('--seconds', type=float, default=600)
    p.add_argument('--eval-seconds', type=float, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--steps', type=int, default=0, help='Positive value selects fixed-step linear-decay protocol')
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--val-nx', type=int, default=128)
    p.add_argument('--val-nt', type=int, default=51)
    args = p.parse_args(argv)
    if args.steps < 0 or args.eval_every < 1:
        p.error('invalid step count or evaluation interval')
    if min(args.seconds, args.eval_seconds, args.lr) <= 0 or min(args.batch, args.width) < 1 or min(args.depth, args.constraints, args.val_nx, args.val_nt) < 2:
        p.error('invalid time, learning rate or sizes')
    return args


def train(args):
    device = torch.device(args.device)
    model = MLP(2, args.width, args.depth, args.seed + 2, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .999), eps=1e-8, foreach=False)
    rng = np.random.default_rng(args.seed + 1)
    data = constraints(args.constraints, device=device)
    constraint_sampler = ConstraintSampler('kdv1d', args.constraints, args.seed)
    resampled = getattr(args, 'constraint_sampling', 'resampled') == 'resampled'
    constraint_hashes = []
    fixed_np = np.random.default_rng(20260999).random((400, 2)).astype(np.float32)
    fixed_np[:, 0] = 2 * fixed_np[:, 0] - 1
    fixed_points = torch.as_tensor(fixed_np, device=device)
    validation_points = grid(args.val_nx, args.val_nt, device=device)
    np.savez(args.out / 'evaluation_points.npz', loss_points=fixed_np,
             solution_points=validation_points.cpu().numpy(),
             **{k: v.cpu().numpy() for k, v in zip(('initial', 'left', 'right'), data)})
    initial_hash = digest(model.parameters())
    metrics, assessments, batch_hashes = [], [], []
    optimizer_seconds = 0.0

    def checkpoint(step, name):
        torch.save(dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        sampler_state=rng.bit_generator.state, constraint_sampler_state=constraint_sampler.state, optimizer_seconds=optimizer_seconds,
                        configuration={**vars(args), 'out': str(args.out)}), args.out / name)

    def assess(step, model_time):
        sync(device)
        tick = time.perf_counter()
        with torch.no_grad():
            value, parts = loss(model, fixed_points, data, 'nested_jvp')
        result, _, _ = validate(model, validation_points, data, fixed_points[:128])
        sync(device)
        result.update(step=step, training_wall_seconds=model_time,
                      optimizer_seconds=optimizer_seconds, fixed_loss=float(value),
                      fixed_parts={k: float(v) for k, v in parts.items()},
                      assessment_seconds=time.perf_counter() - tick)
        if not np.isfinite(result['fixed_loss']):
            raise RuntimeError('nonfinite fixed loss')
        assessments.append(result)
        save(args.out / 'validation.json', assessments)
        return result

    checkpoint(0, 'initial.pt')
    assess(0, 0.0)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    started = time.perf_counter()
    next_assessment = args.eval_seconds
    step, model_time = 0, 0.0
    while (step < args.steps if args.steps else time.perf_counter() - started < args.seconds):
        sync(device)
        tick = time.perf_counter()
        batch = rng.random((args.batch, 2)).astype(np.float32)
        batch[:, 0] = 2 * batch[:, 0] - 1
        points = torch.as_tensor(batch, device=device)
        if resampled:
            constraint_np = constraint_sampler.sample_numpy()
            train_data = as_tensors('kdv1d', constraint_np, device)
        else:
            train_data = data
        optimizer.zero_grad(set_to_none=True)
        lr_used = args.lr * (1 - step / args.steps) if args.steps else args.lr
        for group in optimizer.param_groups:
            group['lr'] = lr_used
        value, parts = loss(model, points, train_data, args.method)
        value.backward()
        optimizer.step()
        sync(device)
        elapsed = time.perf_counter() - tick
        optimizer_seconds += elapsed
        step += 1
        next_lr = args.lr * (1 - step / args.steps) if args.steps else args.lr
        for group in optimizer.param_groups:
            group['lr'] = next_lr
        model_time = time.perf_counter() - started
        batch_hashes.append(digest((points,)))
        if resampled: constraint_hashes.append(batch_hash(constraint_np))
        row = dict(step=step, training_wall_seconds=model_time, optimizer_seconds=optimizer_seconds,
                   step_ms=elapsed * 1000, loss=float(value.detach()), lr_used=lr_used, next_lr=next_lr,
                   **{k: float(v.detach()) for k, v in parts.items()})
        if not all(np.isfinite(v) for v in row.values()):
            raise RuntimeError('nonfinite training metric')
        metrics.append(row)
        if (step >= args.steps if args.steps else model_time >= args.seconds):
            break
        if (step % args.eval_every == 0 if args.steps else model_time >= next_assessment):
            record = assess(step, model_time)
            save(args.out / 'metrics.json', metrics)
            save(args.out / 'batch_hashes.json', batch_hashes)
            checkpoint(step, f'checkpoint_{step:06d}.pt')
            print(f'{args.method} time={model_time:.3f}s step={step} fixed_loss={record["fixed_loss"]:.8g} relative_l2={record["relative_l2"]:.6g}', flush=True)
            next_assessment = (int((time.perf_counter() - started) / args.eval_seconds) + 1) * args.eval_seconds
    training_wall_seconds = time.perf_counter() - started
    final = assess(step, model_time)
    save(args.out / 'metrics.json', metrics)
    save(args.out / 'batch_hashes.json', batch_hashes)
    save(args.out / 'constraint_batch_hashes.json', constraint_hashes)
    checkpoint(step, 'final.pt')
    test_points = grid(257, 101, device=device)
    test, prediction, target = validate(model, test_points, data, fixed_points[:128])
    np.savez(args.out / 'predictions.npz', points=test_points.cpu().numpy(),
             prediction=prediction.cpu().numpy(), target=target.cpu().numpy())
    final_hash = digest(model.parameters())
    if initial_hash == final_hash:
        raise AssertionError('parameters unchanged')
    save(args.out / 'summary.json', dict(method=args.method, seed=args.seed, steps=step, constraint_sampling="resampled" if resampled else "fixed",
         initial_parameter_hash=initial_hash, final_parameter_hash=final_hash,
         budget_seconds=None if args.steps else args.seconds, training_wall_seconds=training_wall_seconds,
         requested_steps=args.steps, final_scheduled_lr=optimizer.param_groups[0]['lr'],
         final_model_time_seconds=model_time, optimizer_seconds=optimizer_seconds,
         training_plus_final_diagnostics_seconds=time.perf_counter() - started,
         termination='fixed_steps_linear_decay' if args.steps else 'wall_time_budget', initial_validation=assessments[0], final_validation=final,
         test=test, evaluation_point_hash=digest((fixed_points,)),
         timing_scope='Training wall clock includes sampling, transfer, optimizer, periodic evaluation, hashing, logging and checkpointing. Initialization/initial assessment and final diagnostics excluded. Fixed-step mode stops at requested updates; otherwise stop after crossing wall-time budget.',
         peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None))
    with (args.out / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(metrics)
    print(f'{args.method} COMPLETE steps={step} training_wall={training_wall_seconds:.6f}s fixed_loss={final["fixed_loss"]:.8g}', flush=True)


if __name__ == '__main__':
    run_record(parse_args(), train)
