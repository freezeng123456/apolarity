"""Paired PINN training to an exact update count, saving numeric curves only."""
import argparse
import csv
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from train_fixed_wall import PROBLEMS, full_loss, prediction_errors, sample
from torch_pinn.constraint_sampling import ConstraintSampler, as_tensors, batch_hash
from torch_pinn.model import MLP
from torch_pinn.records import digest, run_record, save, sync


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', choices=PROBLEMS, required=True)
    parser.add_argument('--method', choices=('nested_jvp', 'shared_jet_linear'), required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--expected-device', choices=('V100', 'T4', 'RTX 3080'), default='V100')
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--batch', type=int, default=400)
    parser.add_argument('--constraint-side', type=int, default=0)
    parser.add_argument('--eval-points', type=int, default=10000)
    parser.add_argument('--eval-seed-base', type=int, default=20261400)
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args(argv)
    if args.lr is None:
        args.lr = 1e-4 if args.case == 'ch2d' else 1e-5
    if min(args.updates, args.eval_every, args.width, args.batch, args.eval_points) < 1:
        parser.error('update, evaluation and network counts must be positive')
    if args.depth < 2 or args.constraint_side < 0 or args.constraint_side == 1:
        parser.error('invalid depth or boundary sample count')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('learning rate must be finite and positive')
    if not 0 <= args.eval_seed_base < 2**32-100:
        parser.error('invalid evaluation seed')
    return args


def train(args):
    process_started = time.perf_counter()
    device = torch.device(args.device)
    module = PROBLEMS[args.case]
    dim = 2 if args.case == 'kdv1d' else 3
    side = args.constraint_side or (128 if dim == 2 else 16)
    model = MLP(dim, args.width, args.depth, args.seed+2, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(.9, .999), eps=1e-8, foreach=False)
    sampler = ConstraintSampler(args.case, side, args.seed)
    interior_rng = np.random.default_rng(args.seed+1)
    tests = sample(np.random.default_rng(args.eval_seed_base+1), args.eval_points, args.case)
    terminal = sample(np.random.default_rng(args.eval_seed_base+2), args.eval_points, args.case)
    terminal[:, -1] = 1
    test_tensors = {
        'spacetime': torch.as_tensor(tests, device=device),
        'terminal': torch.as_tensor(terminal, device=device),
    }
    initial_hash = digest(model.parameters())
    # Derivative warm-up uses neither the training random streams nor an update.
    warm_points = torch.as_tensor(sample(np.random.default_rng(20260001),
                                         min(args.batch, 8), args.case), device=device)
    warm_data = module.constraints(side, device=device)
    warm_loss, warm_parts = full_loss(module, model, warm_points, warm_data, args.method)
    warm_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    sync(device)
    del warm_loss, warm_parts, warm_points, warm_data

    training_seconds = 0.0
    evaluation_seconds = 0.0
    interior_digest = hashlib.sha256()
    constraint_digest = hashlib.sha256()
    curve_path = args.out/'curve.csv'
    fields = ('step', 'time_s', 'spacetime_relative_l2', 'terminal_relative_l2', 'loss')
    with curve_path.open('w', newline='', buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        def evaluate(step, loss):
            nonlocal evaluation_seconds
            sync(device)
            began = time.perf_counter()
            metrics = {key: prediction_errors(module, model, points)[0]
                       for key, points in test_tensors.items()}
            if not all(math.isfinite(value) for values in metrics.values() for value in values.values()):
                raise FloatingPointError('nonfinite test error')
            sync(device)
            evaluation_seconds += time.perf_counter()-began
            writer.writerow(dict(step=step, time_s=training_seconds,
                spacetime_relative_l2=metrics['spacetime']['relative_l2'],
                terminal_relative_l2=metrics['terminal']['relative_l2'], loss=loss))
            stream.flush()
            print(f'{args.case}/{args.method} step={step}/{args.updates} '
                  f'train_s={training_seconds:.3f} '
                  f'RE={metrics["spacetime"]["relative_l2"]:.7g}', flush=True)
            return metrics

        metrics = evaluate(0, None)
        for step in range(1, args.updates+1):
            began = time.perf_counter()
            points_np = sample(interior_rng, args.batch, args.case)
            points = torch.as_tensor(points_np, device=device)
            constraints_np = sampler.sample_numpy()
            data = as_tensors(args.case, constraints_np, device)
            optimizer.zero_grad(set_to_none=True)
            value, parts = full_loss(module, model, points, data, args.method)
            value.backward()
            if not bool(torch.isfinite(value)) or not all(
                    p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError('nonfinite objective or gradient')
            optimizer.step()
            sync(device)
            loss = float(value.detach())
            interior_digest.update(points_np.tobytes())
            constraint_digest.update(batch_hash(constraints_np).encode())
            training_seconds += time.perf_counter()-began
            del value, parts, points, data, constraints_np
            if step % args.eval_every == 0 or step == args.updates:
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise FloatingPointError('nonfinite parameters')
                metrics = evaluate(step, loss)

    final_hash = digest(model.parameters())
    if initial_hash == final_hash:
        raise AssertionError('training did not change parameters')
    save(args.out/'summary.json', dict(case=args.case, method=args.method, seed=args.seed,
        steps=args.updates, target_updates=args.updates, terminal='target_updates',
        wall_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
        process_wall_seconds=time.perf_counter()-process_started,
        steps_per_second=args.updates/training_seconds, final_training_loss=loss,
        initial_hash=initial_hash, final_hash=final_hash, errors=metrics,
        interior_samples_sha256=interior_digest.hexdigest(),
        constraint_samples_sha256=constraint_digest.hexdigest(),
        spacetime_test_sha256=hashlib.sha256(tests.tobytes()).hexdigest(),
        terminal_test_sha256=hashlib.sha256(terminal.tobytes()).hexdigest(),
        constraint_sampling='resampled', checkpoints_saved=False,
        timing_scope='Sum of synchronized training updates: sampling, differentiation, backward, '
                     'Adam, finite checks and sampling digests; warm-up, test evaluation, curve '
                     'output and setup excluded.'))


if __name__ == '__main__':
    run_record(parse_args(), train)
