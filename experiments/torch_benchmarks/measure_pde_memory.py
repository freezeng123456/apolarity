"""Short V100 memory comparison of complete PINN training updates."""
import argparse
import csv
import hashlib
from pathlib import Path
import statistics
import threading
import time

import numpy as np
import torch

from measure_partial_memory import process_memory_mib
from train_fixed_wall import PROBLEMS, full_loss, sample
from torch_pinn.constraint_sampling import ConstraintSampler, as_tensors
from torch_pinn.model import MLP
from torch_pinn.records import digest, run_record, save, sync


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', choices=PROBLEMS, required=True)
    parser.add_argument('--method', choices=('nested_jvp', 'shared_jet_linear'), required=True)
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--batch', type=int, default=400)
    parser.add_argument('--constraint-side', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--interval', type=float, default=.1)
    parser.add_argument('--warmup-updates', type=int, default=2)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--expected-device', default='V100', choices=('V100',))
    args = parser.parse_args(argv)
    if (min(args.width, args.batch, args.seconds, args.interval, args.warmup_updates) <= 0
            or args.depth < 2 or args.constraint_side < 0 or args.constraint_side == 1):
        parser.error('positive sampling and valid network settings required')
    return args


def measure(args):
    module = PROBLEMS[args.case]
    dim = 2 if args.case == 'kdv1d' else 3
    side = args.constraint_side or (128 if dim == 2 else 16)
    model = MLP(dim, args.width, args.depth, args.seed+2, device=args.device)
    optimizer = torch.optim.Adam(model.parameters(),
        lr=1e-4 if args.case == 'ch2d' else 1e-5,
        betas=(.9, .999), eps=1e-8, foreach=False)
    sampler = ConstraintSampler(args.case, side, args.seed)
    interior_rng = np.random.default_rng(args.seed+1)
    initial_hash = digest(model.parameters())
    warm_samples = hashlib.sha256()

    def update(record_samples=False):
        points_np = sample(interior_rng, args.batch, args.case)
        constraints_np = sampler.sample_numpy()
        if record_samples:
            warm_samples.update(points_np.tobytes())
            for name in sorted(constraints_np):
                warm_samples.update(name.encode())
                warm_samples.update(constraints_np[name].tobytes())
        points = torch.as_tensor(points_np, device=args.device)
        data = as_tensors(args.case, constraints_np, args.device)
        optimizer.zero_grad(set_to_none=True)
        value, parts = full_loss(module, model, points, data, args.method)
        value.backward()
        if not bool(torch.isfinite(value)) or not all(
                p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in model.parameters()):
            raise FloatingPointError('nonfinite objective or gradient')
        optimizer.step()
        sync(args.device)
        return float(value.detach())

    for _ in range(args.warmup_updates):
        update(record_samples=True)
    process_memory_mib()
    samples, failures = [], []
    stop, ready = threading.Event(), threading.Event()
    began = time.perf_counter()

    def sample_memory():
        try:
            while not stop.is_set() and time.perf_counter()-began < args.seconds:
                gpu, memory = process_memory_mib()
                elapsed = time.perf_counter()-began
                if elapsed <= args.seconds:
                    samples.append(dict(time_s=elapsed, gpu_uuid=gpu, used_mib=memory))
                ready.set()
                stop.wait(args.interval)
        except Exception as exc:
            failures.append(str(exc))
            ready.set()

    worker = threading.Thread(target=sample_memory, daemon=True)
    worker.start()
    if not ready.wait(6):
        stop.set()
        worker.join(timeout=6)
        raise RuntimeError('memory sampler did not start')
    updates = 0
    try:
        while time.perf_counter()-began < args.seconds and not failures:
            loss = update()
            updates += 1
    finally:
        stop.set()
        worker.join(timeout=6)
    if worker.is_alive() or failures or len(samples) < 2 or not updates:
        raise RuntimeError(f'Invalid memory window: {failures}, {len(samples)} samples')
    final_hash = digest(model.parameters())
    assert final_hash != initial_hash
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    with (args.out/'memory_samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('time_s', 'gpu_uuid', 'used_mib'))
        writer.writeheader()
        writer.writerows(samples)
    values = [r['used_mib'] for r in samples]
    save(args.out/'summary.json', dict(case=args.case, method=args.method, seed=args.seed,
        batch=args.batch, constraint_side=side, warmup_updates=args.warmup_updates,
        sample_count=len(values), mean_memory_mib=statistics.mean(values),
        sample_sd_mib=statistics.stdev(values), min_memory_mib=min(values),
        max_memory_mib=max(values), window_requested_seconds=args.seconds,
        last_sample_seconds=samples[-1]['time_s'], completed_updates=updates,
        initial_hash=initial_hash, warmup_samples_sha256=warm_samples.hexdigest(),
        final_hash=final_hash, final_loss=loss, checkpoints_saved=False,
        metric='Arithmetic mean of nvidia-smi process GPU memory over warmed '
               'complete training updates, including loss, backward and Adam; '
               'includes CUDA context and cached allocator memory.'))


if __name__ == '__main__':
    run_record(parse_args(), measure)
