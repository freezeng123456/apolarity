"""Short process-memory measurement, independent of the existing timing table."""
import argparse
import csv
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

import torch

from first_group import evaluate, make_model_and_points, parse_one_based_pattern
from torch_pinn.records import run_record, save, sync


def process_memory_mib():
    output = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True, timeout=5)
    rows = list(csv.reader(output.splitlines()))
    own = [(r[1].strip(), float(r[2])) for r in rows if int(r[0].strip()) == os.getpid()]
    if len(own) != 1:
        raise RuntimeError(f'Expected one allocated GPU for this process, found {len(own)}')
    return own[0]


def measure(args):
    alpha = parse_one_based_pattern(args.target)
    model, points = make_model_and_points(args.seed, args.batch, args.width, args.depth, args.device)
    with torch.no_grad():
        # One full call warms the exact workload without a long timing rerun.
        value = evaluate(model, points, alpha, args.method)
        sync(args.device)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('nonfinite derivative values')
        del value
        process_memory_mib()  # Validate telemetry before starting the window.
        ready = threading.Event()
        stop = threading.Event()
        samples, failures = [], []
        began = time.perf_counter()

        def sample_memory():
            try:
                while not stop.is_set():
                    elapsed = time.perf_counter()-began
                    if elapsed > args.seconds:
                        break
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
            raise RuntimeError('memory sampler did not start')
        calls = 0
        try:
            while time.perf_counter()-began < args.seconds and not failures:
                value = evaluate(model, points, alpha, args.method)
                sync(args.device)
                calls += 1
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError('nonfinite derivative values')
                del value
        finally:
            stop.set()
            worker.join(timeout=6)
        if worker.is_alive() or failures or len(samples) < 2 or not calls:
            raise RuntimeError(f'Invalid memory sample window: {failures}, {len(samples)} samples')
    with (args.out/'memory_samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('time_s', 'gpu_uuid', 'used_mib'))
        writer.writeheader()
        writer.writerows(samples)
    values = [r['used_mib'] for r in samples]
    save(args.out/'summary.json', dict(target=args.target, method=args.method, seed=args.seed,
        batch=args.batch, sample_count=len(values), mean_memory_mib=statistics.mean(values),
        sample_sd_mib=statistics.stdev(values), min_memory_mib=min(values), max_memory_mib=max(values),
        window_requested_seconds=args.seconds, last_sample_seconds=samples[-1]['time_s'],
        completed_calls=calls, process_elapsed_seconds=time.perf_counter()-began,
        metric='Arithmetic mean of nvidia-smi process used_gpu_memory samples during warmed-up repeated derivative evaluation; includes CUDA context and cached allocator memory.',
        timing_rerun=False, checkpoints_saved=False))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--method', choices=('nested_jvp', 'waring_batched'), required=True)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--interval', type=float, default=.1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--expected-device', default='V100', choices=('V100', 'T4', 'RTX 3080'))
    args = parser.parse_args(argv)
    if min(args.batch, args.width, args.seconds, args.interval) <= 0 or args.depth < 2:
        parser.error('positive memory sampling and network settings required')
    return args


if __name__ == '__main__':
    run_record(parse_args(), measure)
