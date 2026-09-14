"""Isolated KdV candidate timing, with the original timing policy unchanged."""
import argparse
import hashlib
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import jax
import jaxlib
import numpy as np

from pde_benchmark import check, digest, independent_reference, inputs, save
from jax_apolarity_bench.kdv_optimized import METHODS, make_kdv_evaluator

HERE = Path(__file__).resolve().parent


def verify(args):
    rows = []
    for seed in args.seeds:
        params, points = inputs(seed, 'kdv2d', 3, args.width, args.depth)
        expected, components = independent_reference('kdv2d', params, points)
        jax.block_until_ready((expected, components))
        for method in METHODS:
            value, d = make_kdv_evaluator(method, components=True)(params, points)
            row = dict(method=method, seed=seed, value=check(value, expected),
                       components={k: check(d[k], v) for k, v in components.items()})
            check(make_kdv_evaluator(method)(params, points), expected)
            rows.append(row)
            save(args.out / 'verification.json', rows)
            print(f'VERIFIED {method} seed={seed}', flush=True)


def benchmark(args, device):
    evaluate = make_kdv_evaluator(args.method)
    params, points = inputs(args.seeds[0], 'kdv2d', args.batch, args.width, args.depth)
    start = time.perf_counter()
    jax.block_until_ready(evaluate(params, points))
    cold_s = time.perf_counter() - start
    rows = []
    for seed in args.seeds:
        params, points = inputs(seed, 'kdv2d', args.batch, args.width, args.depth)
        hashes = dict(params_sha256=digest(params), points_sha256=digest(points))
        for _ in range(args.warmups):
            jax.block_until_ready(evaluate(params, points))
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            value = evaluate(params, points)
            jax.block_until_ready(value)
            samples.append(1000*(time.perf_counter()-start))
        value = np.asarray(value)
        assert value.shape == (args.batch,) and value.dtype == np.float32 and np.isfinite(value).all()
        stats = device.memory_stats() or {}
        row = dict(case='kdv2d', method=args.method, batch=args.batch, seed=seed,
                   times_ms=samples, median_ms=statistics.median(samples), **hashes,
                   first_process_call_seconds=cold_s, allocator_peak_bytes=stats.get('peak_bytes_in_use'))
        checked_value, components = make_kdv_evaluator(args.method, components=True)(params, points)
        check(checked_value, value)
        arrays = {k: np.asarray(v) for k, v in components.items()}
        assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in arrays.values())
        np.savez(args.out / f'values_{seed}.npz', value=value, **arrays)
        rows.append(row)
        save(args.out / 'results.json', rows)
        print(f'{args.method} batch={args.batch} seed={seed} median_ms={row["median_ms"]:.6f}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--require-gpu', action='store_true')
    p.add_argument('--batch', type=int, default=100)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--seeds', type=int, nargs='+', default=[20260908, 20260909, 20260910])
    p.add_argument('--warmups', type=int, default=10)
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if (not args.verify and args.method is None) or min(args.batch, args.width, args.repeats) < 1 or args.warmups < 0:
        p.error('invalid method or timing sizes')
    if len(set(args.seeds)) != len(args.seeds):
        p.error('duplicate seeds')
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out / 'status.txt'
    status.write_text('PREPARING\n')
    try:
        jax.config.update('jax_default_matmul_precision', 'highest')
        jax.config.update('jax_enable_x64', False)
        device = jax.devices()[0]
        if args.require_gpu:
            assert jax.default_backend() == 'gpu' and 'T4' in device.device_kind
            assert jax.__version__ == jaxlib.__version__ == '0.4.38'
        sources = {str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in sorted([*HERE.glob('*.py'), HERE / 'KDV_OPTIMIZATION.md', *HERE.joinpath('src').rglob('*.py')])}
        save(args.out / 'config.json', {**vars(args), 'out': str(args.out),
             'case': 'kdv2d', 'protocol': 'kdv_bounded_optimization_v1',
             'source_commit': subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip(),
             'source_files_sha256': sources, 'python': sys.executable, 'python_version': sys.version,
             'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'numpy': np.__version__,
             'backend': jax.default_backend(), 'device_kind': device.device_kind,
             'dtype': 'float32', 'precision': 'highest', 'outer_jit': False, 'parameter_backward': False,
             'parameters_and_inputs_dynamic': True,
             'environment': {k: os.environ.get(k) for k in ('PATH', 'CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'OMP_NUM_THREADS')},
             'schedule': {'directions': 6, 'jets_by_order': {'4': 6}} if args.method in METHODS[2:] else None,
             'memory_note': 'Allocator process high-water diagnostic, not an isolated operation peak.'})
        status.write_text('RUNNING\n')
        verify(args) if args.verify else benchmark(args, device)
        status.write_text('COMPLETE\n')
    except Exception as exc:
        save(args.out / 'failure.json', dict(error=repr(exc)))
        status.write_text('FAILED\n')
        raise


if __name__ == '__main__':
    main()
