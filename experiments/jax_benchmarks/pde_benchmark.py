"""Value-only PDE timing cells and independent small-batch correctness gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'src'))
from jax_apolarity_bench.models import init_mlp, mlp_one, sample_upstream_unit_ball_space_time
from jax_apolarity_bench.pde_shared import METHODS, assemble, make_pde_evaluator, schedule_metadata
from jax_apolarity_bench.residuals import KDV2D_TERMS, GKDV1D_TERMS


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(tree):
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        a = np.asarray(leaf)
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def inputs(seed, case, batch, width, depth):
    dim = 3 if case == 'kdv2d' else 2
    params = init_mlp(seed + 2, dim, width, depth)
    points = sample_upstream_unit_ball_space_time(seed + 1, batch, dim - 1)
    jax.block_until_ready((params, points))
    return params, points


def independent_reference(case, params, points):
    """Separate coordinate-JVP term chains; gKdV gradient checked on unexpanded f."""
    dim = points.shape[1]
    basis = jnp.eye(dim, dtype=jnp.float32)
    fun = lambda x: mlp_one(params, x)

    def partial(alpha):
        result = fun
        for axis in alpha:
            previous = result
            result = lambda x, previous=previous, axis=axis: jax.jvp(previous, (x,), (basis[axis],))[1]
        return result

    terms = KDV2D_TERMS if case == 'kdv2d' else GKDV1D_TERMS
    d = {name: jax.vmap(partial(alpha))(points) for name, alpha in terms.items()}
    if case == 'gkdv1d':
        d['u'] = jax.vmap(fun)(points)
        f = lambda x: partial((1,))(x) + fun(x)*partial((0,))(x) + .0025*partial((0, 0, 0))(x)
        fv = jax.vmap(f)(points)
        fg = jax.vmap(jax.grad(f))(points)
        value = jnp.mean(fv**2) + .001*jnp.mean(jnp.sum(fg**2, axis=1))
        return value, {**d, 'f': fv, 'f_x': fg[:, 0], 'f_t': fg[:, 1]}
    value, residual = assemble(case, d)
    return value, {**d, **residual}


def check(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.dtype != np.float32 or not np.isfinite(a).all():
        raise AssertionError(f'invalid array: {a.shape}, {a.dtype}')
    np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-6)
    return {'max_abs_error': float(np.max(np.abs(a-b))),
            'relative_l2_error': float(np.linalg.norm(a-b) / max(float(np.linalg.norm(b)), 1e-30))}


def verify(args):
    rows = []
    for case in ('kdv2d', 'gkdv1d'):
        for seed in args.seeds:
            params, points = inputs(seed, case, 3, args.width, args.depth)
            expected, components = independent_reference(case, params, points)
            jax.block_until_ready((expected, components))
            for method in METHODS:
                value, d = make_pde_evaluator(case, method, components=True)(params, points)
                row = {'case': case, 'seed': seed, 'method': method,
                       'value': check(value, expected),
                       'components': {k: check(d[k], v) for k, v in components.items()}}
                check(make_pde_evaluator(case, method)(params, points), expected)
                rows.append(row)
                save(args.out / 'verification.json', rows)
                print(json.dumps({k: row[k] for k in ('case', 'seed', 'method')}), flush=True)


def benchmark(args, device):
    evaluate = make_pde_evaluator(args.case, args.method)
    params, points = inputs(args.seeds[0], args.case, args.batch, args.width, args.depth)
    start = time.perf_counter()
    jax.block_until_ready(evaluate(params, points))
    first_call_s = time.perf_counter() - start
    rows = []
    for seed in args.seeds:
        params, points = inputs(seed, args.case, args.batch, args.width, args.depth)
        hashes = {'params_sha256': digest(params), 'points_sha256': digest(points)}
        for _ in range(args.warmups):
            jax.block_until_ready(evaluate(params, points))
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            value = evaluate(params, points)
            jax.block_until_ready(value)
            samples.append(1000*(time.perf_counter() - start))
        value = np.asarray(value)
        shape = (args.batch,) if args.case == 'kdv2d' else ()
        if value.shape != shape or not np.isfinite(value).all() or value.dtype != np.float32:
            raise ValueError('invalid output shape, finiteness or dtype')
        stats = device.memory_stats() or {}
        row = dict(case=args.case, method=args.method, batch=args.batch, seed=seed,
                   times_ms=samples, median_ms=statistics.median(samples),
                   first_process_call_seconds=first_call_s, **hashes,
                   allocator_peak_bytes=stats.get('peak_bytes_in_use'),
                   allocator_bytes_in_use=stats.get('bytes_in_use'))
        # Component extraction and host serialization are outside measured intervals.
        checked_value, components = make_pde_evaluator(args.case, args.method, components=True)(params, points)
        check(checked_value, value)
        arrays = {k: np.asarray(v) for k, v in components.items()}
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('nonfinite derivative component')
        np.savez(args.out / f'values_{seed}.npz', value=value, **arrays)
        rows.append(row)
        save(args.out / 'results.json', rows)
        print(json.dumps({k: row[k] for k in ('case', 'method', 'batch', 'seed', 'median_ms')}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', choices=['kdv2d', 'gkdv1d'])
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
    if not args.verify and (args.case is None or args.method is None):
        p.error('--case and --method required for timing')
    if min(args.batch, args.width, args.repeats) < 1 or args.warmups < 0 or len(set(args.seeds)) != len(args.seeds):
        p.error('invalid sizes, repetitions or duplicate seeds')
    args.out.mkdir(parents=True, exist_ok=False)
    status = args.out / 'status.txt'
    status.write_text('PREPARING\n')
    try:
        jax.config.update('jax_default_matmul_precision', 'highest')
        jax.config.update('jax_enable_x64', False)
        device = jax.devices()[0]
        if args.require_gpu and jax.default_backend() != 'gpu':
            raise RuntimeError('GPU required')
        commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip()
        source = {str(f.relative_to(HERE)): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in sorted([*HERE.glob('*pde*py'), HERE / 'PDE_SHARED.md', *HERE.joinpath('src').rglob('*.py')])}
        save(args.out / 'config.json', {**vars(args), 'out': str(args.out),
             'protocol': 'pde_shared_real_no_outer_jit_v1', 'source_commit': commit,
             'source_files_sha256': source, 'python': sys.executable, 'python_version': sys.version,
             'jax': jax.__version__, 'jaxlib': jaxlib.__version__, 'backend': jax.default_backend(),
             'device': str(device), 'device_kind': device.device_kind,
             'dtype': 'float32', 'matmul_precision': 'highest', 'outer_jit': False,
             'parameter_backward': False, 'parameters_and_inputs_dynamic': True,
             'point_sampler': 'uniform radius [0,1], normalized Gaussian spatial direction, uniform time [0,1]',
             'environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'OMP_NUM_THREADS')},
             'schedule': None if args.verify else schedule_metadata(args.case, args.method),
             'memory_note': 'JAX allocator process high-water mark, not isolated operation peak; later seeds may include earlier component extraction.'})
        status.write_text('RUNNING\n')
        verify(args) if args.verify else benchmark(args, device)
        status.write_text('COMPLETE\n')
    except Exception as exc:
        save(args.out / 'failure.json', {'error': repr(exc)})
        status.write_text('FAILED\n')
        raise


if __name__ == '__main__':
    main()
