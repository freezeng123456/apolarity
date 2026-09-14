"""Fixed g-KdV comparison: original JVP/shared controls and linear candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from run_pde_matrix import compare_group, save

SEEDS = (20260908, 20260909, 20260910)
BATCHES = (100, 400, 1600)
METHODS = ('nested_jvp', 'shared_jet', 'shared_jet_linear')


def decision(root):
    comparisons = []
    for batch in BATCHES:
        rows = {m: json.loads((root / f'gkdv1d_{m}_b{batch}' / 'results.json').read_text()) for m in METHODS}
        assert all([r['seed'] for r in rows[m]] == list(SEEDS) for m in METHODS)
        for reference, old, candidate in zip(*(rows[m] for m in METHODS)):
            comparisons.append(dict(batch=batch, seed=reference['seed'],
                speedup_vs_jvp=reference['median_ms']/candidate['median_ms'],
                speedup_vs_original_shared=old['median_ms']/candidate['median_ms']))
    passed = all(r['speedup_vs_original_shared'] >= 1.10 and r['speedup_vs_jvp'] > 1 for r in comparisons)
    return dict(gate='PASS' if passed else 'RETAIN_ORIGINAL', minimum_shared_speedup=1.10,
                all_seeds_and_batches_required=True, comparisons=comparisons)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=3600)
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    commit = subprocess.check_output(['git', '-C', str(here), 'rev-parse', 'HEAD'], text=True).strip()
    args.out.mkdir(parents=True, exist_ok=False)
    cells = [('gkdv1d', m, b) for b in BATCHES for m in METHODS]
    env = os.environ.copy()
    env.update(JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', JAX_DEFAULT_MATMUL_PRECISION='highest')
    env.pop('LD_LIBRARY_PATH', None)
    save(args.out / 'manifest.json', dict(protocol='gkdv_linear_v1', source_commit=commit,
         python=sys.executable, cells=cells, seeds=SEEDS, expected_cells=9,
         expected_records=27, expected_timed_samples=810, expected_verification_records=9,
         width=128, depth=4, warmups=10, repeats=30, baseline='nested_jvp',
         gpu_budget=1, process_concurrency=1, timeout_per_process_s=args.timeout,
         gate='each seed/batch: >=1.10 vs original shared and >1 vs JVP',
         canonical_root=str(args.out.resolve())))
    status = args.out / 'status.txt'
    outcomes = []

    def run(name, options):
        cmd = [sys.executable, str(here / 'gkdv_optimization_benchmark.py'), '--require-gpu',
               '--out', str(args.out / name), *options]
        start = time.time()
        with (args.out / (name + '.log')).open('w') as log:
            try:
                code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = 124
        outcomes.append(dict(cell=name, exit_code=code, elapsed_s=time.time()-start, command=cmd))
        save(args.out / 'progress.json', outcomes)
        print(json.dumps(outcomes[-1]), flush=True)
        if code:
            raise RuntimeError(f'Failed {name}: {code}')

    try:
        status.write_text('VERIFYING\n')
        run('verification', ['--verify'])
        status.write_text('RUNNING\n')
        for case, method, batch in cells:
            run(f'{case}_{method}_b{batch}', ['--method', method, '--batch', str(batch)])
        compare_group(args.out, cells, SEEDS)
        save(args.out / 'decision.json', decision(args.out))
        summary = []
        for _, method, batch in cells:
            rows = json.loads((args.out / f'gkdv1d_{method}_b{batch}' / 'results.json').read_text())
            assert [r['seed'] for r in rows] == list(SEEDS)
            assert all(len(r['times_ms']) == 30 for r in rows)
            summary.append(dict(method=method, batch=batch, median_ms=statistics.median(r['median_ms'] for r in rows)))
        save(args.out / 'summary.json', summary)
        status.write_text('COMPLETE\n')
    except Exception as exc:
        save(args.out / 'failure.json', dict(error=repr(exc)))
        status.write_text('FAILED\n')
        raise
    finally:
        hashes = [hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + str(f.relative_to(args.out))
                  for f in sorted(args.out.rglob('*')) if f.is_file() and f.name != 'SHA256SUMS']
        (args.out / 'SHA256SUMS').write_text('\n'.join(hashes) + '\n')


if __name__ == '__main__':
    main()
