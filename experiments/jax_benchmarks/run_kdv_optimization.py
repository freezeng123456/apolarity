"""Frozen endpoint screen and gated B=400 confirmation, with full artifacts."""
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
METHODS = ('nested_jvp', 'shared_jet', 'shared_jet_fused', 'shared_jet_linear')


def decision(root, batches, candidates):
    details = []
    qualified = []
    for method in candidates:
        ratios = []
        for batch in batches:
            reference = json.loads((root / f'kdv2d_nested_jvp_b{batch}' / 'results.json').read_text())
            current = json.loads((root / f'kdv2d_{method}_b{batch}' / 'results.json').read_text())
            assert [r['seed'] for r in reference] == list(SEEDS)
            assert [r['seed'] for r in current] == list(SEEDS)
            for a, b in zip(reference, current):
                ratio = a['median_ms']/b['median_ms']
                ratios.append(ratio)
                details.append(dict(method=method, batch=batch, seed=a['seed'], speedup_vs_jvp=ratio,
                                    jvp_ms=a['median_ms'], candidate_ms=b['median_ms']))
        if len(ratios) == len(batches)*len(SEEDS) and all(r >= 1.10 for r in ratios):
            qualified.append(method)
    selected = next((m for m in ('shared_jet_linear', 'shared_jet_fused') if m in qualified), None)
    return dict(gate='PASS' if selected else 'STOP_KDV_CASE', minimum_required_ratio=1.10,
                all_seeds_and_batches_required=True, selected_method=selected,
                qualified_methods=qualified, comparisons=details)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['screen', 'confirm'], default='screen')
    p.add_argument('--screen-root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=3600)
    args = p.parse_args()
    here = Path(__file__).resolve().parent
    commit = subprocess.check_output(['git', '-C', str(here), 'rev-parse', 'HEAD'], text=True).strip()
    selected = None
    if args.phase == 'confirm':
        if args.screen_root is None:
            p.error('--screen-root required for confirmation')
        assert (args.screen_root / 'status.txt').read_text().strip() == 'COMPLETE'
        screen = json.loads((args.screen_root / 'decision.json').read_text())
        screen_manifest = json.loads((args.screen_root / 'manifest.json').read_text())
        assert screen['gate'] == 'PASS' and screen_manifest['source_commit'] == commit
        selected = screen['selected_method']
    args.out.mkdir(parents=True, exist_ok=False)
    batches = (100, 1600) if args.phase == 'screen' else (400,)
    methods = METHODS if args.phase == 'screen' else ('nested_jvp', selected)
    cells = [('kdv2d', m, b) for b in batches for m in methods]
    env = os.environ.copy()
    env.update(JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', JAX_DEFAULT_MATMUL_PRECISION='highest')
    env.pop('LD_LIBRARY_PATH', None)
    save(args.out / 'manifest.json', dict(protocol='kdv_bounded_optimization_v1', phase=args.phase,
         source_commit=commit, python=sys.executable, cells=cells, seeds=SEEDS,
         expected_cells=len(cells), expected_records=3*len(cells), expected_timed_samples=90*len(cells),
         width=128, depth=4, warmups=10, repeats=30, baseline='nested_jvp',
         selected_method=selected, screen_root=str(args.screen_root) if args.screen_root else None,
         gpu_budget=1, process_concurrency=1, timeout_per_process_s=args.timeout,
         gate='one candidate must reach >=1.10 speedup on every matched seed/batch',
         canonical_root=str(args.out.resolve())))
    status = args.out / 'status.txt'
    outcomes = []

    def run(name, options):
        cmd = [sys.executable, str(here / 'kdv_optimization_benchmark.py'), '--require-gpu',
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
        if args.phase == 'screen':
            status.write_text('VERIFYING\n')
            run('verification', ['--verify'])
        status.write_text('RUNNING\n')
        for case, method, batch in cells:
            run(f'{case}_{method}_b{batch}', ['--method', method, '--batch', str(batch)])
        compare_group(args.out, cells, SEEDS)
        candidates = METHODS[2:] if args.phase == 'screen' else (selected,)
        save(args.out / 'decision.json', decision(args.out, batches, candidates))
        summary = []
        for _, method, batch in cells:
            rows = json.loads((args.out / f'kdv2d_{method}_b{batch}' / 'results.json').read_text())
            assert [r['seed'] for r in rows] == list(SEEDS)
            assert all(len(r['times_ms']) == 30 for r in rows)
            summary.append(dict(method=method, batch=batch,
                                median_ms=statistics.median(r['median_ms'] for r in rows)))
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
