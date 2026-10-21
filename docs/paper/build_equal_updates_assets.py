"""Build the current KdV figures and tables comparing WDD with STDE++."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
WDD_RESULTS = ROOT / 'results/paper_equal_updates_v100_20261008'
STDE_RESULTS = ROOT / 'results/stdepp_pde_v100_20261016'
TARGETS = {'kdv1d': 25000, 'kdv2d': 20000}
METHODS = ('shared_jet_linear', 'stdepp')
LABELS = ('WDD', 'STDE++')
COLORS = ('#1856a5', '#c63d2f')
SEEDS = tuple(range(20260919, 20260924))
SOURCE_FILES = (
    WDD_RESULTS / 'data/curves.csv', WDD_RESULTS / 'data/pde-results.csv',
    WDD_RESULTS / 'data/provenance.json', STDE_RESULTS / 'curves.csv',
    STDE_RESULTS / 'summaries.json', STDE_RESULTS / 'BASELINE_PAIRS.json',
    STDE_RESULTS / 'SOURCE_MANIFEST.json',
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def scientific(value):
    mantissa, exponent = f'{value:.3e}'.split('e')
    return mantissa + r'\times10^{' + str(int(exponent)) + '}'


def stats(values):
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for line in (WDD_RESULTS / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if name.startswith('data/'):
            assert sha256(WDD_RESULTS / name) == expected, name
    provenance = json.loads((WDD_RESULTS / 'data/provenance.json').read_text())
    assert provenance['status'] == 'PASS'
    assert sha256(WDD_RESULTS / 'data/curves.csv') == provenance['curves_csv_sha256']
    wdd_identities = {(r['case'], int(r['seed'])): r for r in provenance['run_identities']
                      if r['method'] == 'shared_jet_linear'}
    paired_hashes = json.loads((STDE_RESULTS / 'BASELINE_PAIRS.json').read_text())
    stde_summaries = {(r['case'], int(r['seed'])): r
                      for r in json.loads((STDE_RESULTS / 'summaries.json').read_text())}
    datasets = {'shared_jet_linear': read_csv(WDD_RESULTS / 'data/curves.csv'),
                'stdepp': read_csv(STDE_RESULTS / 'curves.csv')}
    endpoints = read_csv(WDD_RESULTS / 'data/pde-results.csv')
    curves, per_seed, checks = {}, [], []
    for case, target in TARGETS.items():
        expected_steps = np.arange(0, target+1, 100)
        for seed in SEEDS:
            baseline = stde_summaries[case, seed]
            matched = {key: baseline[key] == wdd_identities[case, seed][key]
                            == paired_hashes[f'{case}_{seed}'][key]
                       for key in ('initial_hash', 'interior_samples_sha256',
                                   'constraint_samples_sha256', 'spacetime_test_sha256',
                                   'terminal_test_sha256')}
            assert all(matched.values()), (case, seed, matched)
            assert baseline['steps'] == baseline['target_updates'] == target
            assert not baseline['checkpoints_saved']
            checks.append(dict(case=case, seed=seed, matched=matched))
        for method in METHODS:
            times, errors = [], []
            for seed in SEEDS:
                rows = [r for r in datasets[method] if r['case'] == case
                        and r['method'] == method and int(r['seed']) == seed]
                steps = np.array([int(r['step']) for r in rows])
                t = np.array([float(r['time_s']) for r in rows])
                y = np.array([float(r['spacetime_relative_l2']) for r in rows])
                assert np.array_equal(steps, expected_steps), (case, method, seed)
                assert t[0] == 0 and np.all(np.diff(t) > 0)
                assert np.isfinite(t).all() and np.isfinite(y).all() and np.all(y > 0)
                if method == 'shared_jet_linear':
                    end, = [r for r in endpoints if r['case'] == case and r['method'] == method
                            and int(r['seed']) == seed]
                    assert int(end['updates']) == target
                    expected_time, expected_error = float(end['training_seconds']), float(end['spacetime_relative_l2'])
                else:
                    end = stde_summaries[case, seed]
                    expected_time, expected_error = end['wall_seconds'], end['errors']['spacetime']['relative_l2']
                assert math.isclose(t[-1], expected_time, rel_tol=1e-14)
                assert math.isclose(y[-1], expected_error, rel_tol=1e-14)
                times.append(t)
                errors.append(y)
                per_seed.append(dict(case=case, method=method, seed=seed, updates=target,
                                     training_seconds=float(t[-1]), RE=float(y[-1])))
            curves[case, method] = (np.mean(times, axis=0), np.mean(errors, axis=0),
                                    np.std(errors, axis=0, ddof=1))
    summary = {}
    for case in TARGETS:
        group = [r for r in per_seed if r['case'] == case]
        summary[case] = {method: {metric: stats([r[metric] for r in group if r['method'] == method])
                                 for metric in ('training_seconds', 'RE')} for method in METHODS}
        pairs = {s: {r['method']: r['training_seconds'] for r in group if r['seed'] == s} for s in SEEDS}
        summary[case]['paired_speedup'] = stats([pairs[s]['stdepp']/pairs[s]['shared_jet_linear'] for s in SEEDS])
    manifest = dict(protocol='equal_updates_WDD_vs_STDE++', targets=TARGETS,
                    methods=list(METHODS), seeds=list(SEEDS), runs=len(per_seed),
                    source_files={str(p.relative_to(ROOT)): sha256(p) for p in SOURCE_FILES},
                    measurement_batches='WDD: 2026-10-08; STDE++: 2026-10-16. Matched hardware/runtime and paired hashes.',
                    error_metric='Space-time relative L2, denoted RE',
                    aggregation='Arithmetic means of time and error at common update indices',
                    band='Mean to mean plus one sample standard deviation (ddof=1)',
                    smoothing=False, interpolation=False, extrapolation=False,
                    paired_checks=checks, summary=summary, assets={})
    dest = PAPER / 'data/equal_updates'
    if args.check:
        installed = json.loads((dest / 'asset_manifest.json').read_text())
        for key in ('targets', 'methods', 'seeds', 'source_files', 'summary'):
            assert installed[key] == manifest[key], key
        for name, expected in installed['assets'].items():
            assert sha256(PAPER / name) == expected, name
        print(json.dumps(dict(status='PASS', runs=len(per_seed), paired_checks=len(checks), summary=summary)))
        return
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 11,
                         'axes.linewidth': .8, 'path.simplify': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    for case in TARGETS:
        lines = [r'\begin{tabular}{@{}lcc@{}}', r'\toprule',
                 r'Method & \shortstack{Training\\time (s)} & $\operatorname{RE}$ \\', r'\midrule']
        for method, label in zip(METHODS, LABELS):
            row = summary[case][method]
            time, error = row['training_seconds'], row['RE']
            lines.append(label + f" & ${time['mean']:.1f} \\pm {time['sample_sd']:.1f}$ & $" +
                         scientific(error['mean']) + r'\pm' + scientific(error['sample_sd']) + r'$ \\')
        lines.extend([r'\bottomrule', r'\end{tabular}'])
        table = PAPER / f'tables/resampled_wall_{case}.tex'
        table.write_text('\n'.join(lines)+'\n')
        manifest['assets'][str(table.relative_to(PAPER))] = sha256(table)
        fig, ax = plt.subplots(figsize=(5.7, 3.35))
        for method, label, color in zip(METHODS, LABELS, COLORS):
            time, mean, sd = curves[case, method]
            ax.fill_between(time, mean, mean+sd, color=color, alpha=.16, linewidth=0)
            ax.plot(time, mean, color=color, lw=1.8, label=label)
        end = max(curves[case, method][0][-1] for method in METHODS)
        ax.set(yscale='log', xlim=(0, end*1.015), xlabel='Training wall time (s)', ylabel=r'$\operatorname{RE}$')
        ax.grid(which='major', color='.82', linewidth=.65, alpha=.75)
        ax.grid(which='minor', color='.90', linewidth=.45, alpha=.45)
        ax.tick_params(direction='in', top=True, right=True)
        ax.legend(loc='upper right', framealpha=.93, edgecolor='.75')
        fig.tight_layout(pad=.5)
        for ext in ('pdf', 'png'):
            path = PAPER / f'figures/resampled_wall_{case}.{ext}'
            kw = dict(metadata={'CreationDate': None, 'ModDate': None}) if ext == 'pdf' else {}
            fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.04, **kw)
            manifest['assets'][str(path.relative_to(PAPER))] = sha256(path)
        plt.close(fig)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'per_seed_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_seed[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(per_seed)
    (dest / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', runs=len(per_seed), paired_checks=len(checks), summary=summary), indent=2))


if __name__ == '__main__':
    main()
