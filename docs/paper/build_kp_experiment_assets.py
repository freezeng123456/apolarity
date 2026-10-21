"""Reproduce the paper's KP-II table and five-seed mean curve from sealed CSVs."""
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
RESULTS = ROOT / 'results/kp_fp32_v100_20261021'
SEEDS = tuple(range(20260936, 20260941))
METHODS = ('shared_jet_linear', 'stdepp')
LABELS = ('WDD', 'STDE++')
COLORS = ('#1856a5', '#c63d2f')
TARGET = 20000


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scientific(value):
    mantissa, exponent = f'{value:.3e}'.split('e')
    return mantissa + r'\times10^{' + str(int(exponent)) + '}'


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify the installed assets and data.')
    args = parser.parse_args()
    expected_steps = np.arange(0, TARGET + 1, 100)
    selected, all_records, checks = [], [], []
    curves = {}
    source_hashes = json.loads((RESULTS / 'source/SOURCE_MANIFEST.json').read_text())
    for name, expected in source_hashes.items():
        assert sha256(RESULTS / 'source' / name) == expected, name
    for seed in range(20260934, 20260941):
        pair = {}
        for method in METHODS:
            cell = RESULTS / 'cells' / f'kp2d_seed{seed}_{method}'
            for line in (cell / 'SHA256SUMS').read_text().splitlines():
                expected, name = line.split(None, 1)
                assert sha256(cell / name.lstrip('*')) == expected, str(cell / name)
            config = json.loads((cell / 'config.json').read_text())
            summary = json.loads((cell / 'summary.json').read_text())
            assert (cell / 'status.txt').read_text().strip() == 'COMPLETE'
            assert config['source_sha256'] == source_hashes
            assert config['dtype'] == 'float32' and config['lr'] == 1e-4
            assert config['device_kind'] == 'Tesla V100-SXM2-32GB'
            assert config['batch'] == 400 and config['width'] == 128 and config['depth'] == 5
            assert summary['steps'] == summary['target_updates'] == TARGET
            assert not summary['checkpoints_saved']
            with (cell / 'curve.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            steps = np.array([int(r['step']) for r in rows])
            times = np.array([float(r['time_s']) for r in rows])
            errors = np.array([float(r['spacetime_relative_l2']) for r in rows])
            assert np.array_equal(steps, expected_steps)
            assert times[0] == 0 and np.all(np.diff(times) > 0)
            assert np.isfinite(times).all() and np.isfinite(errors).all() and np.all(errors > 0)
            assert math.isclose(times[-1], summary['wall_seconds'], rel_tol=1e-14)
            assert math.isclose(errors[-1], summary['errors']['spacetime']['relative_l2'], rel_tol=1e-14)
            row = dict(seed=seed, method=method, updates=TARGET,
                       training_seconds=float(times[-1]), RE=float(errors[-1]),
                       paper_cohort=seed in SEEDS, curve_sha256=sha256(cell / 'curve.csv'))
            all_records.append(row)
            if seed in SEEDS:
                selected.append(row)
                curves[seed, method] = (times, errors)
            pair[method] = summary
        matched = {k: pair[METHODS[0]][k] == pair[METHODS[1]][k] for k in (
            'initial_hash', 'interior_samples_sha256', 'constraint_samples_sha256', 'spacetime_test_sha256')}
        assert all(matched.values())
        checks.append(dict(seed=seed, matched=matched))

    means = {}
    for method in METHODS:
        times = np.array([curves[seed, method][0] for seed in SEEDS])
        errors = np.array([curves[seed, method][1] for seed in SEEDS])
        means[method] = (times.mean(axis=0), errors.mean(axis=0), errors.std(axis=0, ddof=1))
    summary = {}
    for method in METHODS:
        group = [r for r in selected if r['method'] == method]
        summary[method] = {metric: dict(mean=statistics.mean(r[metric] for r in group),
                                       sample_sd=statistics.stdev(r[metric] for r in group))
                           for metric in ('training_seconds', 'RE')}
    ratios = [curves[s, 'stdepp'][0][-1] / curves[s, 'shared_jet_linear'][0][-1] for s in SEEDS]
    summary['paired_speedup'] = dict(mean=statistics.mean(ratios), sample_sd=statistics.stdev(ratios))
    manifest = dict(seeds=list(SEEDS), retained_seeds=list(range(20260934, 20260941)),
                    cohort_selection='User selected seeds 36-40 after inspecting pilot curves; seeds 34-35 remain in cells/ and all_seed_metrics.csv.',
                    runs=10, updates=TARGET, observations=10*len(expected_steps),
                    x_axis='Arithmetic mean training time at common update indices',
                    aggregation='Arithmetic mean RE at common update indices',
                    band='Mean to mean plus one sample standard deviation (ddof=1)',
                    smoothing=False, interpolation=False, extrapolation=False,
                    summary=summary, paired_checks=checks,
                    source_fingerprint='5363b5b123d0309617d69dc0ff6bae7626a612f3bee7aa64d23818b9076df2e4',
                    curve_hashes={f'{r["seed"]}_{r["method"]}': r['curve_sha256'] for r in all_records})
    asset_manifest = PAPER / 'data/kp/asset_manifest.json'
    if args.check:
        installed = json.loads(asset_manifest.read_text())
        assert installed['summary'] == summary and installed['seeds'] == list(SEEDS)
        assert installed['curve_hashes'] == manifest['curve_hashes']
        for name, expected in installed['assets'].items():
            assert sha256(PAPER / name) == expected, name
        print(json.dumps(dict(status='PASS', seeds=SEEDS, verified_cells=14, summary=summary)))
        return

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 11,
                         'axes.linewidth': .8, 'path.simplify': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(5.7, 3.35))
    for method, label, color in zip(METHODS, LABELS, COLORS):
        time, mean, sd = means[method]
        ax.fill_between(time, mean, mean + sd, color=color, alpha=.16, linewidth=0)
        ax.plot(time, mean, color=color, lw=1.8, label=label)
    end = max(means[method][0][-1] for method in METHODS)
    ax.set(yscale='log', xlim=(0, end*1.015), xlabel='Training wall time (s)', ylabel=r'$\operatorname{RE}$')
    ax.grid(which='major', color='.82', linewidth=.65, alpha=.75)
    ax.grid(which='minor', color='.90', linewidth=.45, alpha=.45)
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(loc='upper right', framealpha=.93, edgecolor='.75')
    fig.tight_layout(pad=.5)
    assets = {}
    for ext in ('pdf', 'png'):
        path = PAPER / f'figures/equal_updates_kp2d.{ext}'
        kwargs = dict(metadata={'CreationDate': None, 'ModDate': None}) if ext == 'pdf' else {}
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=.04, **kwargs)
        assets[str(path.relative_to(PAPER))] = sha256(path)
    plt.close(fig)
    lines = [r'\begin{tabular}{@{}lcc@{}}', r'\toprule',
             r'Method & \shortstack{Training\\time (s)} & $\operatorname{RE}$ \\', r'\midrule']
    for method, label in zip(METHODS, LABELS):
        time, error = summary[method]['training_seconds'], summary[method]['RE']
        lines.append(label + f" & ${time['mean']:.1f} \\pm {time['sample_sd']:.1f}$ & $" +
                     scientific(error['mean']) + r'\pm' + scientific(error['sample_sd']) + r'$ \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    table = PAPER / 'tables/equal_updates_kp2d.tex'
    table.write_text('\n'.join(lines)+'\n')
    assets[str(table.relative_to(PAPER))] = sha256(table)
    write_csv(RESULTS / 'all_seed_metrics.csv', all_records)
    write_csv(RESULTS / 'paper_seed_metrics.csv', selected)
    mean_rows = []
    for method in METHODS:
        time, mean, sd = means[method]
        for index, step in enumerate(expected_steps):
            mean_rows.append(dict(method=method, step=int(step), mean_time_s=float(time[index]),
                                  mean_RE=float(mean[index]), sample_sd_RE=float(sd[index])))
    write_csv(RESULTS / 'mean_curves.csv', mean_rows)
    (RESULTS / 'summary.json').write_text(json.dumps(manifest, indent=2)+'\n')
    asset_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest['assets'] = assets
    asset_manifest.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', seeds=SEEDS, summary=summary), indent=2))


if __name__ == '__main__':
    main()
