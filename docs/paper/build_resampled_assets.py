"""Rebuild the paper's resampled-constraint PDE tables and error curves."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER.parents[1] / 'results/paper_resampled_20260914'
CASES = ('kdv1d', 'kdv2d', 'ch2d')
METHODS = ('nested_jvp', 'shared_jet_linear')
SEEDS = tuple(range(20260919, 20260924))
COMMIT = 'b94fd9ce20f1f6c79bd6195ab6d79097f74c8044'


def stats(values):
    return {'mean': float(np.mean(values)),
            'sample_std': float(np.std(values, ddof=1))}


def scientific(value):
    mantissa, exponent = f'{value:.3e}'.split('e')
    return mantissa + r'\times10^{' + str(int(exponent)) + '}'


def main():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 11,
                         'axes.linewidth': .8, 'path.simplify': False,
                         'pdf.fonttype': 42})
    for folder in ('figures', 'tables', 'data/resampled'):
        (PAPER / folder).mkdir(parents=True, exist_ok=True)
    aggregates, per_seed, manifest = [], [], {}
    for case in CASES:
        records, curves, aggregate = {}, {}, {'case': case}
        for method in METHODS:
            group = []
            for seed in SEEDS:
                cell = RESULTS / f'cells/wall_{case}_seed{seed}_{method}'
                config = json.loads((cell / 'config.json').read_text())
                summary = json.loads((cell / 'summary.json').read_text())
                assert config['constraint_sampling'] == summary['constraint_sampling'] == 'resampled'
                assert config['source_commit'] == COMMIT
                assert config['seconds'] == 1200 and config['max_steps'] == 0
                curve_file = RESULTS / f'curves/{case}_{method}_seed{seed}.json'
                curve = json.loads(curve_file.read_text())
                assert curve['config']['source_commit'] == COMMIT
                assert curve['config']['constraint_sampling'] == 'resampled'
                assert curve['rows'][-1]['step'] == summary['steps']
                for metric in ('spacetime', 'terminal'):
                    assert abs(curve['rows'][-1][metric] - summary['errors'][metric]['relative_l2']) < 1e-7
                row = {'case': case, 'method': method, 'seed': seed,
                       'steps': summary['steps'], 'wall_seconds': summary['wall_seconds'],
                       'spacetime': summary['errors']['spacetime']['relative_l2'],
                       'terminal': summary['errors']['terminal']['relative_l2']}
                group.append(row)
                curves[method, seed] = curve
            records[method] = group
            aggregate[method] = {metric: stats([row[metric] for row in group])
                                 for metric in ('steps', 'spacetime', 'terminal')}
            per_seed.extend(group)
        aggregate['speedup'] = stats([t['steps'] / n['steps'] for n, t in
                                     zip(records[METHODS[0]], records[METHODS[1]])])
        aggregates.append(aggregate)
        lines = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
                 r'Method & Updates & \shortstack{Space--time\\$\operatorname{RE}$} & \shortstack{Terminal\\$\operatorname{RE}$} & $S_{\mathrm{iter}}$ \\',
                 r'\midrule']
        for method, label in zip(METHODS, ('Nested JVP', 'WDD')):
            row = aggregate[method]
            values = [label, f"${row['steps']['mean']:.1f} \\pm {row['steps']['sample_std']:.1f}$"]
            for metric in ('spacetime', 'terminal'):
                values.append('$' + scientific(row[metric]['mean']) + r'\pm' +
                              scientific(row[metric]['sample_std']) + '$')
            speedup = aggregate['speedup']
            values.append('$1$' if method == 'nested_jvp' else
                          f"${speedup['mean']:.2f} \\pm {speedup['sample_std']:.2f}$")
            lines.append(' & '.join(values) + r' \\')
        lines.extend((r'\bottomrule', r'\end{tabular}'))
        (PAPER / f'tables/resampled_wall_{case}.tex').write_text('\n'.join(lines) + '\n')
        grid = np.linspace(max(c['rows'][0]['time'] for c in curves.values()),
                           min(1200., min(c['rows'][-1]['time'] for c in curves.values())), 1201)
        fig, ax = plt.subplots(figsize=(5.7, 3.35))
        for method, color, label in zip(METHODS, ('#c63d2f', '#1856a5'), ('Nested JVP', 'WDD')):
            ys = []
            for seed in SEEDS:
                rows = curves[method, seed]['rows']
                t = np.array([row['time'] for row in rows])
                y = np.array([row['spacetime'] for row in rows])
                assert np.all(np.diff(t) > 0) and np.all(y > 0)
                assert grid[0] >= t[0] and grid[-1] <= t[-1]
                ys.append(10 ** np.interp(grid, t, np.log10(y)))
            mean, sd = np.mean(ys, axis=0), np.std(ys, axis=0, ddof=1)
            ax.fill_between(grid, mean, mean + sd, color=color, alpha=.16, linewidth=0)
            ax.plot(grid, mean, color=color, lw=1.8, label=label)
        ax.set(yscale='log', xlim=(0, 1200), xlabel='Training wall time (s)',
               ylabel=r'Relative $\ell^2$ error (space--time)')
        ax.grid(which='major', color='.82', linewidth=.65, alpha=.75)
        ax.grid(which='minor', color='.90', linewidth=.45, alpha=.45)
        ax.tick_params(direction='in', top=True, right=True)
        ax.legend(loc='upper right', framealpha=.93, edgecolor='.75')
        fig.tight_layout(pad=.5)
        for extension in ('pdf', 'png'):
            fig.savefig(PAPER / f'figures/resampled_wall_{case}.{extension}', dpi=300,
                        bbox_inches='tight', pad_inches=.04)
        plt.close(fig)
        manifest[case] = {'runs': 10, 'observations': sum(len(c['rows']) for c in curves.values()),
                          'display_points': 1201, 'source_commit': COMMIT,
                          'constraint_sampling': 'resampled', 'seeds': SEEDS,
                          'band': 'mean to mean + sample standard deviation',
                          'smoothing': False, 'extrapolation': False}
    (PAPER / 'data/resampled/aggregate_metrics.json').write_text(json.dumps(aggregates, indent=2) + '\n')
    (PAPER / 'data/resampled/asset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (PAPER / 'data/resampled/per_seed_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    print(json.dumps({'status': 'PASS', 'runs': len(per_seed), 'cases': manifest}, indent=2))


if __name__ == '__main__':
    main()
