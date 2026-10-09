"""Install the equal-update V100 figures and build their endpoint tables."""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import statistics

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER.parents[1] / 'results/paper_equal_updates_v100_20261008'
TARGETS = {'kdv1d': 25000, 'kdv2d': 20000, 'ch2d': 1500}
METHODS = ('nested_jvp', 'shared_jet_linear')


def scientific(value):
    mantissa, exponent = f'{value:.3e}'.split('e')
    return mantissa + r'\times10^{' + str(int(exponent)) + '}'


def main():
    for line in (RESULTS / 'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        if name.startswith('data/'):
            assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == sha
    provenance = json.loads((RESULTS / 'data/provenance.json').read_text())
    plotted = json.loads((RESULTS / 'figures/plot_manifest.json').read_text())
    assert provenance['status'] == 'PASS' and provenance['runs'] == 30
    assert plotted['source_curves_sha256'] == provenance['curves_csv_sha256']
    assert plotted['targets'] == TARGETS
    with (RESULTS / 'data/pde-results.csv').open() as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 30
    manifest = dict(source_results=str(RESULTS.relative_to(PAPER.parents[1])),
                    protocol='equal_updates', targets=TARGETS, runs=30,
                    error_metric='Space-time relative L2 error, denoted RE.',
                    source_curves_sha256=provenance['curves_csv_sha256'], assets={})
    for case, target in TARGETS.items():
        lines = [r'\begin{tabular}{@{}lcc@{}}', r'\toprule',
                 r'Method & \shortstack{Training\\time (s)} & $\operatorname{RE}$ \\',
                 r'\midrule']
        for method, label in zip(METHODS, ('Nested JVP', 'WDD')):
            group = [r for r in records if r['case'] == case and r['method'] == method]
            assert sorted(int(r['seed']) for r in group) == list(range(20260919, 20260924))
            assert all(int(r['updates']) == target for r in group)
            values = [label]
            for metric in ('training_seconds', 'spacetime_relative_l2'):
                samples = [float(r[metric]) for r in group]
                assert all(math.isfinite(v) and v > 0 for v in samples)
                mean, sd = statistics.mean(samples), statistics.stdev(samples)
                if metric == 'training_seconds':
                    assert math.isclose(mean, plotted['endpoints'][case][method]['time_s'], rel_tol=1e-14)
                    values.append(f'${mean:.1f} \\pm {sd:.1f}$')
                else:
                    values.append('$' + scientific(mean) + r'\pm' + scientific(sd) + '$')
            lines.append(' & '.join(values) + r' \\')
        lines.extend((r'\bottomrule', r'\end{tabular}'))
        table = PAPER / f'tables/resampled_wall_{case}.tex'
        table.write_text('\n'.join(lines) + '\n')
        paths = [table]
        for ext in ('pdf', 'png'):
            target_file = PAPER / f'figures/resampled_wall_{case}.{ext}'
            shutil.copyfile(RESULTS / f'figures/equal_updates_{case}.{ext}', target_file)
            paths.append(target_file)
        for path in paths:
            manifest['assets'][str(path.relative_to(PAPER))] = hashlib.sha256(path.read_bytes()).hexdigest()
    dest = PAPER / 'data/equal_updates'
    dest.mkdir(exist_ok=True)
    (dest / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', runs=30, assets=len(manifest['assets']), targets=TARGETS)))


if __name__ == '__main__':
    main()
