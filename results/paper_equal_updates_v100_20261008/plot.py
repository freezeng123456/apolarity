"""Redraw the complete equal-update V100 curves from numeric CSV only."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CASES = ('kdv1d', 'kdv2d', 'ch2d')
TITLES = ('1D KdV', '2D KdV', '2D Cahn–Hilliard')
TARGETS = {'kdv1d': 25000, 'kdv2d': 20000, 'ch2d': 1500}
METHODS = ('nested_jvp', 'shared_jet_linear')
SEEDS = tuple(range(20260919, 20260924))
COLORS = ('#c63d2f', '#1856a5')
LABELS = ('Nested JVP', 'WDD')


def main():
    provenance = json.loads((ROOT / 'data/provenance.json').read_text())
    data = ROOT / 'data/curves.csv'
    assert provenance['status'] == 'PASS' and provenance['runs'] == 30
    assert hashlib.sha256(data.read_bytes()).hexdigest() == provenance['curves_csv_sha256']
    with data.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == provenance['observations']
    curves = {}
    for case in CASES:
        interval = 10 if case == 'ch2d' else 100
        expected_steps = np.arange(0, TARGETS[case] + 1, interval)
        for method in METHODS:
            times, errors = [], []
            for seed in SEEDS:
                selected = [r for r in rows if r['case'] == case
                            and r['method'] == method and int(r['seed']) == seed]
                steps = np.array([int(r['step']) for r in selected])
                t = np.array([float(r['time_s']) for r in selected])
                y = np.array([float(r['spacetime_relative_l2']) for r in selected])
                assert np.array_equal(steps, expected_steps)
                assert t[0] == 0 and np.all(np.diff(t) > 0)
                assert np.isfinite(t).all() and np.isfinite(y).all() and np.all(y > 0)
                times.append(t)
                errors.append(y)
            # Average paired seeds at the same update index. No time-grid
            # interpolation, missing-seed averages, or endpoint extrapolation.
            curves[case, method] = (np.mean(times, axis=0),
                                    np.mean(errors, axis=0),
                                    np.std(errors, axis=0, ddof=1))

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 11,
                         'axes.linewidth': .8, 'path.simplify': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)

    def draw(ax, case):
        for method, color, label in zip(METHODS, COLORS, LABELS):
            t, mean, sd = curves[case, method]
            ax.fill_between(t, mean, mean + sd, color=color, alpha=.16, linewidth=0)
            ax.plot(t, mean, color=color, lw=1.8, label=label)
        end = max(curves[case, method][0][-1] for method in METHODS)
        ax.set(yscale='log', xlim=(0, end * 1.015),
               xlabel='Training wall time (s)',
               ylabel=r'$\operatorname{RE}$')
        ax.grid(which='major', color='.82', linewidth=.65, alpha=.75)
        ax.grid(which='minor', color='.90', linewidth=.45, alpha=.45)
        ax.tick_params(direction='in', top=True, right=True)
        ax.legend(loc='upper right', framealpha=.93, edgecolor='.75')

    for case in CASES:
        fig, ax = plt.subplots(figsize=(5.7, 3.35))
        draw(ax, case)
        fig.tight_layout(pad=.5)
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'equal_updates_{case}.{ext}', dpi=300,
                        bbox_inches='tight', pad_inches=.04)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.7))
    for ax, case, title in zip(axes, CASES, TITLES):
        draw(ax, case)
        ax.set_title(f'{title} · {TARGETS[case]:,} updates')
    fig.suptitle('V100 · 5 paired seeds · equal update counts', fontsize=12)
    fig.tight_layout(pad=.7)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'v100_equal_updates.{ext}', dpi=240,
                    bbox_inches='tight', pad_inches=.05)
    plt.close(fig)

    manifest = dict(runs=30, observed_points=len(rows), seeds=SEEDS,
                    targets=TARGETS, x_axis='mean cumulative training seconds',
                    aggregation='Arithmetic means of time and error at common update indices.',
                    band='Mean error to mean + sample standard deviation (ddof=1).',
                    interpolation='None; no smoothing or extrapolation.',
                    source_curves_sha256=provenance['curves_csv_sha256'],
                    matplotlib=matplotlib.__version__, numpy=np.__version__,
                    endpoints={case: {method: dict(time_s=float(curves[case, method][0][-1]),
                                                   relative_l2=float(curves[case, method][1][-1]))
                                      for method in METHODS} for case in CASES})
    (out / 'plot_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
