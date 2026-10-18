"""Rebuild the current five-row partial table from saved B=1000 timings."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

PAPER = Path(__file__).resolve().parent
DATA = PAPER.parents[1]/'results/stdepp_section61_b1000_v100_20261018'
TARGETS = ('123456', '11223344', '111222333', '111122223333', '111112222233333')


def render():
    report = json.loads((DATA/'comparison.json').read_text())
    assert report['completion']['status'] == 'COMPLETE'
    for manifest in (DATA/'evidence').rglob('SHA256SUMS'):
        for line in manifest.read_text().splitlines():
            sha, name = line.split('  ', 1)
            assert hashlib.sha256((manifest.parent/name).read_bytes()).hexdigest() == sha
    rows = {row['target']: row for row in report['cells']}
    text = r'''% Source: stdepp-section61-b1000-20261018/comparison.json; formal job 2385480.
% Counts are order-p directional Taylor coefficients per input point.
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Target & $p$ & $R$ & \multicolumn{2}{c}{Taylor coefficients} & \multicolumn{2}{c}{Time (ms)} & Speedup \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
 & & & STDE++ & WDD & STDE++ & WDD & \\
\midrule
'''
    for target in TARGETS:
        row = rows[target]
        assert (row['status'], row['batch'], row['width']) == ('COMPLETE', 1000, 128)
        exponents = row['exponents']
        assert exponents == sorted(exponents)
        rank = math.prod(a+1 for a in exponents[1:])
        real_directions = 2**sum(a % 2 for a in exponents[1:])
        assert row['wdd_jets'] == (rank+real_directions)//2
        records = json.loads((DATA/'evidence/results/partials'/row['name']/'results.json').read_text())
        assert len(records) == 3
        ratios = []
        for method in ('stde61', 'wdd'):
            medians = []
            for record in records:
                values = record['times_ms'][method]
                assert len(values) == 30
                medians.append(statistics.median(values))
                assert medians[-1] == record['median_ms'][method]
            assert math.isclose(statistics.mean(medians), row[method+'_ms'])
            assert math.isclose(statistics.stdev(medians), row[method+'_sd_ms'])
        ratios = [r['median_ms']['stde61']/r['median_ms']['wdd'] for r in records]
        assert math.isclose(statistics.mean(ratios), row['speedup'])
        derivative = ''.join(r'\partial_'+str(i)+(f'^{{{a}}}' if a > 1 else '')
                             for i, a in enumerate(exponents, 1))+'u'
        fields = [f'${derivative}$', str(row['order']), str(rank),
                  str(row['stde61_jets']), str(row['wdd_jets'])]
        fields += [f"${row[m+'_ms']:.3f}\\pm{row[m+'_sd_ms']:.3f}$" for m in ('stde61', 'wdd')]
        fields.append(f"${row['speedup']:.2f}\\times$")
        text += ' & '.join(fields)+' \\\\\n'
    return text+'\\bottomrule\n\\end{tabular}\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    target = PAPER/'tables/t4_individual_partials.tex'
    expected = render()
    if args.check:
        if target.read_text() != expected:
            raise SystemExit('Paper table differs from verified B=1000 records')
        print('Current paper table matches verified B=1000 records.')
    else:
        target.write_text(expected)
        print('Generated five current B=1000 partial-derivative rows.')


if __name__ == '__main__':
    main()
