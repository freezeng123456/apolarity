"""Regenerate the combined individual-partial table from verified result files."""
import csv
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parents[1]
FIRST = ROOT / 'results/first_group/t4b_pytorch_eager_20260908/report'


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def timing(row, prefix):
    return f"${float(row[prefix + '_mean_ms']):.3f}\\pm{float(row[prefix + '_std_ms']):.3f}$"


def table(path, columns, header, rows):
    text = '\\begin{tabular}{@{}' + columns + '@{}}\n\\toprule\n'
    text += header + ' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(row) + ' \\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    path.write_text(text)


def main():
    rows = csv_rows(FIRST / 'summary.csv')
    rows = [row for row in rows if int(row['batch']) == 100]
    assert len(rows) == 6
    keys = {(r['target'], r['batch']) for r in rows}
    assert len(keys) == 6
    targets = list(dict.fromkeys(r['target'] for r in rows))
    rows.sort(key=lambda r: (targets.index(r['target']), int(r['batch'])))
    combined = []
    for row in rows:
        combined.append([
            f"$u_{{{row['target']}}}$", row['order'], row['directions'], row['batch'],
            timing(row, 'nested'),
            timing(row, 'waring_batched'), f"${float(row['paired_speedup_mean']):.2f}\\times$",
        ])
    path = PAPER / 'tables/t4_individual_partials.tex'
    table(path, 'lrrrrrr',
          r'Target & $p$ & $R$ & $B$ & Nested JVP (ms) & WDD (ms) & Speedup',
          combined)
    print('Generated one table: six target derivatives at B=100, WDD versus nested JVP, paired speedups preserved.')


if __name__ == '__main__':
    main()
