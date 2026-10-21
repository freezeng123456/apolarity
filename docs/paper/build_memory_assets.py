"""Verify all 60 sealed V100 memory cells and build the paper's peak table."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

PAPER = Path(__file__).resolve().parent
DATA = PAPER.parents[1] / 'results/memory_peak30s_v100_20261021'
TARGETS = ('123456', '11223344', '111222333', '111122223333', '111112222233333',
           'kdv1d', 'kdv2d', 'kp2d')


def render():
    archive = DATA / 'evidence.tar.gz'
    receipt = json.loads((DATA / 'recovery-receipt.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['sha256']
    assert archive.stat().st_size == receipt['bytes']
    with tarfile.open(archive, 'r:gz') as stream:
        members = {m.name: m for m in stream.getmembers() if m.isfile()}
        assert not any(Path(n).suffix in ('.pt', '.pth', '.ckpt', '.safetensors') for n in members)
        read = lambda name: stream.extractfile(members[name]).read()
        plan = json.loads(read('RUN_PLAN.json'))
        assert plan == json.loads((DATA / 'RUN_PLAN.json').read_text())
        assert json.loads(read('results/GATE_PASS.json'))['status'] == 'PASS'
        rows, pairs = [], {}
        for task in plan['tasks']:
            prefix = 'results/formal/' + task['name'] + '/'
            for line in read(prefix + 'SHA256SUMS').decode().splitlines():
                expected, name = line.split('  ', 1)
                assert hashlib.sha256(read(prefix + name)).hexdigest() == expected
            assert read(prefix + 'status.txt').decode().strip() == 'COMPLETE'
            row = json.loads(read(prefix + 'summary.json'))
            cfg = json.loads(read(prefix + 'config.json'))
            assert row['task'] == cfg['task'] == task
            assert row['requested_seconds'] == 30 and row['elapsed_seconds'] >= 30
            assert row['complete_calls_or_updates'] > 0 and row['finite']
            assert cfg['gpu'] == 'Tesla V100-SXM2-32GB' and '8255C' in cfg['cpu_model']
            assert cfg['threads'] == 1 and not cfg['tf32'] and not cfg['compile']
            assert cfg['source_hashes'] == plan['source_sha256'][cfg['source_bundle']]
            assert cfg['runner_sha256'] == hashlib.sha256(read('measure_peak.py')).hexdigest()
            assert cfg['plan_sha256'] == hashlib.sha256(read('RUN_PLAN.json')).hexdigest()
            assert not row['checkpoints_saved']
            samples = list(csv.DictReader(io.StringIO(read(prefix+'memory_samples.csv').decode())))
            assert len(samples) == row['sample_count']
            assert max(int(s['allocated_peak_bytes']) for s in samples) <= row['peak_allocated_bytes']
            pairs.setdefault((task['case'], task['seed']), []).append(row)
            rows.append(row)
        assert len(rows) == 60 and len(pairs) == 30
        for pair in pairs.values():
            assert len(pair) == 2
            for key in ('initial_hash', 'warmup_samples_sha256'):
                assert pair[0][key] == pair[1][key]
        report = json.loads((DATA / 'MEMORY_COMPARISON.json').read_text())
        assert report == json.loads(read('results/MEMORY_COMPARISON.json'))
        values = {}
        for lane in plan['lanes']:
            case = lane['case']
            values[case] = {}
            for method in ('wdd', 'stdepp'):
                selected = [r for r in rows if r['task']['case'] == case and r['task']['method'] == method]
                assert sorted(r['task']['seed'] for r in selected) == lane['seeds']
                values[case][method] = max(r['peak_allocated_bytes'] for r in selected) / 2**20
            saved, = [r for r in report['summary'] if r['case'] == case]
            for method in ('wdd', 'stdepp'):
                assert saved[method+'_peak_allocated_mib'] == values[case][method]
    lines = [r'% Source: memory_peak30s_v100_20261021; formal job 2393135; 60 verified cells.',
             r'\begin{tabular}{@{}lrr@{}}', r'\toprule',
             r'Target & STDE++ & WDD \\', r'\midrule',
             r'\multicolumn{3}{@{}l}{Individual partial derivatives} \\']
    names = {'kdv1d': 'One-dimensional KdV', 'kdv2d': 'Two-dimensional KdV', 'kp2d': 'KP-II'}
    for target in TARGETS:
        if target == 'kdv1d':
            lines += [r'\midrule', r'\multicolumn{3}{@{}l}{PDE training} \\']
        if target in names:
            name = names[target]
        else:
            task = next(t for t in plan['tasks'] if t['case'] == target)
            name = '$' + ''.join(r'\partial_' + str(i) + (f'^{{{a}}}' if a > 1 else '')
                                for i, a in enumerate(task['exponents'], 1)) + 'u$'
        lines.append(name + ' & ' + f"{values[target]['stdepp']:.1f} & {values[target]['wdd']:.1f}" + r' \\')
    return '\n'.join(lines + [r'\bottomrule', r'\end{tabular}']) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    expected = render()
    target = PAPER / 'tables/peak_memory.tex'
    if args.check:
        assert target.read_text() == expected, 'Peak memory table differs from sealed records'
    else:
        target.write_text(expected)
    print('PASS: 60 memory cells, 30 paired checks; paper reports maximum per-seed peaks.')
