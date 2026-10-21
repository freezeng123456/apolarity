"""Validate and seal all predefined 30-second peaks without dropping any seed."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT/'RUN_PLAN.json').read_text())
assert json.loads((ROOT/'results/GATE_PASS.json').read_text())['status'] == 'PASS'
rows = []
pairs = {}
for task in plan['tasks']:
    cell = ROOT/'results/formal'/task['name']
    assert (cell/'status.txt').read_text().strip() == 'COMPLETE', task['name']
    for line in (cell/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256((cell/name).read_bytes()).hexdigest() == expected
    summary = json.loads((cell/'summary.json').read_text())
    config = json.loads((cell/'config.json').read_text())
    assert summary['task'] == config['task'] == task
    assert summary['requested_seconds'] == 30 and summary['elapsed_seconds'] >= 30
    assert summary['complete_calls_or_updates'] > 0 and summary['finite']
    assert summary['peak_allocated_bytes'] >= summary['baseline_allocated_bytes'] > 0
    assert not summary['checkpoints_saved'] and not config['checkpoints_saved']
    assert config['runner_sha256'] == hashlib.sha256((ROOT/'measure_peak.py').read_bytes()).hexdigest()
    assert config['plan_sha256'] == hashlib.sha256((ROOT/'RUN_PLAN.json').read_bytes()).hexdigest()
    assert config['source_hashes'] == plan['source_sha256'][config['source_bundle']]
    assert config['gpu'] == 'Tesla V100-SXM2-32GB' and '8255C' in config['cpu_model']
    assert config['torch'] == '2.5.1+cu121' and config['numpy'] == '1.26.4'
    assert config['threads'] == 1 and not config['tf32'] and not config['compile']
    if task['kind'] == 'partial':
        expected_counts = {'123456': (32, 32), '11223344': (14, 40),
                           '111222333': (10, 28), '111122223333': (13, 49),
                           '111112222233333': (20, 100)}
        expected = expected_counts[task['case']][0 if task['method']=='wdd' else 1]
        assert config['preparation']['schedule']['evaluated_count'] == expected
    else:
        assert summary['initial_hash'] != summary['final_parameter_hash']
    pairs.setdefault((task['case'], task['seed']), []).append(summary)
    rows.append(dict(case=task['case'], kind=task['kind'], method=task['method'], seed=task['seed'],
        peak_allocated_mib=summary['peak_allocated_mib'],
        sampled_process_peak_mib=summary['sampled_process_peak_mib'],
        peak_reserved_mib=summary['peak_reserved_bytes']/2**20,
        requested_seconds=summary['requested_seconds'], elapsed_seconds=summary['elapsed_seconds'],
        complete_calls_or_updates=summary['complete_calls_or_updates'], samples=summary['sample_count']))
for key, pair in pairs.items():
    assert len(pair) == 2 and {x['task']['method'] for x in pair} == {'wdd', 'stdepp'}
    assert pair[0]['initial_hash'] == pair[1]['initial_hash'], key
    assert pair[0]['warmup_samples_sha256'] == pair[1]['warmup_samples_sha256'], key
summary_rows = []
for lane in plan['lanes']:
    selected = [row for row in rows if row['case'] == lane['case']]
    assert len(selected) == 2*len(lane['seeds'])
    result = dict(case=lane['case'], seed_count=len(lane['seeds']))
    for method in ('wdd', 'stdepp'):
        group = [row for row in selected if row['method'] == method]
        assert sorted(row['seed'] for row in group) == lane['seeds']
        result[method+'_peak_allocated_mib'] = max(row['peak_allocated_mib'] for row in group)
        result[method+'_min_seed_peak_mib'] = min(row['peak_allocated_mib'] for row in group)
        result[method+'_sampled_process_peak_mib'] = max(row['sampled_process_peak_mib'] for row in group)
    result['wdd_reduction_percent'] = 100*(1-result['wdd_peak_allocated_mib']/result['stdepp_peak_allocated_mib'])
    summary_rows.append(result)
for name, data in [('peak-memory-by-seed.csv', rows), ('peak-memory-summary.csv', summary_rows)]:
    with (ROOT/'results'/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(data)
report = dict(status='PASS', cells=len(rows), paired_checks=len(pairs),
    primary_metric=plan['primary_metric'], aggregation=plan['aggregation'], summary=summary_rows,
    elapsed_seconds_range=[min(r['elapsed_seconds'] for r in rows), max(r['elapsed_seconds'] for r in rows)],
    weights_saved=False, checkpoints_saved=False)
(ROOT/'results/MEMORY_COMPARISON.json').write_text(json.dumps(report, indent=2)+'\n')
assert not any(p.suffix.lower() in ('.pt', '.pth', '.ckpt', '.safetensors') for p in ROOT.rglob('*'))
recovery = ROOT/'recovery'; recovery.mkdir(exist_ok=True)
archive = recovery/'memory-peak30s-v100-20261021.tar.gz'
with tarfile.open(str(archive), 'w:gz') as stream:
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and recovery not in path.parents:
            stream.add(str(path), arcname=str(path.relative_to(ROOT)), recursive=False)
receipt = dict(status='SEALED', archive=str(archive), bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), cells=len(rows), paired_checks=len(pairs),
    checkpoints_saved=False)
(recovery/'RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt), flush=True)
print(json.dumps(report), flush=True)
