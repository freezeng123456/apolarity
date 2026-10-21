"""Isolated cell processes, paired gate, and fixed paper-seed lanes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'RUN_PLAN.json').read_text())


def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def cell(index, mode):
    task = PLAN['tasks'][index]
    out = ROOT / 'results' / mode / task['name']
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.parent / (task['name']+'.log')
    command = [sys.executable, str(ROOT/'measure_peak.py'), '--task', str(index),
               '--out', str(out), '--mode', mode, '--seconds', '2' if mode=='gate' else '30']
    with log.open('x') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=150)
    assert result.returncode == 0, (index, result.returncode, str(log))
    assert (out/'status.txt').read_text().strip() == 'COMPLETE'
    return json.loads((out/'summary.json').read_text())


def gate():
    import numpy as np
    rows, checks = [], []
    for lane in PLAN['lanes']:
        pair = [cell(index, 'gate') for index in lane['task_indices'][:2]]
        assert pair[0]['initial_hash'] == pair[1]['initial_hash']
        assert pair[0]['warmup_samples_sha256'] == pair[1]['warmup_samples_sha256']
        a, b = [np.asarray(row['first_warm_result'], dtype=np.float64) for row in pair]
        relative = float(np.linalg.norm(a-b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-30))
        assert relative < 1e-3, (lane['case'], relative)
        checks.append(dict(case=lane['case'], first_warm_relative_difference=relative,
                           initialization_matches=True, warmup_samples_match=True))
        rows.extend(pair)
        save(ROOT/'results/GATE_PROGRESS.json', dict(completed_pairs=len(checks), total_pairs=8))
    save(ROOT/'results/GATE_PASS.json', dict(status='PASS', cells=len(rows), checks=checks,
         primary_metric=PLAN['primary_metric'], runner_sha256=hashlib.sha256((ROOT/'measure_peak.py').read_bytes()).hexdigest()))
    print('GATE_PASS: 16 isolated cells, eight matched pairs', flush=True)


def lane(index):
    assert json.loads((ROOT/'results/GATE_PASS.json').read_text())['status'] == 'PASS'
    lane_plan = PLAN['lanes'][index]
    rows = []
    for task_index in lane_plan['task_indices']:
        row = cell(task_index, 'formal')
        rows.append(dict(task_index=task_index, name=row['task']['name'], peak_allocated_mib=row['peak_allocated_mib']))
        save(ROOT/'results'/('lane-%d.json'%index), dict(status='RUNNING', case=lane_plan['case'], completed=rows))
    for pos in range(0, len(rows), 2):
        pair = [json.loads((ROOT/'results/formal'/r['name']/'summary.json').read_text()) for r in rows[pos:pos+2]]
        assert pair[0]['initial_hash'] == pair[1]['initial_hash']
        assert pair[0]['warmup_samples_sha256'] == pair[1]['warmup_samples_sha256']
    save(ROOT/'results'/('lane-%d.json'%index), dict(status='COMPLETE', case=lane_plan['case'], completed=rows))
    print('LANE_COMPLETE', lane_plan['case'], len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('gate', 'lane'))
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.stage == 'gate':
        gate()
    else:
        lane(args.index)
