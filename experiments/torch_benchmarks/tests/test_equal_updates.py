"""Check exact stopping, paired samples and curve-only storage on a small network."""
import csv
import json
import pytest

from equal_updates_campaign import matrix
from torch_pinn.records import run_record
from train_equal_updates import parse_args, train


@pytest.mark.parametrize('case', ('kdv1d', 'kdv2d', 'ch2d'))
def test_equal_updates_pair_has_matching_samples_and_no_model_files(tmp_path, case):
    summaries = []
    for method in ('nested_jvp', 'shared_jet_linear'):
        out = tmp_path/method
        args = parse_args(['--case', case, '--method', method, '--out', str(out),
            '--device', 'cpu', '--updates', '3', '--eval-every', '2', '--width', '4',
            '--depth', '2', '--batch', '3', '--constraint-side', '2', '--eval-points', '16'])
        run_record(args, train)
        summary = json.loads((out/'summary.json').read_text())
        curve = list(csv.DictReader((out/'curve.csv').open()))
        assert summary['steps'] == 3 and summary['terminal'] == 'target_updates'
        assert [int(row['step']) for row in curve] == [0, 2, 3]
        assert [float(row['time_s']) for row in curve] == sorted(float(row['time_s']) for row in curve)
        assert float(curve[-1]['time_s']) == summary['wall_seconds']
        assert summary['process_wall_seconds'] >= summary['wall_seconds']+summary['evaluation_seconds']
        assert not list(out.glob('*.pt')) and not list(out.glob('*.pth'))
        assert not list(out.glob('*.npz')) and not summary['checkpoints_saved']
        summaries.append(summary)
    for field in ('initial_hash', 'interior_samples_sha256', 'constraint_samples_sha256',
                  'spacetime_test_sha256', 'terminal_test_sha256'):
        assert summaries[0][field] == summaries[1][field]


def test_matrix_preserves_user_budgets_and_excludes_partial_timing(tmp_path):
    cells = matrix(tmp_path)
    assert len(cells) == 42
    pde = [c for c in cells if c['kind'] == 'pde']
    assert len(pde) == 30
    assert {c['case']: c['updates'] for c in pde} == {'kdv1d':25000, 'kdv2d':20000, 'ch2d':1500}
    assert all('--seconds' not in c['command'] for c in pde)
    assert all('measure_partial_memory.py' in c['command'][1]
               for c in cells if c['kind'] == 'memory')
