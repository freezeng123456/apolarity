import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_equal_time import parse_args, train
from torch_pinn.records import configure


def test_equal_time_small_cpu_pair(tmp_path):
    configure('cpu')
    summaries, hashes = [], []
    for method in ('nested_jvp', 'shared_jet_linear'):
        out = tmp_path / method
        out.mkdir()
        args = parse_args(['--out', str(out), '--method', method, '--width', '4', '--depth', '2',
                           '--batch', '3', '--constraints', '3', '--val-nx', '3', '--val-nt', '3',
                           '--seconds', '.15', '--eval-seconds', '.05'])
        train(args)
        summary = json.loads((out / 'summary.json').read_text())
        metrics = json.loads((out / 'metrics.json').read_text())
        assert summary['termination'] == 'wall_time_budget'
        assert summary['training_wall_seconds'] >= .15
        assert summary['steps'] == len(metrics) > 0
        assert summary['initial_parameter_hash'] != summary['final_parameter_hash']
        assert all(a['training_wall_seconds'] < b['training_wall_seconds'] for a, b in zip(metrics, metrics[1:]))
        assert summary['final_validation']['fixed_loss'] > 0
        summaries.append(summary)
        hashes.append(json.loads((out / 'batch_hashes.json').read_text()))
    assert summaries[0]['initial_parameter_hash'] == summaries[1]['initial_parameter_hash']
    assert summaries[0]['evaluation_point_hash'] == summaries[1]['evaluation_point_hash']
    assert summaries[0]['initial_validation']['fixed_loss'] == summaries[1]['initial_validation']['fixed_loss']
    count = min(map(len, hashes))
    assert hashes[0][:count] == hashes[1][:count]


@pytest.mark.parametrize('flag', ['--seconds', '--eval-seconds', '--lr'])
def test_equal_time_rejects_nonpositive(flag):
    with pytest.raises(SystemExit):
        parse_args(['--out', 'unused', '--method', 'nested_jvp', flag, '0'])


@pytest.mark.parametrize('method', ['nested_jvp', 'shared_jet_linear'])
def test_step_decay_evaluation_does_not_change_training(tmp_path, method):
    configure('cpu')
    states, hashes = [], []
    for cadence in (1, 3):
        out = tmp_path / str(cadence)
        out.mkdir()
        args = parse_args(['--out', str(out), '--method', method, '--steps', '6',
                           '--eval-every', str(cadence), '--width', '4', '--depth', '2',
                           '--batch', '3', '--constraints', '3', '--val-nx', '3', '--val-nt', '3'])
        train(args)
        states.append(torch.load(out / 'final.pt', weights_only=False))
        hashes.append(json.loads((out / 'batch_hashes.json').read_text()))
        rows = json.loads((out / 'metrics.json').read_text())
        assert len(rows) == 6
        for k, row in enumerate(rows):
            assert row['lr_used'] == args.lr * (1 - k / 6)
            assert row['next_lr'] == args.lr * (1 - (k + 1) / 6)
        assert rows[-1]['next_lr'] == 0
        assessments = json.loads((out / 'validation.json').read_text())
        assert [r['step'] for r in assessments] == list(range(0, 7, cadence))
    assert hashes[0] == hashes[1]
    assert states[0]['sampler_state'] == states[1]['sampler_state']
    for key in states[0]['model']:
        assert torch.equal(states[0]['model'][key], states[1]['model'][key])
    for key, state in states[0]['optimizer']['state'].items():
        for field, value in state.items():
            assert torch.equal(value, states[1]['optimizer']['state'][key][field])
