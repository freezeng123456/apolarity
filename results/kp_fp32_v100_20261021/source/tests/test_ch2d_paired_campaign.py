"""Validate the new paired dispatch without changing the PDE trainer."""
import csv
import json

import pytest

import ch2d_paired_campaign as paired
from torch_pinn.records import save


def test_plan_keeps_original_five_seeds_and_complete_gpu_groups(tmp_path):
    plan = paired.plan(tmp_path)
    assert len(plan['cells']) == 15 and len(plan['lanes']) == 5
    assert plan['seeds'] == list(range(20260919, 20260924))
    for lane in plan['lanes']:
        cells = [plan['cells'][i] for i in lane['indices']]
        assert {c['method'] for c in cells} == set(paired.METHODS)
        assert {c['seed'] for c in cells} == {lane['seed']}
        assert all((c['updates'], c['batch'], c['width'], c['depth'], c['constraint_side'])
                   == (1500, 400, 128, 5, 16) for c in cells)
    assert not plan['checkpoints'] and not plan['implementations_changed']


def test_stale_gate_is_rejected_before_training(tmp_path):
    gate = tmp_path/'gate'
    gate.mkdir()
    save(gate/'PASS.json', dict(status='PASS', source_fingerprint='stale'))
    with pytest.raises(RuntimeError, match='frozen source'):
        paired.run_cell(tmp_path/'results', 0, gate, device='cpu')
    assert not (tmp_path/'results').exists()


def test_timing_outlier_marks_complete_seed_group_without_selecting_by_error():
    groups = []
    for lane, seed in enumerate(paired.SEEDS):
        results = {method: dict(training_seconds=100.0, re=.1+.01*lane)
                   for method in paired.METHODS}
        if lane == 4:
            results['shared_jet_linear']['training_seconds'] = 120.0
        groups.append(dict(lane=lane, seed=seed, results=results, issues=[]))
    result = paired.audit_groups(groups)
    assert result['flagged_lanes'] == [4]
    assert not result['selection_by_final_error'] and result['preserve_all_attempts']


def test_three_method_training_dispatch_preserves_samples_and_objective(tmp_path, monkeypatch):
    gate = tmp_path/'gate'
    gate.mkdir()
    fingerprint = paired.campaign.source_fingerprint()
    save(gate/'PASS.json', dict(status='PASS', source_fingerprint=fingerprint))
    original = paired.plan

    def tiny_plan(output):
        obj = original(output)
        for cell in obj['cells']:
            cell.update(updates=2, eval_every=1, width=3, depth=2,
                        batch=2, constraint_side=2, eval_points=8)
        return obj

    monkeypatch.setattr(paired, 'plan', tiny_plan)
    summaries = []
    for index in range(3):
        paired.run_cell(tmp_path/'results', index, gate, device='cpu')
        folder = tmp_path/'results'/tiny_plan(tmp_path/'results')['cells'][index]['name']
        obj = json.loads((folder/'summary.json').read_text())
        rows = list(csv.DictReader((folder/'curve.csv').open()))
        assert [int(r['step']) for r in rows] == [0, 1, 2]
        assert obj['steps'] == 2 and not obj['checkpoints_saved']
        assert not list(folder.glob('*.pt')) and not list(folder.glob('*.pth'))
        summaries.append(obj)
    for field in paired.HASH_FIELDS:
        assert len({s[field] for s in summaries}) == 1
    errors = [s['errors']['spacetime']['relative_l2'] for s in summaries]
    assert max(errors)-min(errors) < 1e-5
