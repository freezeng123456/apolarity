"""Protocol gates for the matched STDE++ campaign, without requiring a GPU."""
import json
from types import SimpleNamespace

import pytest

import stdepp_campaign as campaign
from torch_pinn.records import run_record
import train_equal_updates as trainer


def test_campaign_covers_every_paper_case_with_matched_budgets(tmp_path):
    cells = campaign.matrix(tmp_path)
    assert len(cells) == 33
    partials = [cell for cell in cells if cell['kind'] == 'partial']
    pdes = [cell for cell in cells if cell['kind'] == 'pde']
    assert len(partials) == 18 and len(pdes) == 15
    assert {cell['target'] for cell in partials} == set(campaign.TARGETS)
    assert {cell['method'] for cell in partials} == {'nested_jvp', 'waring_batched', 'stdepp'}
    assert all(cell['batch'] == 100 and cell['depth'] == 4 and cell['width'] == 128
               and cell['warmups'] == 10 and cell['repeats'] == 30
               and cell['seeds'] == [20260908, 20260909, 20260910] for cell in partials)
    assert {cell['case']: cell['updates'] for cell in pdes} == {
        'kdv1d': 25000, 'kdv2d': 20000, 'ch2d': 1500}
    assert {cell['seed'] for cell in pdes} == set(range(20260919, 20260924))
    assert all(cell['method'] == 'stdepp' and cell['batch'] == 400 for cell in pdes)
    plan = campaign.plan(tmp_path)
    assert not plan['checkpoints'] and plan['allocation']['cpus_per_task'] == 10
    assert plan['execution']['threads'] == 1


@pytest.mark.parametrize('case', ('kdv1d', 'kdv2d', 'ch2d'))
def test_wrapper_preserves_training_streams_and_curve_only_storage(tmp_path, monkeypatch, case):
    original = trainer.full_loss

    def reference_backend(case_name, model, points, data):
        return original(trainer.PROBLEMS[case_name], model, points, data, 'nested_jvp')

    monkeypatch.setattr(campaign, '_stdepp', lambda: SimpleNamespace(pde_loss=reference_backend))
    cell = next(cell for cell in campaign.matrix(tmp_path) if cell.get('case') == case)
    cell.update(updates=3, eval_every=2, eval_points=16, width=4, depth=2,
                batch=3, constraint_side=2, out=str(tmp_path/'stdepp'))
    campaign.run_pde(cell, 'cpu')
    baseline_cell = {**cell, 'method': 'nested_jvp', 'out': str(tmp_path/'nested')}
    run_record(campaign.pde_args(baseline_cell, 'cpu'), trainer.train)
    assert trainer.full_loss is original
    candidate = json.loads((tmp_path/'stdepp/summary.json').read_text())
    reference = json.loads((tmp_path/'nested/summary.json').read_text())
    for field in ('initial_hash', 'final_hash', 'interior_samples_sha256',
                  'constraint_samples_sha256', 'spacetime_test_sha256', 'terminal_test_sha256'):
        assert candidate[field] == reference[field]
    assert candidate['steps'] == 3 and not candidate['checkpoints_saved']
    assert not [p for p in tmp_path.rglob('*') if p.suffix in ('.pt', '.pth', '.ckpt', '.npz')]


def test_formal_cells_require_matching_preflight_source(tmp_path):
    with pytest.raises(RuntimeError, match='preflight is required'):
        campaign.check_gate(tmp_path)
    path = tmp_path/'preflight/PASS.json'
    path.parent.mkdir()
    receipt = dict(status='PASS', protocol=campaign.PROTOCOL,
        source_fingerprint='stale', environment={'gpu': 'Tesla V100-SXM2-32GB'})
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match='Source changed'):
        campaign.check_gate(tmp_path)
    receipt['source_fingerprint'] = campaign.source_fingerprint()
    path.write_text(json.dumps(receipt))
    assert campaign.check_gate(tmp_path)['status'] == 'PASS'


def test_dispatch_is_restored_after_failure(monkeypatch):
    original = trainer.full_loss
    with pytest.raises(ValueError, match='test failure'):
        with campaign.stdepp_training_dispatch():
            assert trainer.full_loss is not original
            raise ValueError('test failure')
    assert trainer.full_loss is original
