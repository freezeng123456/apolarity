"""Prepared schedules must be numerically equivalent and absent from timing."""
import json

import pytest
import torch

import stdepp_prepared_partials as runner
from first_group import make_model_and_points, parse_one_based_pattern
from torch_pinn import stdepp
from torch_pinn.monomials import monomial


def test_matrix_retains_original_protocol_and_pairs_methods(tmp_path):
    cells = runner.matrix(tmp_path)
    assert len(cells) == 6 and {cell['target'] for cell in cells} == set(runner.TARGETS)
    assert all(cell['methods'] == ['waring_batched', 'stdepp'] for cell in cells)
    assert all(cell['seeds'] == [20260908, 20260909, 20260910]
               and cell['width'] == 128 and cell['depth'] == 4 and cell['batch'] == 100
               and cell['warmups'] == 10 and cell['repeats'] == 30 for cell in cells)
    plan = runner.plan(tmp_path)
    assert plan['allocation']['cpus_per_task'] == 10 and plan['execution']['threads'] == 1
    assert not plan['parameter_backward'] and not plan['checkpoints']


@pytest.mark.parametrize('target', runner.TARGETS)
def test_prepared_outputs_match_original_apis_on_all_six_targets(target):
    alpha = parse_one_based_pattern(target)
    model, points = make_model_and_points(20260908, 2, 3, 2, 'cpu')
    with torch.no_grad():
        for method in runner.METHODS:
            prepared = runner.prepare(method, alpha)
            original = stdepp.partial(model, points, alpha) if method == 'stdepp' else monomial(model, points, alpha)
            observed = runner.evaluate_prepared(model, points, prepared)
            torch.testing.assert_close(observed, original, rtol=1e-5, atol=1e-7)
            assert not observed.requires_grad
            assert prepared.directions.is_complex() == (method == 'waring_batched' and target not in ('111111', '123456'))


@pytest.mark.parametrize('target', ('111111', '123456'))
def test_same_real_direction_weight_sets_have_same_prepared_outputs(target):
    alpha = parse_one_based_pattern(target)
    left = runner.prepare('waring_batched', alpha)
    right = runner.prepare('stdepp', alpha)
    def rows(prepared):
        return sorted((tuple(v.tolist()), float(w))
                      for v, w in zip(prepared.directions, prepared.weights))
    assert rows(left) == rows(right)
    assert not left.directions.is_complex() and not right.directions.is_complex()


def test_all_six_independent_tiny_nested_references_pass():
    checks = runner.correctness_checks()
    assert len(checks) == 12 and {row['target'] for row in checks} == set(runner.TARGETS)


def test_schedule_constructors_only_run_on_cpu_before_timing(monkeypatch):
    calls = []
    old_wdd, old_stdepp = runner.wdd_schedule, stdepp.schedule
    def constructor(original):
        def wrapped(*args, **kwargs):
            calls.append(kwargs['device'])
            return original(*args, **kwargs)
        return wrapped
    monkeypatch.setattr(runner, 'wdd_schedule', constructor(old_wdd))
    monkeypatch.setattr(stdepp, 'schedule', constructor(old_stdepp))
    alpha = parse_one_based_pattern('111111')
    schedules = {method: runner.prepare(method, alpha) for method in runner.METHODS}
    assert calls == ['cpu', 'cpu']
    identities = {method: runner.storage_identity(value) for method, value in schedules.items()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Schedule construction inside measured operation')
    monkeypatch.setattr(runner, 'wdd_schedule', forbidden)
    monkeypatch.setattr(stdepp, 'schedule', forbidden)
    model, points = make_model_and_points(20260908, 2, 3, 2, 'cpu')
    report, _ = runner.benchmark_pair(model, points, schedules, warmups=1, repeats=30)
    assert all(len(report['times_ms'][method]) == 30 for method in runner.METHODS)
    assert report['round_orders'].count(list(runner.METHODS)) == 15
    assert report['round_orders'].count(list(reversed(runner.METHODS))) == 15
    assert all(runner.storage_identity(value) == identities[method] for method, value in schedules.items())
    report2, _ = runner.benchmark_pair(model, points, schedules, seed_index=1, warmups=0, repeats=2)
    assert report2['round_orders'][0] == list(reversed(runner.METHODS))


def test_cell_requires_successful_gate_for_current_frozen_source(tmp_path):
    with pytest.raises(RuntimeError, match='preflight is required'):
        runner.check_gate(tmp_path)
    path = tmp_path/'preflight/PASS.json'
    path.parent.mkdir()
    receipt = dict(status='PASS', protocol=runner.PROTOCOL, source_fingerprint='stale',
                   environment={'gpu': 'Tesla V100-SXM2-32GB'})
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match='Source changed'):
        runner.check_gate(tmp_path)
    receipt['source_fingerprint'] = runner.source_fingerprint()
    path.write_text(json.dumps(receipt))
    assert runner.check_gate(tmp_path)['status'] == 'PASS'
