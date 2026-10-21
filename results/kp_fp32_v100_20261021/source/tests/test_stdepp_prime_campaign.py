"""Protocol and algebraic guards for the actual Section 6.2.4 comparison."""
from collections import Counter
import json

import pytest

import stdepp_prime_campaign as runner


def test_matrix_is_only_six_matched_partial_cases(tmp_path):
    plan = runner.plan(tmp_path)
    assert len(plan['cells']) == 6
    assert {cell['target'] for cell in plan['cells']} == set(runner.TARGETS)
    for cell in plan['cells']:
        assert cell['methods'] == ['wdd', 'stde61', 'prime624']
        assert cell['seeds'] == [20260908, 20260909, 20260910]
        assert (cell['batch'], cell['width'], cell['depth']) == (100, 128, 4)
        assert (cell['warmups'], cell['repeats']) == (10, 30)
    assert plan['allocation']['cpus_per_task'] == 10
    assert plan['execution']['threads'] == 1
    assert not plan['pde_training'] and not plan['checkpoints']
    assert 'same structural-zero-aware sparse Taylor backend' in plan['timing_scope']


@pytest.mark.parametrize('seed_index', range(3))
def test_three_methods_balance_positions_and_pair_precedence(seed_index):
    orders = runner.round_orders(30, seed_index)
    assert len(orders) == 30
    assert all(set(order) == set(runner.METHODS) for order in orders)
    assert set(Counter(tuple(order) for order in orders).values()) == {5}
    for method in runner.METHODS:
        assert Counter(order.index(method) for order in orders) == {0: 10, 1: 10, 2: 10}
    for a, b in [('wdd', 'stde61'), ('wdd', 'prime624'), ('stde61', 'prime624')]:
        assert sum(order.index(a) < order.index(b) for order in orders) == 15


def test_bad_repeat_count_cannot_silently_unbalance_protocol():
    with pytest.raises(ValueError, match='multiple of six'):
        runner.round_orders(31)


def test_table_one_prime_uniqueness_is_independently_certified():
    certificate = runner.prime_certificate((71, 73, 79, 83, 103, 151))
    assert certificate['output_order'] == 560
    assert certificate['nonnegative_solution_count_capped_at_two'] == 1
    assert certificate['unique_solution_is_all_ones']
    with pytest.raises(ValueError, match='unique'):
        runner.prime_certificate((2, 3, 5))
    with pytest.raises(ValueError, match='distinct primes'):
        runner.prime_certificate((4, 7))


def test_formal_requires_completed_source_matched_pilot(tmp_path):
    with pytest.raises(RuntimeError, match='completed pilot'):
        runner.receipt(tmp_path, 'pilot')
    root = tmp_path/'pilot'
    root.mkdir()
    result = dict(status='PASS', protocol=runner.PROTOCOL, source_fingerprint='stale',
                  environment={'gpu': 'Tesla V100-SXM2-32GB'})
    (root/'PASS.json').write_text(json.dumps(result))
    (root/'status.txt').write_text('RUNNING\n')
    with pytest.raises(RuntimeError, match='completed pilot'):
        runner.receipt(tmp_path, 'pilot')
    (root/'status.txt').write_text('COMPLETE\n')
    with pytest.raises(RuntimeError, match='Source changed'):
        runner.receipt(tmp_path, 'pilot')
    result['source_fingerprint'] = runner.source_fingerprint()
    (root/'PASS.json').write_text(json.dumps(result))
    assert runner.receipt(tmp_path, 'pilot')['status'] == 'PASS'


def test_formal_cannot_start_before_preflight_and_pilot(tmp_path):
    with pytest.raises(RuntimeError, match='completed preflight'):
        runner.main(['cell', '--out', str(tmp_path), '--index', '0'])
    assert not (tmp_path/'partials').exists()
    stage = tmp_path/'preflight'
    stage.mkdir()
    (stage/'PASS.json').write_text(json.dumps(dict(status='PASS', protocol=runner.PROTOCOL,
        source_fingerprint=runner.source_fingerprint(),
        environment={'gpu': 'Tesla V100-SXM2-32GB'})))
    (stage/'status.txt').write_text('COMPLETE\n')
    with pytest.raises(RuntimeError, match='completed pilot'):
        runner.main(['cell', '--out', str(tmp_path), '--index', '0'])
    assert not (tmp_path/'partials').exists()


def test_all_three_methods_dispatch_through_the_same_sparse_backend(monkeypatch):
    import torch
    from first_group import make_model_and_points, parse_one_based_pattern
    from torch_pinn import sparse_jets
    observed = []
    original = sparse_jets.prepare_curve_evaluation

    def trace(*args, **kwargs):
        observed.append((tuple(args[2]), args[4]))
        return original(*args, **kwargs)

    monkeypatch.setattr(sparse_jets, 'prepare_curve_evaluation', trace)
    model, points = make_model_and_points(20260908, 1, 2, 2, 'cpu')
    calls, metadata, setup = runner.build_evaluators(model, points, parse_one_based_pattern('111222'))
    assert len(observed) == 3 and observed[:2] == [((1,), 6), ((1,), 6)]
    assert observed[2] == ((71, 73, 79, 83, 103, 151), 560)
    assert all(row['common_backend'] == 'torch_pinn.sparse_jets' for row in metadata.values())
    assert setup['prime_schedule_and_certificate_seconds'] >= 0
    with torch.no_grad():
        values = {method: call() for method, call in calls.items()}
    runner.validate_values(values, 1)
