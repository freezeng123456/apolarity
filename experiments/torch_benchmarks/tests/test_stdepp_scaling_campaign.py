"""Source matching, balanced timing and numerical exclusion guards."""
import argparse
from collections import Counter
import json

import pytest
import torch

import stdepp_scaling_campaign as runner
from first_group import make_model_and_points, parameters_sha256, tensor_sha256


def test_plan_freezes_eighteen_cells_without_selecting_only_winners(tmp_path):
    plan = runner.plan(tmp_path)
    assert len(plan['cells']) == 18
    assert len({cell['name'] for cell in plan['cells']}) == 18
    assert {tuple(cell['exponents']) for cell in plan['cells']} == set(runner.EXPONENTS)
    assert {(cell['batch'], cell['width']) for cell in plan['cells']} == set(runner.SHAPES)
    for cell in plan['cells']:
        assert cell['methods'] == ['wdd', 'stde61']
        assert cell['seeds'] == [20260908, 20260909, 20260910]
        assert (cell['input_dim'], cell['depth'], cell['warmups'], cell['repeats']) == (16, 4, 10, 30)
    assert plan['allocation']['cpus_per_task'] == 10
    assert plan['execution']['threads'] == 1
    assert not plan['pde_training'] and not plan['checkpoints']
    assert not plan['correctness']['absolute_tolerance_floor']


@pytest.mark.parametrize('seed_index', range(3))
def test_every_seed_has_balanced_ab_ba_rounds(seed_index):
    rounds = runner.round_orders(30, seed_index)
    assert Counter(map(tuple, rounds)) == {('wdd', 'stde61'): 15, ('stde61', 'wdd'): 15}
    assert rounds[0] == list(runner.METHODS[::1 if seed_index % 2 == 0 else -1])
    with pytest.raises(ValueError, match='multiple of two'):
        runner.round_orders(31)


def test_float64_reference_preserves_same_quantized_fp32_values():
    model, points = make_model_and_points(20260908, 12, 5, 3, 'cpu')
    before = parameters_sha256(model), tensor_sha256(points)
    reference_model, reference_points = runner.reference_inputs(model, points)
    assert reference_points.dtype == torch.float64 and len(reference_points) == 8
    torch.testing.assert_close(reference_points, points[:8].double(), rtol=0, atol=0)
    for original, reference in zip(model.parameters(), reference_model.parameters()):
        assert reference.dtype == torch.float64
        torch.testing.assert_close(reference, original.double(), rtol=0, atol=0)
        assert original.data_ptr() != reference.data_ptr()
    assert before == (parameters_sha256(model), tensor_sha256(points))


def test_small_derivatives_cannot_pass_via_absolute_tolerance_floor():
    reference = torch.tensor([1e-12, -2e-12], dtype=torch.float64)
    assert not runner.agreement(torch.zeros_like(reference), reference, 1e-3)['passed']
    error = runner.agreement(reference*1.0001, reference, 1e-3)
    assert error['passed']
    assert error['relative_l2'] == pytest.approx(1e-4)
    assert error['max_abs_over_reference_max_abs'] == pytest.approx(1e-4)
    zero = torch.zeros(2, dtype=torch.float64)
    assert runner.agreement(zero, zero, 1e-3)['passed']
    failure = runner.agreement(reference, zero, 1e-3)
    assert not failure['passed'] and failure['relative_l2'] is None
    json.dumps(failure, allow_nan=False)
    assert not runner.agreement(torch.tensor([float('nan')]), torch.ones(1), 1e-3)['passed']


def test_global_max_error_gate_can_fail_even_when_l2_passes():
    reference = torch.ones(10000, dtype=torch.float64)
    observed = reference.clone()
    observed[0] += .01
    error = runner.agreement(observed, reference, 1e-3)
    assert error['relative_l2'] < 1e-3
    assert error['max_abs_over_reference_max_abs'] > 1e-3
    assert not error['passed']


def test_formal_environment_also_requires_matched_cpu_model(monkeypatch):
    monkeypatch.setattr(runner, 'gpu_environment', lambda: {'lscpu': 'Intel Xeon Platinum 8255C CPU'})
    assert '8255C' in runner.environment('cuda')['lscpu']
    monkeypatch.setattr(runner, 'gpu_environment', lambda: {'lscpu': 'other CPU model'})
    with pytest.raises(RuntimeError, match='8255C'):
        runner.environment('cuda')


@pytest.mark.parametrize('exponents,full,reduced,stde', [
    ((6,), 1, 1, 1), ((2, 2, 2), 9, 5, 13), ((2, 2, 2, 2), 27, 14, 40),
    ((3, 3, 3), 16, 10, 28), ((4, 4, 4), 25, 13, 49), ((5, 5, 5), 36, 20, 100),
])
def test_both_methods_use_same_backend_with_verified_conjugate_counts(monkeypatch, exponents, full, reduced, stde):
    seen = []
    original = runner.sparse_jets.prepare_curve_evaluation
    def trace(*args, **kwargs):
        seen.append((tuple(args[2]), args[4]))
        return original(*args, **kwargs)
    monkeypatch.setattr(runner.sparse_jets, 'prepare_curve_evaluation', trace)
    model, points = make_model_and_points(20260908, 2, 2, 2, 'cpu')
    calls, metadata, _ = runner.build_evaluators(model, points, runner.alpha_from_exponents(exponents))
    assert seen == [((1,), sum(exponents))]*2
    assert metadata['wdd']['schedule']['original_count'] == full
    assert metadata['wdd']['schedule']['evaluated_count'] == reduced
    assert metadata['stde61']['schedule']['evaluated_count'] == stde
    assert all(row['common_backend'] == 'torch_pinn.sparse_jets' for row in metadata.values())
    assert set(calls) == set(runner.METHODS)


def test_independent_one_neuron_float64_checks_all_targets():
    rows = runner.independent_checks('cpu')
    assert len(rows) == 12
    assert all(row['passed'] for row in rows)


def test_numerical_failure_is_not_timed(monkeypatch):
    model, points = make_model_and_points(20260908, 2, 2, 2, 'cpu')
    values = {method: torch.zeros(2) for method in runner.METHODS}
    monkeypatch.setattr(runner, 'numerical_gate', lambda *args: (
        dict(passed=False, reason='injected_numeric_failure'), values, {}))
    def forbidden(*args, **kwargs):
        pytest.fail('A failed numerical gate must prevent timing')
    monkeypatch.setattr(runner, 'time_evaluators', forbidden)
    report, observed, references = runner.benchmark(model, points, (0, 0), warmups=0, repeats=2)
    assert report['status'] == 'INVALID_NUMERICS' and not report['timed']
    assert all(count['timed'] == 0 for count in report['evaluation_counts'].values())
    assert observed is values and references == {}


def test_small_cpu_benchmark_passes_without_mutating_source():
    model, points = make_model_and_points(20260908, 8, 3, 2, 'cpu')
    before = parameters_sha256(model), tensor_sha256(points)
    report, values, reference = runner.benchmark(model, points, (0, 0, 1, 1, 2, 2), warmups=1, repeats=2)
    assert report['status'] == 'COMPLETE' and report['timed']
    assert report['correctness']['passed']
    assert all(len(times) == 2 for times in report['times_ms'].values())
    assert all(value.shape == (8,) for value in values.values())
    assert all(value.dtype == torch.float64 for value in reference.values())
    assert before == (parameters_sha256(model), tensor_sha256(points))


def test_invalid_cell_keeps_all_seeds_and_has_no_success_aggregate(tmp_path, monkeypatch):
    tmp_path.joinpath('config.json').write_text('{}')
    seen = []
    def failed(model, points, alpha, **kwargs):
        seen.append(kwargs['seed_index'])
        return dict(status='INVALID_NUMERICS', timed=False, correctness={'passed': False}), {
            method: torch.zeros(len(points)) for method in runner.METHODS}, {}
    monkeypatch.setattr(runner, 'benchmark', failed)
    args = argparse.Namespace(out=tmp_path, device='cpu', target='111111', exponents=[6],
        seeds=[20260908, 20260909, 20260910], batch=2, width=2, depth=2, warmups=0, repeats=2)
    assert runner._cell_body(args) == 'INVALID_NUMERICS'
    rows = json.loads(tmp_path.joinpath('results.json').read_text())
    summary = json.loads(tmp_path.joinpath('summary.json').read_text())
    assert seen == [0, 1, 2] and len(rows) == 3
    assert summary['status'] == 'INVALID_NUMERICS' and summary['valid_seeds'] == 0
    assert 'methods' not in summary
    assert len(list(tmp_path.glob('values_*.npy'))) == 6


def test_stage_distinguishes_numeric_exclusion_from_infrastructure_error(tmp_path):
    args = argparse.Namespace(out=tmp_path/'invalid', device='cpu', expected_device='V100')
    runner.run_stage(args, lambda _: 'INVALID_NUMERICS')
    assert (args.out/'status.txt').read_text().strip() == 'INVALID_NUMERICS'
    assert (args.out/'SHA256SUMS').is_file()
    args.out = tmp_path/'failure'
    def broken(_):
        raise RuntimeError('infrastructure failure')
    with pytest.raises(RuntimeError, match='infrastructure'):
        runner.run_stage(args, broken)
    assert (args.out/'status.txt').read_text().strip() == 'FAILED'


def test_formal_cell_requires_same_source_and_execution_preflight(tmp_path):
    with pytest.raises(RuntimeError, match='completed preflight'):
        runner.main(['cell', '--out', str(tmp_path), '--index', '0', '--device', 'cpu'])
    root = tmp_path/'preflight'
    root.mkdir()
    (root/'status.txt').write_text('COMPLETE\n')
    receipt = dict(status='PASS', protocol=runner.PROTOCOL, source_fingerprint='stale',
        environment={'gpu': 'CPU_LOCAL_VALIDATION_ONLY'})
    (root/'PASS.json').write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match='Source changed'):
        runner.receipt(tmp_path, 'cpu')
    receipt['source_fingerprint'] = runner.source_fingerprint()
    (root/'PASS.json').write_text(json.dumps(receipt))
    assert runner.receipt(tmp_path, 'cpu')['status'] == 'PASS'
    with pytest.raises(RuntimeError, match='matched execution'):
        runner.receipt(tmp_path, 'cuda')
