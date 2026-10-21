"""Balance, profiler interpretation and fail-closed stages for optimized timing."""
from collections import Counter
import json

import pytest

import stdepp_optimized_partials as runner


def test_only_six_partial_cells_retain_matched_execution_settings(tmp_path):
    plan = runner.plan(tmp_path)
    cells = plan['cells']
    assert len(cells) == 6 and {cell['target'] for cell in cells} == set(runner.TARGETS)
    assert all(cell['methods'] == ['waring_batched', 'stdepp'] and
               cell['variants'] == ['baseline', 'precast', 'optimized'] and
               cell['seeds'] == [20260908, 20260909, 20260910] and
               cell['batch'] == 100 and cell['width'] == 128 and cell['depth'] == 4 and
               cell['warmups'] == 10 and cell['repeats'] == 30 for cell in cells)
    assert plan['allocation']['cpus_per_task'] == 10 and plan['allocation']['gpus'] == 1
    assert plan['execution']['threads'] == 1
    assert not plan['pde_training'] and not plan['checkpoints']


@pytest.mark.parametrize('seed_index', range(3))
def test_every_method_and_variant_has_balanced_formal_position(seed_index):
    rounds = runner.balanced_round_orders(30, seed_index)
    expected = {f'{variant}/{method}' for variant in runner.VARIANTS for method in runner.METHODS}
    assert len(rounds) == 30 and all(set(order) == expected and len(order) == 6 for order in rounds)
    variant_orders = Counter(tuple(key.split('/')[0] for key in order[::2]) for order in rounds)
    assert set(variant_orders.values()) == {10}
    for variant in runner.VARIANTS:
        positions = Counter(next(index//2 for index, key in enumerate(order)
                                 if key == f'{variant}/{runner.METHODS[0]}') for order in rounds)
        assert positions == {0: 10, 1: 10, 2: 10}
        first = Counter(next(key.split('/')[1] for key in order if key.startswith(variant+'/'))
                        for order in rounds)
        assert first == {runner.METHODS[0]: 15, runner.METHODS[1]: 15}
        joint = Counter((next(index//2 for index, key in enumerate(order)
                              if key == f'{variant}/{runner.METHODS[0]}'),
                         next(key.split('/')[1] for key in order if key.startswith(variant+'/')))
                        for order in rounds)
        assert set(joint.values()) == {5} and len(joint) == 6
        for method in runner.METHODS:
            absolute = Counter(order.index(f'{variant}/{method}') for order in rounds)
            assert absolute == {index: 5 for index in range(6)}


def test_repeat_count_cannot_silently_unbalance_variants():
    with pytest.raises(ValueError, match='multiple of six'):
        runner.balanced_round_orders(31)


def test_trace_duration_sums_and_union_are_distinguished():
    events = [dict(ph='X', cat='kernel', name='compute', ts=0, dur=10),
              dict(ph='X', cat='kernel', name='compute', ts=5, dur=10),
              dict(ph='X', cat='kernel', name='other', ts=20, dur=3),
              dict(ph='X', cat='gpu_memcpy', name='Memcpy HtoD', ts=15, dur=2),
              dict(ph='X', cat='cpu_op', name='aten::mul', ts=0, dur=25),
              dict(ph='X', cat='cuda_runtime', name='cudaLaunchKernel', ts=0, dur=1)]
    result = runner.summarize_trace({'traceEvents': events})
    assert result['compute_kernels']['summed_duration_us'] == 23
    assert result['compute_kernels']['active_union_us'] == 18
    assert result['compute_kernels']['span_us'] == 23
    assert result['compute_kernels']['internal_gap_us'] == 5
    assert result['all_gpu_activity']['active_union_us'] == 20
    assert result['gpu_copies']['count'] == 1 and result['cpu_kernel_enqueue']['count'] == 1
    assert 'not wall-time fractions' in result['interpretation']


def test_stage_receipt_requires_successful_exit_and_exact_source(tmp_path):
    with pytest.raises(RuntimeError, match='completed successful profile'):
        runner.receipt(tmp_path, 'profile')
    stage = tmp_path/'profile'
    stage.mkdir()
    result = dict(status='PASS', protocol=runner.PROTOCOL, source_fingerprint='old',
                  environment={'gpu': 'Tesla V100-SXM2-32GB'})
    (stage/'PASS.json').write_text(json.dumps(result))
    (stage/'status.txt').write_text('RUNNING\n')
    with pytest.raises(RuntimeError, match='completed successful profile'):
        runner.receipt(tmp_path, 'profile')
    (stage/'status.txt').write_text('COMPLETE\n')
    with pytest.raises(RuntimeError, match='Source changed'):
        runner.receipt(tmp_path, 'profile')
    result['source_fingerprint'] = runner.source_fingerprint()
    (stage/'PASS.json').write_text(json.dumps(result))
    assert runner.receipt(tmp_path, 'profile')['status'] == 'PASS'


def test_preflight_cannot_start_without_prior_completed_profile(tmp_path):
    with pytest.raises(RuntimeError, match='completed successful profile'):
        runner.main(['preflight', '--out', str(tmp_path)])
    assert not (tmp_path/'preflight').exists()
