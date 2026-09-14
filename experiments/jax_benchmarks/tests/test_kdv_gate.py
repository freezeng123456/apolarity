import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('bad_seed', [False, True])
def test_every_seed_required_and_priority(tmp_path, monkeypatch, bad_seed):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from run_kdv_optimization import SEEDS, decision
    methods = ('nested_jvp', 'shared_jet_fused', 'shared_jet_linear')
    for batch in (100, 1600):
        for method in methods:
            cell = tmp_path / f'kdv2d_{method}_b{batch}'
            cell.mkdir()
            rows = [dict(seed=seed, median_ms=10 if method == 'nested_jvp' else 8) for seed in SEEDS]
            if bad_seed and batch == 1600 and method != 'nested_jvp':
                rows[-1]['median_ms'] = 10
            (cell / 'results.json').write_text(json.dumps(rows))
    result = decision(tmp_path, (100, 1600), methods[1:])
    assert len(result['comparisons']) == 12
    assert result['gate'] == ('STOP_KDV_CASE' if bad_seed else 'PASS')
    assert result['selected_method'] == (None if bad_seed else 'shared_jet_linear')
