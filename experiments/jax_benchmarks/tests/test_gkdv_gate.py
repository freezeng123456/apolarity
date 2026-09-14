import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_gkdv_optimization import BATCHES, METHODS, SEEDS, decision


def test_all_pairs_required(tmp_path):
    for batch in BATCHES:
        for method, ms in zip(METHODS, (40, 30, 20)):
            cell = tmp_path / f'gkdv1d_{method}_b{batch}'
            cell.mkdir()
            (cell / 'results.json').write_text(json.dumps([dict(seed=s, median_ms=ms) for s in SEEDS]))
    assert decision(tmp_path)['gate'] == 'PASS'
    path = tmp_path / 'gkdv1d_shared_jet_linear_b1600' / 'results.json'
    rows = json.loads(path.read_text())
    rows[-1]['median_ms'] = 29
    path.write_text(json.dumps(rows))
    assert decision(tmp_path)['gate'] == 'RETAIN_ORIGINAL'
