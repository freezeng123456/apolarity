import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "poly_dim_suite", ROOT / "experiments" / "archived" / "scripts" / "run_poly_dim_suite.py"
)
suite = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(suite)


def test_load_best_weights(tmp_path):
    ranking = tmp_path / "ranking.json"
    ranking.write_text(json.dumps([{"bc_weights": [0.1, 10.0]}]))
    assert suite.load_best_weights(ranking) == (0.1, 10.0)
