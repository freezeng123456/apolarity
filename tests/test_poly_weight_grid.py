import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "poly_weight_grid", ROOT / "experiments" / "archived" / "scripts" / "run_poly_weight_grid.py"
)
grid_search = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(grid_search)


def test_cartesian_grid_sizes_and_order():
    grid = (0.01, 0.1, 1.0)
    o2 = grid_search.cartesian_weights(2, grid)
    o4 = grid_search.cartesian_weights(4, grid)
    o6 = grid_search.cartesian_weights(6, grid)
    assert len(o2) == 3
    assert len(o4) == 9
    assert len(o6) == 27
    assert o4[0] == (0.01, 0.01)
    assert o4[-1] == (1.0, 1.0)


def test_parse_grid_rejects_invalid_values():
    with pytest.raises(ValueError, match="positive"):
        grid_search.parse_grid("0.1,0")
    with pytest.raises(ValueError, match="unique"):
        grid_search.parse_grid("0.1,0.1")


def test_score_and_complete_result(tmp_path):
    rows = [
        {
            "variant": "vanilla_tanh_direct_ad",
            "bc_weights": [0.1, 1.0],
            "L2_err": 0.04,
        },
        {
            "variant": "complex_sinh",
            "bc_weights": [0.1, 1.0],
            "L2_err": 0.01,
        },
    ]
    path = tmp_path / "point.json"
    path.write_text(json.dumps(rows))
    assert grid_search.load_complete(path, (0.1, 1.0)) == rows
    score = grid_search.score_rows(rows)
    assert score["geometric_mean"] == pytest.approx(0.02)
    assert score["max_error"] == pytest.approx(0.04)
