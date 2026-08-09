from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
MODULE_PATH = SCRIPTS / "run_poly_fixed_weight_formal.py"
SPEC = importlib.util.spec_from_file_location("run_poly_fixed_weight_formal", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
formal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formal)


def test_protocol_contains_only_current_poly_tasks_and_weights():
    assert formal.TASK_ORDER == (
        "poly_d2_o2",
        "poly_d2_o4",
        "poly_d2_o6",
    )
    assert formal.FIXED_WEIGHTS == {
        "poly_d2_o2": (1.0,),
        "poly_d2_o4": (1.0, 1.0),
        "poly_d2_o6": (10.0, 1.0, 1.0),
    }


def test_poly_real_baseline_remains_float32_tanh_without_frequency_init(monkeypatch):
    monkeypatch.setattr(formal, "git_state", lambda: {"sha": "test", "dirty": False})
    monkeypatch.setattr(formal, "hardware_metadata", lambda: {"device": "test"})
    payload = formal.manifest(
        formal.TASK_ORDER,
        1200.0,
        tuple(range(5)),
        smoke=False,
    )
    real = payload["architecture"]["real_tanh_autodiff"]
    war = payload["architecture"]["war"]
    assert real == {
        "representation": "real",
        "activation": "tanh",
        "backend": "direct_autodiff",
        "hidden": 128,
        "depth": 4,
        "init_mode": "common_xavier",
        "frequency_initialization": "disabled",
        "parameter_dtype": "torch.float32",
    }
    assert war["activation"] == "sinh"
    assert war["parameter_dtype"] == "torch.complex64"
    assert war["frequency_initialization"] == "disabled"
    assert payload["method_seed_run_count"] == 30


def test_poly_formal_defaults_to_1200_seconds_and_five_seeds():
    args = formal.build_parser().parse_args(["orchestrate"])
    assert args.seconds == 1200.0
    assert args.seeds == 5
    assert args.tasks == "all"
