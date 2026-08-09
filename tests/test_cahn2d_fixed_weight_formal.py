from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
MODULE_PATH = SCRIPTS / "run_fixed_weight_formal.py"
SPEC = importlib.util.spec_from_file_location("run_fixed_weight_formal", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
formal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formal)


def test_protocol_contains_only_two_dimensional_tasks_and_fixed_weights():
    assert formal.TASK_ORDER == (
        "cahn_hilliard_2d_o4",
        "cahn_hilliard_2d_o6",
    )
    assert set(formal.METHODS) == {"war", "real_sinh_autodiff"}
    assert formal.FIXED_WEIGHTS == {
        "cahn_hilliard_2d_o4": (1.0, 10.0),
        "cahn_hilliard_2d_o6": (1.0, 10.0),
    }


def test_formal_parser_defaults_to_1200_seconds_and_five_seeds():
    args = formal.build_parser().parse_args(["orchestrate"])
    assert args.seconds == 1200.0
    assert args.seeds == 5
    assert formal.sample_counts(args) == formal.FORMAL_SAMPLE_COUNTS


def test_worker_command_carries_every_fixed_protocol_field(tmp_path: Path):
    command = formal.worker_command(
        "cahn_hilliard_2d_o6",
        "real_sinh_autodiff",
        4,
        54321,
        1200.0,
        tmp_path / "result.json",
        formal.FORMAL_SAMPLE_COUNTS,
        smoke=False,
    )
    rendered = " ".join(command)
    for fragment in (
        "--task cahn_hilliard_2d_o6",
        "--method real_sinh_autodiff",
        "--weights 1,10",
        "--seed 4",
        "--seconds 1200.0",
        "--n-int 4096",
        "--n-ic 1024",
        "--n-bc 2048",
        "--n-eval 32768",
        "--history-eval-n 4096",
    ):
        assert fragment in rendered


def test_complete_result_requires_2d_samples_weights_and_formal_id(tmp_path: Path):
    path = tmp_path / "war.json"
    payload = {
        "protocol_id": formal.ENGINE_PROTOCOL_ID,
        "formal_protocol_id": formal.PROTOCOL_ID,
        "status": "complete",
        "task_id": "cahn_hilliard_2d_o4",
        "method": "war",
        "weights": [1.0, 10.0],
        "train_seed": 2,
        "budget_seconds": 1200.0,
        "loss": 0.1,
        "rel_error": 0.2,
        "problem": {
            "n_int": 4096,
            "n_ic": 1024,
            "n_bc_total": 2048,
            "n_eval": 32768,
            "history_eval_n": 4096,
        },
    }
    path.write_text(json.dumps(payload))
    assert formal.complete_formal_result(
        path,
        task="cahn_hilliard_2d_o4",
        method="war",
        seed=2,
        seconds=1200.0,
        samples=formal.FORMAL_SAMPLE_COUNTS,
    )
    payload["weights"] = [10.0, 10.0]
    path.write_text(json.dumps(payload))
    assert not formal.complete_formal_result(
        path,
        task="cahn_hilliard_2d_o4",
        method="war",
        seed=2,
        seconds=1200.0,
        samples=formal.FORMAL_SAMPLE_COUNTS,
    )
