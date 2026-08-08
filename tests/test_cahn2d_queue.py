from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_cahn2d_after_poly.py"
SPEC = importlib.util.spec_from_file_location("run_cahn2d_after_poly", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def test_validate_ephemeral_smoke_conclusion(tmp_path: Path):
    path = tmp_path / "smoke.json"
    cells = []
    for task in ("cahn_hilliard_2d_o4", "cahn_hilliard_2d_o6"):
        for method in ("war", "real_sinh_autodiff"):
            cells.append({
                "task_id": task,
                "method": method,
                "status": "complete",
                "loss": 1.0,
                "rel_error": 0.5,
                "peak_mb": 100.0,
            })
    path.write_text(json.dumps({
        "protocol_id": queue.SEARCH_PROTOCOL_ID,
        "passed": True,
        "raw_artifacts_retained": False,
        "cells": cells,
    }))
    assert queue.validate_smoke(path)["passed"] is True


def test_validate_complete_196_run_search(tmp_path: Path):
    (tmp_path / "SEARCH_COMPLETE").write_text("ok\n")
    task_summaries = [
        {
            "task_id": task,
            "complete": True,
            "paired_complete_candidates": 49,
            "complete_runs": 98,
        }
        for task in ("cahn_hilliard_2d_o4", "cahn_hilliard_2d_o6")
    ]
    (tmp_path / "summary.json").write_text(json.dumps({
        "protocol_id": queue.SEARCH_PROTOCOL_ID,
        "complete": True,
        "task_summaries": task_summaries,
    }))
    assert queue.validate_search(tmp_path)["complete"] is True


def test_search_sized_smoke_command_contains_every_sample_count(tmp_path: Path):
    command = queue.build_smoke_command(
        tmp_path / "conclusion.json",
        seconds=3.0,
        sample_counts={
            "n_int": 512,
            "n_ic": 256,
            "n_bc": 512,
            "n_eval": 4096,
            "history_eval_n": 1024,
        },
    )
    rendered = " ".join(command)
    for fragment in (
        "--n-int 512",
        "--n-ic 256",
        "--n-bc 512",
        "--n-eval 4096",
        "--history-eval-n 1024",
    ):
        assert fragment in rendered
