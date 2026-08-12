#!/usr/bin/env python3
"""Run the HO-01 MBE smoke or shared loss-weight grid.

This wrapper selects ``experiments/mbe_2d/problem.py`` and then delegates to
the audited two-weight orchestration engine.  See ``--help`` for the shared
``worker``, ``smoke``, ``orchestrate``, and ``summarize`` subcommands.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ["APOLARITY_PROBLEM_FAMILY"] = "mbe_2d"
runpy.run_path(
    str(ROOT / "scripts" / "run_cahn2d_weight_search.py"),
    run_name="__main__",
)
