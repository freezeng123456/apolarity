#!/usr/bin/env python3
"""Run the HO-02 MPFC smoke or audited shared loss-weight search."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ["APOLARITY_PROBLEM_FAMILY"] = "mpfc_2d"
runpy.run_path(
    str(ROOT / "scripts" / "run_cahn2d_weight_search.py"),
    run_name="__main__",
)
