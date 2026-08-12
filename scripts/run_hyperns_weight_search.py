#!/usr/bin/env python3
"""Run HO-04 hyperviscous Navier--Stokes smoke or shared weight search."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ["APOLARITY_PROBLEM_FAMILY"] = "hyperviscous_ns_2d"
runpy.run_path(
    str(ROOT / "scripts" / "run_cahn2d_weight_search.py"),
    run_name="__main__",
)

