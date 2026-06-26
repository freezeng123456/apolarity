#!/usr/bin/env python3
"""Aggregate the oscillatory benchmark CSVs and check the acceptance criterion:

    the advantage of complex sinh over the best PARAMETER-MATCHED real baseline
    grows (monotonically) with order / wavenumber on the real-valued problems.

Usage:
  python experiments/aggregate_osc.py            # reads results/*.csv
  python experiments/aggregate_osc.py results/helmholtz_highk.csv ...
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["helmholtz_highk", "kdv_dispersive", "plate_beam",
           "cahn_hilliard", "nls"]
REAL_BASELINES = ["fourier", "siren", "mscale", "tanh", "real_sinh"]
COMPLEX = "complex_sinh"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("inf")
    except (TypeError, ValueError):
        return float("inf")


def mean_std(vals):
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return float("inf"), 0.0
    m = sum(vals) / len(vals)
    s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return m, s


def aggregate_file(path):
    rows = load(path)
    # group L2 by (problem, sweep, order, variant)
    g = defaultdict(list)
    meta = {}
    for r in rows:
        key = (r["problem"], fnum(r.get("sweep", 0)), fnum(r.get("order", 0)))
        g[(key, r["variant"])].append(fnum(r["L2_err"]))
        meta[key] = meta.get(key, r)
    # per problem-instance
    instances = sorted({k for (k, _v) in g}, key=lambda k: (k[2], k[1]))
    print(f"\n{'='*78}\n# {Path(path).stem}\n{'='*78}")
    variants = []
    for r in rows:
        if r["variant"] not in variants:
            variants.append(r["variant"])
    header = f"{'instance':<16}{'order':>6}{'sweep':>7}  " + "".join(f"{v:>13}" for v in variants)
    print(header)
    trend = []
    for key in instances:
        prob, sweep, order = key
        cells = []
        for v in variants:
            m, _s = mean_std(g.get((key, v), [float("inf")]))
            cells.append(m)
        line = f"{prob:<16}{order:>6.0f}{sweep:>7.1f}  " + "".join(
            (f"{c:>13.3e}" if math.isfinite(c) else f"{'fail':>13}") for c in cells)
        print(line)
        cm, _ = mean_std(g.get((key, COMPLEX), [float("inf")]))
        reals = [mean_std(g.get((key, v), [float("inf")]))[0] for v in REAL_BASELINES
                 if (key, v) in g]
        best_real = min(reals) if reals else float("inf")
        adv = best_real / cm if (math.isfinite(best_real) and cm > 0) else float("inf")
        trend.append((prob, order, sweep, cm, best_real, adv))
    # per-family advantage-vs-sweep trend (family = problem name w/o trailing token)
    fams = defaultdict(list)
    for prob, order, sweep, cm, br, adv in trend:
        fams["_".join(prob.split("_")[:-1]) or prob].append((sweep, order, cm, br, adv))
    print(f"\n  {'family':<12}{'sweep':>7}{'order':>6}{'complex_L2':>13}"
          f"{'best_real':>12}{'adv(x)':>9}")
    for fam in sorted(fams):
        items = sorted(fams[fam])
        for sweep, order, cm, br, adv in items:
            cm_s = f"{cm:.3e}" if math.isfinite(cm) else "fail"
            br_s = f"{br:.3e}" if math.isfinite(br) else "fail"
            adv_s = f"{adv:.2f}" if math.isfinite(adv) else "inf"
            print(f"  {fam:<12}{sweep:>7.1f}{order:>6.0f}{cm_s:>13}{br_s:>12}{adv_s:>9}")
        advs = [a for *_x, a in items]
        mono = all(b >= a - 1e-9 for a, b in zip(advs, advs[1:])) if len(advs) > 1 else None
        wins = sum(1 for a in advs if a > 1.0)
        fin = [a for a in advs if math.isfinite(a)]
        print(f"  {'-> '+fam:<12} wins {wins}/{len(advs)}  monotone+:{mono}  "
              f"adv {min(fin):.2f}->{max(fin):.2f}\n" if fin else "")
    return trend


def main(argv):
    args = argv or DEFAULT
    paths = []
    for a in args:
        p = Path(a)
        if not p.exists():
            p = ROOT / "results" / (a if a.endswith(".csv") else f"{a}.csv")
        if p.exists():
            paths.append(p)
        else:
            print(f"[skip] {a}: not found")
    for p in paths:
        aggregate_file(p)


if __name__ == "__main__":
    main(sys.argv[1:])
