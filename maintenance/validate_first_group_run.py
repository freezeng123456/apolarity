"""Audit a recovered, frozen first-group run without changing raw artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    assert (args.root / "status.txt").read_text().strip() == "COMPLETE"
    outcomes = json.loads((args.root / "progress.json").read_text())
    names = [f"{t}_{m}_b{b}" for t, m, b in manifest["cells"]]
    assert len(names) == len(set(names)) == manifest["expected_cells"] == 22
    assert [o["cell"] for o in outcomes] == names
    assert all(o["exit_code"] == 0 for o in outcomes)
    digest = hashlib.sha256()
    for path in sorted([args.source / "first_group.py", *args.source.joinpath("src").rglob("*.py")]):
        digest.update(str(path.relative_to(args.source)).encode())
        digest.update(path.read_bytes())
    expected_hash = digest.hexdigest()
    rows = calls = 0
    direction_dtypes = {}
    for target, method, batch in manifest["cells"]:
        cell = args.root / f"{target}_{method}_b{batch}"
        config = json.loads((cell / "config.json").read_text())
        assert (cell / "status.txt").read_text().strip() == "COMPLETE"
        expected = dict(target=target, method=method, batch=batch,
                        source_sha256=expected_hash, execution="no_outer_jit",
                        whole_function_jit=False, parameter_backward=False,
                        backend="gpu", input_dim=16, width=128, depth=4,
                        warmups=10, repeats=30, dtype_real="float32",
                        jax="0.4.38", jaxlib="0.4.38", precision="highest",
                        parameters_are_dynamic=True, inputs_are_dynamic=True)
        # JAX reports the CUDA backend as gpu on some versions and cuda on others.
        assert config["backend"] in ("gpu", "cuda")
        expected.pop("backend")
        for key, value in expected.items():
            assert config[key] == value, (cell.name, key, config[key], value)
        assert config["seeds"] == manifest["seeds"]
        assert "H20" in config["device_kind"]
        direction_dtypes[target] = config["dtype_direction"]
        records = json.loads((cell / "results.json").read_text())
        assert [r["seed"] for r in records] == manifest["seeds"]
        for record in records:
            assert (record["target"], record["method"], record["batch"]) == (target, method, batch)
            assert record["execution"] == "no_outer_jit" and record["status"] == "ok"
            assert record["compile_seconds"] is None and record["executable_memory"] is None
            samples = record["times_ms"]
            assert len(samples) == 30 and np.isfinite(samples).all() and min(samples) > 0
            assert record["median_ms"] == statistics.median(samples)
            values = np.load(cell / f"values_{record['seed']}.npy", allow_pickle=False)
            assert values.shape == (batch,) and values.dtype == np.float32 and np.isfinite(values).all()
            rows += 1
            calls += len(samples)
    assert rows == manifest["expected_rows"] == 66
    report = dict(status="PASS", cells=len(names), seed_rows=rows,
                  timed_calls=calls, source_sha256=expected_hash,
                  direction_dtypes=direction_dtypes,
                  checks="Complete statuses, effective configs, source identity, seeds, timing samples, finite output shapes and dtypes")
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
