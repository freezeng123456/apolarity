from __future__ import annotations
import csv, hashlib, json, os, platform, resource, statistics, time
from pathlib import Path
import jax
import yaml


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def block_tree(value):
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def rss_bytes():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(v * 1024) if platform.system() != "Darwin" else int(v)


def measure(fn, warmups, repeats):
    start = time.perf_counter()
    first = fn()
    block_tree(first)
    first_ms = (time.perf_counter() - start) * 1000
    for _ in range(warmups):
        block_tree(fn())
    ts = []
    for _ in range(repeats):
        s = time.perf_counter()
        out = fn()
        block_tree(out)
        ts.append((time.perf_counter() - s) * 1000)
    dev = jax.devices()[0]
    peak = None
    method = "process ru_maxrss high-water mark (absolute, not per-call delta)"
    try:
        stats = dev.memory_stats()
        if stats:
            for key in ("peak_bytes_in_use", "peak_pool_bytes", "bytes_in_use"):
                if key in stats:
                    peak = int(stats[key])
                    method = f"jax device.memory_stats()[{key!r}]"
                    break
    except Exception:
        pass
    if peak is None:
        peak = rss_bytes()
    return {
        "compile_plus_first_call_ms": first_ms,
        "times_ms": ts,
        "median_ms": statistics.median(ts),
        "mean_ms": statistics.mean(ts),
        "min_ms": min(ts),
        "max_ms": max(ts),
        "peak_memory_bytes": peak,
        "peak_memory_method": method,
    }


def rel_l2(a, e):
    import numpy as np

    aa = np.asarray(a)
    ee = np.asarray(e)
    den = max(np.linalg.norm(ee.ravel()), 1e-30)
    return float(np.linalg.norm((aa - ee).ravel()) / den)


def max_abs(a, e):
    import numpy as np

    return float(np.max(np.abs(np.asarray(a) - np.asarray(e))))


def source_hash(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if (
            p.is_file()
            and "__pycache__" not in p.as_posix()
            and ".pytest_cache" not in p.as_posix()
            and "/results/" not in p.as_posix()
        ):
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def provenance(root, config):
    devices = jax.devices()
    gpu_visible = any(getattr(d, "platform", "") == "gpu" for d in devices)
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in devices],
        "gpu_visible": gpu_visible,
        "config_path": str(config),
        "source_tree_sha256": source_hash(root),
        "jax_platform_name": os.environ.get("JAX_PLATFORMS"),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    }


def write_results(rows, out):
    out = Path(out)
    stem = out.with_suffix("") if out.suffix else out
    stem.parent.mkdir(parents=True, exist_ok=True)
    jp = stem.with_suffix(".json")
    cp = stem.with_suffix(".csv")
    jp.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    keys = sorted({k for r in rows for k in r if k != "times_ms"})
    with cp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            q = {k: r.get(k) for k in keys}
            for k, v in list(q.items()):
                if isinstance(v, (dict, list, tuple)):
                    q[k] = json.dumps(v, separators=(",", ":"), default=str)
            w.writerow(q)
    return jp, cp
