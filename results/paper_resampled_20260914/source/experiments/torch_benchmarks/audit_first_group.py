"""Read-only audit and formal Chinese report for a completed first-group matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


TARGETS = ('111111', '111222', '112233', '123456', '11223344', '111222333')
SEEDS = (20260908, 20260909, 20260910)
SOURCE_FILES = (
    Path('experiments/torch_benchmarks/first_group.py'),
    Path('experiments/torch_benchmarks/first_group_verify.py'),
    Path('experiments/torch_benchmarks/run_first_group_matrix.py'),
    Path('experiments/torch_benchmarks/audit_first_group.py'),
    Path('experiments/torch_benchmarks/launch_first_group_t4.py'),
    Path('experiments/torch_benchmarks/torch_pinn/model.py'),
    Path('experiments/torch_benchmarks/torch_pinn/monomials.py'),
    Path('experiments/torch_benchmarks/torch_pinn/jets.py'),
    Path('experiments/torch_benchmarks/torch_pinn/operators.py'),
)


def read(path: Path):
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sha256_manifest(root: Path) -> int:
    manifest = root/'SHA256SUMS'
    lines = manifest.read_text().splitlines()
    seen = set()
    for line in lines:
        digest, name = line.split('  ', 1)
        file = root/name
        if not file.resolve().is_relative_to(root.resolve()) or name in seen:
            raise AssertionError(f'invalid hash-manifest path {name!r}')
        if file_sha256(file) != digest:
            raise AssertionError(f'hash mismatch: {name}')
        seen.add(name)
    actual = {str(file.relative_to(root)) for file in root.rglob('*') if file.is_file() and file.name != 'SHA256SUMS'}
    if seen != actual:
        raise AssertionError({'manifest_only': sorted(seen-actual), 'disk_only': sorted(actual-seen)})
    return len(seen)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='raw results directory')
    parser.add_argument('--source', type=Path, required=True, help='repository root for source-hash verification')
    parser.add_argument('--out', type=Path, required=True, help='canonical run directory, outside raw root')
    parser.add_argument('--read-only', action='store_true', help='validate existing report/artifacts without writing')
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve()
    if (root/'status.txt').read_text().strip() != 'COMPLETE':
        raise AssertionError('raw matrix is not complete')
    raw_hash_files = validate_sha256_manifest(root)
    manifest = read(root/'manifest.json')
    assert manifest['protocol'] == 'torch_first_group_values_v1'
    assert manifest['expected_cells'] == 22 and manifest['expected_rows'] == 66 and manifest['expected_timed_calls'] == 1980
    assert manifest['baseline'] == 'nested_jvp' and not manifest['automatic_baseline_selection']
    assert manifest['execution'] == 'eager_no_outer_jit' and manifest['seeds'] == list(SEEDS)
    expected_cells = {(cell['target'], cell['method'], cell['batch']) for cell in manifest['cells']}
    if len(expected_cells) != 22:
        raise AssertionError('manifest has duplicate or missing cells')
    verification = read(root/'verification'/'verification.json')
    if (root/'verification'/'status.txt').read_text().strip() != 'PASS' or verification['status'] != 'PASS':
        raise AssertionError('pre-timing verification did not pass')
    rows = []
    lookup = {}
    source_hash_ok = True
    output_checks = []
    for target, method, batch in sorted(expected_cells, key=lambda item: (item[2], len(item[0]), item[0], item[1])):
        cell = root/f'{target}_{method}_b{batch}'
        if (cell/'status.txt').read_text().strip() != 'COMPLETE':
            raise AssertionError(f'incomplete cell: {cell.name}')
        config = read(cell/'config.json')
        assert config['source_commit'] == manifest['source_commit']
        assert config['target'] == target and config['method'] == method and config['batch'] == batch
        assert config['width'] == 128 and config['depth'] == 4 and config['input_dim'] == 16
        assert config['warmups'] == 10 and config['repeats'] == 30 and config['seeds'] == list(SEEDS)
        assert config['backend'] == 'gpu' and 'T4' in config['device_kind']
        assert config['torch'] == '2.0.1+cu118' and config['torch_cuda'] == '11.8'
        assert not config['parameter_backward'] and not config['outer_jit'] and not config['torch_compile']
        assert not config['third_party_differentiation_library'] and not config['direction_chunking']
        for path, digest in config['source_files_sha256'].items():
            source_hash_ok &= file_sha256(args.source/path) == digest
        records = read(cell/'results.json')
        if [record['seed'] for record in records] != list(SEEDS):
            raise AssertionError(f'bad seed rows: {cell.name}')
        for record in records:
            times = record['times_ms']
            if len(times) != 30 or not np.isfinite(times).all() or min(times) <= 0:
                raise AssertionError(f'bad timed samples: {cell.name}/{record["seed"]}')
            if record['median_ms'] != statistics.median(times):
                raise AssertionError(f'bad median: {cell.name}/{record["seed"]}')
            value = np.load(cell/f'values_{record["seed"]}.npy')
            if value.shape != (batch,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise AssertionError(f'bad value: {cell.name}/{record["seed"]}')
            if hashlib.sha256(value.tobytes()).hexdigest() != record['output_sha256']:
                raise AssertionError(f'bad value hash: {cell.name}/{record["seed"]}')
            lookup[(target, method, batch, record['seed'])] = record
            rows.append(record)
    if len(rows) != 66 or not source_hash_ok:
        raise AssertionError({'records': len(rows), 'source_hash_ok': source_hash_ok})
    for target, batch in sorted({(key[0], key[2]) for key in lookup}):
        for seed in SEEDS:
            baseline = np.load(root/f'{target}_nested_jvp_b{batch}'/f'values_{seed}.npy')
            candidate = np.load(root/f'{target}_waring_batched_b{batch}'/f'values_{seed}.npy')
            close = np.allclose(candidate, baseline, rtol=1e-3, atol=1e-6)
            output_checks.append({
                'target': target, 'batch': batch, 'seed': seed, 'method': 'waring_batched',
                'reference': 'nested_jvp', 'max_absolute': float(np.max(np.abs(candidate-baseline))),
                'relative_l2': float(np.linalg.norm(candidate-baseline)/max(np.linalg.norm(baseline), 1e-30)),
                'consistent': bool(close),
            })
            if not close:
                raise AssertionError(output_checks[-1])
    for target in ('123456', '111222333'):
        for seed in SEEDS:
            serial = np.load(root/f'{target}_waring_serial_b100'/f'values_{seed}.npy')
            batched = np.load(root/f'{target}_waring_batched_b100'/f'values_{seed}.npy')
            close = np.allclose(serial, batched, rtol=1e-3, atol=1e-6)
            output_checks.append({
                'target': target, 'batch': 100, 'seed': seed, 'method': 'waring_serial',
                'reference': 'waring_batched', 'max_absolute': float(np.max(np.abs(serial-batched))),
                'relative_l2': float(np.linalg.norm(serial-batched)/max(np.linalg.norm(batched), 1e-30)),
                'consistent': bool(close),
            })
            if not close:
                raise AssertionError(output_checks[-1])
    summary = []
    for target, batch in sorted({(key[0], key[2]) for key in lookup}, key=lambda item: (item[1], len(item[0]), item[0])):
        nested = [lookup[(target, 'nested_jvp', batch, seed)]['median_ms'] for seed in SEEDS]
        waring = [lookup[(target, 'waring_batched', batch, seed)]['median_ms'] for seed in SEEDS]
        speeds = [left/right for left, right in zip(nested, waring)]
        summary.append({
            'target': target, 'batch': batch, 'order': len(target), 'directions': lookup[(target, 'nested_jvp', batch, SEEDS[0])]['rank'],
            'nested_mean_ms': statistics.mean(nested), 'nested_std_ms': statistics.stdev(nested),
            'nested_seed_medians_ms': ';'.join(f'{value:.9f}' for value in nested),
            'waring_batched_mean_ms': statistics.mean(waring), 'waring_batched_std_ms': statistics.stdev(waring),
            'waring_batched_seed_medians_ms': ';'.join(f'{value:.9f}' for value in waring),
            'paired_speedup_mean': statistics.mean(speeds),
            'paired_speedups': ';'.join(f'{value:.9f}' for value in speeds),
        })
    ablation = []
    for target in ('123456', '111222333'):
        serial = [lookup[(target, 'waring_serial', 100, seed)]['median_ms'] for seed in SEEDS]
        batched = [lookup[(target, 'waring_batched', 100, seed)]['median_ms'] for seed in SEEDS]
        ratios = [left/right for left, right in zip(serial, batched)]
        ablation.append({
            'target': target, 'batch': 100,
            'serial_mean_ms': statistics.mean(serial), 'serial_std_ms': statistics.stdev(serial),
            'serial_seed_medians_ms': ';'.join(f'{value:.9f}' for value in serial),
            'batched_mean_ms': statistics.mean(batched), 'batched_std_ms': statistics.stdev(batched),
            'batched_seed_medians_ms': ';'.join(f'{value:.9f}' for value in batched),
            'serial_to_batched_speedup_mean': statistics.mean(ratios),
            'serial_to_batched_speedups': ';'.join(f'{value:.9f}' for value in ratios),
        })
    # Per-seed records intentionally store only timing/value provenance.  The
    # immutable per-cell config is the authoritative hardware record.
    hardware = read(root/'111111_nested_jvp_b100'/'config.json')['device_kind']
    audit = {
        'status': 'PASS',
        'source_commit': manifest['source_commit'],
        'hardware': hardware,
        'raw_hash_manifest_files': raw_hash_files,
        'expected_cells': 22, 'observed_cells': len(expected_cells),
        'expected_records': 66, 'observed_records': len(rows),
        'expected_timed_calls': 1980, 'observed_timed_calls': sum(len(row['times_ms']) for row in rows),
        'source_file_hashes_match': True,
        'pre_timing_verification': verification,
        'all_output_comparisons_pass': all(item['consistent'] for item in output_checks),
        'output_comparisons': output_checks,
        'aggregation': 'mean and sample standard deviation of three per-seed medians; paired ratios use matching seeds',
        'formal_summary': summary,
        'direction_parallelism_ablation': ablation,
        'interpretation': 'value-only derivative evaluation throughput, not PINN training throughput',
    }
    if args.read_only:
        print(json.dumps(audit, indent=2, allow_nan=False))
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'audit_summary.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    (out/'output_validation.json').write_text(json.dumps(output_checks, indent=2, allow_nan=False)+'\n')
    write_csv(out/'summary.csv', summary)
    write_csv(out/'direction_parallelism_ablation.csv', ablation)
    report = [
        '# 固定高阶混合偏导：T4 / PyTorch eager 正式矩阵', '',
        f"- 源码提交：`{manifest['source_commit']}`。设备：{hardware}。",
        '- PyTorch 2.0.1+cu118，float32 实数输入；需要复根方向时方向和权重为 complex64。',
        '- 所有测量均为 value-only：不调用网络参数 backward，不含 PINN 训练、优化器或损失计算。',
        '- 无 torch.compile、无外层整函数 JIT、无第三方求导库。每个样本 10 次预热后取 30 次 CUDA 同步墙钟时间；冷启动另存且不计入。',
        f"- 覆盖：{len(expected_cells)}/22 单元，{len(rows)}/66 种子记录，{sum(len(row['times_ms']) for row in rows)}/1980 次正式计时；预计和实际完全一致。", '',
        '## 正式矩阵', '',
        '| 目标 | Batch | 阶数 | 方向数 | Coordinate nested JVP (ms) | Waring batched (ms) | 配对加速比 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary:
        report.append(
            f"| {row['target']} | {row['batch']} | {row['order']} | {row['directions']} | "
            f"{row['nested_mean_ms']:.4f} ± {row['nested_std_ms']:.4f} | "
            f"{row['waring_batched_mean_ms']:.4f} ± {row['waring_batched_std_ms']:.4f} | "
            f"{row['paired_speedup_mean']:.3f}x |"
        )
    report += ['', '三种子中位数与每个同种子配对比值完整保存在 `summary.csv`；本表的均值和标准差均从这三个中位数计算。', '', '## 同方向 serial / batched 消融', '', '| 目标 | Serial (ms) | Batched (ms) | Serial / batched |', '|---|---:|---:|---:|']
    for row in ablation:
        report.append(
            f"| {row['target']} | {row['serial_mean_ms']:.4f} ± {row['serial_std_ms']:.4f} | "
            f"{row['batched_mean_ms']:.4f} ± {row['batched_std_ms']:.4f} | {row['serial_to_batched_speedup_mean']:.3f}x |"
        )
    report += [
        '', '## 验证与边界', '',
        '- 正式计时前，六个目标均通过解析单项式的独立方向 Taylor 系数/阶乘归一化校验，并以 float64 小网络将 batched 与 serial Waring 值分别对照官方 coordinate nested JVP。',
        '- 正式矩阵中，所有 30 个 baseline-vs-batched 配对输出和 6 个 serial-vs-batched 配对输出均以 rtol=1e-3、atol=1e-6 通过；输入、参数、源码和输出哈希均被审计。',
        '- 方向数是固定 Waring 表示的代数计数，不能单独解释为运行时间最优性。本报告不将 T4/PyTorch 的绝对时间同 H20/JAX 结果混合或据此声称后端优劣。',
        '- 分配器峰值只是单进程诊断，不当作运行期显存峰值；性能数据保留全部单元，不筛选不占优结果。', '',
    ]
    (out/'REPORT.md').write_text('\n'.join(report))
    readme = [
        '# T4 / PyTorch 第一部分结果', '',
        '本目录为指定高阶混合偏导 value-only 矩阵的完整回收副本。它与第二部分共享 T4-B 与 PyTorch 2.0.1+cu118 的环境口径，但不重跑、修改或混合第二部分 PINN 实验。', '',
        '- `results/`：逐单元配置、每次同步计时、三种子输出数组、输入/参数/输出哈希及原始校验。',
        '- `REPORT.md`：正式中文矩阵和解释边界。',
        '- `summary.csv`、`direction_parallelism_ablation.csv`：保留三种子中位数与配对比值的机器可读表。',
        '- `audit_summary.json`、`output_validation.json`：独立审计；`SHA256SUMS` 是整个回收目录的传输完整性清单。', '',
        '该工作负载只测导数求值吞吐，明确不含参数 backward，也不是 PINN 训练吞吐。', '',
    ]
    (out/'README.md').write_text('\n'.join(readme))
    print(json.dumps({key: audit[key] for key in ('status', 'observed_cells', 'observed_records', 'observed_timed_calls')}, indent=2))


if __name__ == '__main__':
    main()
