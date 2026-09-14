"""Durable, single-GPU launcher for the fixed first-group T4 matrix.

This program intentionally has no scheduler logic.  It checks that the one
T4 is idle, records the actual environment, starts the one sequential matrix,
then audits and hashes the canonical output root.  It is designed to be run
under ``nohup`` by a short SSH submission command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def save_text(path: Path, text: str) -> None:
    path.write_text(text)


def run_capture(command: list[str], path: Path) -> str:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    save_text(path, result.stdout)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result.stdout


def full_sha256_manifest(root: Path) -> None:
    entries = []
    for file in sorted(root.rglob('*')):
        if file.is_file() and file.name != 'SHA256SUMS':
            entries.append(f'{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.relative_to(root)}')
    (root/'SHA256SUMS').write_text('\n'.join(entries)+'\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    code = args.code.resolve()
    runner = code/'experiments/torch_benchmarks/run_first_group_matrix.py'
    auditor = code/'experiments/torch_benchmarks/audit_first_group.py'
    if not root.is_dir() or (root/'results').exists():
        raise FileExistsError('root must exist and have no results directory')
    status = root/'status.txt'
    status.write_text('PREPARING\n')
    try:
        if subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() != args.commit:
            raise AssertionError('code worktree commit mismatch')
        subprocess.run(['git', '-C', str(code), 'diff', '--quiet'], check=True)
        gpu = run_capture(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version', '--format=csv,noheader'], root/'gpu-preflight.txt')
        apps = run_capture(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], root/'compute-apps-preflight.txt')
        if apps.strip():
            raise RuntimeError(f'GPU occupied; refusing formal launch: {apps.strip()}')
        runtime = run_capture([
            str(args.python), '-c',
            'import sys,torch,numpy; print(sys.executable); print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"); print(numpy.__version__)',
        ], root/'runtime-preflight.txt').splitlines()
        if len(runtime) < 6 or runtime[2] != '2.0.1+cu118' or runtime[3] != '11.8' or runtime[4] != 'True' or 'Tesla T4' not in runtime[5]:
            raise RuntimeError(f'unsupported formal runtime: {runtime}')
        run_capture([str(args.python), '-m', 'pip', 'freeze'], root/'requirements-lock.txt')
        (root/'provenance.json').write_text(json.dumps({
            'protocol': 'torch_first_group_values_v1', 'source_commit': args.commit,
            'code_worktree': str(code), 'canonical_root': str(root),
            'expected_cells': 20, 'expected_seed_records': 60, 'expected_timed_calls': 1800,
            'python': str(args.python), 'gpu_preflight': gpu.strip(),
            'workload': 'value-only fixed mixed partial derivatives; no parameter backward',
            'execution': 'eager; no torch.compile; no outer whole-function JIT',
        }, indent=2)+'\n')
        (root/'launcher.pid').write_text(f'{os.getpid()}\n')
        status.write_text('RUNNING\n')
        with (root/'launcher.log').open('w') as log:
            result = subprocess.run([str(args.python), str(runner), '--out', str(root/'results'), '--source-commit', args.commit],
                                    cwd=code, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.write_text('FAILED_MATRIX\n')
            full_sha256_manifest(root)
            raise SystemExit(result.returncode)
        with (root/'audit.log').open('w') as log:
            result = subprocess.run([str(args.python), str(auditor), '--root', str(root/'results'), '--source', str(code), '--out', str(root/'report')],
                                    cwd=code, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.write_text('FAILED_AUDIT\n')
            full_sha256_manifest(root)
            raise SystemExit(result.returncode)
        status.write_text('COMPLETE\n')
        full_sha256_manifest(root)
    except BaseException:
        if not status.exists() or status.read_text().strip() not in ('FAILED_MATRIX', 'FAILED_AUDIT'):
            status.write_text('FAILED_PREPARATION\n')
            full_sha256_manifest(root)
        raise


if __name__ == '__main__':
    main()
