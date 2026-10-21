"""Validate the actual V100 workloads before the exact-update campaign."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from equal_updates_campaign import METHODS, TARGETS

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0)
    assert torch.__version__ == '2.5.1+cu121' and np.__version__ == '1.26.4'
    environment = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True))
    (args.out/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    for case in ('kdv1d', 'kdv2d', 'ch2d'):
        pair = []
        for method in METHODS:
            root = args.out/f'{case}_{method}'
            subprocess.run([sys.executable, str(HERE/'train_equal_updates.py'),
                '--case', case, '--method', method, '--updates', '3',
                '--eval-every', '2', '--eval-points', '16', '--out', str(root)], check=True)
            obj = json.loads((root/'summary.json').read_text())
            assert obj['steps'] == 3 and obj['initial_hash'] != obj['final_hash']
            pair.append(obj)
        for key in ('initial_hash', 'interior_samples_sha256', 'constraint_samples_sha256',
                    'spacetime_test_sha256', 'terminal_test_sha256'):
            assert pair[0][key] == pair[1][key], (case, key)
    for target in TARGETS:
        for method in ('nested_jvp', 'waring_batched'):
            root = args.out/f'memory_{target}_{method}'
            subprocess.run([sys.executable, str(HERE/'measure_partial_memory.py'),
                '--target', target, '--method', method, '--seconds', '2',
                '--out', str(root)], check=True)
            obj = json.loads((root/'summary.json').read_text())
            assert obj['sample_count'] >= 2 and obj['mean_memory_mib'] > 0
    assert not [p for p in args.out.rglob('*') if p.suffix in ('.pt', '.pth', '.ckpt', '.npz')]
    (args.out/'PASS.json').write_text(json.dumps(dict(
        status='PASS', full_size_pde_workloads=6, memory_workloads=12,
        checkpoints_saved=False), indent=2)+'\n')
    print('V100_PREFLIGHT_PASS', flush=True)


if __name__ == '__main__':
    main()
