"""Small reproducibility helpers; no framework or compiler switching."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import torch
import numpy as np

HERE=Path(__file__).resolve().parents[1]


def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)


def digest(tensors):
    h=hashlib.sha256()
    for t in tensors:
        a=t.detach().cpu().numpy()
        h.update(str((a.shape,str(a.dtype))).encode());h.update(a.tobytes())
    return h.hexdigest()


def configure(device, expected_device="T4"):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    if torch.device(device).type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        if expected_device not in ('T4','V100','RTX 3080'):
            raise ValueError('unsupported expected GPU family')
        if expected_device not in torch.cuda.get_device_name(device):
            raise RuntimeError(f'protocol requires {expected_device}')
    return torch.device(device)


def source_identity():
    """Allow a checksummed source export when cluster access has no Git remote."""
    manifest = HERE.parents[1] / 'SOURCE_MANIFEST.json'
    if manifest.is_file():
        import json
        obj = json.loads(manifest.read_text())
        expected = obj['files']
        for name, sha in expected.items():
            f = manifest.parent / name
            if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != sha:
                raise RuntimeError(f'source snapshot mismatch: {name}')
        actual = {str(f.relative_to(manifest.parent)) for f in HERE.rglob('*.py')}
        if actual != {name for name in expected if name.endswith('.py')}:
            raise RuntimeError('source snapshot Python inventory mismatch')
        return obj['source_commit'], 'sha256_verified_export'
    return subprocess.check_output(['git','-C',str(HERE),'rev-parse','HEAD'],text=True).strip(), 'git'


def provenance(args):
    sources={str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(HERE.rglob('*')) if f.suffix in ('.py','.md')}
    source_commit,source_kind=source_identity()
    return {**vars(args),'out':str(args.out),'torch':torch.__version__,'numpy':np.__version__,
            'python':platform.python_version(),'cuda_runtime':torch.version.cuda,
            'device_kind':torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else 'CPU',
            'source_commit':source_commit,'source_kind':source_kind,
            'source_sha256':sources,'execution':'eager','compile':False,'dtype':'float32',
            'tf32':False,'autocast':False,'threads':torch.get_num_threads(),
            'environment':{k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG','SLURM_JOB_ID','SLURM_ARRAY_TASK_ID','SLURM_JOB_GPUS')}}


def check(actual,expected):
    a,b=actual.detach(),expected.detach()
    if a.shape!=b.shape or not bool(torch.isfinite(a).all()):raise AssertionError('invalid tensor')
    torch.testing.assert_close(a,b,rtol=1e-3,atol=1e-6)
    return dict(max_abs=float((a-b).abs().max()),relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30)))


def seal(root):
    lines=[hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(root))
           for f in sorted(root.rglob('*')) if f.is_file() and f.name!='SHA256SUMS']
    (root/'SHA256SUMS').write_text('\n'.join(lines)+'\n')


def run_record(args,body):
    args.out.mkdir(parents=True,exist_ok=False)
    status=args.out/'status.txt'
    started=time.time()
    try:
        configure(args.device, getattr(args,"expected_device","T4"))
        save(args.out/'config.json',provenance(args))
        status.write_text('RUNNING\n')
        body(args)
        save(args.out/'completion.json',dict(elapsed_s=time.time()-started))
        status.write_text('COMPLETE\n')
    except Exception as exc:
        save(args.out/'failure.json',dict(error=repr(exc)))
        status.write_text('FAILED\n')
        raise
    finally:seal(args.out)
