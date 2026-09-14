"""One-time, recoverable 2026-09-08 cleanup; explicit paths, checksum every move."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
legacy = 'archive/legacy_code/project/'
pairs = [(s, legacy + s) for s in (
    'src', 'tests', 'tests_active', 'scripts', 'configs', 'pyproject.toml',
    'README.md', 'METHOD_IMPLEMENTATION_REVIEW.md', 'progress.md',
    'experiments/README.md', 'experiments/benchmarks', 'experiments/torch_benchmarks',
    'experiments/stde_comparison', 'experiments/polyharmonic', 'experiments/chirp',
    'experiments/maxwell', 'experiments/cahn_hilliard_2d', 'experiments/common',
    'experiments/tools', 'experiments/logs', 'logs',
)]
pairs += [
    ('experiments/archived', 'archive/prior_archive'),
    ('experiments/results', 'archive/legacy_results/experiments/results'),
    ('outputs', 'archive/legacy_results/outputs'),
    ('tmp', 'archive/web_reviews/tmp'),
    ('docs/paper/tmp', 'archive/paper_previews/paper_tmp'),
    ('docs/beamer', 'archive/paper_previews/beamer'),
    ('docs/canvas', 'archive/paper_previews/canvas'),
    ('docs/archive', 'archive/paper_previews/prior_archive'),
    ('experiments/jax_benchmarks/plot_first_group.py', 'archive/legacy_code/jax_plot_first_group.py'),
]
for p in sorted((root/'output').iterdir()):
    if p.name.startswith('h20_'):
        dest = 'results/diagnostics/' if 'diagnosis' in p.name else 'results/historical/'
    elif p.name == 'pdf':
        dest = 'archive/paper_previews/output/'
    else:
        dest = 'archive/web_reviews/output/'
    pairs.append((str(p.relative_to(root)), dest + p.name))

manifest = root/'archive/MOVES_20260908.json'
if manifest.exists():
    raise FileExistsError('Cleanup already executed; use manifest for recovery')
for old,new in pairs:
    if not (root/old).exists() or (root/new).exists():
        raise ValueError((old,new))
records=[]
for old,new in pairs:
    src,dst=root/old,root/new
    files=sorted(p for p in src.rglob('*') if p.is_file()) if src.is_dir() else [src]
    hashes=[(str(p.relative_to(src)) if p != src else '', hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(src),str(dst))
    for rel,sha in hashes:
        p=dst/rel if rel else dst
        if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
            raise ValueError('checksum mismatch: '+str(p))
    records.append(dict(old=old,new=new,files=[dict(path=p,sha256=h) for p,h in hashes]))
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(dict(moved_groups=len(records),verified_files=sum(len(r['files']) for r in records))))
